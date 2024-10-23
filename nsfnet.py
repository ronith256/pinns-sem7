import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class DenseBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers with skip connections
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h_new = F.silu(layer(h))  # Using SiLU (Swish) activation
            h = h_new + h if h.shape == h_new.shape else h_new
        return h

class NSFNet(nn.Module):
    def __init__(self, domain_params: Dict[str, float]):
        super().__init__()
        self.L = domain_params['L']
        self.H = domain_params['H']
        
        # Physical parameters
        self.rho = 997.07  # Density
        self.mu = 8.9e-4   # Dynamic viscosity
        self.k = 0.606     # Thermal conductivity
        self.cp = 4200     # Specific heat capacity
        self.Q_flux = 20000  # Heat flux
        
        # Network architecture
        input_dim = 3  # x, y, t
        hidden_dim = 64
        num_dense_layers = 4
        
        # Encoding blocks
        self.encoder = nn.ModuleList([
            DenseBlock(input_dim, hidden_dim, num_dense_layers),
            DenseBlock(hidden_dim, hidden_dim, num_dense_layers)
        ])
        
        # Decoding heads for u, v, p, T
        self.u_head = nn.Linear(hidden_dim, 1)
        self.v_head = nn.Linear(hidden_dim, 1)
        self.p_head = nn.Linear(hidden_dim, 1)
        self.T_head = nn.Linear(hidden_dim, 1)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Normalize inputs
        x_norm = x[:, 0:1] / self.L
        y_norm = x[:, 1:2] / self.H
        t_norm = t / 30.0  # Normalize time by simulation duration
        
        # Combine inputs
        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        
        # Forward pass through encoder blocks
        h = inputs
        for block in self.encoder:
            h = block(h)
        
        # Generate outputs through heads
        u = self.u_head(h)
        v = self.v_head(h)
        p = self.p_head(h)
        T = self.T_head(h)
        
        return u, v, p, T
    
    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute PDE residuals for physics-informed training."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T = self.forward(x, t)
        
        # Calculate gradients
        grad_outputs = tuple(torch.ones_like(output) for output in (u, v, p, T))
        derivatives = torch.autograd.grad(
            outputs=(u, v, p, T),
            inputs=(x, t),
            grad_outputs=grad_outputs,
            create_graph=True
        )
        
        # Spatial derivatives
        dx = derivatives[0]
        dt = derivatives[1]
        
        # Extract individual derivatives
        u_x = dx[:, 0:1, 0:1]
        u_y = dx[:, 0:1, 1:2]
        v_x = dx[:, 1:2, 0:1]
        v_y = dx[:, 1:2, 1:2]
        p_x = dx[:, 2:3, 0:1]
        p_y = dx[:, 2:3, 1:2]
        T_x = dx[:, 3:4, 0:1]
        T_y = dx[:, 3:4, 1:2]
        
        # Time derivatives
        u_t = dt[:, 0:1]
        v_t = dt[:, 1:2]
        T_t = dt[:, 3:4]
        
        # Compute residuals
        # Continuity equation
        res_continuity = u_x + v_y
        
        # Momentum equations
        nu = self.mu / self.rho
        res_momentum_x = u_t + u * u_x + v * u_y + (1/self.rho) * p_x - nu * (u_x * u_x + u_y * u_y)
        res_momentum_y = v_t + u * v_x + v * v_y + (1/self.rho) * p_y - nu * (v_x * v_x + v_y * v_y)
        
        # Energy equation
        alpha = self.k / (self.rho * self.cp)
        res_energy = T_t + u * T_x + v * T_y - alpha * (T_x * T_x + T_y * T_y)
        
        return torch.cat([
            res_continuity,
            res_momentum_x,
            res_momentum_y,
            res_energy
        ], dim=1)
    
    def compute_bc_residuals(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition residuals."""
        # Inlet conditions (x = 0)
        inlet_mask = x[:, 0] < 1e-5
        u_in = 200 * self.mu / (self.rho * 0.00116)  # Based on Re = 200
        inlet_res = torch.zeros_like(x[:, 0:1])
        if inlet_mask.any():
            u, v, _, T = self.forward(x[inlet_mask], t[inlet_mask])
            inlet_res[inlet_mask] = torch.cat([
                u - u_in,
                v,
                T - 300.0
            ], dim=1)
        
        # Wall conditions (y = 0 or y = H)
        wall_mask = (x[:, 1] < 1e-5) | (torch.abs(x[:, 1] - self.H) < 1e-5)
        wall_res = torch.zeros_like(x[:, 0:1])
        if wall_mask.any():
            u, v, _, _ = self.forward(x[wall_mask], t[wall_mask])
            wall_res[wall_mask] = torch.cat([u, v], dim=1)
        
        return torch.cat([inlet_res, wall_res], dim=1)
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Make predictions for visualization."""
        return self.forward(x, t)

def train_nsfnet(model: NSFNet, num_epochs: int = 1000) -> NSFNet:
    """Train the NSFNet model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Generate training points
    num_points = 10000
    x = torch.linspace(0, model.L, 100, device=model.device)
    y = torch.linspace(0, model.H, 50, device=model.device)
    t = torch.linspace(0, 30, 30, device=model.device)
    
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    training_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    training_times = T.flatten().unsqueeze(1)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute losses
        pde_residuals = model.compute_pde_residuals(training_points, training_times)
        bc_residuals = model.compute_bc_residuals(training_points, training_times)
        
        pde_loss = torch.mean(torch.square(pde_residuals))
        bc_loss = torch.mean(torch.square(bc_residuals))
        
        total_loss = pde_loss + bc_loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step(total_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.6f}")
    
    return model