import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from utils import *

class VanillaPINN(nn.Module):
    def __init__(self, layers: List[int] = [3, 64, 64, 64, 4]):
        super().__init__()
        self.layers = layers
        
        # Build the neural network
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                modules.append(nn.Tanh())
        
        self.network = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Ensure inputs are on the same device as the model
        x = x.to(self.network[0].weight.device)
        t = t.to(self.network[0].weight.device)
        
        # Combine spatial coordinates and time
        inputs = torch.cat([x, t.reshape(-1, 1)], dim=1)
        
        # Get network output
        outputs = self.network(inputs)
        
        # Split outputs into velocity components, pressure, and temperature
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        T = outputs[:, 3:4]
        
        return u, v, p, T
    
    def to(self, device: torch.device) -> 'VanillaPINN':
        """Move model to specified device"""
        super().to(device)
        return self

class VanillaPINNSolver:
    def __init__(self, domain_params: dict, physics_params: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        # Initialize model and move to device
        self.model = VanillaPINN().to(self.device)
        self.physics_loss = PhysicsLoss(**physics_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def to(self, device: torch.device) -> 'VanillaPINNSolver':
        """Move solver to specified device"""
        self.device = device
        self.model = self.model.to(device)
        self.physics_loss = self.physics_loss.to(device)
        return self
        
    def compute_gradients(self, u: torch.Tensor, v: torch.Tensor, 
                         p: torch.Tensor, T: torch.Tensor, 
                         x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute spatial and temporal gradients with proper device handling"""
        # Ensure inputs require gradients
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Helper function for gradient computation"""
            return compute_gradient(y, x, allow_unused=True)
        
        # First derivatives
        u_x = grad(u.sum(), x)[:, 0:1]
        u_y = grad(u.sum(), x)[:, 1:2]
        v_x = grad(v.sum(), x)[:, 0:1]
        v_y = grad(v.sum(), x)[:, 1:2]
        T_x = grad(T.sum(), x)[:, 0:1]
        T_y = grad(T.sum(), x)[:, 1:2]
        
        # Second derivatives
        u_xx = grad(u_x.sum(), x)[:, 0:1]
        u_yy = grad(u_y.sum(), x)[:, 1:2]
        v_xx = grad(v_x.sum(), x)[:, 0:1]
        v_yy = grad(v_y.sum(), x)[:, 1:2]
        T_xx = grad(T_x.sum(), x)[:, 0:1]
        T_yy = grad(T_y.sum(), x)[:, 1:2]
        
        # Time derivatives
        u_t = grad(u.sum(), t)
        v_t = grad(v.sum(), t)
        T_t = grad(T.sum(), t)
        
        return u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t

    def boundary_loss(self, x_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss with device handling"""
        # Ensure tensors are on correct device and require gradients
        x_b = x_b.to(self.device).requires_grad_(True)
        t = t.to(self.device).requires_grad_(True)
        
        u, v, p, T = self.model(x_b, t)
        
        # Inlet conditions (x = 0)
        inlet_mask = (x_b[:, 0] == 0)
        u_in = self.compute_inlet_velocity()
        inlet_loss = (
            torch.mean(torch.square(u[inlet_mask] - u_in)) +
            torch.mean(torch.square(v[inlet_mask])) +
            torch.mean(torch.square(T[inlet_mask] - 300))
        )
        
        # Wall conditions (y = 0 or y = H)
        wall_mask = (x_b[:, 1] == 0) | (x_b[:, 1] == self.H)
        wall_loss = (
            torch.mean(torch.square(u[wall_mask])) +
            torch.mean(torch.square(v[wall_mask]))
        )
        
        # Bottom wall heat flux condition
        bottom_wall_mask = (x_b[:, 1] == 0)
        if torch.any(bottom_wall_mask):
            T_masked = T[bottom_wall_mask]
            if T_masked.requires_grad:
                T_grad = compute_gradient(T_masked.sum(), x_b, allow_unused=True)
                if T_grad is not None:
                    heat_flux_loss = torch.mean(torch.square(
                        -self.physics_loss.k * T_grad[:, 1] - self.Q_flux
                    ))
                else:
                    heat_flux_loss = torch.tensor(0.0, device=self.device)
            else:
                heat_flux_loss = torch.tensor(0.0, device=self.device)
        else:
            heat_flux_loss = torch.tensor(0.0, device=self.device)
        
        return inlet_loss + wall_loss + heat_flux_loss

    def compute_inlet_velocity(self) -> torch.Tensor:
        """Compute inlet velocity based on Reynolds number"""
        return (self.Re * self.physics_loss.mu / 
                (self.physics_loss.rho * self.H)).to(self.device)
    
    def train_step(self, x_domain: torch.Tensor, x_boundary: torch.Tensor, 
                  t_domain: torch.Tensor, t_boundary: torch.Tensor) -> float:
        """Perform one training step with proper device handling"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move inputs to device
        x_domain = x_domain.to(self.device).requires_grad_(True)
        t_domain = t_domain.to(self.device).requires_grad_(True)
        x_boundary = x_boundary.to(self.device)
        t_boundary = t_boundary.to(self.device)
        
        try:
            # Forward pass
            u, v, p, T = self.model(x_domain, t_domain)
            
            # Check for numerical issues
            if any(check_nan_inf(tensor, name) 
                  for tensor, name in zip([u, v, p, T], ['u', 'v', 'p', 'T'])):
                raise ValueError("NaN or Inf values detected in model outputs")
            
            # Compute gradients
            grads = self.compute_gradients(u, v, p, T, x_domain, t_domain)
            u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t = grads
            
            # Physics losses
            losses = {
                'continuity': self.physics_loss.continuity_loss(
                    u, v, lambda x: u_x, lambda x: v_y),
                'momentum_x': self.physics_loss.momentum_x_loss(
                    u, v, p, lambda x: u_x, lambda x: u_y,
                    lambda x: u_xx, lambda x: u_yy, u_t),
                'momentum_y': self.physics_loss.momentum_y_loss(
                    u, v, p, lambda x: v_x, lambda x: v_y,
                    lambda x: v_xx, lambda x: v_yy, v_t),
                'energy': self.physics_loss.energy_loss(
                    T, u, v, lambda x: T_x, lambda x: T_y,
                    lambda x: T_xx, lambda x: T_yy, T_t),
                'boundary': self.boundary_loss(x_boundary, t_boundary)
            }
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return float('inf')

    def train(self, epochs: int, nx: int = 501, ny: int = 51) -> List[float]:
        """Train the model with proper device handling"""
        try:
            # Create domain and boundary points on device
            x_domain = create_domain_points(nx, ny, self.L, self.H, self.device)
            x_boundary = create_boundary_points(nx, ny, self.L, self.H, self.device)
            
            # Create time tensors
            t_domain = torch.zeros(x_domain.shape[0], 1, device=self.device)
            t_boundary = torch.zeros(x_boundary.shape[0], 1, device=self.device)
            
            losses = []
            for epoch in range(epochs):
                loss = self.train_step(x_domain, x_boundary, t_domain, t_boundary)
                losses.append(loss)
                
                if epoch:
                    print(f"Epoch {epoch}, Loss: {loss:.6f}")
                    # Check for divergence
                    if loss > 1e5:
                        print("Training diverged, stopping early")
                        break
            
            return losses
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return []
                
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Make predictions with proper device handling"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            t = t.to(self.device)
            return self.model(x, t)