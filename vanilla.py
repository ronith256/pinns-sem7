import torch
import torch.nn as nn
from typing import Tuple
from utils import * 

class VanillaPINN(nn.Module):
    def __init__(self, layers: list[int] = [3, 64, 64, 64, 4]):
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

class VanillaPINNSolver:
    def __init__(self, domain_params, physics_params):
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        self.model = VanillaPINN()
        self.physics_loss = PhysicsLoss(**physics_params)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def compute_gradients(self, u, v, p, T, x, t):
        """Compute spatial and temporal gradients"""
        def grad(y, x, allow_unused=False):
            """Compute gradient of y with respect to x"""
            grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,
                                     create_graph=True, allow_unused=allow_unused)[0]
            return grad
        
        # First spatial derivatives
        u_x = grad(torch.sum(u), x, allow_unused=True)[:, 0:1]
        u_y = grad(torch.sum(u), x, allow_unused=True)[:, 1:2]
        v_x = grad(torch.sum(v), x, allow_unused=True)[:, 0:1]
        v_y = grad(torch.sum(v), x, allow_unused=True)[:, 1:2]
        T_x = grad(torch.sum(T), x, allow_unused=True)[:, 0:1]
        T_y = grad(torch.sum(T), x, allow_unused=True)[:, 1:2]
        
        # Second spatial derivatives
        u_xx = grad(torch.sum(u_x), x, allow_unused=True)[:, 0:1]
        u_yy = grad(torch.sum(u_y), x, allow_unused=True)[:, 1:2]
        v_xx = grad(torch.sum(v_x), x, allow_unused=True)[:, 0:1]
        v_yy = grad(torch.sum(v_y), x, allow_unused=True)[:, 1:2]
        T_xx = grad(torch.sum(T_x), x, allow_unused=True)[:, 0:1]
        T_yy = grad(torch.sum(T_y), x, allow_unused=True)[:, 1:2]
        
        # Time derivatives
        u_t = grad(torch.sum(u), t, allow_unused=True)
        v_t = grad(torch.sum(v), t, allow_unused=True)
        T_t = grad(torch.sum(T), t, allow_unused=True)
        
        return u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t

    def boundary_loss(self, x_b, t):
        """Compute boundary condition loss"""
        u, v, p, T = self.model(x_b, t)
        
        # Inlet conditions (x = 0)
        inlet_mask = x_b[:, 0] == 0
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
            T_grad = torch.autograd.grad(torch.sum(T[bottom_wall_mask]), x_b,
                                       create_graph=True, allow_unused=True)[0][:, 1]
            heat_flux_loss = torch.mean(torch.square(
                -self.physics_loss.k * T_grad - self.Q_flux))
        else:
            heat_flux_loss = torch.tensor(0.0)
        
        return inlet_loss + wall_loss + heat_flux_loss

    def compute_inlet_velocity(self):
        """Compute inlet velocity based on Reynolds number"""
        return self.Re * self.physics_loss.mu / (self.physics_loss.rho * self.H)
    
    def train_step(self, x_domain, x_boundary, t_domain, t_boundary):
        self.optimizer.zero_grad()
        
        # Ensure inputs require gradients
        x_domain.requires_grad_(True)
        t_domain.requires_grad_(True)
        
        # Forward pass for domain points
        u, v, p, T = self.model(x_domain, t_domain)
        
        # Compute gradients
        grads = self.compute_gradients(u, v, p, T, x_domain, t_domain)
        u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t = grads
        
        # Physics losses
        continuity = self.physics_loss.continuity_loss(u, v, lambda x: u_x, lambda x: v_y)
        momentum_x = self.physics_loss.momentum_x_loss(u, v, p, lambda x: u_x, lambda x: u_y,
                                                     lambda x: u_xx, lambda x: u_yy, u_t)
        momentum_y = self.physics_loss.momentum_y_loss(u, v, p, lambda x: v_x, lambda x: v_y,
                                                     lambda x: v_xx, lambda x: v_yy, v_t)
        energy = self.physics_loss.energy_loss(T, u, v, lambda x: T_x, lambda x: T_y,
                                             lambda x: T_xx, lambda x: T_yy, T_t)
        
        # Boundary losses using boundary time tensor
        bc_loss = self.boundary_loss(x_boundary, t_boundary)
        
        # Total loss
        loss = continuity + momentum_x + momentum_y + energy + bc_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, epochs, nx=501, ny=51):
        # Create domain and boundary points
        x_domain = create_domain_points(nx, ny, self.L, self.H)
        x_boundary = create_boundary_points(nx, ny, self.L, self.H)
        
        # Create time tensors with matching batch sizes
        t_domain = torch.zeros(x_domain.shape[0], 1)
        t_boundary = torch.zeros(x_boundary.shape[0], 1)
        
        losses = []
        for epoch in range(epochs):
            loss = self.train_step(x_domain, x_boundary, t_domain, t_boundary)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
                
    def predict(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)