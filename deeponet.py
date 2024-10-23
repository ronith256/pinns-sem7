import torch
import torch.nn as nn
from typing import Tuple
from utils import *

class BranchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepONetPINN(nn.Module):
    def __init__(self, branch_input_dim: int = 3, trunk_input_dim: int = 3,
                 hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        
        self.branch_net = BranchNet(branch_input_dim, hidden_dim, output_dim * hidden_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dim, output_dim)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Combine spatial coordinates and time for both networks
        inputs = torch.cat([x, t.reshape(-1, 1)], dim=1)
        
        # Get branch and trunk outputs
        branch_out = self.branch_net(inputs)
        trunk_out = self.trunk_net(inputs)
        
        # Reshape branch output for dot product
        branch_out = branch_out.reshape(-1, self.output_dim, self.hidden_dim)
        trunk_out = trunk_out.reshape(-1, self.hidden_dim, 1)
        
        # Compute dot product
        outputs = torch.bmm(branch_out, trunk_out).squeeze(-1)
        
        # Split outputs into velocity components, pressure, and temperature
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        T = outputs[:, 3:4]
        
        return u, v, p, T

class DeepONetPINNSolver:
    def __init__(self, domain_params, physics_params):
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        self.model = DeepONetPINN()
        self.physics_loss = PhysicsLoss(**physics_params)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def compute_gradients(self, u, v, p, T, x, t):
        """Compute spatial and temporal gradients"""
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        def grad(y, x):
            return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                    create_graph=True)[0]
        
        # First derivatives
        u_x = grad(u, x)[:, 0:1]
        u_y = grad(u, x)[:, 1:2]
        v_x = grad(v, x)[:, 0:1]
        v_y = grad(v, x)[:, 1:2]
        T_x = grad(T, x)[:, 0:1]
        T_y = grad(T, x)[:, 1:2]
        
        # Second derivatives
        u_xx = grad(u_x, x)[:, 0:1]
        u_yy = grad(u_y, x)[:, 1:2]
        v_xx = grad(v_x, x)[:, 0:1]
        v_yy = grad(v_y, x)[:, 1:2]
        T_xx = grad(T_x, x)[:, 0:1]
        T_yy = grad(T_y, x)[:, 1:2]
        
        # Time derivatives
        u_t = grad(u, t)
        v_t = grad(v, t)
        T_t = grad(T, t)
        
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
        T_grad = torch.autograd.grad(T[bottom_wall_mask].sum(), x_b,
                                   create_graph=True)[0][:, 1]
        heat_flux_loss = torch.mean(torch.square(
            -self.physics_loss.k * T_grad - self.Q_flux))
        
        return inlet_loss + wall_loss + heat_flux_loss

    def compute_inlet_velocity(self):
        """Compute inlet velocity based on Reynolds number"""
        return self.Re * self.physics_loss.mu / (self.physics_loss.rho * self.H)

    def train_step(self, x_domain, x_boundary, t):
        self.optimizer.zero_grad()
        
        # Forward pass for domain points
        u, v, p, T = self.model(x_domain, t)
        
        # Compute gradients
        grads = self.compute_gradients(u, v, p, T, x_domain, t)
        u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t = grads
        
        # Physics losses
        continuity = self.physics_loss.continuity_loss(u, v, lambda x: u_x, lambda x: v_y)
        momentum_x = self.physics_loss.momentum_x_loss(u, v, p, lambda x: u_x, lambda x: u_y,
                                                     lambda x: u_xx, lambda x: u_yy, u_t)
        momentum_y = self.physics_loss.momentum_y_loss(u, v, p, lambda x: v_x, lambda x: v_y,
                                                     lambda x: v_xx, lambda x: v_yy, v_t)
        energy = self.physics_loss.energy_loss(T, u, v, lambda x: T_x, lambda x: T_y,
                                             lambda x: T_xx, lambda x: T_yy, T_t)
        
        # Boundary losses
        bc_loss = self.boundary_loss(x_boundary, t)
        
        # Total loss
        loss = continuity + momentum_x + momentum_y + energy + bc_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, epochs, nx=501, ny=51):
        x_domain = create_domain_points(nx, ny, self.L, self.H)
        x_boundary = create_boundary_points(nx, ny, self.L, self.H)
        t = torch.zeros(x_domain.shape[0], 1)
        history = []
        for epoch in range(epochs):
            loss = self.train_step(x_domain, x_boundary, t)
            history.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return history         
    def predict(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)