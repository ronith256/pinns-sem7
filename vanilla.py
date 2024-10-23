import torch
import torch.nn as nn
from typing import Tuple
from utils import * 

class VanillaPINN(nn.Module):
    def __init__(self, layers: list[int] = [3, 64, 64, 64, 4]):
        super().__init__()
        self.name = "Vanilla PINN"
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
        inlet_loss = (
            torch.mean(torch.square(u[inlet_mask] - self.compute_inlet_velocity())) +
            torch.mean(torch.square(v[inlet_mask])) +
            torch.mean(torch.square(T[inlet_mask] - 300))
        )
        
        # Wall conditions (y = 0 or y = H)
        wall_mask = (x_b[:, 1] == 0) | (x_b[:, 1] == self.H)
        wall_loss = (
            torch.mean(torch.square(u[wall_mask])) +
            torch.mean(torch.square(v[wall_mask]))
        )
        
        return inlet_loss + wall_loss

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
                
    def predict(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)