import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils import * 

class StyleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))

class NSFNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = StyleLayer(input_channels, 64)
        self.enc2 = StyleLayer(64, 128)
        self.enc3 = StyleLayer(128, 256)
        
        # Flow-specific layers
        self.flow1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.flow2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Decoder
        self.dec3 = StyleLayer(256, 128)
        self.dec2 = StyleLayer(128, 64)
        self.dec1 = nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 4 channels for u, v, p, T
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Reshape input to match convolution expectations
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.H, self.W)
        t = t.reshape(batch_size, 1, 1, 1).expand(-1, -1, self.H, self.W)
        x = torch.cat([x, t], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Flow processing
        f = F.relu(self.flow1(e3))
        f = F.relu(self.flow2(f))
        
        # Decoder
        d3 = self.dec3(f)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        # Split outputs and reshape
        u = d1[:, 0:1].reshape(batch_size, -1)
        v = d1[:, 1:2].reshape(batch_size, -1)
        p = d1[:, 2:3].reshape(batch_size, -1)
        T = d1[:, 3:4].reshape(batch_size, -1)
        
        return u, v, p, T

class NSFNetSolver:
    def __init__(self, domain_params, physics_params):
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        self.model = NSFNet()
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