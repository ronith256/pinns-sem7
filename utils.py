import torch
import torch.nn as nn
import numpy as np

class PhysicsLoss:
    def __init__(self, rho=997.07, mu=8.9e-4, k=0.606, cp=4200):
        self.rho = rho
        self.mu = mu
        self.k = k
        self.cp = cp
        self.nu = mu/rho

    def continuity_loss(self, u, v, x_grad, y_grad):
        # ∂u/∂x + ∂v/∂y = 0
        return torch.mean(torch.square(x_grad(u) + y_grad(v)))

    def momentum_x_loss(self, u, v, p, x_grad, y_grad, x_grad2, y_grad2, u_t):
        # ∂u/∂t + u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
        convection = u * x_grad(u) + v * y_grad(u)
        pressure = (1/self.rho) * x_grad(p)
        diffusion = self.nu * (x_grad2(u) + y_grad2(u))
        return torch.mean(torch.square(u_t + convection + pressure - diffusion))

    def momentum_y_loss(self, u, v, p, x_grad, y_grad, x_grad2, y_grad2, v_t):
        # Similar to momentum_x but for v component
        convection = u * x_grad(v) + v * y_grad(v)
        pressure = (1/self.rho) * y_grad(p)
        diffusion = self.nu * (x_grad2(v) + y_grad2(v))
        return torch.mean(torch.square(v_t + convection + pressure - diffusion))

    def energy_loss(self, T, u, v, x_grad, y_grad, x_grad2, y_grad2, T_t):
        # ρcp(∂T/∂t + u∂T/∂x + v∂T/∂y) = k(∂²T/∂x² + ∂²T/∂y²)
        convection = u * x_grad(T) + v * y_grad(T)
        diffusion = (self.k/(self.rho * self.cp)) * (x_grad2(T) + y_grad2(T))
        return torch.mean(torch.square(T_t + convection - diffusion))

def create_domain_points(nx, ny, L, H):
    """Create spatial domain points"""
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    return torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

def create_boundary_points(nx, ny, L, H):
    """Create boundary points for BC enforcement"""
    # Inlet (x = 0)
    x_inlet = np.zeros(ny)
    y_inlet = np.linspace(0, H, ny)
    inlet = np.stack([x_inlet, y_inlet], axis=1)
    
    # Outlet (x = L)
    x_outlet = np.ones(ny) * L
    y_outlet = np.linspace(0, H, ny)
    outlet = np.stack([x_outlet, y_outlet], axis=1)
    
    # Bottom wall (y = 0)
    x_bottom = np.linspace(0, L, nx)
    y_bottom = np.zeros(nx)
    bottom = np.stack([x_bottom, y_bottom], axis=1)
    
    # Top wall (y = H)
    x_top = np.linspace(0, L, nx)
    y_top = np.ones(nx) * H
    top = np.stack([x_top, y_top], axis=1)
    
    return torch.tensor(np.vstack([inlet, outlet, bottom, top]), dtype=torch.float32)