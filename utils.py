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
    """Create boundary points for BC enforcement with proper ordering"""
    # Calculate number of points for each boundary
    n_inlet = ny  # Points on inlet
    n_outlet = ny  # Points on outlet
    n_bottom = nx  # Points on bottom wall
    n_top = nx     # Points on top wall
    
    # Create boundary points arrays
    inlet_points = torch.tensor([(0, y) for y in np.linspace(0, H, ny)], dtype=torch.float32)
    outlet_points = torch.tensor([(L, y) for y in np.linspace(0, H, ny)], dtype=torch.float32)
    bottom_points = torch.tensor([(x, 0) for x in np.linspace(0, L, nx)], dtype=torch.float32)
    top_points = torch.tensor([(x, H) for x in np.linspace(0, L, nx)], dtype=torch.float32)
    
    # Combine all boundary points
    boundary_points = torch.cat([inlet_points, outlet_points, bottom_points, top_points], dim=0)
    
    return boundary_points