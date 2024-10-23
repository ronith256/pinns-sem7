import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class PhysicsLoss:
    def __init__(self, rho=997.07, mu=8.9e-4, k=0.606, cp=4200, device=None):
        # Initialize parameters as tensors on the specified device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rho = torch.tensor(rho, dtype=torch.float32, device=self.device)
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.k = torch.tensor(k, dtype=torch.float32, device=self.device)
        self.cp = torch.tensor(cp, dtype=torch.float32, device=self.device)
        self.nu = self.mu / self.rho

    def to(self, device: torch.device) -> 'PhysicsLoss':
        """Move all tensors to specified device"""
        self.device = device
        self.rho = self.rho.to(device)
        self.mu = self.mu.to(device)
        self.k = self.k.to(device)
        self.cp = self.cp.to(device)
        self.nu = self.nu.to(device)
        return self

    def continuity_loss(self, u: torch.Tensor, v: torch.Tensor, 
                       x_grad: callable, y_grad: callable) -> torch.Tensor:
        """Compute continuity equation loss: ∂u/∂x + ∂v/∂y = 0"""
        try:
            divergence = x_grad(u) + y_grad(v)
            return torch.mean(torch.square(divergence))
        except RuntimeError as e:
            print(f"Error in continuity loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=self.device)

    def momentum_x_loss(self, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                       x_grad: callable, y_grad: callable, 
                       x_grad2: callable, y_grad2: callable,
                       u_t: torch.Tensor) -> torch.Tensor:
        """Compute x-momentum equation loss"""
        try:
            convection = u * x_grad(u) + v * y_grad(u)
            pressure = (1/self.rho) * x_grad(p)
            diffusion = self.nu * (x_grad2(u) + y_grad2(u))
            residual = u_t + convection + pressure - diffusion
            return torch.mean(torch.square(residual))
        except RuntimeError as e:
            print(f"Error in x-momentum loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=self.device)

    def momentum_y_loss(self, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                       x_grad: callable, y_grad: callable, 
                       x_grad2: callable, y_grad2: callable,
                       v_t: torch.Tensor) -> torch.Tensor:
        """Compute y-momentum equation loss"""
        try:
            convection = u * x_grad(v) + v * y_grad(v)
            pressure = (1/self.rho) * y_grad(p)
            diffusion = self.nu * (x_grad2(v) + y_grad2(v))
            residual = v_t + convection + pressure - diffusion
            return torch.mean(torch.square(residual))
        except RuntimeError as e:
            print(f"Error in y-momentum loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=self.device)

    def energy_loss(self, T: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                   x_grad: callable, y_grad: callable,
                   x_grad2: callable, y_grad2: callable,
                   T_t: torch.Tensor) -> torch.Tensor:
        """Compute energy equation loss"""
        try:
            convection = u * x_grad(T) + v * y_grad(T)
            diffusion = (self.k/(self.rho * self.cp)) * (x_grad2(T) + y_grad2(T))
            residual = T_t + convection - diffusion
            return torch.mean(torch.square(residual))
        except RuntimeError as e:
            print(f"Error in energy loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=self.device)

def create_domain_points(nx: int, ny: int, L: float, H: float, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
    """Create spatial domain points on specified device"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    try:
        x = np.linspace(0, L, nx)
        y = np.linspace(0, H, ny)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        return torch.tensor(points, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"Error creating domain points: {str(e)}")
        raise

def create_boundary_points(nx: int, ny: int, L: float, H: float,
                         device: Optional[torch.device] = None) -> torch.Tensor:
    """Create boundary points with proper ordering on specified device"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    try:
        # Create boundary points arrays
        inlet_points = torch.tensor([(0, y) for y in np.linspace(0, H, ny)],
                                  dtype=torch.float32, device=device)
        outlet_points = torch.tensor([(L, y) for y in np.linspace(0, H, ny)],
                                   dtype=torch.float32, device=device)
        bottom_points = torch.tensor([(x, 0) for x in np.linspace(0, L, nx)],
                                   dtype=torch.float32, device=device)
        top_points = torch.tensor([(x, H) for x in np.linspace(0, L, nx)],
                                dtype=torch.float32, device=device)
        
        # Combine all boundary points
        return torch.cat([inlet_points, outlet_points, bottom_points, top_points], dim=0)
    except Exception as e:
        print(f"Error creating boundary points: {str(e)}")
        raise

def compute_gradient(y: torch.Tensor, x: torch.Tensor, 
                    grad_outputs: Optional[torch.Tensor] = None,
                    allow_unused: bool = False) -> torch.Tensor:
    """Safely compute gradient of y with respect to x"""
    try:
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,
                                 create_graph=True, allow_unused=allow_unused)[0]
        return grad if grad is not None else torch.zeros_like(x)
    except RuntimeError as e:
        print(f"Error computing gradient: {str(e)}")
        return torch.zeros_like(x)

def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f"Warning: NaN values detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"Warning: Inf values detected in {name}")
        return True
    return False

def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, 
                eps: float = 1e-8) -> torch.Tensor:
    """Safely divide tensors avoiding division by zero"""
    return numerator / (denominator + eps)

def moving_average(values: list, window: int = 100) -> np.ndarray:
    """Compute moving average of values"""
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')