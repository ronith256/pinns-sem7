import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class LaminarFlowPINN(nn.Module):
    def __init__(self, domain_params: dict):
        super().__init__()
        # Domain parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Dh = domain_params['Dh']
        
        # Physical parameters
        self.rho = 997.07  # Density
        self.mu = 8.9e-4   # Dynamic viscosity
        self.nu = self.mu / self.rho  # Kinematic viscosity
        self.k = 0.606     # Thermal conductivity
        self.cp = 4200     # Specific heat capacity
        self.Q_flux = 20000  # Heat flux
        self.T0 = 300      # Initial temperature
        
        # Neural network architecture
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # Input: (x, y, t)
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 4)   # Output: (u, v, p, T)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, t], dim=1)
        outputs = self.net(inputs)
        return outputs
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass with separate outputs"""
        outputs = self.forward(x, t)
        return outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
    
    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute spatial and temporal derivatives using autograd"""
        inputs = torch.cat([x, t], dim=1)
        inputs.requires_grad_(True)
        
        outputs = self.net(inputs)
        u, v, p, T = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        
        # First derivatives
        grads = torch.autograd.grad(
            [u.sum(), v.sum(), p.sum(), T.sum()],
            inputs,
            create_graph=True
        )
        
        u_x = grads[0][:, 0]
        u_y = grads[0][:, 1]
        u_t = grads[0][:, 2]
        
        v_x = grads[1][:, 0]
        v_y = grads[1][:, 1]
        v_t = grads[1][:, 2]
        
        p_x = grads[2][:, 0]
        p_y = grads[2][:, 1]
        
        T_x = grads[3][:, 0]
        T_y = grads[3][:, 1]
        T_t = grads[3][:, 2]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), inputs, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y.sum(), inputs, create_graph=True)[0][:, 1]
        
        v_xx = torch.autograd.grad(v_x.sum(), inputs, create_graph=True)[0][:, 0]
        v_yy = torch.autograd.grad(v_y.sum(), inputs, create_graph=True)[0][:, 1]
        
        T_xx = torch.autograd.grad(T_x.sum(), inputs, create_graph=True)[0][:, 0]
        T_yy = torch.autograd.grad(T_y.sum(), inputs, create_graph=True)[0][:, 1]
        
        return (u, v, p, T, 
                u_x, u_y, u_t, v_x, v_y, v_t, p_x, p_y, T_x, T_y, T_t,
                u_xx, u_yy, v_xx, v_yy, T_xx, T_yy)
    
    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute residuals for all governing equations"""
        derivatives = self.compute_derivatives(x, t)
        (u, v, p, T, 
         u_x, u_y, u_t, v_x, v_y, v_t, p_x, p_y, T_x, T_y, T_t,
         u_xx, u_yy, v_xx, v_yy, T_xx, T_yy) = derivatives
        
        # Continuity equation
        continuity = u_x + v_y
        
        # Momentum equations
        momentum_x = (u_t + u * u_x + v * u_y + 
                     (1/self.rho) * p_x - self.nu * (u_xx + u_yy))
        
        momentum_y = (v_t + u * v_x + v * v_y + 
                     (1/self.rho) * p_y - self.nu * (v_xx + v_yy))
        
        # Energy equation
        energy = (self.rho * self.cp * (T_t + u * T_x + v * T_y) - 
                 self.k * (T_xx + T_yy))
        
        return continuity, momentum_x, momentum_y, energy
    
    def compute_bc_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute residuals for boundary conditions"""
        # Get all variables and derivatives
        derivatives = self.compute_derivatives(x, t)
        (u, v, p, T, 
         u_x, u_y, u_t, v_x, v_y, v_t, p_x, p_y, T_x, T_y, T_t,
         u_xx, u_yy, v_xx, v_yy, T_xx, T_yy) = derivatives
        
        # Calculate inlet velocity based on Reynolds number
        u_in = self.Re * self.mu / (self.rho * self.Dh)
        
        # Create masks for different boundaries
        inlet_mask = torch.abs(x[:, 0]) < 1e-5
        outlet_mask = torch.abs(x[:, 0] - self.L) < 1e-5
        bottom_mask = torch.abs(x[:, 1]) < 1e-5
        top_mask = torch.abs(x[:, 1] - self.H) < 1e-5
        
        # Inlet conditions (x = 0)
        inlet_u = (u - u_in) * inlet_mask
        inlet_v = v * inlet_mask
        inlet_T = (T - self.T0) * inlet_mask
        inlet_p = (p - 101325) * inlet_mask  # Reference pressure
        
        # Outlet conditions (x = L)
        outlet_u = u_x * outlet_mask
        outlet_v = v_x * outlet_mask
        outlet_T = T_x * outlet_mask
        outlet_p = p_x * outlet_mask
        
        # Bottom wall conditions (y = 0)
        bottom_u = u * bottom_mask
        bottom_v = v * bottom_mask
        bottom_T = (self.k * T_y + self.Q_flux) * bottom_mask
        bottom_p = p_y * bottom_mask
        
        # Top wall conditions (y = H)
        top_u = u * top_mask
        top_v = v * top_mask
        top_T = T_y * top_mask
        top_p = p_y * top_mask
        
        return (
            inlet_u, inlet_v, inlet_T, inlet_p,
            outlet_u, outlet_v, outlet_T, outlet_p,
            bottom_u, bottom_v, bottom_T, bottom_p,
            top_u, top_v, top_T, top_p
        )
    
    def loss_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute total loss"""
        # PDE residuals
        continuity, momentum_x, momentum_y, energy = self.compute_pde_residuals(x, t)
        pde_loss = (torch.mean(continuity**2) + 
                   torch.mean(momentum_x**2) + 
                   torch.mean(momentum_y**2) + 
                   torch.mean(energy**2))
        
        # Boundary condition residuals
        bc_residuals = self.compute_bc_residuals(x, t)
        bc_loss = sum(torch.mean(residual**2) for residual in bc_residuals)
        
        return pde_loss + bc_loss

    def train_model(self, num_epochs: int, batch_size: int):
        """Train the PINN"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        # Generate training points
        x = torch.linspace(0, self.L, 100)
        y = torch.linspace(0, self.H, 20)
        t = torch.linspace(0, 30, 30)
        
        X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
        training_points = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
        
        # Training loop
        for epoch in range(num_epochs):
            # Mini-batch training
            indices = torch.randperm(training_points.shape[0])
            total_loss = 0
            num_batches = 0
            
            for i in range(0, training_points.shape[0], batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_points = training_points[batch_indices].to(self.device)
                
                optimizer.zero_grad()
                loss = self.loss_fn(batch_points[:, :2], batch_points[:, 2:3])
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            if epoch % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")