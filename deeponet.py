import torch
import torch.nn as nn
from typing import Tuple, Optional, List
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
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        try:
            # Ensure inputs are on the same device as the model
            device = next(self.parameters()).device
            x = x.to(device)
            t = t.to(device)
            
            # Combine spatial coordinates and time
            inputs = torch.cat([x, t.reshape(-1, 1)], dim=1)
            batch_size = inputs.shape[0]
            
            # Get branch and trunk outputs with proper reshaping
            branch_out = self.branch_net(inputs)
            trunk_out = self.trunk_net(inputs)
            
            # Reshape branch output for dot product
            branch_out = branch_out.reshape(batch_size, self.output_dim, self.hidden_dim)
            trunk_out = trunk_out.reshape(batch_size, self.hidden_dim, 1)
            
            # Compute dot product
            outputs = torch.bmm(branch_out, trunk_out).squeeze(-1)
            
            # Split outputs
            u = outputs[:, 0:1]
            v = outputs[:, 1:2]
            p = outputs[:, 2:3]
            T = outputs[:, 3:4]
            
            return u, v, p, T
            
        except Exception as e:
            print(f"Error in DeepONet forward pass: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=device)
            return zeros, zeros, zeros, zeros

class DeepONetPINNSolver:
    def __init__(self, domain_params: dict, physics_params: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        # Initialize model and move to device
        self.model = DeepONetPINN().to(self.device)
        self.physics_loss = PhysicsLoss(**physics_params).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=1000,
            verbose=True
        )
        
    def to(self, device: torch.device) -> 'DeepONetPINNSolver':
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
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Helper function for gradient computation"""
            return compute_gradient(y, x, allow_unused=True)
        
        try:
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
            
        except Exception as e:
            print(f"Error computing gradients: {str(e)}")
            zeros = torch.zeros_like(x[:, 0:1])
            return tuple([zeros] * 15)  # Return 15 zero tensors

    def boundary_loss(self, x_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss with device handling"""
        x_b = x_b.to(self.device).requires_grad_(True)
        t = t.to(self.device).requires_grad_(True)
        
        try:
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
            
        except Exception as e:
            print(f"Error computing boundary loss: {str(e)}")
            return torch.tensor(float('inf'), device=self.device)

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
        x_domain = x_domain.to(self.device)
        t_domain = t_domain.to(self.device)
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
            
            # Total loss with loss scaling for stability
            total_loss = sum(losses.values())
            
            # Backward pass and optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step(total_loss)
            
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
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                loss = self.train_step(x_domain, x_boundary, t_domain, t_boundary)
                losses.append(loss)
                
                # Early stopping logic
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Check for divergence or convergence
                if loss > 1e5 or patience_counter > 2000:
                    print("Training stopped early due to divergence or lack of improvement")
                    break
            
            return losses
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return []
        
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Make predictions with proper device handling"""
        self.model.eval()
        try:
            with torch.no_grad():
                x = x.to(self.device)
                t = t.to(self.device)
                u, v, p, T = self.model(x, t)
                
                # Check for numerical issues
                if any(check_nan_inf(tensor, name) 
                      for tensor, name in zip([u, v, p, T], ['u', 'v', 'p', 'T'])):
                    raise ValueError("NaN or Inf values detected in predictions")
                
                return u, v, p, T
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=self.device)
            return zeros, zeros, zeros, zeros
    
    def save_model(self, path: str):
        """Save model state with error handling"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'device': self.device,
                'L': self.L,
                'H': self.H,
                'Re': self.Re,
                'Q_flux': self.Q_flux
            }, path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, path: str):
        """Load model state with error handling"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load parameters
            self.L = checkpoint['L']
            self.H = checkpoint['H']
            self.Re = checkpoint['Re']
            self.Q_flux = checkpoint['Q_flux']
            
            # Move model to correct device
            self.to(self.device)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def validate(self, x_val: torch.Tensor, t_val: torch.Tensor) -> dict:
        """Validate model performance"""
        self.model.eval()
        try:
            with torch.no_grad():
                x_val = x_val.to(self.device)
                t_val = t_val.to(self.device)
                
                # Forward pass
                u, v, p, T = self.model(x_val, t_val)
                
                # Compute gradients for validation
                grads = self.compute_gradients(u, v, p, T, x_val, t_val)
                u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t = grads
                
                # Compute individual losses
                validation_losses = {
                    'continuity': self.physics_loss.continuity_loss(
                        u, v, lambda x: u_x, lambda x: v_y).item(),
                    'momentum_x': self.physics_loss.momentum_x_loss(
                        u, v, p, lambda x: u_x, lambda x: u_y,
                        lambda x: u_xx, lambda x: u_yy, u_t).item(),
                    'momentum_y': self.physics_loss.momentum_y_loss(
                        u, v, p, lambda x: v_x, lambda x: v_y,
                        lambda x: v_xx, lambda x: v_yy, v_t).item(),
                    'energy': self.physics_loss.energy_loss(
                        T, u, v, lambda x: T_x, lambda x: T_y,
                        lambda x: T_xx, lambda x: T_yy, T_t).item()
                }
                
                validation_losses['total'] = sum(validation_losses.values())
                return validation_losses
                
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return {'total': float('inf')}

    def compute_field_statistics(self, u: torch.Tensor, v: torch.Tensor, 
                               p: torch.Tensor, T: torch.Tensor) -> dict:
        """Compute statistics for the flow fields"""
        try:
            with torch.no_grad():
                stats = {
                    'velocity_magnitude_mean': torch.mean(torch.sqrt(u**2 + v**2)).item(),
                    'velocity_magnitude_max': torch.max(torch.sqrt(u**2 + v**2)).item(),
                    'pressure_mean': torch.mean(p).item(),
                    'pressure_range': (torch.min(p).item(), torch.max(p).item()),
                    'temperature_mean': torch.mean(T).item(),
                    'temperature_range': (torch.min(T).item(), torch.max(T).item())
                }
                return stats
        except Exception as e:
            print(f"Error computing field statistics: {str(e)}")
            return {}