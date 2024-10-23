import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from utils import *

class StyleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights for better convergence"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.conv(x)
            x = self.norm(x)
            return F.relu(x)
        except RuntimeError as e:
            print(f"Error in StyleLayer forward pass: {str(e)}")
            return torch.zeros_like(x)

class NSFNet(nn.Module):
    def __init__(self, input_channels: int = 3, H: int = 51, W: int = 501):
        super().__init__()
        self.H = H
        self.W = W
        
        # Encoder
        self.enc1 = StyleLayer(input_channels, 64)
        self.enc2 = StyleLayer(64, 128)
        self.enc3 = StyleLayer(128, 256)
        
        # Flow-specific layers with residual connections
        self.flow1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.flow2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.flow_norm = nn.InstanceNorm2d(256)
        
        # Decoder with skip connections
        self.dec3 = StyleLayer(256 + 128, 128)  # Added input channels for skip connection
        self.dec2 = StyleLayer(128 + 64, 64)    # Added input channels for skip connection
        self.dec1 = nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 4 channels for u, v, p, T
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights for better convergence"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        try:
            # Ensure inputs are on the correct device
            device = next(self.parameters()).device
            x = x.to(device)
            t = t.to(device)
            
            # Reshape input to match convolution expectations
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, self.H, self.W)
            t = t.reshape(batch_size, 1, 1, 1).expand(-1, -1, self.H, self.W)
            x = torch.cat([x, t], dim=1)
            
            # Encoder with skip connections
            e1 = self.enc1(x)
            e1 = self.dropout(e1)
            
            e2 = self.enc2(e1)
            e2 = self.dropout(e2)
            
            e3 = self.enc3(e2)
            e3 = self.dropout(e3)
            
            # Flow processing with residual connection
            f = F.relu(self.flow_norm(self.flow1(e3)))
            f = F.relu(self.flow_norm(self.flow2(f)))
            f = f + e3  # Residual connection
            
            # Decoder with skip connections
            d3 = self.dec3(torch.cat([f, e2], dim=1))
            d2 = self.dec2(torch.cat([d3, e1], dim=1))
            d1 = self.dec1(d2)
            
            # Check for numerical issues
            if check_nan_inf(d1, "decoder_output"):
                raise ValueError("NaN or Inf values detected in network output")
            
            # Split outputs and reshape
            u = d1[:, 0:1].reshape(batch_size, -1)
            v = d1[:, 1:2].reshape(batch_size, -1)
            p = d1[:, 2:3].reshape(batch_size, -1)
            T = d1[:, 3:4].reshape(batch_size, -1)
            
            return u, v, p, T
            
        except Exception as e:
            print(f"Error in NSFNet forward pass: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=device)
            return zeros, zeros, zeros, zeros

class NSFNetSolver:
    def __init__(self, domain_params: dict, physics_params: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        # Initialize model and move to device
        self.model = NSFNet().to(self.device)
        self.physics_loss = PhysicsLoss(**physics_params).to(self.device)
        
        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=500,
            verbose=True,
            min_lr=1e-6
        )
        
        # Initialize loss weights
        self.loss_weights = {
            'continuity': 1.0,
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'energy': 1.0,
            'boundary': 1.0
        }
        
    def to(self, device: torch.device) -> 'NSFNetSolver':
        """Move solver to specified device"""
        self.device = device
        self.model = self.model.to(device)
        self.physics_loss = self.physics_loss.to(device)
        return self

    def compute_gradients(self, u: torch.Tensor, v: torch.Tensor,
                         p: torch.Tensor, T: torch.Tensor,
                         x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute spatial and temporal gradients with proper device handling"""
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        try:
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
            
        except Exception as e:
            print(f"Error computing gradients: {str(e)}")
            zeros = torch.zeros_like(x[:, 0:1])
            return tuple([zeros] * 15)

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
                  t_domain: torch.Tensor, t_boundary: torch.Tensor) -> dict:
        """Perform one training step with proper device handling and detailed loss tracking"""
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
            
            # Compute individual losses with weights
            losses = {
                'continuity': self.loss_weights['continuity'] * self.physics_loss.continuity_loss(
                    u, v, lambda x: u_x, lambda x: v_y),
                'momentum_x': self.loss_weights['momentum_x'] * self.physics_loss.momentum_x_loss(
                    u, v, p, lambda x: u_x, lambda x: u_y,
                    lambda x: u_xx, lambda x: u_yy, u_t),
                'momentum_y': self.loss_weights['momentum_y'] * self.physics_loss.momentum_y_loss(
                    u, v, p, lambda x: v_x, lambda x: v_y,
                    lambda x: v_xx, lambda x: v_yy, v_t),
                'energy': self.loss_weights['energy'] * self.physics_loss.energy_loss(
                    T, u, v, lambda x: T_x, lambda x: T_y,
                    lambda x: T_xx, lambda x: T_yy, T_t),
                'boundary': self.loss_weights['boundary'] * self.boundary_loss(x_boundary, t_boundary)
            }
            
            # Compute total loss
            total_loss = sum(losses.values())
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step(total_loss)
            
            # Convert losses to float for logging
            losses = {k: v.item() for k, v in losses.items()}
            losses['total'] = total_loss.item()
            
            return losses
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return {k: float('inf') for k in ['continuity', 'momentum_x', 'momentum_y', 
                                            'energy', 'boundary', 'total']}

    def train(self, epochs: int, nx: int = 501, ny: int = 51,
              validation_freq: int = 100) -> Tuple[List[dict], List[dict]]:
        """Train the model with validation and early stopping"""
        try:
            # Create domain and boundary points
            x_domain = create_domain_points(nx, ny, self.L, self.H, self.device)
            x_boundary = create_boundary_points(nx, ny, self.L, self.H, self.device)
            
            # Create time tensors
            t_domain = torch.zeros(x_domain.shape[0], 1, device=self.device)
            t_boundary = torch.zeros(x_boundary.shape[0], 1, device=self.device)
            
            # Create validation points (using a subset of domain points)
            val_indices = torch.randperm(x_domain.shape[0])[:1000]
            x_val = x_domain[val_indices]
            t_val = t_domain[val_indices]
            
            # Initialize tracking variables
            train_history = []
            val_history = []
            best_val_loss = float('inf')
            patience = 1000
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training step
                train_losses = self.train_step(x_domain, x_boundary, t_domain, t_boundary)
                train_history.append(train_losses)
                
                # Validation step
                if epoch % validation_freq == 0:
                    val_losses = self.validate(x_val, t_val)
                    val_history.append(val_losses)
                    
                    # Early stopping check
                    if val_losses['total'] < best_val_loss:
                        best_val_loss = val_losses['total']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Print progress
                    print(f"Epoch {epoch}")
                    print(f"Training Loss: {train_losses['total']:.6f}")
                    print(f"Validation Loss: {val_losses['total']:.6f}")
                    print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Check stopping conditions
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
                
                if train_losses['total'] > 1e5:
                    print("Training diverged")
                    break
            
            return train_history, val_history
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return [], []

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
                losses = {
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
                
                losses['total'] = sum(losses.values())
                return losses
                
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return {k: float('inf') for k in ['continuity', 'momentum_x', 'momentum_y', 
                                            'energy', 'total']}

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
                'loss_weights': self.loss_weights,
                'device': self.device,
                'domain_params': {
                    'L': self.L,
                    'H': self.H,
                    'Re': self.Re,
                    'Q_flux': self.Q_flux
                }
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
            self.loss_weights = checkpoint['loss_weights']
            
            # Load domain parameters
            params = checkpoint['domain_params']
            self.L = params['L']
            self.H = params['H']
            self.Re = params['Re']
            self.Q_flux = params['Q_flux']
            
            # Move model to correct device
            self.to(self.device)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")