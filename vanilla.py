import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from utils import *

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm1(x)
        x = F.silu(x)  # SiLU/Swish activation for better gradient flow
        x = self.linear1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x + identity

class VanillaPINN(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, num_blocks: int = 4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output heads with separate networks for better specialization
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2)  # u, v
        )
        
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)  # p
        )
        
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)  # T
        )
        
        # Initialize weights carefully
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Smaller initialization for better gradient flow
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        try:
            # Ensure inputs are on the same device as the model
            device = next(self.parameters()).device
            x = x.to(device)
            t = t.to(device)
            
            # Compute normalization statistics for current batch
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-8
            t_mean = t.mean()
            t_std = t.std() + 1e-8
            
            # Normalize inputs
            x_normalized = (x - x_mean) / x_std
            t_normalized = (t - t_mean) / t_std
            
            # Combine inputs
            inputs = torch.cat([x_normalized, t_normalized.reshape(-1, 1)], dim=1)
            
            # Input normalization and projection
            x = self.input_norm(inputs)
            x = self.input_proj(x)
            
            # Process through residual blocks
            for block in self.blocks:
                x = block(x)
            
            # Get outputs from different heads
            velocity = self.velocity_head(x)
            pressure = self.pressure_head(x)
            temperature = self.temperature_head(x)
            
            # Split velocity into components
            u = velocity[:, 0:1]
            v = velocity[:, 1:2]
            p = pressure
            T = temperature
            
            # Scale outputs to reasonable ranges
            u = u * 0.1  # Velocity scaling
            v = v * 0.1
            p = p * 100  # Pressure scaling
            T = T * 10 + 300  # Temperature scaling with offset
            
            return u, v, p, T
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            return (torch.zeros(x.shape[0], 1, device=device),) * 4

class VanillaPINNSolver:
    def __init__(self, domain_params: dict, physics_params: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        # Initialize model
        self.model = VanillaPINN().to(self.device)
        self.physics_loss = PhysicsLoss(**physics_params).to(self.device)
        
        # Use Adam optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Dynamic loss weights
        self.loss_weights = {
            'continuity': 1.0,
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'energy': 0.1,  # Reduced weight for energy equation
            'boundary': 10.0  # Increased weight for boundary conditions
        }
        
        # Initialize best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        
        # Loss history for adaptive weighting
        self.loss_history = []
        self.adaptation_interval = 100
        
    def to(self, device: torch.device) -> 'VanillaPINNSolver':
        """Move solver to specified device"""
        self.device = device
        self.model = self.model.to(device)
        self.physics_loss = self.physics_loss.to(device)
        return self

    def compute_gradients(self, u: torch.Tensor, v: torch.Tensor,
                         p: torch.Tensor, T: torch.Tensor,
                         x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute spatial and temporal gradients with improved stability"""
        try:
            # Enable gradient computation
            x = x.requires_grad_(True)
            t = t.requires_grad_(True)
            
            def grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
                """Compute gradient with stability checks"""
                g = compute_gradient(y, x, allow_unused=True)
                if g is None:
                    return torch.zeros_like(x)
                # Clip gradients to prevent explosions
                return torch.clamp(g, min=-100, max=100)
            
            # First derivatives with gradient clipping
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

    def update_loss_weights(self, losses: dict):
        """Dynamically adjust loss weights based on their relative magnitudes"""
        if len(self.loss_history) >= self.adaptation_interval:
            # Compute average losses over the interval
            avg_losses = {
                k: sum(hist[k] for hist in self.loss_history) / len(self.loss_history)
                for k in losses.keys() if k != 'total'
            }
            
            # Update weights inversely proportional to loss magnitudes
            max_loss = max(avg_losses.values())
            for k in self.loss_weights.keys():
                if k in avg_losses and avg_losses[k] > 0:
                    self.loss_weights[k] = max_loss / avg_losses[k]
            
            # Normalize weights
            total_weight = sum(self.loss_weights.values())
            for k in self.loss_weights:
                self.loss_weights[k] /= total_weight
                
            # Clear history
            self.loss_history = []
        else:
            self.loss_history.append(losses)

    def train_step(self, x_domain: torch.Tensor, x_boundary: torch.Tensor,
                  t_domain: torch.Tensor, t_boundary: torch.Tensor) -> dict:
        """Perform one training step with improved stability"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        try:
            # Forward pass
            u, v, p, T = self.model(x_domain, t_domain)
            
            # Compute gradients
            grads = self.compute_gradients(u, v, p, T, x_domain, t_domain)
            u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t = grads
            
            # Compute individual losses with gradient masking
            with torch.cuda.amp.autocast(enabled=True):
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
            
            # Compute total loss with loss scaling
            total_loss = sum(losses.values())
            
            # Check for invalid loss values
            if not torch.isfinite(total_loss):
                raise ValueError("Loss is not finite")
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update loss weights
            self.update_loss_weights({k: v.item() for k, v in losses.items()})
            
            # Convert losses to float for logging
            losses = {k: v.item() for k, v in losses.items()}
            losses['total'] = total_loss.item()
            
            return losses
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return {k: float('inf') for k in ['continuity', 'momentum_x', 'momentum_y', 
                                            'energy', 'boundary', 'total']}
    
    def train(self, epochs: int, nx: int = 501, ny: int = 51,
              batch_size: Optional[int] = 1024) -> List[dict]:
        """Train the model with batching and early stopping"""
        try:
            # Create domain and boundary points
            x_domain = create_domain_points(nx, ny, self.L, self.H, self.device)
            x_boundary = create_boundary_points(nx, ny, self.L, self.H, self.device)
            
            # Create time tensors
            t_domain = torch.zeros(x_domain.shape[0], 1, device=self.device)
            t_boundary = torch.zeros(x_boundary.shape[0], 1, device=self.device)
            
            # Initialize variables for early stopping
            patience = 1000
            patience_counter = 0
            min_delta = 1e-6
            best_loss = float('inf')
            history = []
            
            # Training loop with batching
            for epoch in range(epochs):
                # Shuffle data
                idx = torch.randperm(x_domain.shape[0])
                x_domain = x_domain[idx]
                t_domain = t_domain[idx]
                
                # Train by batches
                batch_losses = []
                for i in range(0, x_domain.shape[0], batch_size):
                    end = min(i + batch_size, x_domain.shape[0])
                    batch_loss = self.train_step(
                        x_domain[i:end],
                        x_boundary[:(end-i)],
                        t_domain[i:end],
                        t_boundary[:(end-i)]
                    )
                    batch_losses.append(batch_loss)
                
                # Compute average loss for the epoch
                avg_loss = {
                    k: sum(loss[k] for loss in batch_losses) / len(batch_losses)
                    for k in batch_losses[0].keys()
                }
                total_loss = avg_loss['total']
                history.append(avg_loss)
                
                # Update learning rate scheduler
                self.scheduler.step(total_loss)
                
                # Early stopping check
                if total_loss < best_loss - min_delta:
                    best_loss = total_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {
                        key: value.cpu() for key, value in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                # Print progress
                if epoch % 100 == 0:
                    print(f"\nEpoch {epoch}")
                    print(f"Total Loss: {total_loss:.6f}")
                    for k, v in avg_loss.items():
                        if k != 'total':
                            print(f"{k}: {v:.6f}")
                    print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Check stopping conditions
                if patience_counter >= patience:
                    print("\nEarly stopping triggered")
                    break
                
                if total_loss > 1e5 or not torch.isfinite(torch.tensor(total_loss)):
                    print("\nTraining diverged")
                    break
            
            # Restore best model
            if self.best_model_state is not None:
                self.model.load_state_dict({
                    k: v.to(self.device) for k, v in self.best_model_state.items()
                })
            
            return history
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return []

    def predict(self, x: torch.Tensor, t: torch.Tensor, 
                batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """Make predictions with batching support and error handling"""
        self.model.eval()
        try:
            with torch.no_grad():
                x = x.to(self.device)
                t = t.to(self.device)
                
                if batch_size is None or batch_size >= x.shape[0]:
                    # Single batch prediction
                    return self.model(x, t)
                else:
                    # Batch predictions
                    n_samples = x.shape[0]
                    n_batches = (n_samples + batch_size - 1) // batch_size
                    
                    u_list, v_list, p_list, T_list = [], [], [], []
                    
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, n_samples)
                        
                        u_batch, v_batch, p_batch, T_batch = self.model(
                            x[start_idx:end_idx], 
                            t[start_idx:end_idx]
                        )
                        
                        u_list.append(u_batch)
                        v_list.append(v_batch)
                        p_list.append(p_batch)
                        T_list.append(T_batch)
                    
                    return (torch.cat(u_list, dim=0), 
                            torch.cat(v_list, dim=0),
                            torch.cat(p_list, dim=0), 
                            torch.cat(T_list, dim=0))
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=self.device)
            return zeros, zeros, zeros, zeros

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
                        lambda x: T_xx, lambda x: T_yy, T_t).item(),
                    'boundary': self.boundary_loss(x_val, t_val).item()
                }
                
                validation_losses['total'] = sum(validation_losses.values())
                return validation_losses
                
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return {k: float('inf') for k in ['continuity', 'momentum_x', 'momentum_y', 
                                            'energy', 'boundary', 'total']}

    def save_model(self, path: str):
        """Save model state with error handling"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss_weights': self.loss_weights,
                'best_loss': self.best_loss,
                'best_model_state': self.best_model_state,
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
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load other parameters
            self.loss_weights = checkpoint['loss_weights']
            self.best_loss = checkpoint['best_loss']
            self.best_model_state = checkpoint['best_model_state']
            
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