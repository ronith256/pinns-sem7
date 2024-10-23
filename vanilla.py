import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from utils import *

class VanillaPINN(nn.Module):
    def __init__(self, layers: List[int] = [3, 128, 128, 128, 128, 4]):
        super().__init__()
        self.layers = layers
        
        # Build the neural network with batch normalization and proper initialization
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                modules.append(nn.BatchNorm1d(layers[i+1]))
                modules.append(nn.Tanh())
        
        self.network = nn.Sequential(*modules)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        try:
            # Ensure inputs are on the same device as the model
            device = next(self.parameters()).device
            x = x.to(device)
            t = t.to(device)
            
            # Normalize inputs to improve training stability
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
            x_normalized = (x - x_mean) / x_std
            
            t_mean = torch.mean(t)
            t_std = torch.std(t) + 1e-8
            t_normalized = (t - t_mean) / t_std
            
            # Combine normalized spatial coordinates and time
            inputs = torch.cat([x_normalized, t_normalized.reshape(-1, 1)], dim=1)
            
            # Get network output
            outputs = self.network(inputs)
            
            # Split outputs into velocity components, pressure, and temperature
            u = outputs[:, 0:1]
            v = outputs[:, 1:2]
            p = outputs[:, 2:3]
            T = outputs[:, 3:4]
            
            return u, v, p, T
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=device, requires_grad=True)
            return zeros, zeros, zeros, zeros

class VanillaPINNSolver:
    def __init__(self, domain_params: dict, physics_params: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        # Initialize model and move to device
        self.model = VanillaPINN().to(self.device)
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
        
        # Loss weights
        self.loss_weights = {
            'continuity': 1.0,
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'energy': 1.0,
            'boundary': 10.0
        }
        
        # Initialize best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def to(self, device: torch.device) -> 'VanillaPINNSolver':
        self.device = device
        self.model = self.model.to(device)
        self.physics_loss = self.physics_loss.to(device)
        return self

    def compute_gradients(self, u: torch.Tensor, v: torch.Tensor,
                         p: torch.Tensor, T: torch.Tensor,
                         x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute spatial and temporal gradients"""
        try:
            # Create computation graph
            x.requires_grad_(True)
            t.requires_grad_(True)
            
            # Recompute outputs with gradients enabled
            u_pred, v_pred, p_pred, T_pred = self.model(x, t)
            
            # First derivatives
            u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:, 0:1]
            u_y = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:, 1:2]
            v_x = torch.autograd.grad(v_pred.sum(), x, create_graph=True)[0][:, 0:1]
            v_y = torch.autograd.grad(v_pred.sum(), x, create_graph=True)[0][:, 1:2]
            T_x = torch.autograd.grad(T_pred.sum(), x, create_graph=True)[0][:, 0:1]
            T_y = torch.autograd.grad(T_pred.sum(), x, create_graph=True)[0][:, 1:2]
            
            # Second derivatives
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
            u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
            v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0:1]
            v_yy = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0][:, 1:2]
            T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0][:, 0:1]
            T_yy = torch.autograd.grad(T_y.sum(), x, create_graph=True)[0][:, 1:2]
            
            # Time derivatives
            u_t = torch.autograd.grad(u_pred.sum(), t, create_graph=True)[0]
            v_t = torch.autograd.grad(v_pred.sum(), t, create_graph=True)[0]
            T_t = torch.autograd.grad(T_pred.sum(), t, create_graph=True)[0]
            
            return u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t
            
        except Exception as e:
            print(f"Error computing gradients: {str(e)}")
            zeros = torch.zeros_like(x[:, 0:1])
            return tuple([zeros] * 15)

    def compute_inlet_velocity(self) -> torch.Tensor:
        """Compute inlet velocity based on Reynolds number"""
        return (self.Re * self.physics_loss.mu / 
                (self.physics_loss.rho * self.H)).to(self.device)

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
        
    def train_step(self, x_domain: torch.Tensor, x_boundary: torch.Tensor,
                  t_domain: torch.Tensor, t_boundary: torch.Tensor) -> dict:
        """Perform one training step with improved stability"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move inputs to device
        x_domain = x_domain.to(self.device)
        t_domain = t_domain.to(self.device)
        x_boundary = x_boundary.to(self.device)
        t_boundary = t_boundary.to(self.device)
        
        try:
            # Forward pass with gradient computation
            u, v, p, T = self.model(x_domain, t_domain)
            
            # Compute gradients
            grads = self.compute_gradients(u, v, p, T, x_domain, t_domain)
            u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t = grads
            
            # Compute individual losses with weights and gradient masking
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
            
            # Compute total loss with loss scaling for stability
            total_loss = sum(losses.values())
            
            # Check for invalid loss values
            if not torch.isfinite(total_loss):
                raise ValueError("Loss is not finite")
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
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