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
            return (torch.zeros(x.shape[0], 1, device=device),) * 4

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
        
        # Use AdamW optimizer with weight decay and gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warm-up and decay
        self.scheduler = torch.optim.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            steps_per_epoch=1,
            epochs=10000,
            pct_start=0.1,
            div_factor=25.0
        )
        
        # Loss weights for balancing different terms
        self.loss_weights = {
            'continuity': 1.0,
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'energy': 1.0,
            'boundary': 10.0  # Increased weight for boundary conditions
        }
        
        # Initialize best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        
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
            
            # Compute total loss with loss scaling for stability
            total_loss = sum(losses.values())
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step and scheduler update
            self.optimizer.step()
            self.scheduler.step()
            
            # Update best model if needed
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.best_model_state = {
                    key: value.cpu() for key, value in self.model.state_dict().items()
                }
            
            # Convert losses to float for logging
            return {k: v.item() for k, v in losses.items()}
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return {k: float('inf') for k in self.loss_weights.keys()}
    
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
                total_loss = sum(avg_loss.values())
                history.append(avg_loss)
                
                # Early stopping check
                if total_loss < best_loss - min_delta:
                    best_loss = total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Print progress
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}")
                    print(f"Total Loss: {total_loss:.6f}")
                    for k, v in avg_loss.items():
                        print(f"{k}: {v:.6f}")
                    print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Check stopping conditions
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
                
                if total_loss > 1e5:
                    print("Training diverged")
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