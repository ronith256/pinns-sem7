import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from utils import *

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, x direction
        self.modes2 = modes2  # Number of Fourier modes to multiply, y direction

        # Initialize weights with scaling for better gradient flow
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 
                                   dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 
                                   dtype=torch.cfloat))

    def complex_matmul_2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Perform complex matrix multiplication in 2D"""
        try:
            return torch.einsum("bixy,ioxy->boxy", input, weights)
        except Exception as e:
            print(f"Error in complex matrix multiplication: {str(e)}")
            return torch.zeros_like(input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved stability and error handling"""
        try:
            batchsize = x.shape[0]
            
            # Compute Fourier coefficients
            x_ft = torch.fft.rfft2(x, norm='ortho')
            
            # Initialize output tensor
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), 
                               x.size(-1)//2 + 1, dtype=torch.cfloat, 
                               device=x.device)
            
            # Multiply relevant Fourier modes
            # Lower frequencies
            out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul_2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            
            # Higher frequencies
            out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul_2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
            # Return to physical space
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
            
            return x
            
        except Exception as e:
            print(f"Error in SpectralConv2d forward pass: {str(e)}")
            return torch.zeros_like(x, device=x.device)

class FNO2d(nn.Module):
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 64, 
                 H: int = 51, W: int = 501):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.H = H
        self.W = W
        
        # Padding for input standardization
        self.padding = nn.ReplicationPad2d(1)
        
        # Input projection
        self.fc0 = nn.Linear(4, self.width)  # 4 input channels for x, y, t coordinates
        
        # Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Regular convolutions for local features
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Batch normalization for stability
        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.bn3 = nn.BatchNorm2d(self.width)
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 output channels for u, v, p, T
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
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
            x_reshaped = x.reshape(batch_size, 2, self.H, self.W)  # 2 channels for x,y coordinates
            t = t.reshape(batch_size, 1, 1, 1).expand(-1, 1, self.H, self.W)
            inputs = torch.cat([x_reshaped, t], dim=1)  # 3 channels total
            
            # Process through network
            # Initial projection
            x = self.fc0(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = self.padding(x)
            
            # Fourier layer 0: spectral conv -> regular conv -> nonlinearity
            x1 = self.conv0(x)
            x2 = self.w0(x)
            x = self.bn0(x1 + x2)
            x = F.gelu(x)
            x = self.dropout(x)
            
            # Fourier layer 1
            x1 = self.conv1(x)
            x2 = self.w1(x)
            x = self.bn1(x1 + x2)
            x = F.gelu(x)
            x = self.dropout(x)
            
            # Fourier layer 2
            x1 = self.conv2(x)
            x2 = self.w2(x)
            x = self.bn2(x1 + x2)
            x = F.gelu(x)
            x = self.dropout(x)
            
            # Fourier layer 3
            x1 = self.conv3(x)
            x2 = self.w3(x)
            x = self.bn3(x1 + x2)
            x = F.gelu(x)
            x = self.dropout(x)
            
            # Output projection
            x = x.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            
            # Check for numerical issues
            if check_nan_inf(x, "network_output"):
                raise ValueError("NaN or Inf values detected in network output")
            
            # Split into separate outputs
            u = x[..., 0].reshape(batch_size, -1)
            v = x[..., 1].reshape(batch_size, -1)
            p = x[..., 2].reshape(batch_size, -1)
            T = x[..., 3].reshape(batch_size, -1)
            
            return u, v, p, T
            
        except Exception as e:
            print(f"Error in FNO2d forward pass: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=device)
            return zeros, zeros, zeros, zeros

class FNOPINNSolver:
    def __init__(self, domain_params: dict, physics_params: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store domain parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = domain_params['Re']
        self.Q_flux = domain_params['Q_flux']
        
        # Initialize model and move to device
        self.model = FNO2d().to(self.device)
        self.physics_loss = PhysicsLoss(**physics_params).to(self.device)
        
        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with cosine annealing and warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,  # Reset every 1000 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Loss weights for different physics components
        self.loss_weights = {
            'continuity': 1.0,
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'energy': 1.0,
            'boundary': 1.0
        }
        
        # Initialize best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def to(self, device: torch.device) -> 'FNOPINNSolver':
        """Move solver to specified device"""
        self.device = device
        self.model = self.model.to(device)
        self.physics_loss = self.physics_loss.to(device)
        return self

    def compute_gradients(self, u: torch.Tensor, v: torch.Tensor,
                         p: torch.Tensor, T: torch.Tensor,
                         x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute spatial and temporal gradients with FFT-based differentiation"""
        try:
            # Reshape inputs for FFT
            batch_size = u.shape[0]
            H, W = self.model.H, self.model.W
            
            # Helper function for FFT-based derivative
            def fft_derivative(field: torch.Tensor, direction: str) -> torch.Tensor:
                field = field.reshape(batch_size, H, W)
                
                if direction == 'x':
                    # Wavenumbers in x direction
                    kx = torch.fft.fftfreq(W, d=self.L/W, device=self.device) * 2 * torch.pi
                    # Compute derivative in Fourier space
                    field_ft = torch.fft.fft(field, dim=-1)
                    field_dx_ft = 1j * kx[None, None, :] * field_ft
                    return torch.fft.ifft(field_dx_ft, dim=-1).real.reshape(batch_size, -1)
                    
                else:  # direction == 'y'
                    # Wavenumbers in y direction
                    ky = torch.fft.fftfreq(H, d=self.H/H, device=self.device) * 2 * torch.pi
                    # Compute derivative in Fourier space
                    field_ft = torch.fft.fft(field, dim=1)
                    field_dy_ft = 1j * ky[None, :, None] * field_ft
                    return torch.fft.ifft(field_dy_ft, dim=1).real.reshape(batch_size, -1)
            
            # First derivatives
            u_x = fft_derivative(u, 'x')
            u_y = fft_derivative(u, 'y')
            v_x = fft_derivative(v, 'x')
            v_y = fft_derivative(v, 'y')
            T_x = fft_derivative(T, 'x')
            T_y = fft_derivative(T, 'y')
            
            # Second derivatives
            u_xx = fft_derivative(u_x.reshape(batch_size, H, W), 'x')
            u_yy = fft_derivative(u_y.reshape(batch_size, H, W), 'y')
            v_xx = fft_derivative(v_x.reshape(batch_size, H, W), 'x')
            v_yy = fft_derivative(v_y.reshape(batch_size, H, W), 'y')
            T_xx = fft_derivative(T_x.reshape(batch_size, H, W), 'x')
            T_yy = fft_derivative(T_y.reshape(batch_size, H, W), 'y')
            
            # Time derivatives using autograd
            def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return compute_gradient(y, x, allow_unused=True)
            
            t.requires_grad_(True)
            u_t = grad(u.sum(), t)
            v_t = grad(v.sum(), t)
            T_t = grad(T.sum(), t)
            
            return u_x, u_y, v_x, v_y, T_x, T_y, u_xx, u_yy, v_xx, v_yy, T_xx, T_yy, u_t, v_t, T_t
            
        except Exception as e:
            print(f"Error computing gradients: {str(e)}")
            zeros = torch.zeros_like(x[:, 0:1])
            return tuple([zeros] * 15)

    def boundary_loss(self, x_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss with improved stability"""
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
            
            # Bottom wall heat flux condition with FFT-based derivative
            bottom_wall_mask = (x_b[:, 1] == 0)
            if torch.any(bottom_wall_mask):
                T_masked = T[bottom_wall_mask]
                
                # Reshape for FFT
                T_wall = T_masked.reshape(-1, 1, self.model.W)
                
                # Compute y-derivative using FFT
                ky = torch.fft.fftfreq(2, d=self.H/2, device=self.device) * 2 * torch.pi
                T_ft = torch.fft.fft(F.pad(T_wall, (0, 0, 0, 1)), dim=1)
                T_dy_ft = 1j * ky[None, :, None] * T_ft
                T_dy = torch.fft.ifft(T_dy_ft, dim=1).real[:, 0]
                
                heat_flux_loss = torch.mean(torch.square(
                    -self.physics_loss.k * T_dy - self.Q_flux
                ))
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
        """Perform one training step with improved stability"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Move inputs to device
        x_domain = x_domain.to(self.device)
        t_domain = t_domain.to(self.device)
        x_boundary = x_boundary.to(self.device)
        t_boundary = t_boundary.to(self.device)
        
        try:
            # Forward pass with gradient computation
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
            
            # Total loss with gradient scaling for stability
            total_loss = sum(losses.values())
            
            # Mixed precision backward pass
            with torch.cuda.amp.autocast():
                total_loss.backward()
            
            # Gradient clipping
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
            losses = {k: v.item() for k, v in losses.items()}
            losses['total'] = total_loss.item()
            
            return losses
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return {k: float('inf') for k in ['continuity', 'momentum_x', 'momentum_y', 
                                            'energy', 'boundary', 'total']}
    def train(self, epochs: int, nx: int = 501, ny: int = 51,
              validation_freq: int = 100, batch_size: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
        """Train the model with batching, validation, and early stopping"""
        try:
            # Create domain and boundary points
            x_domain = create_domain_points(nx, ny, self.L, self.H, self.device)
            x_boundary = create_boundary_points(nx, ny, self.L, self.H, self.device)
            
            # Create time tensors
            t_domain = torch.zeros(x_domain.shape[0], 1, device=self.device)
            t_boundary = torch.zeros(x_boundary.shape[0], 1, device=self.device)
            
            # Set up batching if specified
            if batch_size is not None:
                n_batches = x_domain.shape[0] // batch_size
                indices = torch.randperm(x_domain.shape[0], device=self.device)
            else:
                batch_size = x_domain.shape[0]
                n_batches = 1
                indices = torch.arange(x_domain.shape[0], device=self.device)
            
            # Create validation points (using a subset of domain points)
            val_indices = torch.randperm(x_domain.shape[0])[:1000]
            x_val = x_domain[val_indices]
            t_val = t_domain[val_indices]
            
            # Initialize tracking variables
            train_history = []
            val_history = []
            best_epoch = 0
            patience = 2000
            patience_counter = 0
            min_lr = 1e-6
            
            # Set up mixed precision training
            scaler = torch.cuda.amp.GradScaler()
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Shuffle data for each epoch
                if batch_size < x_domain.shape[0]:
                    indices = torch.randperm(x_domain.shape[0], device=self.device)
                
                # Train by batches
                for batch in range(n_batches):
                    batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
                    
                    # Prepare batch data
                    x_batch = x_domain[batch_indices]
                    t_batch = t_domain[batch_indices]
                    x_boundary_batch = x_boundary[batch_indices % x_boundary.shape[0]]
                    t_boundary_batch = t_boundary[batch_indices % x_boundary.shape[0]]
                    
                    # Train step with mixed precision
                    with torch.cuda.amp.autocast():
                        batch_losses = self.train_step(
                            x_batch, x_boundary_batch, t_batch, t_boundary_batch)
                    
                    epoch_losses.append(batch_losses)
                
                # Compute average losses for the epoch
                avg_losses = {
                    k: sum(loss[k] for loss in epoch_losses) / len(epoch_losses)
                    for k in epoch_losses[0].keys()
                }
                train_history.append(avg_losses)
                
                # Validation step
                if epoch % validation_freq == 0:
                    val_losses = self.validate(x_val, t_val)
                    val_history.append(val_losses)
                    
                    # Early stopping check
                    if val_losses['total'] < self.best_loss:
                        self.best_loss = val_losses['total']
                        best_epoch = epoch
                        patience_counter = 0
                        # Save best model state
                        self.best_model_state = {
                            key: value.cpu() for key, value in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                    
                    # Print progress
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"\nEpoch {epoch + 1}/{epochs}")
                    print(f"Training Loss: {avg_losses['total']:.6f}")
                    print(f"Validation Loss: {val_losses['total']:.6f}")
                    print(f"Learning Rate: {current_lr:.2e}")
                    print(f"Best Epoch: {best_epoch}")
                    
                    # Check stopping conditions
                    if patience_counter >= patience:
                        print("\nEarly stopping triggered!")
                        break
                    
                    if current_lr <= min_lr:
                        print("\nMinimum learning rate reached!")
                        break
                
                # Check for divergence
                if not torch.isfinite(torch.tensor(avg_losses['total'])):
                    print("\nTraining diverged!")
                    break
            
            # Restore best model
            if self.best_model_state is not None:
                self.model.load_state_dict({
                    k: v.to(self.device) for k, v in self.best_model_state.items()
                })
            
            return train_history, val_history
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return [], []

    def validate(self, x_val: torch.Tensor, t_val: torch.Tensor) -> dict:
        """Validate model performance with error handling"""
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
            return {k: float('inf') for k in ['continuity', 'momentum_x', 'momentum_y', 
                                            'energy', 'total']}

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
                            x[start_idx:end_idx], t[start_idx:end_idx])
                        
                        u_list.append(u_batch)
                        v_list.append(v_batch)
                        p_list.append(p_batch)
                        T_list.append(T_batch)
                    
                    return (torch.cat(u_list, dim=0), torch.cat(v_list, dim=0),
                            torch.cat(p_list, dim=0), torch.cat(T_list, dim=0))
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            zeros = torch.zeros(x.shape[0], 1, device=self.device)
            return zeros, zeros, zeros, zeros

    def save_model(self, path: str, save_optimizer: bool = True):
        """Save model state with comprehensive checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'best_model_state': self.best_model_state,
                'best_loss': self.best_loss,
                'loss_weights': self.loss_weights,
                'domain_params': {
                    'L': self.L,
                    'H': self.H,
                    'Re': self.Re,
                    'Q_flux': self.Q_flux
                }
            }
            
            if save_optimizer:
                checkpoint.update({
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                })
            
            torch.save(checkpoint, path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, path: str, load_optimizer: bool = True):
        """Load model state with error handling"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_model_state = checkpoint['best_model_state']
            self.best_loss = checkpoint['best_loss']
            self.loss_weights = checkpoint['loss_weights']
            
            # Load domain parameters
            params = checkpoint['domain_params']
            self.L = params['L']
            self.H = params['H']
            self.Re = params['Re']
            self.Q_flux = params['Q_flux']
            
            # Optionally load optimizer state
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Move model to correct device
            self.to(self.device)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")