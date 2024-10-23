import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BranchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        logger.info(f"Initializing BranchNet with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"BranchNet input shape: {x.shape}")
        output = self.net(x)
        logger.debug(f"BranchNet output shape: {output.shape}")
        return output

class TrunkNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        logger.info(f"Initializing TrunkNet with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"TrunkNet input shape: {x.shape}")
        output = self.net(x)
        logger.debug(f"TrunkNet output shape: {output.shape}")
        return output

class FlowDeepONet(nn.Module):
    def __init__(self, domain_params: Dict[str, Any]):
        super().__init__()
        
        # Physical parameters
        self.rho = 997.07
        self.mu = 8.9e-4
        self.nu = self.mu / self.rho
        self.k = 0.606
        self.cp = 4200
        self.q_flux = 20000
        
        # Domain parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = 200
        self.Dh = 0.00116
        self.U_in = (self.Re * self.mu) / (self.rho * self.Dh)

        # Network parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.hidden_dim = 256
        self.output_dim = 64
        
        # Number of sensor points for each variable
        self.n_sensors = 10  # Number of sensors per variable
        self.total_sensors = self.n_sensors * 4  # Total sensors for u, v, p, T
        
        logger.info(f"Network dimensions - Hidden: {self.hidden_dim}, Output: {self.output_dim}")
        logger.info(f"Number of sensors: {self.n_sensors} per variable")

        # Branch and Trunk networks for each variable
        self.branch_u = BranchNet(self.total_sensors, self.hidden_dim, self.output_dim).to(self.device)
        self.branch_v = BranchNet(self.total_sensors, self.hidden_dim, self.output_dim).to(self.device)
        self.branch_p = BranchNet(self.total_sensors, self.hidden_dim, self.output_dim).to(self.device)
        self.branch_T = BranchNet(self.total_sensors, self.hidden_dim, self.output_dim).to(self.device)

        # Trunk networks take coordinates (x, y, t) as input
        self.trunk_u = TrunkNet(3, self.hidden_dim, self.output_dim).to(self.device)
        self.trunk_v = TrunkNet(3, self.hidden_dim, self.output_dim).to(self.device)
        self.trunk_p = TrunkNet(3, self.hidden_dim, self.output_dim).to(self.device)
        self.trunk_T = TrunkNet(3, self.hidden_dim, self.output_dim).to(self.device)

    def forward(self, sensor_data: torch.Tensor, coords: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        logger.debug(f"Forward pass input shapes - sensor_data: {sensor_data.shape}, coords: {coords.shape}, t: {t.shape}")
        
        # Ensure sensor_data is on the correct device
        sensor_data = sensor_data.to(self.device)
        coords = coords.to(self.device)
        t = t.to(self.device)
        
        # Ensure sensor_data has batch dimension
        if sensor_data.dim() == 1:
            sensor_data = sensor_data.unsqueeze(0)
            logger.debug(f"Added batch dimension to sensor_data: {sensor_data.shape}")
        
        # Combine coordinates and time
        inputs = torch.cat([coords, t], dim=1)
        logger.debug(f"Combined inputs shape: {inputs.shape}")

        # Expand sensor data if necessary to match coords batch size
        if sensor_data.size(0) == 1 and coords.size(0) > 1:
            sensor_data = sensor_data.expand(coords.size(0), -1)
            logger.debug(f"Expanded sensor_data shape: {sensor_data.shape}")

        # Branch outputs
        branch_u_out = self.branch_u(sensor_data)
        branch_v_out = self.branch_v(sensor_data)
        branch_p_out = self.branch_p(sensor_data)
        branch_T_out = self.branch_T(sensor_data)
        
        logger.debug(f"Branch outputs shape: {branch_u_out.shape}")

        # Trunk outputs
        trunk_u_out = self.trunk_u(inputs)
        trunk_v_out = self.trunk_v(inputs)
        trunk_p_out = self.trunk_p(inputs)
        trunk_T_out = self.trunk_T(inputs)
        
        logger.debug(f"Trunk outputs shape: {trunk_u_out.shape}")

        # DeepONet dot product for each variable
        u = torch.sum(branch_u_out * trunk_u_out, dim=-1, keepdim=True)
        v = torch.sum(branch_v_out * trunk_v_out, dim=-1, keepdim=True)
        p = torch.sum(branch_p_out * trunk_p_out, dim=-1, keepdim=True)
        T = torch.sum(branch_T_out * trunk_T_out, dim=-1, keepdim=True)
        
        logger.debug(f"Final output shapes - u: {u.shape}, v: {v.shape}, p: {p.shape}, T: {T.shape}")

        return u, v, p, T

    def generate_sensor_data(self) -> torch.Tensor:
        """Generate synthetic sensor data for training."""
        logger.info("Generating synthetic sensor data")
        
        # Generate sensor positions
        x_sensors = torch.linspace(0, self.L, self.n_sensors, device=self.device)
        y_sensors = torch.linspace(0, self.H, self.n_sensors, device=self.device)
        X, Y = torch.meshgrid(x_sensors, y_sensors, indexing='ij')
        
        # Generate synthetic measurements for each variable
        U = torch.sin(2 * np.pi * X / self.L) * torch.cos(2 * np.pi * Y / self.H)
        V = -torch.cos(2 * np.pi * X / self.L) * torch.sin(2 * np.pi * Y / self.H)
        P = torch.sin(2 * np.pi * X / self.L) * torch.sin(2 * np.pi * Y / self.H)
        T = 300 + 20 * torch.sin(np.pi * Y / self.H)
        
        # Flatten and combine all measurements
        sensor_data = torch.cat([
            U.flatten(),
            V.flatten(),
            P.flatten(),
            T.flatten()
        ])
        
        logger.debug(f"Generated sensor data shape: {sensor_data.shape}")
        return sensor_data

    def predict(self, sensor_data: torch.Tensor, coords: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self.eval()
        with torch.no_grad():
            return self.forward(sensor_data, coords, t)

    def compute_derivatives(self, sensor_data: torch.Tensor, coords: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        coords.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T = self.forward(sensor_data, coords, t)
        
        # First derivatives
        u_x = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 1:2]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        v_x = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 1:2]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p.sum(), coords, create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p.sum(), coords, create_graph=True)[0][:, 1:2]
        
        T_x = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 0:1]
        T_y = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 1:2]
        T_t = torch.autograd.grad(T.sum(), t, create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1:2]
        
        v_xx = torch.autograd.grad(v_x.sum(), coords, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), coords, create_graph=True)[0][:, 1:2]
        
        T_xx = torch.autograd.grad(T_x.sum(), coords, create_graph=True)[0][:, 0:1]
        T_yy = torch.autograd.grad(T_y.sum(), coords, create_graph=True)[0][:, 1:2]
        
        return {
            'u': u, 'v': v, 'p': p, 'T': T,
            'u_x': u_x, 'u_y': u_y, 'u_t': u_t,
            'v_x': v_x, 'v_y': v_y, 'v_t': v_t,
            'p_x': p_x, 'p_y': p_y,
            'T_x': T_x, 'T_y': T_y, 'T_t': T_t,
            'u_xx': u_xx, 'u_yy': u_yy,
            'v_xx': v_xx, 'v_yy': v_yy,
            'T_xx': T_xx, 'T_yy': T_yy
        }

    def compute_pde_residuals(self, sensor_data: torch.Tensor, coords: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = self.compute_derivatives(sensor_data, coords, t)
        
        # Continuity equation
        continuity = d['u_x'] + d['v_y']
        
        # Momentum equations
        momentum_x = (d['u_t'] + d['u']*d['u_x'] + d['v']*d['u_y'] + 
                     (1/self.rho)*d['p_x'] - self.nu*(d['u_xx'] + d['u_yy']))
        
        momentum_y = (d['v_t'] + d['u']*d['v_x'] + d['v']*d['v_y'] + 
                     (1/self.rho)*d['p_y'] - self.nu*(d['v_xx'] + d['v_yy']))
        
        # Energy equation
        energy = (d['T_t'] + d['u']*d['T_x'] + d['v']*d['T_y'] - 
                 (self.k/(self.rho*self.cp))*(d['T_xx'] + d['T_yy']))
        
        return {
            'continuity': continuity,
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'energy': energy
        }

    def compute_bc_residuals(self, sensor_data: torch.Tensor, coords: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = self.compute_derivatives(sensor_data, coords, t)
        
        # Inlet conditions (x = 0)
        inlet_mask = coords[:, 0:1] < 1e-5
        inlet_u = d['u'] - self.U_in * torch.ones_like(d['u'])
        inlet_v = d['v']
        inlet_T = d['T'] - 300.0
        
        # Outlet conditions (x = L)
        outlet_mask = torch.abs(coords[:, 0:1] - self.L) < 1e-5
        outlet_u = d['u_x']
        outlet_v = d['v_x']
        outlet_T = d['T_x']
        
        # Bottom wall conditions (y = 0)
        bottom_mask = coords[:, 1:2] < 1e-5
        bottom_u = d['u']
        bottom_v = d['v']
        bottom_T = -self.k * d['T_y'] - self.q_flux
        
        # Top wall conditions (y = H)
        top_mask = torch.abs(coords[:, 1:2] - self.H) < 1e-5
        top_u = d['u']
        top_v = d['v']
        top_T = d['T_y']
        
        return {
            'inlet_u': inlet_u * inlet_mask,
            'inlet_v': inlet_v * inlet_mask,
            'inlet_T': inlet_T * inlet_mask,
            'outlet_u': outlet_u * outlet_mask,
            'outlet_v': outlet_v * outlet_mask,
            'outlet_T': outlet_T * outlet_mask,
            'bottom_u': bottom_u * bottom_mask,
            'bottom_v': bottom_v * bottom_mask,
            'bottom_T': bottom_T * bottom_mask,
            'top_u': top_u * top_mask,
            'top_v': top_v * top_mask,
            'top_T': top_T * top_mask
        }

    def compute_loss(self, sensor_data: torch.Tensor, coords: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logger.debug("Computing loss")
        logger.debug(f"Input shapes - sensor_data: {sensor_data.shape}, coords: {coords.shape}, t: {t.shape}")
        
        # PDE residuals
        pde_residuals = self.compute_pde_residuals(sensor_data, coords, t)
        pde_loss = (torch.mean(pde_residuals['continuity']**2) +
                   torch.mean(pde_residuals['momentum_x']**2) +
                   torch.mean(pde_residuals['momentum_y']**2) +
                   torch.mean(pde_residuals['energy']**2))
        
        # Boundary condition residuals
        bc_residuals = self.compute_bc_residuals(sensor_data, coords, t)
        bc_loss = sum(torch.mean(r**2) for r in bc_residuals.values())
        
        # Total loss
        total_loss = pde_loss + 10.0 * bc_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item()
        }
        
        logger.debug(f"Loss values: {loss_dict}")
        return total_loss, loss_dict

def train_model(model: FlowDeepONet, num_epochs: int = 10000, batch_size: int = 1000):
    """
    Train the DeepONet model with batch processing and improved logging.
    """
    logger.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
    
    # Generate training points
    nx, ny, nt = 50, 20, 10
    x = torch.linspace(0, model.L, nx, device=model.device)
    y = torch.linspace(0, model.H, ny, device=model.device)
    t = torch.linspace(0, 30, nt, device=model.device)
    
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
    t_train = T.flatten().unsqueeze(1)
    
    logger.info(f"Generated training points - Total points: {coords.shape[0]}")
    
    # Generate sensor data
    sensor_data = model.generate_sensor_data()
    logger.info(f"Generated sensor data shape: {sensor_data.shape}")
    
    # Initialize loss history
    loss_history = {
        'total_loss': [],
        'pde_loss': [],
        'bc_loss': []
    }
    
    # Training loop with batch processing
    num_batches = coords.shape[0] // batch_size + (1 if coords.shape[0] % batch_size != 0 else 0)
    
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            # Shuffle the data
            indices = torch.randperm(coords.shape[0], device=model.device)
            coords_shuffled = coords[indices]
            t_train_shuffled = t_train[indices]
            
            for batch in range(num_batches):
                optimizer.zero_grad()
                
                # Get batch data
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, coords.shape[0])
                coords_batch = coords_shuffled[start_idx:end_idx]
                t_batch = t_train_shuffled[start_idx:end_idx]
                
                # Compute loss
                loss, loss_dict = model.compute_loss(sensor_data, coords_batch, t_batch)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss_dict['total_loss']
            
            # Average loss for the epoch
            epoch_loss /= num_batches
            
            # Store losses
            for key in loss_dict:
                loss_history[key].append(loss_dict[key])
            
            # Update learning rate
            scheduler.step(epoch_loss)
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}:")
                logger.info(f"  Total Loss = {loss_dict['total_loss']:.6f}")
                logger.info(f"  PDE Loss = {loss_dict['pde_loss']:.6f}")
                logger.info(f"  BC Loss = {loss_dict['bc_loss']:.6f}")
                logger.info(f"  Learning Rate = {optimizer.param_groups[0]['lr']:.6e}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    return model, loss_history

def test_model(model: FlowDeepONet, domain_params: Dict[str, float], t: float) -> None:
    """
    Test the trained model at a specific time with improved visualization and error handling.
    """
    logger.info(f"Testing model at t = {t}")
    
    try:
        model.eval()
        
        # Generate test points
        nx, ny = 100, 50
        x = torch.linspace(0, domain_params['L'], nx, device=model.device)
        y = torch.linspace(0, domain_params['H'], ny, device=model.device)
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        t_test = torch.full((coords.shape[0], 1), t, device=model.device)
        
        # Generate sensor data for testing
        sensor_data = model.generate_sensor_data()
        
        # Get predictions
        with torch.no_grad():
            u, v, p, T = model.predict(sensor_data, coords, t_test)
        
        # Reshape predictions
        u = u.reshape(nx, ny).cpu().numpy()
        v = v.reshape(nx, ny).cpu().numpy()
        p = p.reshape(nx, ny).cpu().numpy()
        T = T.reshape(nx, ny).cpu().numpy()
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        
        # Create visualization
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        # Velocity magnitude plot
        plt.subplot(2, 2, 1)
        vel_mag = np.sqrt(u**2 + v**2)
        plt.contourf(X_np, Y_np, vel_mag, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(X_np, Y_np, u, v, color='k', density=1.5)
        plt.title('Velocity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Pressure plot
        plt.subplot(2, 2, 2)
        plt.contourf(X_np, Y_np, p, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Pressure (Pa)')
        plt.title('Pressure Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Temperature plot
        plt.subplot(2, 2, 3)
        plt.contourf(X_np, Y_np, T, levels=50, cmap='hot')
        plt.colorbar(label='Temperature (K)')
        plt.title('Temperature Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Vorticity plot
        plt.subplot(2, 2, 4)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dudy = np.gradient(u, dy, axis=1)
        dvdx = np.gradient(v, dx, axis=0)
        vorticity = dvdx - dudy
        plt.contourf(X_np, Y_np, vorticity, levels=50, cmap='RdBu')
        plt.colorbar(label='Vorticity (1/s)')
        plt.title('Vorticity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        plt.suptitle(f'DeepONet Flow Field Results at t = {t:.2f} s')
        plt.tight_layout()
        
        # Save the figure
        save_path = f'flow_field_t{t:.1f}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Set domain parameters
        domain_params = {
            'L': 25 * 0.00116,  # Channel length (25 * Dh)
            'H': 0.5 * 0.00116  # Channel height (0.5 * Dh)
        }
        
        # Initialize model
        model = FlowDeepONet(domain_params)
        
        # Train model
        logger.info("Starting model training...")
        trained_model, loss_history = train_model(model, num_epochs=1000)
        
        # Test model
        logger.info("Testing trained model...")
        test_model(trained_model, domain_params, t=15.0)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise