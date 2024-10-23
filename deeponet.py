import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowDeepONet:
    def __init__(self, domain_params: Dict[str, float]):
        """
        Initialize FlowDeepONet with domain parameters.
        
        Args:
            domain_params: Dictionary containing domain parameters (L, H)
        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Physical parameters
        self.rho = 997.07  # Density
        self.mu = 8.9e-4   # Dynamic viscosity
        self.nu = self.mu / self.rho  # Kinematic viscosity
        self.k = 0.606     # Thermal conductivity
        self.cp = 4200     # Specific heat capacity
        self.q_flux = 20000  # Heat flux
        
        # Domain parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = 200
        self.Dh = 0.00116
        self.U_in = (self.Re * self.mu) / (self.rho * self.Dh)
        
        # Initialize the model
        self._setup_geometry()
        self._setup_training_data()
        self._create_model()
        
    def _setup_geometry(self):
        """Set up the geometry and time domain."""
        # Spatial domain: [0, L] x [0, H]
        self.geom = dde.geometry.Rectangle([0, 0], [self.L, self.H])
        
        # Time domain: [0, 30]
        self.timedomain = dde.geometry.TimeDomain(0, 30)
        
        # Spatio-temporal domain
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)
        
    def _navier_stokes_pde(self, x, y):
        """Define the Navier-Stokes PDEs."""
        # Split variables
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        T = y[:, 3:4]
        
        # Derivatives
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        du_t = dde.grad.jacobian(y, x, i=0, j=2)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)
        dT_x = dde.grad.jacobian(y, x, i=3, j=0)
        dT_y = dde.grad.jacobian(y, x, i=3, j=1)
        dT_t = dde.grad.jacobian(y, x, i=3, j=2)
        
        # Second derivatives
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        dT_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
        dT_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)
        
        # Equations
        continuity = du_x + dv_y
        momentum_x = self.rho * (du_t + u * du_x + v * du_y) + dp_x - self.mu * (du_xx + du_yy)
        momentum_y = self.rho * (dv_t + u * dv_x + v * dv_y) + dp_y - self.mu * (dv_xx + dv_yy)
        energy = self.rho * self.cp * (dT_t + u * dT_x + v * dT_y) - self.k * (dT_xx + dT_yy)
        
        return [continuity, momentum_x, momentum_y, energy]
    
    def _generate_data(self, nx: int, ny: int, nt: int):
        """Generate training and testing data."""
        # Create grid points
        x = np.linspace(0, self.L, nx)
        y = np.linspace(0, self.H, ny)
        t = np.linspace(0, 30, nt)
        
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
        
        # Initial analytical solution
        u = self.U_in * (1 - (points[:, 1]/self.H)**2)
        v = np.zeros_like(u)
        p = self.rho * self.U_in**2 * (1 - points[:, 0]/self.L)
        T = 300 + 20 * (points[:, 1]/self.H)
        
        # Convert to torch tensors and move to GPU if available
        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        values = torch.tensor(np.stack([u, v, p, T], axis=1), dtype=torch.float32, device=self.device)
        
        return points, values
        
    def _setup_training_data(self):
        """Set up training data."""
        train_points, train_values = self._generate_data(20, 10, 5)
        test_points, test_values = self._generate_data(10, 5, 3)
        
        # Create observation points for BC
        self.observe_x = dde.icbc.PointSetBC(
            points=train_points.cpu().numpy(),
            values=train_values[:, 0:1].cpu().numpy(),
            component=0
        )
        self.observe_y = dde.icbc.PointSetBC(
            points=train_points.cpu().numpy(),
            values=train_values[:, 1:2].cpu().numpy(),
            component=1
        )
        
    def _create_model(self):
        """Create the PINN model."""
        # Network architecture
        layers = [3] + [50] * 6 + [4]
        activation = "tanh"
        
        # Create neural network with updated initialization
        self.net = dde.nn.FNN(
            input_size=layers[0],
            output_size=layers[-1],
            hidden_layers=layers[1:-1],
            activation=activation
        )
        
        # Move network to GPU if available
        if torch.cuda.is_available():
            self.net.cuda()
        
        # Create data
        data = dde.data.TimePDE(
            geometry=self.geomtime,
            pde=self._navier_stokes_pde,
            bcs=[self.observe_x, self.observe_y],
            num_domain=1000,
            num_boundary=200,
            num_initial=100
        )
        
        # Create model
        self.model = dde.Model(data, self.net)
        
    def train(self, epochs: int = 10000) -> Dict:
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs")
        
        # First stage training with Adam
        self.model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        loss_history, train_state = self.model.train(
            iterations=epochs//2,
            display_every=100
        )
        
        # Second stage training with L-BFGS
        self.model.compile("L-BFGS", metrics=["l2 relative error"])
        loss_history, train_state = self.model.train(
            iterations=epochs//2,
            display_every=100
        )
        
        return loss_history
    
    def predict(self, x_star: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Make predictions at given points."""
        # Convert input to tensor and move to GPU if available
        x_star_tensor = torch.tensor(x_star, dtype=torch.float32, device=self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model.predict(x_star_tensor.cpu().numpy())
            predictions = torch.tensor(predictions, device=self.device)
        
        return (
            predictions[:, 0:1].cpu().numpy(),  # u
            predictions[:, 1:2].cpu().numpy(),  # v
            predictions[:, 2:3].cpu().numpy(),  # p
            predictions[:, 3:4].cpu().numpy()   # T
        )
    
    def plot_results(self, t: float, save_path: str = None) -> None:
        """Plot the flow field results at a specific time."""
        logger.info(f"Plotting results at t = {t}")
        
        # Create grid
        nx, ny = 100, 50
        x = np.linspace(0, self.L, nx)
        y = np.linspace(0, self.H, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create input points
        x_star = np.vstack((
            X.flatten(),
            Y.flatten(),
            t * np.ones_like(X.flatten())
        )).T
        
        # Get predictions
        u, v, p, T = self.predict(x_star)
        
        # Reshape predictions
        u = u.reshape((nx, ny))
        v = v.reshape((nx, ny))
        p = p.reshape((nx, ny))
        T = T.reshape((nx, ny))
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Velocity magnitude plot
        plt.subplot(2, 2, 1)
        vel_mag = np.sqrt(u**2 + v**2)
        plt.contourf(X, Y, vel_mag, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(x, y, u.T, v.T, color='k', density=1.5)
        plt.title('Velocity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Pressure plot
        plt.subplot(2, 2, 2)
        plt.contourf(X, Y, p, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Pressure (Pa)')
        plt.title('Pressure Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Temperature plot
        plt.subplot(2, 2, 3)
        plt.contourf(X, Y, T, levels=50, cmap='hot')
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
        plt.contourf(X, Y, vorticity, levels=50, cmap='RdBu')
        plt.colorbar(label='Vorticity (1/s)')
        plt.title('Vorticity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        plt.suptitle(f'Flow Field Results at t = {t:.2f} s')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.close()

if __name__ == "__main__":
    # Set domain parameters
    domain_params = {
        'L': 25 * 0.00116,  # Channel length
        'H': 0.5 * 0.00116  # Channel height
    }
    
    try:
        # Create and train model
        model = FlowDeepONet(domain_params)
        loss_history = model.train(epochs=10000)
        
        # Plot results at different times
        for t in [5.0, 15.0, 30.0]:
            model.plot_results(t, save_path=f'flow_field_t{t:.1f}.png')
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise