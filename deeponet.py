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
        
        # Network parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_branch = 40  # Number of branch neurons
        self.n_trunk = 128  # Number of trunk neurons
        
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
        
    def _generate_sensor_locations(self):
        """Generate sensor locations for training data."""
        nx_sensors = 10
        ny_sensors = 5
        
        x = np.linspace(0, self.L, nx_sensors)
        y = np.linspace(0, self.H, ny_sensors)
        t = np.linspace(0, 30, 10)
        
        X, Y, T = np.meshgrid(x, y, t)
        sensor_locations = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
        
        return sensor_locations
        
    def _setup_training_data(self):
        """Set up training data for DeepONet."""
        # Generate sensor locations
        sensor_locations = self._generate_sensor_locations()
        
        # Generate synthetic data at sensor locations
        def analytical_solution(x, y, t):
            """Simple analytical solution for testing."""
            u = self.U_in * (1 - np.exp(-0.1 * t)) * (1 - (y/self.H)**2)
            v = np.zeros_like(u)
            p = self.rho * self.U_in**2 * (1 - x/self.L)
            T = 300 + 20 * (y/self.H)
            return np.stack([u, v, p, T], axis=1)
        
        sensor_data = analytical_solution(
            sensor_locations[:, 0],
            sensor_locations[:, 1],
            sensor_locations[:, 2]
        )
        
        # Create training data
        self.train_x = sensor_locations
        self.train_y = sensor_data
        
    def _create_model(self):
        """Create the DeepONet model."""
        # Define branch and trunk nets
        branch_net = dde.nn.FNN(
            [self.train_x.shape[1]] + [128] * 3 + [self.n_branch],
            "tanh",
            "Glorot normal"
        )
        
        trunk_net = dde.nn.FNN(
            [3] + [self.n_trunk] * 3 + [4],
            "tanh",
            "Glorot normal"
        )
        
        # Create data
        data = dde.data.TripleCartesianProd(
            X_train=self.train_x,
            y_train=self.train_y,
            X_test=None
        )
        
        # Create DeepONet
        self.net = dde.nn.DeepONet(
            branch_net=branch_net,
            trunk_net=trunk_net,
            data=data,
            output_dim=4
        )
        
        # Create loss function combining data loss and physics constraints
        def custom_loss(inputs, outputs, model):
            loss_data = torch.mean((outputs - model(inputs))**2)
            
            # Add physics-based constraints
            x, y, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
            u, v, p, T = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4]
            
            # Compute derivatives
            du_dx = dde.grad.jacobian(u, x)
            dv_dy = dde.grad.jacobian(v, y)
            
            # Continuity equation
            loss_continuity = torch.mean((du_dx + dv_dy)**2)
            
            # Simplified momentum and energy constraints
            loss_physics = loss_continuity
            
            return loss_data + 0.1 * loss_physics
        
        self.net.compile("adam", lr=1e-3, loss=custom_loss)
        
    def train(self, epochs: int = 10000) -> Dict:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        losshistory, train_state = self.net.train(
            epochs=epochs,
            display_every=100
        )
        
        return losshistory
    
    def predict(self, x_star: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Make predictions at given points.
        
        Args:
            x_star: Points to evaluate at, shape (n_points, 3)
            
        Returns:
            Tuple of (u, v, p, T) predictions
        """
        y_pred = self.net.predict(x_star)
        return (
            y_pred[:, 0:1],  # u
            y_pred[:, 1:2],  # v
            y_pred[:, 2:3],  # p
            y_pred[:, 3:4]   # T
        )
    
    def plot_results(self, t: float, save_path: str = None) -> None:
        """
        Plot the flow field results at a specific time.
        
        Args:
            t: Time to plot at
            save_path: Optional path to save the figure
        """
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
        u = u.reshape(nx, ny)
        v = v.reshape(nx, ny)
        p = p.reshape(nx, ny)
        T = T.reshape(nx, ny)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Velocity magnitude plot
        plt.subplot(2, 2, 1)
        vel_mag = np.sqrt(u**2 + v**2)
        plt.contourf(X, Y, vel_mag, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(X, Y, u, v, color='k', density=1.5)
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
        logger.error(f"Error in main execution: {str(e)}")
        raise