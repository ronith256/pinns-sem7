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
    
    def _generate_data(self, nx: int, ny: int, nt: int):
        """Generate training and testing data."""
        # Create grid points
        x = np.linspace(0, self.L, nx)
        y = np.linspace(0, self.H, ny)
        t = np.linspace(0, 30, nt)
        
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
        
        # Synthetic solution function
        def analytical_solution(points):
            x, y, t = points[:, 0], points[:, 1], points[:, 2]
            
            # Velocity components
            u = self.U_in * (1 - np.exp(-0.1 * t)) * (1 - (y/self.H)**2)
            v = np.zeros_like(u)
            
            # Pressure
            p = self.rho * self.U_in**2 * (1 - x/self.L)
            
            # Temperature
            T = 300 + 20 * (y/self.H) * (1 - np.exp(-0.05 * t))
            
            return np.stack([u, v, p, T], axis=1)
        
        # Generate solutions
        values = analytical_solution(points)
        
        return points, values
        
    def _setup_training_data(self):
        """Set up training data for DeepONet."""
        # Generate training data
        train_points, train_values = self._generate_data(20, 10, 5)
        
        # Generate testing data (coarser grid)
        test_points, test_values = self._generate_data(10, 5, 3)
        
        # Create input features for branch net (sensor measurements)
        n_sensors = 10
        sensor_points, sensor_values = self._generate_data(n_sensors, n_sensors, 1)
        self.sensor_data = sensor_values.flatten()
        
        # Store data
        self.train_x = (self.sensor_data, train_points)
        self.train_y = train_values
        self.test_x = (self.sensor_data, test_points)
        self.test_y = test_values
        
    def _create_model(self):
        """Create the DeepONet model."""
        # Set up networks
        branch_net = dde.nn.FNN(
            layer_sizes=[self.sensor_data.size, 128, 128, self.n_branch],
            activation="tanh",
            kernel_initializer="Glorot uniform"
        )
        
        trunk_net = dde.nn.FNN(
            layer_sizes=[3, self.n_trunk, self.n_trunk, self.n_trunk, 4],
            activation="tanh",
            kernel_initializer="Glorot uniform"
        )
        
        # Create data
        data = dde.data.Triple(
            X_train=self.train_x,
            y_train=self.train_y,
            X_test=self.test_x,
            y_test=self.test_y
        )
        
        # Create model
        self.net = dde.nn.DeepONet(
            branch_net=branch_net,
            trunk_net=trunk_net,
            output_dim=4
        )
        
        # Define loss weights
        weights = {
            "data": 1.0,
            "continuity": 0.1,
            "momentum_x": 0.1,
            "momentum_y": 0.1,
            "energy": 0.1
        }
        
        # Custom loss function including physics constraints
        def custom_loss(model, batch_data):
            x_branch, x_trunk = batch_data[0]
            y_true = batch_data[1]
            
            # Data loss
            y_pred = model(x_branch, x_trunk)
            loss_data = torch.mean((y_pred - y_true) ** 2)
            
            # Physics loss (simplified)
            u = y_pred[:, 0:1]
            v = y_pred[:, 1:2]
            
            # Compute derivatives
            du_dx = dde.grad.jacobian(u, x_trunk[:, 0:1])
            dv_dy = dde.grad.jacobian(v, x_trunk[:, 1:2])
            
            # Continuity equation
            loss_continuity = torch.mean((du_dx + dv_dy) ** 2)
            
            # Total loss
            total_loss = (
                weights["data"] * loss_data +
                weights["continuity"] * loss_continuity
            )
            
            return total_loss
        
        # Compile model
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
            epochs=epochs
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
        # Prepare input data
        x_branch = np.tile(self.sensor_data, (x_star.shape[0], 1))
        predictions = self.net.predict((x_branch, x_star))
        
        return (
            predictions[:, 0:1],  # u
            predictions[:, 1:2],  # v
            predictions[:, 2:3],  # p
            predictions[:, 3:4]   # T
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
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))
        p = p.reshape((ny, nx))
        T = T.reshape((ny, nx))
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Velocity magnitude plot
        plt.subplot(2, 2, 1)
        vel_mag = np.sqrt(u**2 + v**2)
        plt.contourf(X, Y, vel_mag.T, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(x, y, u.T, v.T, color='k', density=1.5)
        plt.title('Velocity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Pressure plot
        plt.subplot(2, 2, 2)
        plt.contourf(X, Y, p.T, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Pressure (Pa)')
        plt.title('Pressure Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Temperature plot
        plt.subplot(2, 2, 3)
        plt.contourf(X, Y, T.T, levels=50, cmap='hot')
        plt.colorbar(label='Temperature (K)')
        plt.title('Temperature Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Vorticity plot
        plt.subplot(2, 2, 4)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        vorticity = dvdx - dudy
        plt.contourf(X, Y, vorticity.T, levels=50, cmap='RdBu')
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