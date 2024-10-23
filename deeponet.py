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
        self._setup_pde()
        self._create_model()
        
    def _setup_geometry(self):
        """Set up the geometry and time domain."""
        # Spatial domain: [0, L] x [0, H]
        self.geom = dde.geometry.Rectangle([0, 0], [self.L, self.H])
        
        # Time domain: [0, 30]
        self.timedomain = dde.geometry.TimeDomain(0, 30)
        
        # Spatio-temporal domain
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)
        
    def _setup_pde(self):
        """Set up the PDE system."""
        def pde(x, y):
            """
            Define the PDE system.
            x: input coordinates (x, y, t)
            y: network output (u, v, p, T)
            """
            u, v, p, T = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
            
            # Get derivatives
            du_x = dde.grad.jacobian(y, x, i=0, j=0)
            du_y = dde.grad.jacobian(y, x, i=0, j=1)
            du_t = dde.grad.jacobian(y, x, i=0, j=2)
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_yy = dde.grad.hessian(y, x, i=0, j=1)
            
            dv_x = dde.grad.jacobian(y, x, i=1, j=0)
            dv_y = dde.grad.jacobian(y, x, i=1, j=1)
            dv_t = dde.grad.jacobian(y, x, i=1, j=2)
            dv_xx = dde.grad.hessian(y, x, i=1, j=0)
            dv_yy = dde.grad.hessian(y, x, i=1, j=1)
            
            dp_x = dde.grad.jacobian(y, x, i=2, j=0)
            dp_y = dde.grad.jacobian(y, x, i=2, j=1)
            
            dT_x = dde.grad.jacobian(y, x, i=3, j=0)
            dT_y = dde.grad.jacobian(y, x, i=3, j=1)
            dT_t = dde.grad.jacobian(y, x, i=3, j=2)
            dT_xx = dde.grad.hessian(y, x, i=3, j=0)
            dT_yy = dde.grad.hessian(y, x, i=3, j=1)
            
            # Continuity equation
            continuity = du_x + dv_y
            
            # Momentum equations
            momentum_x = (du_t + u * du_x + v * du_y + 
                        (1/self.rho) * dp_x - self.nu * (du_xx + du_yy))
            
            momentum_y = (dv_t + u * dv_x + v * dv_y + 
                        (1/self.rho) * dp_y - self.nu * (dv_xx + dv_yy))
            
            # Energy equation
            energy = (dT_t + u * dT_x + v * dT_y - 
                     (self.k/(self.rho * self.cp)) * (dT_xx + dT_yy))
            
            return [continuity, momentum_x, momentum_y, energy]
        
        self.pde = pde
        
    def _create_model(self):
        """Create the DeepONet model."""
        # Define branch and trunk nets
        branch_net = dde.nn.DeepONetCartesianProd(
            [self.n_branch, self.n_branch, self.n_branch, self.n_branch],
            [128, 128, 128],
            "tanh",
            "Glorot normal"
        )
        
        trunk_net = dde.nn.FNN(
            [3] + [self.n_trunk] * 3 + [4],
            "tanh",
            "Glorot normal"
        )
        
        # Create DeepONet
        self.net = dde.nn.DeepONet(
            branch_net,
            trunk_net,
            self.pde,
            self.geomtime,
            num_domain=10000,
            num_boundary=2000,
            num_initial=1000,
            num_test=1000
        )
        
        # Add boundary conditions
        def inlet_bc(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)
        
        def outlet_bc(x, on_boundary):
            return on_boundary and np.isclose(x[0], self.L)
        
        def bottom_bc(x, on_boundary):
            return on_boundary and np.isclose(x[1], 0)
        
        def top_bc(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.H)
        
        # Inlet conditions
        self.net.add_boundary_condition(
            lambda x: self.U_in - dde.grad.jacobian(self.net(x), x, i=0, j=0),
            inlet_bc
        )
        self.net.add_boundary_condition(
            lambda x: dde.grad.jacobian(self.net(x), x, i=1, j=0),
            inlet_bc
        )
        self.net.add_boundary_condition(
            lambda x: 300 - self.net(x)[:, 3:4],
            inlet_bc
        )
        
        # Outlet conditions
        self.net.add_boundary_condition(
            lambda x: dde.grad.jacobian(self.net(x), x, i=0, j=0),
            outlet_bc
        )
        self.net.add_boundary_condition(
            lambda x: dde.grad.jacobian(self.net(x), x, i=1, j=0),
            outlet_bc
        )
        self.net.add_boundary_condition(
            lambda x: dde.grad.jacobian(self.net(x), x, i=3, j=0),
            outlet_bc
        )
        
        # Bottom wall conditions
        self.net.add_boundary_condition(
            lambda x: self.net(x)[:, 0:1],
            bottom_bc
        )
        self.net.add_boundary_condition(
            lambda x: self.net(x)[:, 1:2],
            bottom_bc
        )
        self.net.add_boundary_condition(
            lambda x: -self.k * dde.grad.jacobian(self.net(x), x, i=3, j=1) - self.q_flux,
            bottom_bc
        )
        
        # Top wall conditions
        self.net.add_boundary_condition(
            lambda x: self.net(x)[:, 0:1],
            top_bc
        )
        self.net.add_boundary_condition(
            lambda x: self.net(x)[:, 1:2],
            top_bc
        )
        self.net.add_boundary_condition(
            lambda x: dde.grad.jacobian(self.net(x), x, i=3, j=1),
            top_bc
        )
        
    def train(self, epochs: int = 10000) -> Dict:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
        
        # Train the model
        loss_history = self.net.train(
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            display_every=100
        )
        
        return loss_history
    
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