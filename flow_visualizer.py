import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, Optional, Union
from vanilla import FlowPINN

class FlowVisualizer:
    def __init__(self, domain_params: Dict[str, float]):
        """
        Initialize the flow visualizer with domain parameters.
        
        Args:
            domain_params: Dictionary containing 'L' (length) and 'H' (height) of the domain
        """
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up plotting parameters
        plt.style.use('seaborn')
        self.figsize = (15, 10)
        self.cmap = 'RdYlBu_r'
        
    def _create_mesh(self, nx: int = 100, ny: int = 50) -> tuple:
        """Create a mesh for visualization."""
        x = torch.linspace(0, self.L, nx, device=self.device)
        y = torch.linspace(0, self.H, ny, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return X, Y, xy
    
    def plot_flow_field(self, model: FlowPINN, t: float, 
                       title: str = "Flow Field", 
                       save_path: Optional[str] = None,
                       nx: int = 100, ny: int = 50) -> plt.Figure:
        """
        Plot the flow field at a specific time.
        
        Args:
            model: Trained FlowPINN model
            t: Time at which to visualize the flow
            title: Plot title
            save_path: Optional path to save the figure
            nx: Number of points in x direction
            ny: Number of points in y direction
        
        Returns:
            matplotlib Figure object
        """
        model.eval()
        X, Y, xy = self._create_mesh(nx, ny)
        t_tensor = torch.full((xy.shape[0], 1), t, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            u, v, p, T = model.predict(xy, t_tensor)
            
        # Reshape predictions
        u = u.reshape(nx, ny).cpu().numpy()
        v = v.reshape(nx, ny).cpu().numpy()
        p = p.reshape(nx, ny).cpu().numpy()
        T = T.reshape(nx, ny).cpu().numpy()
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        
        # Velocity magnitude plot
        plt.subplot(2, 2, 1)
        vel_mag = np.sqrt(u**2 + v**2)
        plt.contourf(X, Y, vel_mag, levels=50, cmap=self.cmap)
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(X, Y, u, v, color='k', density=1.5, linewidth=0.5)
        plt.title('Velocity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Pressure plot
        plt.subplot(2, 2, 2)
        plt.contourf(X, Y, p, levels=50, cmap=self.cmap)
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
        dudy, dudx = np.gradient(u, Y[:, 0], X[0, :])
        dvdy, dvdx = np.gradient(v, Y[:, 0], X[0, :])
        vorticity = dvdx - dudy
        plt.contourf(X, Y, vorticity, levels=50, cmap='RdBu')
        plt.colorbar(label='Vorticity (1/s)')
        plt.title('Vorticity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        plt.suptitle(f'{title} at t = {t:.2f} s')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_animation(self, model: FlowPINN, t_range: Union[list, np.ndarray], 
                        save_path: str, nx: int = 100, ny: int = 50) -> None:
        """
        Create an animation of the flow field over time.
        
        Args:
            model: Trained FlowPINN model
            t_range: Array of time points to animate
            save_path: Path to save the animation
            nx: Number of points in x direction
            ny: Number of points in y direction
        """
        fig = plt.figure(figsize=self.figsize)
        
        X, Y, xy = self._create_mesh(nx, ny)
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        
        def update(frame):
            plt.clf()
            t = t_range[frame]
            t_tensor = torch.full((xy.shape[0], 1), t, device=self.device)
            
            with torch.no_grad():
                u, v, p, T = model.predict(xy, t_tensor)
            
            u = u.reshape(nx, ny).cpu().numpy()
            v = v.reshape(nx, ny).cpu().numpy()
            vel_mag = np.sqrt(u**2 + v**2)
            
            plt.contourf(X, Y, vel_mag, levels=50, cmap=self.cmap)
            plt.colorbar(label='Velocity Magnitude (m/s)')
            plt.streamplot(X, Y, u, v, color='k', density=1.5, linewidth=0.5)
            plt.title(f'Flow Field at t = {t:.2f} s')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            
        anim = FuncAnimation(fig, update, frames=len(t_range), 
                           interval=100, blit=False)
        anim.save(save_path, writer='pillow', fps=10)
        plt.close()
    
    def plot_history(self, loss_history: dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            loss_history: Dictionary containing loss values over epochs
            save_path: Optional path to save the figure
        
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(10, 6))
        epochs = range(1, len(loss_history['total_loss']) + 1)
        
        plt.semilogy(epochs, loss_history['total_loss'], label='Total Loss')
        plt.semilogy(epochs, loss_history['pde_loss'], label='PDE Loss')
        plt.semilogy(epochs, loss_history['bc_loss'], label='BC Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig