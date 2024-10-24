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
        print("\nCreating mesh...")
        x = torch.linspace(0, self.L, nx, device=self.device)
        y = torch.linspace(0, self.H, ny, device=self.device)
        print(f"x range: [{x[0].item():.6f}, {x[-1].item():.6f}], length: {len(x)}")
        print(f"y range: [{y[0].item():.6f}, {y[-1].item():.6f}], length: {len(y)}")
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        print(f"X shape after meshgrid: {X.shape}")
        print(f"Y shape after meshgrid: {Y.shape}")
        
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
        print(f"xy shape: {xy.shape}\n")
        return X, Y, xy
    
    def _compute_vorticity(self, u: np.ndarray, v: np.ndarray, 
                          dx: float, dy: float) -> np.ndarray:
        """
        Compute vorticity using central differences with handling for edge cases.
        """
        dudy = np.zeros_like(u)
        dvdx = np.zeros_like(v)
        
        # Interior points - central difference
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        
        # Edge points - forward/backward difference
        dudy[0, :] = (u[1, :] - u[0, :]) / dy
        dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy
        dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx
        dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx
        
        return dvdx - dudy
    
    def plot_flow_field(self, model: FlowPINN, t: float, 
                       title: str = "Flow Field", 
                       save_path: Optional[str] = None,
                       nx: int = 100, ny: int = 50) -> plt.Figure:
        """
        Plot the flow field at a specific time.
        """
        model.eval()
        X, Y, xy = self._create_mesh(nx, ny)
        t_tensor = torch.full((xy.shape[0], 1), t, device=self.device)
        print(f"Time tensor shape: {t_tensor.shape}")
        
        # Get predictions
        with torch.no_grad():
            u, v, p, T = model.predict(xy, t_tensor)
            print(f"\nPrediction shapes:")
            print(f"u: {u.shape}, v: {v.shape}, p: {p.shape}, T: {T.shape}")
        
        # Reshape predictions and convert to numpy
        u = u.reshape(nx, ny).cpu().numpy()
        v = v.reshape(nx, ny).cpu().numpy()
        p = p.reshape(nx, ny).cpu().numpy()
        T = T.reshape(nx, ny).cpu().numpy()
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        
        print(f"\nArray shapes after reshape:")
        print(f"u: {u.shape}, v: {v.shape}")
        print(f"X: {X.shape}, Y: {Y.shape}")
        
        # Calculate grid spacing
        dx = X[1, 0] - X[0, 0]
        dy = Y[0, 1] - Y[0, 0]
        print(f"\nGrid spacing - dx: {dx:.6f}, dy: {dy:.6f}")
        
        # Create meshgrid for streamplot
        x_1d = np.linspace(0, self.L, nx)
        y_1d = np.linspace(0, self.H, ny)
        X_stream, Y_stream = np.meshgrid(x_1d, y_1d)
        
        print(f"\nStreamplot grid shapes:")
        print(f"X_stream: {X_stream.shape}, Y_stream: {Y_stream.shape}")
        print("Sample X_stream first row:", X_stream[0, :3])
        print("Sample Y_stream first column:", Y_stream[:3, 0])
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        
        # Velocity magnitude plot
        plt.subplot(2, 2, 1)
        vel_mag = np.sqrt(u**2 + v**2)
        vel_plot = plt.contourf(X, Y, vel_mag, levels=50, cmap=self.cmap)
        plt.colorbar(vel_plot, label='Velocity Magnitude (m/s)')
        
        # Add streamlines
        # Transpose arrays for streamplot
        u_stream = u.T
        v_stream = v.T
        print(f"\nStreamplot input shapes:")
        print(f"u_stream: {u_stream.shape}, v_stream: {v_stream.shape}")
        
        plt.streamplot(X_stream[0, :], Y_stream[:, 0],
                      u_stream, v_stream,
                      color='k', density=1.5, linewidth=0.5,
                      arrowsize=0.5)
        
        plt.title('Velocity Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Pressure plot
        plt.subplot(2, 2, 2)
        p_plot = plt.contourf(X, Y, p, levels=50, cmap=self.cmap)
        plt.colorbar(p_plot, label='Pressure (Pa)')
        plt.title('Pressure Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Temperature plot
        plt.subplot(2, 2, 3)
        T_plot = plt.contourf(X, Y, T, levels=50, cmap='hot')
        plt.colorbar(T_plot, label='Temperature (K)')
        plt.title('Temperature Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # Vorticity plot
        plt.subplot(2, 2, 4)
        vorticity = self._compute_vorticity(u, v, dx, dy)
        vmax = np.max(np.abs(vorticity))
        v_plot = plt.contourf(X, Y, vorticity, levels=50,
                             cmap='RdBu', vmin=-vmax, vmax=vmax)
        plt.colorbar(v_plot, label='Vorticity (1/s)')
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
        """
        fig = plt.figure(figsize=self.figsize)
        
        X, Y, xy = self._create_mesh(nx, ny)
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        
        # Create 1D arrays for streamplot
        x_1d = np.linspace(0, self.L, nx)
        y_1d = np.linspace(0, self.H, ny)
        
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
            
            plt.streamplot(x_1d, y_1d, u.T, v.T,
                         color='k', density=1.5, linewidth=0.5,
                         arrowsize=0.5)
            
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