import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from typing import Dict, Any, Tuple

class FlowVisualizer:
    def __init__(self, domain_params: Dict[str, Any]):
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.nx = 501
        self.ny = 51
        
        # Create grid
        self.x = np.linspace(0, self.L, self.nx)
        self.y = np.linspace(0, self.H, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def create_input_tensor(self, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input tensor for the models"""
        x_flat = torch.tensor(np.stack([self.X.flatten(), self.Y.flatten()], axis=1),
                            dtype=torch.float32)
        t_tensor = torch.ones(x_flat.shape[0], 1) * t
        return x_flat, t_tensor
    
    def plot_flow_field(self, model: torch.nn.Module, t: float, title: str):
        """Plot flow field at a given time"""
        x_flat, t_tensor = self.create_input_tensor(t)
        u, v, p, T = model.predict(x_flat, t_tensor)
        
        # Reshape predictions
        u = u.numpy().reshape(self.ny, self.nx)
        v = v.numpy().reshape(self.ny, self.nx)
        p = p.numpy().reshape(self.ny, self.nx)
        T = T.numpy().reshape(self.ny, self.nx)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{title} at t = {t:.2f}s')
        
        # Velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        im1 = ax1.pcolormesh(self.X, self.Y, vel_mag, shading='auto')
        ax1.set_title('Velocity Magnitude')
        plt.colorbar(im1, ax=ax1)
        
        # Streamlines
        ax2.streamplot(self.x, self.y, u.T, v.T, density=1.5)
        ax2.set_title('Streamlines')
        
        # Pressure
        im3 = ax3.pcolormesh(self.X, self.Y, p, shading='auto')
        ax3.set_title('Pressure')
        plt.colorbar(im3, ax=ax3)
        
        # Temperature
        im4 = ax4.pcolormesh(self.X, self.Y, T, shading='auto')
        ax4.set_title('Temperature')
        plt.colorbar(im4, ax=ax4)
        
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, model: torch.nn.Module, t_range: np.ndarray, 
                        save_path: str = None):
        """Create animation of the flow field evolution"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        def update(frame):
            t = t_range[frame]
            x_flat, t_tensor = self.create_input_tensor(t)
            # Move tensors to model's device
            x_flat = x_flat.to(model.device)
            t_tensor = t_tensor.to(model.device)
            u, v, p, T = model.predict(x_flat, t_tensor)
            
            # Move results back to CPU for plotting
            u = u.cpu().numpy().reshape(self.ny, self.nx)
            v = v.cpu().numpy().reshape(self.ny, self.nx)
            p = p.cpu().numpy().reshape(self.ny, self.nx)
            T = T.cpu().numpy().reshape(self.ny, self.nx)
            
            # Clear previous plots
            for ax in (ax1, ax2, ax3, ax4):
                ax.clear()
            
            # Update plots
            vel_mag = np.sqrt(u**2 + v**2)
            ax1.pcolormesh(self.X, self.Y, vel_mag, shading='auto')
            ax1.set_title('Velocity Magnitude')
            
            ax2.streamplot(self.x, self.y, u.T, v.T, density=1.5)
            ax2.set_title('Streamlines')
            
            ax3.pcolormesh(self.X, self.Y, p, shading='auto')
            ax3.set_title('Pressure')
            
            ax4.pcolormesh(self.X, self.Y, T, shading='auto')
            ax4.set_title('Temperature')
            
            for ax in (ax1, ax2, ax3, ax4):
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal')
            
            fig.suptitle(f't = {t:.2f}s')
        
        anim = FuncAnimation(fig, update, frames=len(t_range),
                           interval=100, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        return anim
    
    def compare_models(self, models: Dict[str, torch.nn.Module], t: float):
        """Compare results from different models"""
        n_models = len(models)
        fig = plt.figure(figsize=(15, 4*n_models))
        
        for i, (name, model) in enumerate(models.items()):
            x_flat, t_tensor = self.create_input_tensor(t)
            x_flat = x_flat.to(model.device)  # Move to model's device
            t_tensor = t_tensor.to(model.device)
            u, v, p, T = model.predict(x_flat, t_tensor)
            
            # Move results back to CPU for plotting
            u = u.cpu().numpy().reshape(self.ny, self.nx)
            v = v.cpu().numpy().reshape(self.ny, self.nx)
            vel_mag = np.sqrt(u**2 + v**2)
            
            ax1 = fig.add_subplot(n_models, 2, 2*i + 1)
            im1 = ax1.pcolormesh(self.X, self.Y, vel_mag, shading='auto')
            ax1.set_title(f'{name} - Velocity Magnitude')
            plt.colorbar(im1, ax=ax1)
            
            ax2 = fig.add_subplot(n_models, 2, 2*i + 2)
            ax2.streamplot(self.x, self.y, u.T, v.T, density=1.5)
            ax2.set_title(f'{name} - Streamlines')
            
            for ax in (ax1, ax2):
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig