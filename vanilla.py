import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any

class FlowPINN(nn.Module):
    def __init__(self, domain_params: Dict[str, Any]):
        super().__init__()
        
        # Physical parameters
        self.rho = 997.07  # Density of water
        self.mu = 8.9e-4   # Dynamic viscosity
        self.nu = self.mu / self.rho  # Kinematic viscosity
        self.k = 0.606     # Thermal conductivity
        self.cp = 4200     # Specific heat capacity
        self.q_flux = 20000  # Heat flux
        
        # Domain parameters
        self.L = domain_params['L']
        self.H = domain_params['H']
        self.Re = 200      # Reynolds number
        
        # Calculate inlet velocity from Reynolds number
        self.Dh = 0.00116  # Hydraulic diameter
        self.U_in = (self.Re * self.mu) / (self.rho * self.Dh)
        
        # Neural network architecture
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
            nn.Linear(3, 128),  # Input: (x, y, t)
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 4)   # Output: (u, v, p, T)
        ).to(self.device)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Combine inputs
        inputs = torch.cat([x, t], dim=1)
        outputs = self.net(inputs)
        
        return (outputs[:, 0:1],   # u velocity
                outputs[:, 1:2],   # v velocity
                outputs[:, 2:3],   # pressure
                outputs[:, 3:4])   # temperature
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self.eval()
        with torch.no_grad():
            return self.forward(x, t)
    
    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u, v, p, T = self.forward(x, t)
        
        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 1:2]
        
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 0:1]
        T_y = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 1:2]
        T_t = torch.autograd.grad(T.sum(), t, create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
        
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0][:, 1:2]
        
        T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0][:, 0:1]
        T_yy = torch.autograd.grad(T_y.sum(), x, create_graph=True)[0][:, 1:2]
        
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
    
    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = self.compute_derivatives(x, t)
        
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
    
    def compute_bc_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = self.compute_derivatives(x, t)
        
        # Inlet conditions (x = 0)
        inlet_mask = x[:, 0:1] < 1e-5
        inlet_u = d['u'] - self.U_in * torch.ones_like(d['u'])
        inlet_v = d['v']
        inlet_T = d['T'] - 300.0
        
        # Outlet conditions (x = L)
        outlet_mask = torch.abs(x[:, 0:1] - self.L) < 1e-5
        outlet_u = d['u_x']
        outlet_v = d['v_x']
        outlet_T = d['T_x']
        
        # Bottom wall conditions (y = 0)
        bottom_mask = x[:, 1:2] < 1e-5
        bottom_u = d['u']
        bottom_v = d['v']
        bottom_T = -self.k * d['T_y'] - self.q_flux
        
        # Top wall conditions (y = H)
        top_mask = torch.abs(x[:, 1:2] - self.H) < 1e-5
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
    
    def compute_loss(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # PDE residuals
        pde_residuals = self.compute_pde_residuals(x, t)
        pde_loss = (torch.mean(pde_residuals['continuity']**2) +
                   torch.mean(pde_residuals['momentum_x']**2) +
                   torch.mean(pde_residuals['momentum_y']**2) +
                   torch.mean(pde_residuals['energy']**2))
        
        # Boundary condition residuals
        bc_residuals = self.compute_bc_residuals(x, t)
        bc_loss = sum(torch.mean(r**2) for r in bc_residuals.values())
        
        # Total loss
        total_loss = pde_loss + 10.0 * bc_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item()
        }
        
        return total_loss, loss_dict

def train_model(model: FlowPINN, num_epochs: int = 10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Generate training points
    nx, ny, nt = 50, 20, 10
    x = torch.linspace(0, model.L, nx, device=model.device)
    y = torch.linspace(0, model.H, ny, device=model.device)
    t = torch.linspace(0, 30, nt, device=model.device)
    
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    x_train = torch.stack([X.flatten(), Y.flatten()], dim=1)
    t_train = T.flatten().unsqueeze(1)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(x_train, t_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss_dict['total_loss']:.6f}, "
                  f"PDE Loss = {loss_dict['pde_loss']:.6f}, "
                  f"BC Loss = {loss_dict['bc_loss']:.6f}")
    
    return model

# Example usage
# if __name__ == "__main__":
#     domain_params = {
#         'L': 25 * 0.00116,  # Channel length (25 * Dh)
#         'H': 0.5 * 0.00116  # Channel height (0.5 * Dh)
#     }
    
#     # Initialize and train model
#     model = FlowPINN(domain_params)
#     trained_model = train_model(model)
    
#     # Initialize visualizer
#     visualizer = FlowVisualizer(domain_params)
    
#     # Create plots at different times
#     times = [0.0, 15.0, 30.0]
#     for t in times:
#         fig = visualizer.plot_flow_field(trained_model, t, f"Flow Field")
#         plt.savefig(f"flow_field_t{t:.1f}.png")
#         plt.close()
    
#     # Create animation
#     t_range = np.linspace(0, 30, 31)
#     anim = visualizer.create_animation(trained_model, t_range, "flow_animation.gif")