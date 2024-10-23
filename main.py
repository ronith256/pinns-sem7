from deeponet import * 
from flow_visualizer import *
from fno import * 
from nsfnet import *
from vanilla import * 

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import os

def get_device():
    """Get the available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(model_name: str, history: list, fig: plt.Figure, save_dir: str = 'results'):
    """Save training history and figures"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    np.save(os.path.join(save_dir, f'{model_name}_history.npy'), np.array(history))
    
    # Save figure
    fig.savefig(os.path.join(save_dir, f'{model_name}_comparison.png'))
    plt.close(fig)

def compare_and_analyze_models():
    # Get device and set up parameters
    device = get_device()
    
    # Set up domain and physics parameters
    domain_params = {
        'L': 25 * 0.00116,  # Length
        'H': 0.5 * 0.00116,  # Height
        'Re': 200,  # Reynolds number
        'Q_flux': 20000  # Heat flux
    }

    physics_params = {
        'rho': 997.07,  # Density
        'mu': 8.9e-4,   # Dynamic viscosity
        'k': 0.606,     # Thermal conductivity
        'cp': 4200      # Specific heat capacity
    }

    # Initialize solvers with device
    solvers = {
        'Vanilla PINN': VanillaPINNSolver(domain_params, physics_params).to(device)
        # 'DeepONet': DeepONetPINNSolver(domain_params, physics_params).to(device),
        # 'NSFNet': NSFNetSolver(domain_params, physics_params).to(device),
        # 'FNO': FNOPINNSolver(domain_params, physics_params).to(device)
    }

    # Dictionary to store training histories
    training_histories = {}

    # Train models
    epochs = 100
    print("Training models...")
    
    try:
        for name, solver in solvers.items():
            print(f"\nTraining {name}...")
            history = solver.train(epochs)
            training_histories[name] = history
            
            # Save intermediate results
            save_results(name.lower().replace(" ", "_"), 
                       history, 
                       plt.figure(),  # Create empty figure as placeholder
                       'intermediate_results')
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None, None, None

    # Initialize visualizer
    visualizer = FlowVisualizer(domain_params)
    
    # Compare models at different time steps
    time_points = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0]
    
    for t in time_points:
        print(f"\nComparing models at t = {t}s")
        try:
            fig = visualizer.compare_models(solvers, t)
            plt.savefig(f'results/comparison_t{t}.png')
            plt.close()
        except Exception as e:
            print(f"Error generating comparison at t={t}: {str(e)}")

    # Plot training histories
    try:
        plt.figure(figsize=(10, 6))
        for name, history in training_histories.items():
            plt.plot(history, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training History Comparison')
        plt.legend()
        plt.savefig('results/training_history.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

    # Compute error metrics
    def compute_error_metrics(prediction, reference):
        return {
            'mae': torch.mean(torch.abs(prediction - reference)).item(),
            'rmse': torch.sqrt(torch.mean(torch.square(prediction - reference))).item(),
            'max_error': torch.max(torch.abs(prediction - reference)).item()
        }

    # Compare solutions at t = 30s
    t_final = 30.0
    try:
        x_eval, t_eval = visualizer.create_input_tensor(t_final)
        x_eval = x_eval.to(device)
        t_eval = t_eval.to(device)
        
        # Use FNO solution as reference
        u_ref, v_ref, p_ref, T_ref = solvers['FNO'].predict(x_eval, t_eval)
        
        error_metrics = {}
        for name, solver in solvers.items():
            if name == 'FNO':
                continue
            
            u, v, p, T = solver.predict(x_eval, t_eval)
            error_metrics[name] = {
                'velocity': compute_error_metrics(torch.stack([u, v]), torch.stack([u_ref, v_ref])),
                'pressure': compute_error_metrics(p, p_ref),
                'temperature': compute_error_metrics(T, T_ref)
            }

        # Save error metrics
        np.save('results/error_metrics.npy', error_metrics)
        
        # Print error metrics
        print("\nError Metrics (compared to FNO solution):")
        for model_name, metrics in error_metrics.items():
            print(f"\n{model_name}:")
            for field_name, field_metrics in metrics.items():
                print(f"  {field_name}:")
                for metric_name, value in field_metrics.items():
                    print(f"    {metric_name}: {value:.6f}")
                    
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        error_metrics = None

    return solvers, visualizer, training_histories, error_metrics

def create_animations(solvers, visualizer):
    """Create animations for all models"""
    t_range = np.linspace(0, 30, 31)
    device = next(iter(solvers.values())).device  # Get device from first solver
    
    for name, solver in solvers.items():
        print(f"\nCreating animation for {name}...")
        try:
            anim = visualizer.create_animation(
                solver, 
                t_range, 
                device=device,
                save_path=f'results/{name.lower().replace(" ", "_")}_animation.gif'
            )
        except Exception as e:
            print(f"Error creating animation for {name}: {str(e)}")

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)

    # Run comparison
    solvers, visualizer, training_histories, error_metrics = compare_and_analyze_models()
    
    # Create animations if training was successful
    if solvers is not None:
        create_animations(solvers, visualizer)