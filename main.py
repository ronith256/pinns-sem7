from deeponet import * 
from flow_visualizer import *
from fno import * 
from nsfnet import *
from vanilla import * 

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def compare_and_analyze_models():
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

    # Initialize solvers
    vanilla_solver = VanillaPINNSolver(domain_params, physics_params)
    deeponet_solver = DeepONetPINNSolver(domain_params, physics_params)
    nsfnet_solver = NSFNetSolver(domain_params, physics_params)
    fno_solver = FNOPINNSolver(domain_params, physics_params)

    # Dictionary to store training histories
    training_histories = {
        'Vanilla PINN': [],
        'DeepONet': [],
        'NSFNet': [],
        'FNO': []
    }

    # Train models
    epochs = 100
    print("Training models...")
    
    solvers = {
        'Vanilla PINN': vanilla_solver,
        'DeepONet': deeponet_solver,
        'NSFNet': nsfnet_solver,
        'FNO': fno_solver
    }

    # for name, solver in solvers.items():
    #     print(f"\nTraining {name}...")
    #     history = []
    #     for epoch in range(epochs):
    #         loss = solver.train_step(solver.x_domain, solver.x_boundary, solver.t)
    #         history.append(loss)
    #         if epoch % 100 == 0:
    #             print(f"Epoch {epoch}, Loss: {loss:.6f}")
    #     training_histories[name] = history
    for name, solver in solver.items():
        history = solver.train(epochs)
        training_histories[name] = history

    # Initialize visualizer
    visualizer = FlowVisualizer(domain_params)

    # Compare models at different time steps
    time_points = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0]
    
    for t in time_points:
        print(f"\nComparing models at t = {t}s")
        fig = visualizer.compare_models(solvers, t)
        plt.savefig(f'comparison_t{t}.png')
        plt.close()

    # Plot training histories
    plt.figure(figsize=(10, 6))
    for name, history in training_histories.items():
        plt.plot(history, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training History Comparison')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    # Compute error metrics
    def compute_error_metrics(prediction, reference):
        return {
            'mae': torch.mean(torch.abs(prediction - reference)),
            'rmse': torch.sqrt(torch.mean(torch.square(prediction - reference))),
            'max_error': torch.max(torch.abs(prediction - reference))
        }

    # Compare solutions at t = 30s
    t_final = 30.0
    x_eval, t_eval = visualizer.create_input_tensor(t_final)
    
    # Use FNO solution as reference (you might want to use actual experimental data if available)
    u_ref, v_ref, p_ref, T_ref = fno_solver.predict(x_eval, t_eval)
    
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

    # Print error metrics
    print("\nError Metrics (compared to FNO solution):")
    for model_name, metrics in error_metrics.items():
        print(f"\n{model_name}:")
        for field_name, field_metrics in metrics.items():
            print(f"  {field_name}:")
            for metric_name, value in field_metrics.items():
                print(f"    {metric_name}: {value:.6f}")

    return solvers, visualizer, training_histories, error_metrics

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Run comparison
    solvers, visualizer, training_histories, error_metrics = compare_and_analyze_models()

    # Create animation for each model
    t_range = np.linspace(0, 30, 31)
    for name, solver in solvers.items():
        print(f"\nCreating animation for {name}...")
        anim = visualizer.create_animation(solver, t_range, save_path=f'{name.lower().replace(" ", "_")}_animation.gif')