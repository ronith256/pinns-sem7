import torch
from flow_visualizer import FlowVisualizer
from vanilla import * 

# Define domain parameters
domain_params = {
    'L': 25 * 0.00116,  # Length = 25*Dh
    'H': 0.5 * 0.00116,  # Height = 0.5*Dh
    'Re': 200,          # Reynolds number
    'Dh': 0.00116      # Hydraulic diameter
}

# Create and train the model
model = LaminarFlowPINN(domain_params)
model.train_model(num_epochs=5000, batch_size=1024)

# Create visualizer
visualizer = FlowVisualizer(domain_params)

# Create plots at different times
times = [0.0, 10.0, 20.0, 30.0]
for t in times:
    fig = visualizer.plot_flow_field(model, t, f"Laminar Flow Results")
    fig.savefig(f"flow_field_t_{t:.1f}.png")

# Create animation
t_range = np.linspace(0, 30, 31)
anim = visualizer.create_animation(model, t_range, save_path="flow_animation.gif")