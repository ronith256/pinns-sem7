# from vanilla import FlowPINN, train_model
# from flow_visualizer import FlowVisualizer
# import matplotlib.pyplot as plt

# # Initialize and train
# domain_params = {'L': 25 * 0.00116, 'H': 0.5 * 0.00116}
# model = FlowPINN(domain_params)
# trained_model = train_model(model, num_epochs=300)

# # Visualize
# visualizer = FlowVisualizer(domain_params)
# fig = visualizer.plot_flow_field(trained_model, t=15.0, title="Flow Field")
# plt.show()  # This will display the figure

from nsfnet import NSFNet, train_nsfnet
from flow_visualizer import FlowVisualizer
import matplotlib.pyplot as plt
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize parameters
domain_params = {
    'L': 25 * 0.00116,  # Channel length
    'H': 0.5 * 0.00116  # Channel height
}

# Initialize and train model
model = NSFNet(domain_params)
trained_model = train_nsfnet(model, num_epochs=1000)

# Create visualizer
visualizer = FlowVisualizer(domain_params)

# Create visualizations at different time steps
times = [0.0, 5.0, 15.0, 30.0]
for t in times:
    fig = visualizer.plot_flow_field(
        trained_model, 
        t=t, 
        title=f"NSFNet Flow Field",
        save_path=f"flow_field_t{t:.1f}.png"
    )
    plt.close()

# Create animation
import numpy as np
t_range = np.linspace(0, 30, 31)
visualizer.create_animation(
    trained_model,
    t_range,
    save_path="flow_animation.gif"
)