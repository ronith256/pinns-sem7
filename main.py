from vanilla import FlowPINN, train_model
from flow_visualizer import FlowVisualizer

# Initialize and train
domain_params = {'L': 25 * 0.00116, 'H': 0.5 * 0.00116}
model = FlowPINN(domain_params)
trained_model = train_model(model, num_epochs=300)

# Visualize
visualizer = FlowVisualizer(domain_params)
visualizer.plot_flow_field(trained_model, t=15.0, title="Flow Field")