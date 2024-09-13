# Spiking Neural Network Maze Solver

This Streamlit application demonstrates the use of Spiking Neural Networks (SNNs) in solving maze navigation problems. It allows users to experiment with different neuron models, tune SNN parameters, and visualize the network's behavior before applying it to a maze-solving task.

## Features

- Choose from multiple neuron models:
  - Leaky Integrate-and-Fire (LIF)
  - Izhikevich
  - Hodgkin-Huxley (HH)
- Adjust SNN hyperparameters
- Visualize SNN activity:
  - Spike raster plots
  - Membrane potentials
  - Synaptic weight changes
- Generate random mazes
- Simulate an agent navigating through the maze using the SNN

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/snn-maze-solver.git
   cd snn-maze-solver
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run snn_maze_solver_multi_model.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Use the sidebar to select a neuron model and adjust its parameters.

4. Click "Run SNN Simulation" to visualize the network's behavior.

5. Once satisfied with the SNN parameters, proceed to the Maze Solver section.

6. Adjust maze parameters and click "Generate New Maze and Start Solving" to watch the SNN-controlled agent navigate the maze.

## Neuron Models

### Leaky Integrate-and-Fire (LIF)
A simple neuron model that integrates input and fires when a threshold is reached.

Parameters:
- Resting Potential
- Reset Potential
- Threshold Potential
- Membrane Time Constant
- Membrane Resistance

### Izhikevich
A versatile model that can exhibit various firing patterns observed in biological neurons.

Parameters:
- a: Time scale of the recovery variable
- b: Sensitivity of the recovery variable
- c: After-spike reset value of the membrane potential
- d: After-spike reset of the recovery variable

### Hodgkin-Huxley (HH)
A biophysically detailed model that describes the initiation and propagation of action potentials.

Parameters:
- gNa: Sodium conductance
- gK: Potassium conductance
- gL: Leak conductance
- ENa: Sodium reversal potential
- EK: Potassium reversal potential
- EL: Leak reversal potential
- C: Membrane capacitance

## Contributing

Contributions to improve the application or add new features are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
