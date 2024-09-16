import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import random

# Streamlit app
st.title('Spiking Neural Network Simulation')

# Parameters
num_neurons = st.sidebar.number_input('Number of Neurons', min_value=10, max_value=200, value=100)
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
simulation_time = st.sidebar.slider('Simulation Time (ms)', min_value=10, max_value=1000, value=100) * ms
time_step = 0.1*ms

# Input current
input_current = st.sidebar.text_input('Input Current', '0.8 + 0.4*rand()')

# Neuron Model (LIF - Leaky Integrate-and-Fire)
eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''
threshold = 'v>1'
reset = 'v = 0'
refractory = 12*ms

# Create neurons with adjusted parameters
neurons = NeuronGroup(num_neurons, eqs, threshold=threshold, reset=reset, refractory=refractory, method='exact')
neurons.v = 'rand()'  # Initial random membrane potentials
neurons.I = input_current  # User-defined input currents
neurons.tau = '10*ms + 2*ms*rand()'  # Introduce variability in tau

# STDP parameters
taupre = 20*ms
taupost = 20*ms
dApre = 0.005  # Lower learning rate for more gradual changes
dApost = -dApre

# Synaptic connections with STDP (Hebbian learning)
synapses = Synapses(neurons, neurons,
                    '''
                    w : 1
                    dApre/dt = -Apre/taupre : 1 (event-driven)
                    dApost/dt = -Apost/taupost : 1 (event-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    Apre += dApre
                    w = clip(w + Apost, 0, 2)
                    ''',
                    on_post='''
                    Apost += dApost
                    w = clip(w + Apre, 0, 2)
                    ''')

synapses.connect(condition='i != j', p=0.1)  # Random connections with 10% probability
synapses.w = '0.05*rand()'  # Even lower initial weights to prevent saturation
synapses.Apre = 0
synapses.Apost = 0

# Monitor neuron spikes, state variables, and synaptic weights
spike_mon = SpikeMonitor(neurons)
state_mon = StateMonitor(neurons, ['v'], record=True)
weight_mon = StateMonitor(synapses, 'w', record=True)

# Simulation
if st.button('Run Simulation'):
    run(simulation_time)

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    # Plot spikes
    axs[0].plot(spike_mon.t/ms, spike_mon.i, '.k')
    axs[0].set_xlim(0, 200)  # Zoom into the first 200ms for better visibility
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Neuron index')
    axs[0].set_title('Neuron Spikes')

    # Plot membrane potentials for a subset of neurons
    for i in range(5):  # Plot first 5 neurons for clarity
        axs[1].plot(state_mon.t/ms, state_mon.v[i])
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Membrane potential (v)')
    axs[1].set_title('Membrane Potentials')

    # Plot synaptic weights
    for i in range(5):  # Plot first 5 synapses for clarity
        axs[2].plot(weight_mon.t/ms, weight_mon.w[i])
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('Synaptic weight (w)')
    axs[2].set_title('Synaptic Weights')

    st.pyplot(fig)

    # Option to run the SNN to solve the maze
    if st.button('Run SNN to Solve Maze'):
        st.session_state.run_maze = True

# Maze solving step
if 'run_maze' in st.session_state and st.session_state.run_maze:
    st.title('Spiking Neural Network Maze Solver')

    # Maze parameters
    maze_size = (5, 5)
    start_position = (0, 0)
    goal_position = (4, 4)

    # Create a simple maze environment
    def create_maze(maze_size):
        maze = np.zeros(maze_size)
        maze[goal_position] = 1  # Goal position
        # Randomly add walls
        num_walls = 5
        for _ in range(num_walls):
            x, y = random.randint(0, maze_size[0] - 1), random.randint(0, maze_size[1] - 1)
            if (x, y) != start_position and (x, y) != goal_position:
                maze[x, y] = -1  # Wall
        return maze

    maze = create_maze(maze_size)

    # Visualize the maze
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(maze, cmap='gray', origin='lower')
    ax.scatter(start_position[1], start_position[0], color='green', label='Start Position')
    ax.scatter(goal_position[1], goal_position[0], color='red', label='Goal Position')
    ax.set_title("Maze Layout")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Here you can add the logic to run the SNN to solve the maze
    # For now, we just display the maze
