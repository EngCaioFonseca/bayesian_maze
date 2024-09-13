import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import random

# Maze creation function
def create_maze(size, num_walls):
    maze = np.zeros((size, size))
    walls = random.sample([(i, j) for i in range(size) for j in range(size) if (i, j) != (0, 0) and (i, j) != (size-1, size-1)], num_walls)
    for wall in walls:
        maze[wall] = 1
    return maze

# SNN class
class SpikingNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Neuron groups
        self.input_group = NeuronGroup(input_size, 'v : 1', threshold='v > 0.5', reset='v = 0', method='euler')
        self.hidden_group = NeuronGroup(hidden_size, 'dv/dt = (I - v) / tau : 1', threshold='v > 0.5', reset='v = 0', method='euler')
        self.output_group = NeuronGroup(output_size, 'dv/dt = (I - v) / tau : 1', threshold='v > 0.5', reset='v = 0', method='euler')
        
        # Synapses
        self.input_hidden = Synapses(self.input_group, self.hidden_group, 'w : 1', on_pre='v_post += w')
        self.input_hidden.connect(p=0.5)
        self.input_hidden.w = 'rand()'
        
        self.hidden_output = Synapses(self.hidden_group, self.output_group, 'w : 1', on_pre='v_post += w')
        self.hidden_output.connect(p=0.5)
        self.hidden_output.w = 'rand()'
        
        # STDP
        self.stdp = STDP(self.hidden_output, (0.1, 0.1))
        
        # Monitors
        self.hidden_monitor = SpikeMonitor(self.hidden_group)
        self.output_monitor = SpikeMonitor(self.output_group)
    
    def run(self, duration):
        run(duration * ms)
    
    def get_decision(self):
        spike_counts = self.output_monitor.count
        return np.argmax(spike_counts)

# Agent class
class Agent:
    def __init__(self, maze):
        self.position = (0, 0)
        self.maze = maze
        self.snn = SpikingNeuralNetwork(input_size=4, hidden_size=20, output_size=4)
    
    def move(self):
        # Get surrounding state
        state = self.get_surrounding_state()
        
        # Set input to SNN
        self.snn.input_group.v = state
        
        # Run SNN
        self.snn.run(10)
        
        # Get decision from SNN
        decision = self.snn.get_decision()
        
        # Move based on decision
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        new_position = (self.position[0] + directions[decision][0], self.position[1] + directions[decision][1])
        
        # Check if move is valid
        if 0 <= new_position[0] < self.maze.shape[0] and 0 <= new_position[1] < self.maze.shape[1] and self.maze[new_position] == 0:
            self.position = new_position
        
        return self.position
    
    def get_surrounding_state(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        state = []
        for dx, dy in directions:
            x, y = self.position[0] + dx, self.position[1] + dy
            if 0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1]:
                state.append(1 - self.maze[x, y])  # 1 for open path, 0 for wall
            else:
                state.append(0)  # Treat out of bounds as wall
        return state

# Plotting function for maze and agent
def plot_maze_and_agent(maze, agent_positions):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze, cmap='binary')
    ax.set_title('Maze with Agent Path')
    
    # Plot agent path
    path_x, path_y = zip(*agent_positions)
    ax.plot(path_y, path_x, 'r-', linewidth=2, alpha=0.7)
    
    # Plot start and end positions
    ax.plot(0, 0, 'go', markersize=12, label='Start')
    ax.plot(maze.shape[1]-1, maze.shape[0]-1, 'bo', markersize=12, label='Goal')
    
    ax.legend()
    return fig

# Plotting function for SNN activity
def plot_snn_activity(snn):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(snn.hidden_monitor.t/ms, snn.hidden_monitor.i, '.k')
    ax1.set_title('Hidden Layer Spikes')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron Index')
    
    ax2.plot(snn.output_monitor.t/ms, snn.output_monitor.i, '.k')
    ax2.set_title('Output Layer Spikes')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Neuron Index')
    
    plt.tight_layout()
    return fig

# Streamlit app
st.title('Spiking Neural Network Maze Solver')

# Maze parameters
st.sidebar.header('Maze Parameters')
maze_size = st.sidebar.slider('Maze Size', min_value=5, max_value=20, value=10)
num_walls = st.sidebar.slider('Number of Walls', min_value=0, max_value=maze_size**2-2, value=(maze_size**2)//4)

# Create maze
if st.sidebar.button('Generate New Maze'):
    st.session_state.maze = create_maze(maze_size, num_walls)
    st.session_state.agent = Agent(st.session_state.maze)
    st.session_state.agent_positions = [st.session_state.agent.position]

# Display maze
if 'maze' in st.session_state:
    st.write('Maze:')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(st.session_state.maze, cmap='binary')
    ax.set_title('Generated Maze')
    st.pyplot(fig)

# Agent movement
if 'agent' in st.session_state and st.button('Move Agent'):
    for _ in range(10):  # Move agent 10 small steps
        new_position = st.session_state.agent.move()
        st.session_state.agent_positions.append(new_position)
    
    # Display updated maze with agent path
    fig = plot_maze_and_agent(st.session_state.maze, st.session_state.agent_positions)
    st.pyplot(fig)
    
    # Display SNN activity
    fig = plot_snn_activity(st.session_state.agent.snn)
    st.pyplot(fig)

    # Check if agent reached the goal
    if st.session_state.agent.position == (maze_size-1, maze_size-1):
        st.success('Agent reached the goal!')

# Run full simulation
if 'agent' in st.session_state and st.button('Run Full Simulation'):
    max_steps = maze_size * maze_size * 2  # Set a maximum number of steps
    for _ in range(max_steps):
        new_position = st.session_state.agent.move()
        st.session_state.agent_positions.append(new_position)
        if new_position == (maze_size-1, maze_size-1):
            st.success('Agent reached the goal!')
            break
    
    # Display final maze with agent path
    fig = plot_maze_and_agent(st.session_state.maze, st.session_state.agent_positions)
    st.pyplot(fig)
    
    # Display final SNN activity
    fig = plot_snn_activity(st.session_state.agent.snn)
    st.pyplot(fig)

    if new_position != (maze_size-1, maze_size-1):
        st.warning('Agent did not reach the goal within the maximum number of steps.')
# Agent class
class Agent:
    def __init__(self, maze):
        self.position = (0, 0)
        self.maze = maze
        self.snn = SpikingNeuralNetwork(input_size=4, hidden_size=20, output_size=4)
    
    def move(self):
        # Get surrounding state
        state = self.get_surrounding_state()
        
        # Set input to SNN
        self.snn.input_group.v = state
        
        # Run SNN
        self.snn.run(10)
        
        # Get decision from SNN
        decision = self.snn.get_decision()
        
        # Move based on decision
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        new_position = (self.position[0] + directions[decision][0], self.position[1] + directions[decision][1])
        
        # Check if move is valid
        if 0 <= new_position[0] < self.maze.shape[0] and 0 <= new_position[1] < self.maze.shape[1] and self.maze[new_position] == 0:
            self.position = new_position
        
        return self.position
    
    def get_surrounding_state(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        state = []
        for dx, dy in directions:
            x, y = self.position[0] + dx, self.position[1] + dy
            if 0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1]:
                state.append(1 - self.maze[x, y])  # 1 for open path, 0 for wall
            else:
                state.append(0)  # Treat out of bounds as wall
        return state

# Plotting function for maze and agent
def plot_maze_and_agent(maze, agent_positions):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze, cmap='binary')
    ax.set_title('Maze with Agent Path')
    
    # Plot agent path
    path_x, path_y = zip(*agent_positions)
    ax.plot(path_y, path_x, 'r-', linewidth=2, alpha=0.7)
    
    # Plot start and end positions
    ax.plot(0, 0, 'go', markersize=12, label='Start')
    ax.plot(maze.shape[1]-1, maze.shape[0]-1, 'bo', markersize=12, label='Goal')
    
    ax.legend()
    return fig

# Plotting function for SNN activity
def plot_snn_activity(snn):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(snn.hidden_monitor.t/ms, snn.hidden_monitor.i, '.k')
    ax1.set_title('Hidden Layer Spikes')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron Index')
    
    ax2.plot(snn.output_monitor.t/ms, snn.output_monitor.i, '.k')
    ax2.set_title('Output Layer Spikes')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Neuron Index')
    
    plt.tight_layout()
    return fig

# Streamlit app
st.title('Spiking Neural Network Maze Solver')

# Maze parameters
st.sidebar.header('Maze Parameters')
maze_size = st.sidebar.slider('Maze Size', min_value=5, max_value=20, value=10)
num_walls = st.sidebar.slider('Number of Walls', min_value=0, max_value=maze_size**2-2, value=(maze_size**2)//4)

# Create maze
if st.sidebar.button('Generate New Maze'):
    st.session_state.maze = create_maze(maze_size, num_walls)
    st.session_state.agent = Agent(st.session_state.maze)
    st.session_state.agent_positions = [st.session_state.agent.position]

# Display maze
if 'maze' in st.session_state:
    st.write('Maze:')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(st.session_state.maze, cmap='binary')
    ax.set_title('Generated Maze')
    st.pyplot(fig)

# Agent movement
if 'agent' in st.session_state and st.button('Move Agent'):
    for _ in range(10):  # Move agent 10 small steps
        new_position = st.session_state.agent.move()
        st.session_state.agent_positions.append(new_position)
    
    # Display updated maze with agent path
    fig = plot_maze_and_agent(st.session_state.maze, st.session_state.agent_positions)
    st.pyplot(fig)
    
    # Display SNN activity
    fig = plot_snn_activity(st.session_state.agent.snn)
    st.pyplot(fig)

    # Check if agent reached the goal
    if st.session_state.agent.position == (maze_size-1, maze_size-1):
        st.success('Agent reached the goal!')

# Run full simulation
if 'agent' in st.session_state and st.button('Run Full Simulation'):
    max_steps = maze_size * maze_size * 2  # Set a maximum number of steps
    for _ in range(max_steps):
        new_position = st.session_state.agent.move()
        st.session_state.agent_positions.append(new_position)
        if new_position == (maze_size-1, maze_size-1):
            st.success('Agent reached the goal!')
            break
    
    # Display final maze with agent path
    fig = plot_maze_and_agent(st.session_state.maze, st.session_state.agent_positions)
    st.pyplot(fig)
    
    # Display final SNN activity
    fig = plot_snn_activity(st.session_state.agent.snn)
    st.pyplot(fig)

    if new_position != (maze_size-1, maze_size-1):
        st.warning('Agent did not reach the goal within the maximum number of steps.')