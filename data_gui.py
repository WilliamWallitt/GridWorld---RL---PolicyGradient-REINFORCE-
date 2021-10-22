import tkinter as tk
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame
import tkinter.messagebox
from tkinter import *
from data_visualisation import DataVisualization
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use("TkAgg")
from PIL import Image


class DataGUI(tk.Frame):

    def __init__(self, states, steps, agent_positions, rewards, training_data, map_size=10):
        self.root = tk.Tk()
        tk.Frame.__init__(self, master=self.root)
        self.data_visualization = DataVisualization(states, self.root, map_size)
        self.data_visualization.run_animation()
        self.states = states
        self.training_data = training_data
        self.steps, self.agent_positions, self.rewards = steps, agent_positions, rewards
        self.current_state_index = 0
        self.current_state_representation = FigureCanvasTkAgg(self.data_visualization.figure, self.root)
        # button stuff
        self.prev = tk.Button(self.root, text="Previous State", width=10, height=2, command=self.prev_callback)
        self.next = tk.Button(self.root, text="Next State", width=10, height=2, command=self.next_callback)
        # gif stuff
        self.im = None
        self.gif_label = tk.Label(image="")
        self.gif_button = tk.Button(text="start gif", command=self.gif_animation)
        self.gif_count = 0
        self.frames = 0
        # text stuff
        self.text = tk.Label(self.root, text="Agent Position: " + str(self.agent_positions[0]) + "   Agent Curr Rewards: "
                                             + str(self.rewards[0]) +
                                             "   Step: " + str(self.steps[0]))

    def draw_state(self):
        current_state = self.states[self.current_state_index]
        current_step, current_agent_pos, current_reward = \
            self.steps[self.current_state_index], self.agent_positions[self.current_state_index], self.rewards[self.current_state_index]

        self.text['text'] = "Agent Position: " + str(current_agent_pos) + \
                            "   Agent Curr Rewards: " + str(current_reward) + \
                            "   Step: " + str(current_step)

        plt.pcolormesh(self.data_visualization.X, self.data_visualization.Y, current_state, shading='auto')
        self.current_state_representation.get_tk_widget()
        self.current_state_representation.draw()

    def draw_gif(self, file):

        info = Image.open(file)
        self.frames = info.n_frames
        self.im = [tk.PhotoImage(file=file, format=f'gif - {i}') for i in range(self.frames)]
        self.gif_label.grid(column=0, row=2)

    def gif_animation(self):
        self.gif_button.pack_forget()
        self.gif_button.grid_forget()
        im2 = self.im[self.gif_count]
        self.gif_label.configure(image=im2)
        self.gif_count += 1
        if self.gif_count == self.frames:
            self.gif_count = 0

        self.root.after(50, lambda: self.gif_animation())

    def run_gui(self):
        # self.root.resizable(0, 0)
        self.text.grid(column=0, row=1, sticky=tk.S)
        self.prev.grid(column=0, row=0, sticky=tk.W)
        self.next.grid(column=0, row=0, sticky=tk.E)

        # self.current_state_representation.get_tk_widget().pack(fill=tk.BOTH)
        self.current_state_representation.get_tk_widget().grid(column=0, row=1, sticky=tk.N)
        self.current_state_representation.draw()

        self.draw_gif("game.gif")
        self.gif_button.grid(column=0, row=2)

        self.data_visualization.plot_graph(x=[i for i in range(len(self.rewards))], y=self.rewards,
                                           title="Rewards gained for each step",
                                           x_lab="episodes", y_lab="rewards", col=1, row=1)

        self.data_visualization.plot_graph(x=self.training_data[0], y=self.training_data[1], title="Long term rewards",
                                           x_lab="episodes", y_lab="rewards", col=1, row=2)

        self.root.mainloop()

    def prev_callback(self):
        self.current_state_index = self.current_state_index - 1 if self.current_state_index > 0 else self.current_state_index
        self.draw_state()

    def next_callback(self):
        self.current_state_index = self.current_state_index + 1 if self.current_state_index < self.states.shape[0] -1 \
            else self.current_state_index
        self.draw_state()
