import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DataVisualization:

    def __init__(self, states, root, map_size):
        self.figure = plt.figure(figsize=(4, 4))
        self.X = [x for x in range(map_size)]
        self.Y = [y for y in range(map_size)]
        self.map_size = map_size
        self.Z = states
        self.root = root
        plt.pcolormesh(self.X, self.Y, self.Z[0], shading='auto')

    def plot_graph(self, *, x, y, title, x_lab, y_lab, col, row):
        plt.rcParams["figure.autolayout"] = True

        figure = plt.Figure(figsize=(4, 4))
        ax = figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        line = FigureCanvasTkAgg(figure, self.root)
        line.get_tk_widget().grid(column=col, row=row)
        line.draw()

    def animate(self, i):

        plt.title('State: %.2f' % i)
        self.X = [x for x in range(self.map_size)]
        self.Y = [y for y in range(self.map_size)]
        # This is where new data is inserted into the plot.
        plt.pcolormesh(self.X, self.Y, self.Z[i], shading='auto')

    def run_animation(self):
        anim_ = animation.FuncAnimation(self.figure, self.animate, frames=self.Z.shape[0], blit=False)
        # python no like mp4
        writer_gif = animation.PillowWriter(fps=1)
        anim_.save('game.gif', writer=writer_gif)
        # plt.show()

