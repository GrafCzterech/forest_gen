import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .visualiser import Visualizer


class MatplotlibVisualizer(Visualizer):
    def show_heightmap(self, heightmap):
        fig, ax = plt.subplots()
        ax.contourf(heightmap)
        ax.set_title("Heightmap")
        plt.show()

    def show_flow(self, flow):
        fig, ax = plt.subplots()
        ax.imshow(flow, cmap="Blues")
        ax.set_title("Flow Accumulation")
        plt.show()

    def show_moisture(self, moisture):
        fig, ax = plt.subplots()
        ax.imshow(moisture, cmap="YlGn")
        ax.set_title("Moisture Index")
        plt.show()

    def show_moisture_classes(self, classes):
        fig, ax = plt.subplots()
        cmap = ListedColormap(("sandybrown", "lightgreen", "steelblue"))
        ax.imshow(classes, cmap=cmap)
        ax.set_title("Moisture Classes")
        plt.show()
