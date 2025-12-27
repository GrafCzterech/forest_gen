import matplotlib.pyplot as plt

from .visualiser import Visualizer


class HeightmapVisualizer(Visualizer):
    def visualize(self, data, title="Heightmap"):
        fig, ax = plt.subplots()
        cs = ax.contourf(data, levels=20, cmap="terrain", origin="lower")
        ax.set_title(title)
        fig.colorbar(cs, ax=ax, label="Elevation")
        plt.show()
