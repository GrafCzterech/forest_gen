import matplotlib.pyplot as plt

from .visualiser import Visualizer


class FlowVisualizer(Visualizer):
    def visualize(self, data, title="Flow Accumulation"):
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap="Blues_r", origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Accumulation")
        plt.show()
