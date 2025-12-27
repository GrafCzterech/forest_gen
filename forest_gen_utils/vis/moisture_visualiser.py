import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .visualiser import Visualizer


class MoistureVisualizer(Visualizer):
    def __init__(self, dry_thresh=0.33, wet_thresh=0.66):
        self.colors = ("sandybrown", "lightgreen", "steelblue")
        self.norm = BoundaryNorm([0, 1, 2, 3], 3)
        self.dry, self.wet = dry_thresh, wet_thresh

    def visualize(self, moisture, title="Moisture Classes"):
        classes = (moisture > self.wet).astype(int) * 2
        mid = (moisture > self.dry) & (moisture <= self.wet)
        classes[mid] = 1

        cmap = ListedColormap(self.colors)
        fig, ax = plt.subplots()
        ax.imshow(classes, cmap=cmap, norm=self.norm, origin="lower")
        ax.set_title(title)

        legend = [
            Patch(color=c, label=l)
            for c, l in zip(self.colors, ["Dry", "Normal", "Wet"])
        ]
        ax.legend(handles=legend, loc="upper right")

        plt.show()
