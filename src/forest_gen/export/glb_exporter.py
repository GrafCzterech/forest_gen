import numpy as np
from PIL import Image
from .export_strategy import ExportStrategy
import trimesh
from trimesh.visual.texture import TextureVisuals
from pathlib import Path


class GLBExporter(ExportStrategy):
    """
    Exports terrain as a GLB mesh with a random texture.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        max_elevation: float = 100.0,
        seed: int | None = None,
    ):
        self.resolution = resolution
        self.max_elevation = max_elevation
        self.seed = seed

    def export(self, heightmap: np.ndarray, path: str) -> None:
        rows, cols = heightmap.shape
        rng = np.random.default_rng(self.seed)
        tex_arr = rng.integers(0, 255, size=(rows, cols, 3), dtype=np.uint8)

        p = Path(path)
        tex_path = f"{p.with_suffix('')}_texture.png"
        Image.fromarray(tex_arr).save(tex_path)

        j_, i_ = np.meshgrid(np.arange(cols), np.arange(rows))
        verts = np.column_stack(
            (
                j_.ravel() * self.resolution,
                heightmap.ravel() * self.max_elevation,
                i_.ravel() * self.resolution,
            )
        ).astype(np.float32)
        uvs = np.column_stack(
            (j_.ravel() / (cols - 1), 1 - i_.ravel() / (rows - 1))
        ).astype(np.float32)

        idx = np.arange(rows * cols).reshape(rows, cols)
        f1 = np.column_stack(
            (idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel())
        )
        f2 = np.column_stack(
            (idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel())
        )
        faces = np.vstack((f1, f2)).astype(np.int32)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        material = TextureVisuals(image=tex_arr).material
        mesh.visual = TextureVisuals(uv=uvs, material=material)
        mesh.rezero()
        mesh.fix_normals()

        with open(path, "wb") as f:
            trimesh.Scene(mesh).export(file_obj=f, file_type="glb")
