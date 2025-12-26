import numpy as np
from PIL import Image
from .export_strategy import ExportStrategy
import trimesh
from trimesh.visual.texture import TextureVisuals
from pathlib import Path


class GLBExporter(ExportStrategy):
    """
    Concrete Strategy exporting terrain as a textured GLB mesh.
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
        """
        Export a heightmap as a GLB mesh with a generated texture.

        Vertices are laid out on a regular grid using the configured
        resolution, heights are scaled by ``max_elevation``, and a
        random RGB texture is generated for visualization.

        :param heightmap: Heightmap array with values in ``[0.0, 1.0]``.
        :type heightmap: numpy.ndarray
        :param path: Output ``.glb`` file path.
        :type path: str
        """
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
        # expanded_verts = verts[faces].reshape((-1, 3)).astype(np.float32)
        # expanded_uvs = uvs[faces].reshape((-1, 2)).astype(np.float32)
        # expanded_faces = (
            # np.arange(expanded_verts.shape[0], dtype=np.int32).reshape(-1, 3)
        # )

        # mesh = trimesh.Trimesh(vertices=expanded_verts, faces=expanded_faces, process=False)
        mesh.visual = TextureVisuals(uv=uvs, material=material)
        mesh.rezero()
        mesh.fix_normals()

        with open(path, "wb") as f:
            trimesh.Scene(mesh).export(file_obj=f, file_type="glb")
