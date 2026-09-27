"""Interactive 2D Anisotropic Green Coordinates (AGC).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog

import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.path import Path as MatplotlibPath
from matplotlib.widgets import Button
import numpy as np
from scipy import ndimage

LOGGER = logging.getLogger("agc")
EDGE_EPS = 1e-12
PROXIMITY_THRESHOLD = 1e-6
ANISOTROPIC_PROXIMITY_THRESHOLD = 1e-3


def load_image_rgba(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a grayscale, RGB, or RGBA image and return RGB plus optional alpha."""
    image = iio.imread(path)
    if image.ndim == 2:
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        alpha = None
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3] if image.shape[2] >= 4 else None
    else:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, alpha


def load_cage(path: Path) -> np.ndarray:
    """Load an N-by-2 cage without repeating the first vertex."""
    values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
    if values.size % 2:
        raise ValueError(f"Cage file must contain coordinate pairs: {path}")
    cage = values.reshape(-1, 2)
    if cage.shape[0] < 3:
        raise ValueError("A cage must contain at least three vertices.")
    if not np.all(np.isfinite(cage)):
        raise ValueError("Cage coordinates must be finite.")
    return cage


def polygon_mask(x, y, height, width):
    yy, xx = np.mgrid[1 : height + 1, 1 : width + 1]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    polygon = MatplotlibPath(np.column_stack((x, y)))
    return polygon.contains_points(points, radius=1e-9).reshape(height, width)


def boundary_mask(mask):
    eroded = ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return mask & ~eroded


def apply_green_coordinates(coordinates, values):
    return np.sum(coordinates * np.asarray(values).reshape(1, 1, -1), axis=2)


def anisotropic_matrix_square_root(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("The anisotropy matrix must have shape 2 x 2.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The anisotropy matrix must contain finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The anisotropy matrix must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("The anisotropy matrix must be positive definite.")
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return root, inverse_root


def isotropic_edge_integral(vi, vj, normal, query_points):
    """Evaluate the original closed-form isotropic edge integral.
    """
    point_count = query_points.shape[1]
    edge = vj - vi
    edge_length = np.linalg.norm(edge, axis=0)
    b = vi - query_points
    q_value = np.sum(edge * edge, axis=0)
    s_value = np.sum(b * b, axis=0)
    r_value = 2.0 * np.sum(edge * b, axis=0)

    discriminant = np.maximum(4.0 * s_value * q_value - r_value**2, EDGE_EPS)
    square_root = np.sqrt(discriminant)
    normal_term = edge_length * np.sum(b * np.asarray(normal).reshape(2, 1), axis=0)

    log0 = np.log(np.maximum(s_value, EDGE_EPS))
    log1 = np.log(np.maximum(s_value + q_value + r_value, EDGE_EPS))
    log_difference = log1 - log0
    q_safe = np.maximum(q_value, EDGE_EPS)
    square_root_safe = np.maximum(square_root, EDGE_EPS)
    atan0 = np.arctan(r_value / square_root_safe) / square_root_safe
    atan1 = np.arctan((2.0 * q_value + r_value) / square_root_safe) / square_root_safe
    atan_difference = atan1 - atan0
    atan_difference[square_root < EDGE_EPS] = 0.0

    d_value = edge_length * (
        (4.0 * s_value - r_value**2 / q_safe) * atan_difference
        + r_value / (2.0 * q_safe) * log_difference
        + log1
        - 2.0
    ) / (4.0 * np.pi)
    cj_value = -normal_term * (
        log_difference / (2.0 * q_safe) - atan_difference * r_value / q_safe
    ) / (2.0 * np.pi)
    ci_value = normal_term * (
        log_difference / (2.0 * q_safe)
        - atan_difference * (2.0 + r_value / q_safe)
    ) / (2.0 * np.pi)

    distance_to_start = np.linalg.norm(b, axis=0)
    distance_to_end = np.linalg.norm(vj - query_points, axis=0)
    close_start = distance_to_start < PROXIMITY_THRESHOLD
    close_end = distance_to_end < PROXIMITY_THRESHOLD
    ci_value[close_start] = 0.0
    cj_value[close_end] = 0.0
    d_value[close_start | close_end] = 0.0

    for values in (d_value, ci_value, cj_value):
        np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if np.ndim(d_value) == 0:
        d_value = np.full(point_count, d_value, dtype=float)
        ci_value = np.full(point_count, ci_value, dtype=float)
        cj_value = np.full(point_count, cj_value, dtype=float)
    return d_value, ci_value, cj_value


def edge_integral_gc(vi, vj, query_points):
    """Evaluate one classical Green Coordinate edge contribution."""
    edge = vj - vi
    edge_length = float(np.linalg.norm(edge))
    if edge_length < EDGE_EPS:
        zeros = np.zeros(query_points.shape[1], dtype=float)
        return zeros, zeros.copy(), zeros.copy()
    normal = np.array([edge[1], -edge[0]], dtype=float) / edge_length
    return isotropic_edge_integral(
        vi.reshape(2, 1) - query_points,
        vj.reshape(2, 1) - query_points,
        normal,
        np.zeros_like(query_points),
    )


def edge_integral_agc(vi, vj, query_points, anisotropy, root, inverse_root):
    """Evaluate one anisotropic Green Coordinate edge contribution."""
    edge = vj - vi
    edge_length = float(np.linalg.norm(edge))
    if edge_length < EDGE_EPS:
        zeros = np.zeros(query_points.shape[1], dtype=float)
        return zeros, zeros.copy(), zeros.copy()

    normal = np.array([edge[1], -edge[0]], dtype=float) / edge_length
    normal_metric = float(normal.T @ anisotropy @ normal)
    if normal_metric < EDGE_EPS:
        zeros = np.zeros(query_points.shape[1], dtype=float)
        return zeros, zeros.copy(), zeros.copy()

    transformed_start = inverse_root @ (vi.reshape(2, 1) - query_points)
    transformed_end = inverse_root @ (vj.reshape(2, 1) - query_points)
    transformed_normal = root @ normal
    transformed_normal_norm = float(np.linalg.norm(transformed_normal))
    if transformed_normal_norm < EDGE_EPS:
        zeros = np.zeros(query_points.shape[1], dtype=float)
        return zeros, zeros.copy(), zeros.copy()
    transformed_normal /= transformed_normal_norm

    d_iso, ci_value, cj_value = isotropic_edge_integral(
        transformed_start,
        transformed_end,
        transformed_normal,
        np.zeros_like(query_points),
    )
    d_value = d_iso / np.sqrt(normal_metric)
    np.nan_to_num(d_value, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return d_value, ci_value, cj_value


def points_close_to_edge(vi, vj, points, threshold):
    edge = vj - vi
    edge_length = float(np.linalg.norm(edge))
    if edge_length < EDGE_EPS:
        return np.zeros(points.shape[1], dtype=bool)
    normal = np.array([edge[1], -edge[0]], dtype=float) / edge_length
    relative = points - vi.reshape(2, 1)
    distance = np.abs(np.sum(relative * normal.reshape(2, 1), axis=0))
    parameter = np.sum(relative * edge.reshape(2, 1), axis=0) / edge_length**2
    return (distance < threshold) & (parameter >= 0.0) & (parameter <= 1.0)


def precompute_coordinates(
    source_cage_math,
    all_points,
    height,
    width,
    valid_mask,
    integral,
):
    valid_indices = np.flatnonzero(valid_mask.ravel())
    vertex_count = source_cage_math.shape[1]
    c_weights = np.zeros((height, width, vertex_count), dtype=float)
    d_weights = np.zeros_like(c_weights)
    if not valid_indices.size:
        return c_weights, d_weights

    points = all_points[:, valid_indices]
    c_valid = np.zeros((valid_indices.size, vertex_count), dtype=float)
    d_valid = np.zeros_like(c_valid)
    for i in range(vertex_count):
        j = (i + 1) % vertex_count
        vi = source_cage_math[:, i]
        vj = source_cage_math[:, j]
        d_value, ci_value, cj_value = integral(vi, vj, points)
        threshold = (
            ANISOTROPIC_PROXIMITY_THRESHOLD
            if integral.__name__ == "agc_integral"
            else PROXIMITY_THRESHOLD
        )
        close = points_close_to_edge(vi, vj, points, threshold)
        d_value[close] = 0.0
        ci_value[close] = 0.0
        cj_value[close] = 0.0
        d_valid[:, i] = d_value
        c_valid[:, i] += ci_value
        c_valid[:, j] += cj_value

    c_weights.reshape(-1, vertex_count)[valid_indices] = c_valid
    d_weights.reshape(-1, vertex_count)[valid_indices] = d_valid
    return c_weights, d_weights


def make_gc_integral():
    def gc_integral(vi, vj, points):
        return edge_integral_gc(vi, vj, points)
    gc_integral.__name__ = "gc_integral"
    return gc_integral


def make_agc_integral(anisotropy, root, inverse_root):
    def agc_integral(vi, vj, points):
        return edge_integral_agc(vi, vj, points, anisotropy, root, inverse_root)
    agc_integral.__name__ = "agc_integral"
    return agc_integral


class AGCApplication:
    """Interactive comparison between classical GC and AGC deformation."""

    def __init__(self, args):
        self.args = args
        self.image_path, self.cage_path = resolve_input_paths(args)
        if not self.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if not self.cage_path.is_file():
            raise FileNotFoundError(f"Cage not found: {self.cage_path}")

        self.image, alpha = load_image_rgba(self.image_path)
        self.image_height, self.image_width = self.image.shape[:2]
        self.valid_mask = (
            np.ones((self.image_height, self.image_width), dtype=bool)
            if alpha is None else alpha > 0
        )
        self.source_cage = load_cage(self.cage_path)
        self.cage_points = self.source_cage.copy()
        self.vertex_count = self.source_cage.shape[0]
        cage_math = np.vstack((
            self.source_cage[:, 0],
            self.image_height - self.source_cage[:, 1],
        ))
        self.v0_math = cage_math  # 只保留这一份，不再做归一化

        self.anisotropy = np.asarray(args.A, dtype=float).reshape(2, 2)
        self.root, self.inverse_root = anisotropic_matrix_square_root(self.anisotropy)

        x_grid, y_grid = np.meshgrid(
            np.arange(1, self.image_width + 1),
            np.arange(1, self.image_height + 1),
        )
        self.original_grid = (x_grid, y_grid)
        self.all_math_points = np.vstack((
            x_grid.ravel(), self.image_height - y_grid.ravel()
        ))
        inside = polygon_mask(
            self.source_cage[:, 0], self.source_cage[:, 1],
            self.image_height, self.image_width,
        )
        distance = ndimage.distance_transform_edt(~boundary_mask(inside))
        self.safe_inside_mask = inside & (distance >= 2.0) & self.valid_mask
        if not np.any(self.safe_inside_mask):
            raise ValueError("The cage has no valid interior pixels after boundary filtering.")

        started = time.perf_counter()
        self.c_baseline, self.d_baseline = precompute_coordinates(
            self.v0_math, self.all_math_points,
            self.image_height, self.image_width, self.safe_inside_mask,
            make_gc_integral(),
        )
        LOGGER.info("Classical GC coordinates computed in %.3f s", time.perf_counter() - started)
        started = time.perf_counter()
        self.c_agc, self.d_agc = precompute_coordinates(
            self.v0_math, self.all_math_points,
            self.image_height, self.image_width, self.safe_inside_mask,
            make_agc_integral(self.anisotropy, self.root, self.inverse_root),
        )
        LOGGER.info("AGC coordinates computed in %.3f s", time.perf_counter() - started)

        self.left_coordinates = None
        self.right_coordinates = None
        self.dragging = False
        self.dragged_vertex = None
        self.orange = (1.0, 0.5, 0.0)
        self.figure = plt.figure("AGC deformation", figsize=(12, 7))
        self.left_axis = self.figure.add_subplot(1, 2, 1)
        self.right_axis = self.figure.add_subplot(1, 2, 2)
        self.left_axis.set_title("Classical Green Coordinates")
        self.right_axis.set_title(
            f"Anisotropic Green Coordinates (A={self.anisotropy.tolist()})"
        )
        self.left_surface, self.left_line, self.left_points = self.setup_axis(self.left_axis)
        self.right_surface, self.right_line, self.right_points = self.setup_axis(self.right_axis)
        self.setup_controls()
        self.setup_events()
        if args.target_cage:
            self.set_target_cage(load_cage(Path(args.target_cage)))
            self.apply_deformation()

    def draw_texture(self, axis, x_data=None, y_data=None, visible_mask=None):
        if x_data is None or y_data is None:
            x_data, y_data = self.original_grid
        else:
            x_data = np.asarray(x_data, dtype=float).copy()
            y_data = np.asarray(y_data, dtype=float).copy()
            finite = np.isfinite(x_data) & np.isfinite(y_data)
            x_data[~finite] = self.original_grid[0][~finite]
            y_data[~finite] = self.original_grid[1][~finite]

        rgba = np.ones((self.image_height, self.image_width, 4), dtype=float)
        rgba[:, :, :3] = self.image.astype(float) / 255.0
        alpha_mask = self.valid_mask if visible_mask is None else visible_mask & self.valid_mask
        rgba[:, :, 3] = alpha_mask.astype(float)
        mesh = axis.pcolormesh(
            x_data, y_data, rgba, shading="nearest",
            edgecolors="none", antialiased=False,
        )
        mesh.set_gid("deformation_texture")
        mesh.set_zorder(1)
        return mesh

    @staticmethod
    def clear_texture(axis):
        for artist in list(axis.collections):
            if isinstance(artist, QuadMesh) and artist.get_gid() == "deformation_texture":
                artist.remove()

    @staticmethod
    def fill_invalid_nearest(x_data, y_data, valid_mask):
        if np.all(valid_mask):
            return x_data, y_data
        _, indices = ndimage.distance_transform_edt(
            ~valid_mask, return_indices=True
        )
        x_filled, y_filled = x_data.copy(), y_data.copy()
        invalid = ~valid_mask
        x_filled[invalid] = x_data[indices[0][invalid], indices[1][invalid]]
        y_filled[invalid] = y_data[indices[0][invalid], indices[1][invalid]]
        return x_filled, y_filled

    def setup_axis(self, axis):
        axis.set_aspect("equal")
        surface = self.draw_texture(axis)
        closed = np.vstack((self.cage_points, self.cage_points[0]))
        line, = axis.plot(
            closed[:, 0], closed[:, 1], "-o", linewidth=2, markersize=6,
            markerfacecolor=self.orange, color=self.orange, zorder=3,
        )
        points, = axis.plot(
            self.cage_points[:, 0], self.cage_points[:, 1], "o", markersize=8,
            color=self.orange, markerfacecolor=self.orange, picker=8, zorder=3,
        )
        self.apply_axis_limits(axis)
        return surface, line, points

    def apply_axis_limits(self, axis):
        padding = self.args.axis_padding
        axis.set_xlim(
            np.min(self.cage_points[:, 0]) - padding,
            np.max(self.cage_points[:, 0]) + padding,
        )
        axis.set_ylim(
            np.max(self.cage_points[:, 1]) + padding,
            np.min(self.cage_points[:, 1]) - padding,
        )

    def setup_controls(self):
        plt.subplots_adjust(bottom=0.12)
        apply_axis = self.figure.add_axes([0.02, 0.03, 0.14, 0.04])
        save_axis = self.figure.add_axes([0.19, 0.03, 0.13, 0.04])
        load_axis = self.figure.add_axes([0.35, 0.03, 0.15, 0.04])
        self.apply_button = Button(apply_axis, "Apply deformation")
        self.save_button = Button(save_axis, "Save cage")
        self.load_button = Button(load_axis, "Load target cage")
        self.apply_button.on_clicked(self.apply_deformation)
        self.save_button.on_clicked(self.save_cage)
        self.load_button.on_clicked(self.load_target_cage)

    def setup_events(self):
        self.figure.canvas.mpl_connect("button_press_event", self.start_drag)
        self.figure.canvas.mpl_connect("motion_notify_event", self.drag_vertex)
        self.figure.canvas.mpl_connect("button_release_event", self.stop_drag)

    def update_cage_display(self):
        closed = np.vstack((self.cage_points, self.cage_points[0]))
        for points, line in (
            (self.left_points, self.left_line),
            (self.right_points, self.right_line),
        ):
            points.set_data(self.cage_points[:, 0], self.cage_points[:, 1])
            line.set_data(closed[:, 0], closed[:, 1])
        self.figure.canvas.draw_idle()

    def start_drag(self, event):
        if event.inaxes is not self.left_axis or event.xdata is None or event.ydata is None:
            return
        distances = np.sum(
            (self.cage_points - np.array([event.xdata, event.ydata])) ** 2, axis=1
        )
        index = int(np.argmin(distances))
        if distances[index] <= 20.0**2:
            self.dragging, self.dragged_vertex = True, index

    def drag_vertex(self, event):
        if (
            not self.dragging or event.inaxes is not self.left_axis
            or event.xdata is None or event.ydata is None
        ):
            return
        self.cage_points[self.dragged_vertex] = [event.xdata, event.ydata]
        self.update_cage_display()

    def stop_drag(self, _event):
        self.dragging, self.dragged_vertex = False, None

    def deformed_coordinates(self, c_weights, d_weights, anisotropy=None):
        target = np.vstack((
            self.cage_points[:, 0],
            self.image_height - self.cage_points[:, 1],
        ))
        source = self.v0_math

        target_edges = np.roll(target, -1, axis=1) - target
        source_edges = np.roll(source, -1, axis=1) - source
        target_lengths = np.linalg.norm(target_edges, axis=0)
        source_lengths = np.linalg.norm(source_edges, axis=0)
        scales = (target_lengths / np.maximum(source_lengths, 1e-10))[None, None, :]

        normals = np.vstack((target_edges[1], -target_edges[0]))
        normals /= np.maximum(target_lengths, 1e-10)

        if anisotropy is not None:
            normals = anisotropy @ normals

        x_value = apply_green_coordinates(c_weights, target[0])
        x_value += apply_green_coordinates(scales * d_weights, normals[0])
        y_value = apply_green_coordinates(c_weights, target[1])
        y_value += apply_green_coordinates(scales * d_weights, normals[1])

        x_image = x_value.reshape(self.image_height, self.image_width)
        y_image = self.image_height - y_value.reshape(self.image_height, self.image_width)

        x_image = x_image.copy()
        y_image = y_image.copy()
        x_image[~self.safe_inside_mask] = np.nan
        y_image[~self.safe_inside_mask] = np.nan
        return self.fill_invalid_nearest(x_image, y_image, self.safe_inside_mask)

    def update_axis(self, axis, c_weights, d_weights, anisotropy=None):
        x_image, y_image = self.deformed_coordinates(c_weights, d_weights, anisotropy)
        self.clear_texture(axis)
        surface = self.draw_texture(axis, x_image, y_image, self.valid_mask)
        self.apply_axis_limits(axis)
        return x_image, y_image, surface

    def apply_deformation(self, _event=None):
        started = time.perf_counter()
        x_left, y_left, self.left_surface = self.update_axis(
            self.left_axis, self.c_baseline, self.d_baseline,
            anisotropy=None,  # 各向同性：不乘 A
        )
        self.left_coordinates = (x_left, y_left)
        x_right, y_right, self.right_surface = self.update_axis(
            self.right_axis, self.c_agc, self.d_agc,
            anisotropy=self.anisotropy,  # 各向异性：乘 A
        )
        self.right_coordinates = (x_right, y_right)
        self.update_cage_display()
        self.figure.canvas.draw_idle()
        LOGGER.info("Deformation updated in %.3f s", time.perf_counter() - started)

    def set_target_cage(self, cage):
        if cage.shape != self.source_cage.shape:
            raise ValueError(
                f"Target cage shape {cage.shape} does not match source "
                f"{self.source_cage.shape}."
            )
        self.cage_points = np.asarray(cage, dtype=float)
        self.update_cage_display()
        self.apply_axis_limits(self.left_axis)
        self.apply_axis_limits(self.right_axis)

    @staticmethod
    def create_hidden_tk_root():
        root = tk.Tk()
        root.withdraw()
        return root

    def save_cage(self, _event=None):
        root = self.create_hidden_tk_root()
        try:
            filename = filedialog.asksaveasfilename(
                title="Save target cage",
                initialfile="target_cage.txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
        finally:
            root.destroy()
        if filename:
            np.savetxt(filename, self.cage_points, fmt="%.10f")
            LOGGER.info("Saved cage to %s", filename)

    def load_target_cage(self, _event=None):
        root = self.create_hidden_tk_root()
        try:
            filename = filedialog.askopenfilename(
                title="Load target cage",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
        finally:
            root.destroy()
        if not filename:
            return
        try:
            self.set_target_cage(load_cage(Path(filename)))
            self.apply_deformation()
            LOGGER.info("Loaded target cage from %s", filename)
        except Exception:
            LOGGER.exception("Failed to load target cage from %s", filename)

    def show(self):
        plt.show()


def resolve_input_paths(args):
    if args.image is not None or args.cage is not None:
        if args.image is None or args.cage is None:
            raise ValueError("--image and --cage must be provided together.")
        return Path(args.image), Path(args.cage)
    directory = Path(args.data_root) / args.folder
    return directory / "img.png", directory / "cage.txt"


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Interactive 2D Anisotropic Green Coordinates deformation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder", default="giraffe", help="Example folder under --data-root."
    )
    parser.add_argument(
        "--data-root", default="data", help="Root directory for examples."
    )
    parser.add_argument("--image", help="Explicit image path; requires --cage.")
    parser.add_argument("--cage", help="Explicit source cage path; requires --image.")
    parser.add_argument("--target-cage", help="Optional target cage loaded at startup.")
    parser.add_argument(
        "--A", nargs=4, type=float,
        default=[1.5, 0.5, 0.5, 1.5],
        metavar=("A11", "A12", "A21", "A22"),
        help="Symmetric positive-definite 2x2 matrix in row-major order.",
    )
    parser.add_argument(
        "--axis-padding", type=nonnegative_float, default=100.0,
        help="Padding around the cage in display coordinates.",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        AGCApplication(args).show()
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()