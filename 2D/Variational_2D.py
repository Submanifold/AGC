"""
Variational Anisotropic Green Coordinates for 2D shape deformation.
"""

import argparse
import math
import os
import re
import time

import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.path import Path
from matplotlib.widgets import Button
import numpy as np
from scipy import ndimage



def anisotropic_matrix_sqrt(A):
    A = np.asarray(A, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(A)
    if np.any(eigvals <= 0):
        raise ValueError("A must be symmetric positive definite.")
    B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    inv_B = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return B, inv_B


def load_image_rgba(path):
    img = iio.imread(path)
    if img.ndim == 2:
        rgb = np.repeat(img[:, :, None], 3, axis=2)
        alpha = None
    elif img.shape[2] == 4:
        rgb = img[:, :, :3].copy()
        alpha = img[:, :, 3]
        for c in range(3):
            rgb[:, :, c][alpha == 0] = 255
    else:
        rgb = img[:, :, :3].copy()
        alpha = None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, alpha


def load_cage(path):
    data = np.loadtxt(path)
    flat = np.asarray(data, dtype=float).ravel()
    if flat.size % 2 != 0:
        raise ValueError("Cage file must contain an even number of values.")
    return flat.reshape(-1, 2)


def parse_point_data(data_str):
    pts = []
    for item in data_str.split(";"):
        parts = [p for p in re.split(r"[,\s]+", item.strip()) if p]
        if len(parts) >= 2:
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def read_constraints_file(path):
    result = {
        "user_constraints_origin": np.zeros((0, 2), dtype=float),
        "user_constraints_deformed": np.zeros((0, 2), dtype=float),
        "medial_points": np.zeros((0, 2), dtype=float),
    }
    if not os.path.exists(path):
        return result
    pattern = re.compile(r"^\s*(\w+)\s*=\s*\[(.*)\]\s*;\s*$")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%") or stripped.startswith("#"):
                continue
            m = pattern.match(line)
            if not m:
                continue
            name, body = m.group(1), m.group(2)
            if name in result:
                result[name] = parse_point_data(body)
    return result


def apply_coordinate_transform(points, h_img):
    points = np.asarray(points, dtype=float).copy().reshape(-1, 2)
    if points.size:
        points[:, 1] = h_img - points[:, 1] + 1.0
    return points


def poly2mask(x, y, h, w):
    yy, xx = np.mgrid[1:h + 1, 1:w + 1]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    mask = Path(np.vstack([x, y]).T).contains_points(pts, radius=1e-9)
    return mask.reshape(h, w)


def bwperim(mask):
    eroded = ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return mask & ~eroded


def is_point_in_cage(point_math, V0_math, h_img):
    point_pixel = np.array([point_math[0], h_img - point_math[1] + 1.0])
    cage_pixel = np.column_stack([V0_math[0], h_img - V0_math[1] + 1.0])
    return bool(Path(cage_pixel).contains_point(point_pixel, radius=1e-9))


def compute_edge_normals(V0):
    edge_vec = np.roll(V0, -1, axis=1) - V0
    n = np.vstack([edge_vec[1], -edge_vec[0]])
    n = -n / np.maximum(np.linalg.norm(n, axis=0, keepdims=True), 1e-10)
    return n


def generate_boundary_samples(V0, h_img, epsilon=5.0, samples_per_edge=10):
    k = V0.shape[1]
    normals = -compute_edge_normals(V0)
    samples = []
    for i in range(k):
        j = (i + 1) % k
        vi = V0[:, i]
        vj = V0[:, j]
        for t in np.linspace(0, 1, samples_per_edge + 2)[1:-1]:
            p = vi + t * (vj - vi) + epsilon * normals[:, i]
            if is_point_in_cage(p, V0, h_img):
                samples.append(p)
    return np.asarray(samples, dtype=float).reshape(-1, 2)


def isotropic_edge_integral(vi, vj, ni, w):
    a = vj - vi
    edge_len = np.linalg.norm(a, axis=0)
    b = vi - w
    Q = np.sum(a * a, axis=0)
    S = np.sum(b * b, axis=0)
    R = 2.0 * np.sum(a * b, axis=0)
    disc = np.maximum(4.0 * S * Q - R ** 2, 1e-10)
    SRT = np.sqrt(disc)
    BA = edge_len * np.sum(b * ni.reshape(2, 1), axis=0)
    L0 = np.log(np.maximum(S, 1e-10))
    L1 = np.log(np.maximum(S + Q + R, 1e-10))
    L10 = L1 - L0
    Qs = np.maximum(Q, 1e-10)
    SRTs = np.maximum(SRT, 1e-10)
    A0 = np.arctan(R / SRTs) / SRTs
    A1 = np.arctan((2.0 * Q + R) / SRTs) / SRTs
    A10 = A1 - A0
    A10[SRT < 1e-10] = 0.0
    d = edge_len * (
        (4.0 * S - R ** 2 / Qs) * A10
        + R / (2.0 * Qs) * L10
        + L1
        - 2.0
    ) / (4.0 * np.pi)
    cj = -BA * (L10 / (2.0 * Qs) - A10 * (R / Qs)) / (2.0 * np.pi)
    ci = +BA * (L10 / (2.0 * Qs) - A10 * (2.0 + R / Qs)) / (2.0 * np.pi)
    close_i = np.linalg.norm(b, axis=0) < 1e-3
    close_j = np.linalg.norm(vj - w, axis=0) < 1e-3
    ci[close_i] = 0.0
    cj[close_j] = 0.0
    d[close_i | close_j] = 0.0
    return np.nan_to_num(d), np.nan_to_num(ci), np.nan_to_num(cj)


def anisotropic_edge_integral(vi, vj, w, A, B, inv_B):
    P = w.shape[1]
    edge_vec = vj - vi
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-10:
        return np.zeros(P), np.zeros(P), np.zeros(P)
    ni = np.array([edge_vec[1], -edge_vec[0]], dtype=float) / edge_len
    ni_A_ni = float(ni.T @ A @ ni)
    if ni_A_ni < 1e-10:
        return np.zeros(P), np.zeros(P), np.zeros(P)
    factor_d = 1.0 / np.sqrt(ni_A_ni)
    new_vi = inv_B @ (vi.reshape(2, 1) - w)
    new_vj = inv_B @ (vj.reshape(2, 1) - w)
    niA = B @ ni
    niA = niA / max(np.linalg.norm(niA), 1e-10)
    d_iso, ci_iso, cj_iso = isotropic_edge_integral(
        new_vi, new_vj, niA, np.zeros((2, P))
    )
    return (
        np.nan_to_num(d_iso * factor_d),
        np.nan_to_num(ci_iso),
        np.nan_to_num(cj_iso),
    )


def compute_anisotropic_green_coordinates(V0, W, A, B, inv_B, h, w):
    k = V0.shape[1]
    C = np.zeros((h, w, k), dtype=float)
    D = np.zeros((h, w, k), dtype=float)
    edge_stats = np.zeros((k, 5), dtype=float)
    for i in range(k):
        j = (i + 1) % k
        vi = V0[:, i]
        vj = V0[:, j]
        edge_vec = vj - vi
        edge_length = np.linalg.norm(edge_vec)
        if edge_length < 1e-10:
            continue
        edge_normal = np.array([edge_vec[1], -edge_vec[0]], dtype=float) / edge_length
        rel = W - vi.reshape(2, 1)
        dist = np.abs(np.sum(rel * edge_normal.reshape(2, 1), axis=0))
        t = np.sum(rel * edge_vec.reshape(2, 1), axis=0) / (edge_length ** 2)
        close = (dist < 1e-3) & (t >= 0.0) & (t <= 1.0)
        d_val, ci_val, cj_val = anisotropic_edge_integral(vi, vj, W, A, B, inv_B)
        d_val[close] = 0.0
        ci_val[close] = 0.0
        cj_val[close] = 0.0
        D[:, :, i] = d_val.reshape(h, w)
        C[:, :, i] += ci_val.reshape(h, w)
        C[:, :, j] += cj_val.reshape(h, w)
        edge_stats[i, :] = [i + 1, j + 1, edge_length, np.min(ci_val), np.max(cj_val)]
    return C, D, edge_stats


def compute_log_difference(omega):
    omega = complex(omega)
    if abs(omega) < 1e-10:
        return np.log(1.0) - np.log(1e-10)
    if abs(1.0 - omega) < 1e-10:
        return np.log(1e-10) - np.log(-1.0 + 0j)
    return np.log(1.0 - omega) - np.log(-omega)


def compute_G_integral(a, b, omega):
    omega = complex(omega)
    if b == -1:
        if a == 0:
            return compute_log_difference(omega)
        val = (omega ** a) * compute_log_difference(omega)
        for kk in range(a):
            val += (omega ** kk) / (a - kk)
        return val
    if a == 0:
        if b + 1 == 0:
            return compute_log_difference(omega)
        return ((1.0 - omega) ** (b + 1) - (-omega) ** (b + 1)) / (b + 1)
    if b + 1 == 0:
        return compute_G_integral(a, -1, omega)
    return (
        (1.0 - omega) ** (b + 1)
        - a * compute_G_integral(a - 1, b + 1, omega)
    ) / (b + 1)


def computeF_Kn_double_root(A, K, n, omega):
    return compute_G_integral(n, -2 * K, omega) / (A ** K)


def computeF_Kn_closed_form(Q, R, S, K, n, omega1, omega2):
    A = Q
    if abs(omega1 - omega2) < 1e-12:
        return computeF_Kn_double_root(A, K, n, omega1)
    val = 0j
    for kk in range(1, K + 1):
        coeff = (
            ((-1) ** (K - kk))
            * math.factorial(2 * K - kk - 1)
            / (math.factorial(K - 1) * math.factorial(K - kk))
        )
        alpha = coeff * (omega1 - omega2) ** (-(2 * K - kk))
        beta = coeff * (omega2 - omega1) ** (-(2 * K - kk))
        val += alpha * compute_G_integral(n, -kk, omega1) + beta * compute_G_integral(
            n, -kk, omega2
        )
    return val / (A ** K)


def real_clean(x):
    x = np.real_if_close(x, tol=1000)
    return np.real(x).astype(float)


def compute_curve_coordinates_closed_form(c_coeffs, eta, n):
    vi = c_coeffs[:, 0]
    vj = c_coeffs[:, 1]
    a = vj - vi
    L = np.linalg.norm(a)
    if L < 1e-12:
        return np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))
    b = vi - eta
    Q = float(np.dot(a, a))
    R = float(2.0 * np.dot(a, b))
    S = float(np.dot(b, b))
    if Q < 1e-14:
        return np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))
    sqrt_disc = np.sqrt(complex(R * R - 4.0 * Q * S))
    omega1 = (-R + sqrt_disc) / (2.0 * Q)
    omega2 = (-R - sqrt_disc) / (2.0 * Q)
    two_pi = 2.0 * np.pi
    F10 = computeF_Kn_closed_form(Q, R, S, 1, 0, omega1, omega2)
    F11 = computeF_Kn_closed_form(Q, R, S, 1, 1, omega1, omega2)
    grad_D = -(L / two_pi) * (b * F10 + a * F11)
    F20 = computeF_Kn_closed_form(Q, R, S, 2, 0, omega1, omega2)
    F21 = computeF_Kn_closed_form(Q, R, S, 2, 1, omega1, omega2)
    F22 = computeF_Kn_closed_form(Q, R, S, 2, 2, omega1, omega2)
    I_term = np.eye(2) * F10
    term_bb = np.outer(b, b) * F20
    term_ab_ba = (np.outer(a, b) + np.outer(b, a)) * F21
    term_aa = np.outer(a, a) * F22
    hess_D = (L / two_pi) * (I_term - 2.0 * (term_bb + term_ab_ba + term_aa))
    a_dot_n = float(np.dot(a, n))
    b_dot_n = float(np.dot(b, n))
    F12 = computeF_Kn_closed_form(Q, R, S, 1, 2, omega1, omega2)
    F13 = computeF_Kn_closed_form(Q, R, S, 1, 3, omega1, omega2)
    G10 = F11
    G20 = F21
    G21 = F22
    G22 = computeF_Kn_closed_form(Q, R, S, 2, 3, omega1, omega2)
    term1 = n * G10
    int_bb_n = (b_dot_n * b) * G20
    int_ab_n = (b_dot_n * a + a_dot_n * b) * G21
    int_aa_n = (a_dot_n * a) * G22
    grad_C = (L / two_pi) * (term1 - 2.0 * (int_bb_n + int_ab_n + int_aa_n))
    F30 = computeF_Kn_closed_form(Q, R, S, 3, 0, omega1, omega2)
    F31 = computeF_Kn_closed_form(Q, R, S, 3, 1, omega1, omega2)
    F32 = computeF_Kn_closed_form(Q, R, S, 3, 2, omega1, omega2)
    F33 = computeF_Kn_closed_form(Q, R, S, 3, 3, omega1, omega2)
    G30 = F31
    G31 = F32
    G32 = F33
    G33 = computeF_Kn_closed_form(Q, R, S, 3, 4, omega1, omega2)
    term_H1 = 2.0 * (
        (np.outer(n, b) + np.outer(b, n)) * G20
        + (np.outer(n, a) + np.outer(a, n)) * G21
    )
    term_H2 = 2.0 * np.eye(2) * (b_dot_n * G20 + a_dot_n * G21)
    c0 = b_dot_n * np.outer(b, b) * G30
    c1 = b_dot_n * (np.outer(a, b) + np.outer(b, a)) * G31 + a_dot_n * np.outer(
        b, b
    ) * G31
    c2 = b_dot_n * np.outer(a, a) * G32 + a_dot_n * (
        np.outer(a, b) + np.outer(b, a)
    ) * G32
    c3 = a_dot_n * np.outer(a, a) * G33
    hess_C = (L / two_pi) * (term_H1 + term_H2 - 8.0 * (c0 + c1 + c2 + c3))
    return (
        real_clean(grad_C),
        real_clean(grad_D),
        real_clean(hess_C),
        real_clean(hess_D),
    )


def compute_green_coord_values_for_point(V0, eta, A, B, inv_B):
    """
    Compute the anisotropic Green coordinate values at a point eta.
    """
    eta = np.asarray(eta, dtype=float).reshape(2)
    k = V0.shape[1]
    W = eta.reshape(2, 1)

    C_val = np.zeros(k, dtype=float)
    D_val = np.zeros(k, dtype=float)

    for i in range(k):
        j = (i + 1) % k
        vi = V0[:, i]
        vj = V0[:, j]
        d_val, ci_val, cj_val = anisotropic_edge_integral(vi, vj, W, A, B, inv_B)
        D_val[i] = float(d_val[0])
        C_val[i] += float(ci_val[0])
        C_val[j] += float(cj_val[0])

    return np.nan_to_num(C_val), np.nan_to_num(D_val)


def finite_difference_green_derivatives(V0, eta, A, B, inv_B, step=1e-4):
    """
    Compute numerical Jacobian and Hessian of C and D by central differences.
    """
    eta = np.asarray(eta, dtype=float).reshape(2)
    h = float(step)

    def eval_CD(p):
        return compute_green_coord_values_for_point(V0, p, A, B, inv_B)

    e_x = np.array([h, 0.0])
    e_y = np.array([0.0, h])

    C0, D0 = eval_CD(eta)

    C_px, D_px = eval_CD(eta + e_x)
    C_mx, D_mx = eval_CD(eta - e_x)
    C_py, D_py = eval_CD(eta + e_y)
    C_my, D_my = eval_CD(eta - e_y)

    C_pp, D_pp = eval_CD(eta + e_x + e_y)
    C_pm, D_pm = eval_CD(eta + e_x - e_y)
    C_mp, D_mp = eval_CD(eta - e_x + e_y)
    C_mm, D_mm = eval_CD(eta - e_x - e_y)

    k = V0.shape[1]

    num_grad_C = np.zeros((k, 2), dtype=float)
    num_grad_D = np.zeros((k, 2), dtype=float)

    num_hess_C = np.zeros((k, 2, 2), dtype=float)
    num_hess_D = np.zeros((k, 2, 2), dtype=float)

    num_grad_C[:, 0] = (C_px - C_mx) / (2.0 * h)
    num_grad_C[:, 1] = (C_py - C_my) / (2.0 * h)

    num_grad_D[:, 0] = (D_px - D_mx) / (2.0 * h)
    num_grad_D[:, 1] = (D_py - D_my) / (2.0 * h)

    num_hess_C[:, 0, 0] = (C_px - 2.0 * C0 + C_mx) / (h * h)
    num_hess_C[:, 1, 1] = (C_py - 2.0 * C0 + C_my) / (h * h)

    num_hess_D[:, 0, 0] = (D_px - 2.0 * D0 + D_mx) / (h * h)
    num_hess_D[:, 1, 1] = (D_py - 2.0 * D0 + D_my) / (h * h)

    num_hess_C[:, 0, 1] = (C_pp - C_pm - C_mp + C_mm) / (4.0 * h * h)
    num_hess_C[:, 1, 0] = num_hess_C[:, 0, 1]

    num_hess_D[:, 0, 1] = (D_pp - D_pm - D_mp + D_mm) / (4.0 * h * h)
    num_hess_D[:, 1, 0] = num_hess_D[:, 0, 1]

    return (
        np.nan_to_num(num_grad_C),
        np.nan_to_num(num_grad_D),
        np.nan_to_num(num_hess_C),
        np.nan_to_num(num_hess_D),
    )


def relative_error(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), eps)


def max_abs_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0

def validate_closed_form_derivatives(
    V0,
    points,
    A,
    B,
    inv_B,
    check_count=5,
    step=1e-3,
    rtol=1e-3,
    atol=1e-3,
):
    """
    Validate closed-form Jacobian/Hessian against numerical.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)

    if points.shape[0] == 0:
        print("[CHECK] no derivative check points, skipped.")
        return True

    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]

    if points.shape[0] == 0:
        print("[CHECK] no finite derivative check points, skipped.")
        return True

    check_count = min(int(check_count), points.shape[0])
    check_points = points[:check_count]

    print("")
    print("[CHECK] validating closed-form Jacobian / Hessian against numerical")
    print(f"[CHECK] points = {check_count}, numerical step = {step:g}")
    print(f"[CHECK] tolerance: rtol = {rtol:g}, atol = {atol:g}")

    cf_grad_C, cf_grad_D, cf_hess_C, cf_hess_D = compute_green_coord_derivatives_for_points(
        V0, check_points, A, B, inv_B
    )

    all_ok = True

    for idx, p in enumerate(check_points):
        (
            num_grad_C,
            num_grad_D,
            num_hess_C,
            num_hess_D,
        ) = finite_difference_green_derivatives(V0, p, A, B, inv_B, step=step)

        err_grad_C_abs = max_abs_error(cf_grad_C[idx], num_grad_C)
        err_grad_D_abs = max_abs_error(cf_grad_D[idx], num_grad_D)
        err_hess_C_abs = max_abs_error(cf_hess_C[idx], num_hess_C)
        err_hess_D_abs = max_abs_error(cf_hess_D[idx], num_hess_D)

        err_grad_C_rel = relative_error(cf_grad_C[idx], num_grad_C)
        err_grad_D_rel = relative_error(cf_grad_D[idx], num_grad_D)
        err_hess_C_rel = relative_error(cf_hess_C[idx], num_hess_C)
        err_hess_D_rel = relative_error(cf_hess_D[idx], num_hess_D)

        ok_grad_C = np.allclose(cf_grad_C[idx], num_grad_C, rtol=rtol, atol=atol)
        ok_grad_D = np.allclose(cf_grad_D[idx], num_grad_D, rtol=rtol, atol=atol)
        ok_hess_C = np.allclose(cf_hess_C[idx], num_hess_C, rtol=rtol, atol=atol)
        ok_hess_D = np.allclose(cf_hess_D[idx], num_hess_D, rtol=rtol, atol=atol)

        ok = ok_grad_C and ok_grad_D and ok_hess_C and ok_hess_D
        all_ok = all_ok and ok

        status = "OK" if ok else "FAIL"

        print(f"[CHECK] point {idx + 1:03d} eta = ({p[0]:.6f}, {p[1]:.6f}) -> {status}")
        print(
            f"        grad_C: abs={err_grad_C_abs:.3e}, rel={err_grad_C_rel:.3e}, ok={ok_grad_C}"
        )
        print(
            f"        grad_D: abs={err_grad_D_abs:.3e}, rel={err_grad_D_rel:.3e}, ok={ok_grad_D}"
        )
        print(
            f"        hess_C: abs={err_hess_C_abs:.3e}, rel={err_hess_C_rel:.3e}, ok={ok_hess_C}"
        )
        print(
            f"        hess_D: abs={err_hess_D_abs:.3e}, rel={err_hess_D_rel:.3e}, ok={ok_hess_D}"
        )

        if not ok:
            def report_max(name, closed, numeric):
                diff = np.abs(closed - numeric)
                max_id = np.unravel_index(np.argmax(diff), diff.shape)
                print(
                    f"        [MAX] {name}: index={max_id}, "
                    f"closed={closed[max_id]:.12e}, numeric={numeric[max_id]:.12e}, "
                    f"diff={diff[max_id]:.3e}"
                )

            report_max("grad_C", cf_grad_C[idx], num_grad_C)
            report_max("grad_D", cf_grad_D[idx], num_grad_D)
            report_max("hess_C", cf_hess_C[idx], num_hess_C)
            report_max("hess_D", cf_hess_D[idx], num_hess_D)

    print(f"[CHECK] final result: {'PASS' if all_ok else 'FAIL'}")
    print("")

    return all_ok


def compute_edge_contribution_simple(V0, edge_idx, eta, A, B, inv_B, inv_flag):
    V02 = inv_B @ V0
    eta2 = inv_B @ eta
    k = V0.shape[1]
    if inv_flag:
        i = (edge_idx + 1) % k
        j = edge_idx
    else:
        i = edge_idx
        j = (edge_idx + 1) % k
    edge_vec = V0[:, j] - V0[:, i]
    n = np.array([edge_vec[1], -edge_vec[0]], dtype=float)
    n = n / max(np.linalg.norm(n), 1e-10)
    if inv_flag:
        n = -n
    edge_vec2 = V02[:, j] - V02[:, i]
    n2 = np.array([edge_vec2[1], -edge_vec2[0]], dtype=float)
    n2 = n2 / max(np.linalg.norm(n2), 1e-10)
    if inv_flag:
        n2 = -n2
    grad_C, grad_D, hess_C, hess_D = compute_curve_coordinates_closed_form(
        V02[:, [i, j]], eta2, n2
    )
    nAn = float(n.T @ A @ n)
    factor = 1.0 / np.sqrt(nAn) if nAn > 1e-10 else 0.0
    grad_D *= factor
    hess_D *= factor
    grad_C = inv_B.T @ grad_C
    hess_C = inv_B.T @ hess_C @ inv_B
    grad_D = inv_B.T @ grad_D
    hess_D = inv_B.T @ hess_D @ inv_B
    return grad_C, grad_D, hess_C, hess_D


def unique_points(points, tol=1e-5):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    out = []
    for p in points:
        if not out or np.all(
            np.linalg.norm(np.asarray(out) - p.reshape(1, 2), axis=1) >= tol
        ):
            out.append(p)
    return np.asarray(out, dtype=float).reshape(-1, 2)


def point_key(p):
    return f"{p[0]:.10f}_{p[1]:.10f}"


def compute_green_coord_derivatives_for_points(V0, points, A, B, inv_B):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    k = V0.shape[1]
    n = points.shape[0]
    grad_C = np.zeros((n, k, 2), dtype=float)
    grad_D = np.zeros((n, k, 2), dtype=float)
    hess_C = np.zeros((n, k, 2, 2), dtype=float)
    hess_D = np.zeros((n, k, 2, 2), dtype=float)
    for pi, p in enumerate(points):
        eta = p.reshape(2)
        for edge in range(k):
            gCf, gDf, hCf, hDf = compute_edge_contribution_simple(
                V0, edge, eta, A, B, inv_B, False
            )
            j = (edge + 1) % k
            grad_D[pi, edge, :] = gDf
            hess_D[pi, edge, :, :] = hDf
            grad_C[pi, j, :] += gCf
            hess_C[pi, j, :, :] += hCf
            gCr, _gDr, hCr, _hDr = compute_edge_contribution_simple(
                V0, edge, eta, A, B, inv_B, True
            )
            j_rev = edge
            grad_C[pi, j_rev, :] += gCr
            hess_C[pi, j_rev, :, :] += hCr
    return grad_C, grad_D, hess_C, hess_D


def solve_local_step(a, b, grad_C, grad_D, point_indices, anchors):
    k = a.shape[1]
    m = b.shape[1]
    R = np.zeros((anchors.shape[0], 2, 2), dtype=float)
    for i, p in enumerate(anchors):
        idx = point_indices[point_key(p)]
        gC = grad_C[idx]
        gD = grad_D[idx]
        J = np.zeros((2, 2), dtype=float)
        for v in range(k):
            J[0, 0] += a[0, v] * gC[v, 0]
            J[0, 1] += a[1, v] * gC[v, 0]
            J[1, 0] += a[0, v] * gC[v, 1]
            J[1, 1] += a[1, v] * gC[v, 1]
        for e in range(m):
            J[0, 0] += b[0, e] * gD[e, 0]
            J[0, 1] += b[1, e] * gD[e, 0]
            J[1, 0] += b[0, e] * gD[e, 1]
            J[1, 1] += b[1, e] * gD[e, 1]
        U, _, Vt = np.linalg.svd(J)
        Ri = U @ Vt
        if np.linalg.det(Ri) < 0:
            Vt[-1, :] *= -1.0
            Ri = U @ Vt
        R[i] = Ri
    return R


def solve_soft_constraints(A_ls, B_ls, x0):
    if A_ls.size == 0:
        return x0.copy()
    lam_fidelity = 1.0
    lam_change = 0.1
    n = x0.size
    A_aug = np.vstack([np.sqrt(lam_fidelity) * A_ls, np.sqrt(lam_change) * np.eye(n)])
    B_aug = np.concatenate([np.sqrt(lam_fidelity) * B_ls, np.sqrt(lam_change) * x0])
    U, s, Vt = np.linalg.svd(A_aug, full_matrices=False)
    tol = max(A_aug.shape) * np.finfo(float).eps * max(np.max(s), 1.0)
    s_inv = np.zeros_like(s)
    s_inv[s > tol] = 1.0 / s[s > tol]
    return Vt.T @ (s_inv * (U.T @ B_aug))


def sample_C_D_at_point(C, D, p_math):
    h, w, k = C.shape
    x_idx = int(round(p_math[0])) - 1
    y_idx = int(round(h - p_math[1]))
    x_idx = max(0, min(x_idx, w - 1))
    y_idx = max(0, min(y_idx, h - 1))
    return C[y_idx, x_idx, :], D[y_idx, x_idx, :]


def solve_global_step(
    a,
    b,
    R,
    C,
    D,
    all_points,
    point_indices,
    grad_C,
    grad_D,
    hess_C,
    hess_D,
    user_origin,
    user_deformed,
    boundary_samples,
    anchors,
    lambda_hess=100.0,
    lambda_user=100.0,
):
    k = a.shape[1]
    m = b.shape[1]
    a_new = np.zeros_like(a)
    b_new = np.zeros_like(b)
    for dim in range(2):
        rows = []
        rhs = []
        for i, p in enumerate(anchors):
            idx = point_indices[point_key(p)]
            row = np.zeros((2, k + m), dtype=float)
            row[:, :k] = grad_C[idx].T
            row[:, k:] = grad_D[idx].T
            rows.append(row)
            rhs.append(R[i, :, dim].reshape(2))
        for p in boundary_samples:
            idx = point_indices[point_key(p)]
            row = np.zeros((3, k + m), dtype=float)
            row[0, :k] = hess_C[idx, :, 0, 0]
            row[1, :k] = hess_C[idx, :, 0, 1]
            row[2, :k] = hess_C[idx, :, 1, 1]
            row[0, k:] = hess_D[idx, :, 0, 0]
            row[1, k:] = hess_D[idx, :, 0, 1]
            row[2, k:] = hess_D[idx, :, 1, 1]
            rows.append(np.sqrt(lambda_hess) * row)
            rhs.append(np.zeros(3))
        for p, q in zip(user_origin, user_deformed):
            cval, dval = sample_C_D_at_point(C, D, p)
            row = np.zeros(k + m, dtype=float)
            row[:k] = cval
            row[k:] = dval
            rows.append(np.sqrt(lambda_user) * row.reshape(1, -1))
            rhs.append(np.array([np.sqrt(lambda_user) * q[dim]]))
        A_ls = np.vstack(rows) if rows else np.zeros((0, k + m))
        B_ls = np.concatenate(rhs) if rhs else np.zeros(0)
        x0 = np.concatenate([a[dim], b[dim]])
        sol = solve_soft_constraints(A_ls, B_ls, x0)
        a_new[dim] = sol[:k]
        b_new[dim] = sol[k:]
    return a_new, b_new


def variational_harmonic_map(
    V,
    C,
    D,
    A,
    B,
    inv_B,
    user_origin,
    user_deformed,
    medial_points,
    boundary_samples,
    max_iter=100,
    tol=1e-4,
    lambda_user=100.0,
    lambda_hess=100.0,
):
    k = V.shape[1]
    all_points = unique_points(
        np.vstack(
            [
                x
                for x in [medial_points, boundary_samples, user_origin]
                if x is not None and len(x)
            ]
        )
    )
    print(f"[INFO] derivative points: {all_points.shape[0]}")
    grad_C, grad_D, hess_C, hess_D = compute_green_coord_derivatives_for_points(
        V, all_points, A, B, inv_B
    )
    point_indices = {point_key(p): i for i, p in enumerate(all_points)}
    a = V.copy()
    edge_vec = np.roll(V, -1, axis=1) - V
    edge_len = np.linalg.norm(edge_vec, axis=0)
    N = np.vstack([edge_vec[1], -edge_vec[0]]) / np.maximum(
        edge_len.reshape(1, -1), 1e-10
    )
    b = A @ N
    anchors = np.asarray(medial_points, dtype=float).reshape(-1, 2)
    if anchors.shape[0] == 0:
        anchors = boundary_samples
    prev = np.concatenate([a.ravel(), b.ravel()])
    for it in range(max_iter):
        R = solve_local_step(a, b, grad_C, grad_D, point_indices, anchors)
        a, b = solve_global_step(
            a,
            b,
            R,
            C,
            D,
            all_points,
            point_indices,
            grad_C,
            grad_D,
            hess_C,
            hess_D,
            user_origin,
            user_deformed,
            boundary_samples,
            anchors,
            lambda_hess=lambda_hess,
            lambda_user=lambda_user,
        )
        cur = np.concatenate([a.ravel(), b.ravel()])
        rel = np.linalg.norm(cur - prev) / max(np.linalg.norm(prev), 1e-10)
        print(f"[VHM] iter {it + 1:03d}, relative change = {rel:.3e}")
        if rel < tol:
            break
        prev = cur
    return a, b


def apply_green_coords(coords, x):
    return np.sum(coords * np.asarray(x).reshape(1, 1, -1), axis=2)


def make_safe_mask(cage_points_img, alpha_mask, h, w):
    cage_mask = poly2mask(cage_points_img[:, 0], cage_points_img[:, 1], h, w)
    boundary = bwperim(cage_mask)
    dist = ndimage.distance_transform_edt(~boundary)
    return cage_mask & (dist >= 2.0) & alpha_mask


class VariationalAnisotropicApp:
    def __init__(
        self,
        folder,
        A,
        epsilon=5.0,
        samples_per_edge=10,
        lambda_user=100.0,
        lambda_hess=100.0,
    ):
        self.folder = folder
        self.lambda_user = float(lambda_user)
        self.lambda_hess = float(lambda_hess)

        self.img_path = os.path.join(".", "data", folder, "img.png")
        self.cage_path = os.path.join(".", "data", folder, "cage.txt")
        if not os.path.exists(self.cage_path):
            self.cage_path = os.path.join(".", "data", folder, "cage_variational.txt")
        self.constraints_path = os.path.join(".", "data", folder, "constraints.txt")

        self.img, alpha = load_image_rgba(self.img_path)
        self.h, self.w = self.img.shape[:2]

        if alpha is None:
            self.valid_mask = np.ones((self.h, self.w), dtype=bool)
        else:
            self.valid_mask = alpha > 0

        self.cage_points = load_cage(self.cage_path)
        self.current_cage = self.cage_points.copy()
        self.A = np.asarray(A, dtype=float)
        self.B, self.inv_B = anisotropic_matrix_sqrt(self.A)

        self.V0_math = np.vstack(
            [self.cage_points[:, 0], self.h - self.cage_points[:, 1] + 1.0]
        )

        X, Y = np.meshgrid(np.arange(1, self.w + 1), np.arange(1, self.h + 1))
        self.X0_grid = X
        self.Y0_grid = Y
        self.W_math = np.vstack([X.ravel(), self.h - Y.ravel() + 1.0])

        self.safe_in_mask = make_safe_mask(
            self.cage_points, self.valid_mask, self.h, self.w
        )

        cons = read_constraints_file(self.constraints_path)
        self.user_origin = apply_coordinate_transform(
            cons["user_constraints_origin"], self.h
        )
        self.user_deformed = apply_coordinate_transform(
            cons["user_constraints_deformed"], self.h
        )
        self.medial_points = apply_coordinate_transform(cons["medial_points"], self.h)

        self.boundary_samples = generate_boundary_samples(
            self.V0_math, self.h, epsilon, samples_per_edge
        )

        print(f"[INFO] image: {self.h} x {self.w}, cage vertices: {self.cage_points.shape[0]}")
        print(
            f"[INFO] position constraints: {self.user_origin.shape[0]}, "
            f"medial anchors: {self.medial_points.shape[0]}, "
            f"Hessian samples: {self.boundary_samples.shape[0]}"
        )
        print("[INFO] precomputing anisotropic Green coordinates...")
        t0 = time.perf_counter()
        self.C, self.D, _ = compute_anisotropic_green_coordinates(
            self.V0_math, self.W_math, self.A, self.B, self.inv_B, self.h, self.w
        )
        print(f"[INFO] precompute done in {time.perf_counter() - t0:.2f}s")

        self.fig = plt.figure("Variational Harmonic Map - Anisotropic", figsize=(8, 7))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.texture = None
        self.h_cage = None
        self.h_points = None
        self.h_user_points = None
        self.h_user_points_inner = None
        self.dragging = False
        self.dragged_point = None

        self.setup_axis()
        self.setup_controls()
        self.setup_events()

    def make_rgba(self, visible_mask=None):
        rgba = np.ones((self.h, self.w, 4), dtype=float)
        rgba[:, :, :3] = self.img.astype(float) / 255.0
        rgba[:, :, 3] = (
            self.valid_mask if visible_mask is None else visible_mask & self.valid_mask
        ).astype(float)
        return rgba

    def draw_texture(self, x_data=None, y_data=None, visible_mask=None):
        if x_data is None or y_data is None:
            x_data, y_data = self.X0_grid, self.Y0_grid
        x_data = np.asarray(x_data, dtype=float).copy()
        y_data = np.asarray(y_data, dtype=float).copy()
        finite = np.isfinite(x_data) & np.isfinite(y_data)
        x_data[~finite] = self.X0_grid[~finite]
        y_data[~finite] = self.Y0_grid[~finite]
        mesh = self.ax.pcolormesh(
            x_data,
            y_data,
            self.make_rgba(visible_mask),
            shading="nearest",
            edgecolors="none",
            antialiased=False,
            zorder=1,
        )
        mesh.set_gid("deformation_texture")
        return mesh

    def clear_texture(self):
        for artist in list(self.ax.collections):
            if (
                isinstance(artist, QuadMesh)
                and artist.get_gid() == "deformation_texture"
            ):
                artist.remove()

    def setup_axis(self):
        white = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        self.ax.imshow(
            white,
            origin="upper",
            extent=[1, self.w, self.h, 1],
            zorder=0,
        )
        self.ax.set_aspect("equal")
        self.texture = self.draw_texture()
        lines = np.vstack([self.current_cage, self.current_cage[0]])
        orange = (1.0, 0.5, 0.0)
        self.h_cage, = self.ax.plot(
            lines[:, 0], lines[:, 1], "-", linewidth=2, color=orange, zorder=3
        )
        self.h_points, = self.ax.plot(
            self.current_cage[:, 0],
            self.current_cage[:, 1],
            "o",
            markersize=12,
            markeredgewidth=3,
            markerfacecolor="none",
            color=orange,
            picker=8,
            zorder=4,
        )
        if self.user_origin.size:
            user_disp = apply_coordinate_transform(self.user_origin, self.h)
            self.h_user_points, = self.ax.plot(
                user_disp[:, 0],
                user_disp[:, 1],
                "o",
                markersize=12,
                markeredgewidth=3,
                markerfacecolor="none",
                color="red",
                zorder=5,
            )
            self.h_user_points_inner, = self.ax.plot(
                user_disp[:, 0],
                user_disp[:, 1],
                ".",
                markersize=3,
                color="red",
                zorder=5,
            )
        else:
            self.h_user_points = None
            self.h_user_points_inner = None
        self.apply_limits()

    def setup_controls(self):
        plt.subplots_adjust(bottom=0.12)
        ax_apply = self.fig.add_axes([0.02, 0.03, 0.20, 0.05])
        self.btn_apply = Button(ax_apply, "Apply Deformation")
        self.btn_apply.on_clicked(self.apply_deformation)

    def setup_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self.start_drag)
        self.fig.canvas.mpl_connect("motion_notify_event", self.dragging_fcn)
        self.fig.canvas.mpl_connect("button_release_event", self.stop_drag)

    def apply_limits(self, extra_points=None):
        buf = 100
        points = [self.current_cage]
        if extra_points is not None:
            extra_points = np.asarray(extra_points, dtype=float).reshape(-1, 2)
            finite = np.isfinite(extra_points[:, 0]) & np.isfinite(extra_points[:, 1])
            if np.any(finite):
                points.append(extra_points[finite])
        points = np.vstack(points)
        xmin = np.min(points[:, 0]) - buf
        xmax = np.max(points[:, 0]) + buf
        ymin = np.min(points[:, 1]) - buf
        ymax = np.max(points[:, 1]) + buf
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymax, ymin)

    def update_cage_display(self):
        lines = np.vstack([self.current_cage, self.current_cage[0]])
        self.h_cage.set_data(lines[:, 0], lines[:, 1])
        self.h_points.set_data(self.current_cage[:, 0], self.current_cage[:, 1])
        self.fig.canvas.draw_idle()

    def start_drag(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        d = np.sum(
            (self.current_cage - np.array([event.xdata, event.ydata])) ** 2, axis=1
        )
        idx = int(np.argmin(d))
        if d[idx] <= 20 ** 2:
            self.dragging = True
            self.dragged_point = idx

    def dragging_fcn(self, event):
        if (
            not self.dragging
            or event.inaxes is not self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self.current_cage[self.dragged_point] = [event.xdata, event.ydata]
        self.update_cage_display()

    def stop_drag(self, event):
        if self.dragging:
            self.dragging = False
            self.dragged_point = None

    def fill_invalid_nearest(self, x, y, mask):
        if np.all(mask):
            return x, y
        _, inds = ndimage.distance_transform_edt(~mask, return_indices=True)
        xx = x.copy()
        yy = y.copy()
        bad = ~mask
        xx[bad] = x[inds[0][bad], inds[1][bad]]
        yy[bad] = y[inds[0][bad], inds[1][bad]]
        return xx, yy

    def update_axis_deformation(self, a, b):
        k = a.shape[1]
        X1 = apply_green_coords(self.C, a[0]) + apply_green_coords(self.D, b[0])
        Y1 = apply_green_coords(self.C, a[1]) + apply_green_coords(self.D, b[1])
        X1_img = X1.reshape(self.h, self.w)
        Y1_img = self.h - Y1.reshape(self.h, self.w) + 1.0
        bad = ~np.isfinite(X1_img) | ~np.isfinite(Y1_img)
        X1_img[bad | ~self.safe_in_mask] = np.nan
        Y1_img[bad | ~self.safe_in_mask] = np.nan
        X1_img, Y1_img = self.fill_invalid_nearest(
            X1_img, Y1_img, self.safe_in_mask & ~bad
        )
        self.clear_texture()
        self.texture = self.draw_texture(
            X1_img, Y1_img, visible_mask=self.safe_in_mask
        )

        self.h_cage.set_visible(False)
        self.h_points.set_visible(False)

        extra_points = np.column_stack([X1_img.ravel(), Y1_img.ravel()])

        if self.user_deformed.size:
            user_deformed_disp = apply_coordinate_transform(self.user_deformed, self.h)
            if self.h_user_points is None:
                self.h_user_points, = self.ax.plot(
                    user_deformed_disp[:, 0],
                    user_deformed_disp[:, 1],
                    "o",
                    markersize=12,
                    markeredgewidth=3,
                    markerfacecolor="none",
                    color="blue",
                    linestyle="None",
                    zorder=5,
                )
                self.h_user_points_inner, = self.ax.plot(
                    user_deformed_disp[:, 0],
                    user_deformed_disp[:, 1],
                    ".",
                    markersize=3,
                    color="blue",
                    linestyle="None",
                    zorder=5,
                )
            else:
                self.h_user_points.set_data(
                    user_deformed_disp[:, 0], user_deformed_disp[:, 1]
                )
                self.h_user_points.set_color("blue")
                self.h_user_points.set_markerfacecolor("none")
                self.h_user_points.set_visible(True)
                if self.h_user_points_inner is None:
                    self.h_user_points_inner, = self.ax.plot(
                        user_deformed_disp[:, 0],
                        user_deformed_disp[:, 1],
                        ".",
                        markersize=12,
                        color="blue",
                        linestyle="None",
                        zorder=5,
                    )
                else:
                    self.h_user_points_inner.set_data(
                        user_deformed_disp[:, 0], user_deformed_disp[:, 1]
                    )
                    self.h_user_points_inner.set_color("blue")
                    self.h_user_points_inner.set_visible(True)
            extra_points = np.vstack([extra_points, user_deformed_disp])

        self.apply_limits(extra_points)
        self.fig.canvas.draw_idle()

    def apply_deformation(self, event=None):
        print("[INFO] applying variational anisotropic deformation...")
        t0 = time.perf_counter()
        a, b = variational_harmonic_map(
            self.V0_math,
            self.C,
            self.D,
            self.A,
            self.B,
            self.inv_B,
            self.user_origin,
            self.user_deformed,
            self.medial_points,
            self.boundary_samples,
            lambda_user=self.lambda_user,
            lambda_hess=self.lambda_hess,
        )
        self.update_axis_deformation(a, b)
        print(f"[INFO] done in {time.perf_counter() - t0:.2f}s")

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Variational harmonic map anisotropic deformation."
        )
    )
    parser.add_argument(
        "--folder",
        default="tower",
        help=(
            "data/<folder> containing img.png, cage_variational.txt or cage.txt, "
            "and constraints.txt"
        ),
    )
    parser.add_argument(
        "--A", nargs=4, type=float,
        default=[1.5, 0.5, 0.5, 1.5],
        metavar=("A11", "A12", "A21", "A22"),
        help="Symmetric positive-definite 2x2 matrix in row-major order.",
    )
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--samples-per-edge", type=int, default=10)
    parser.add_argument(
        "--lambda-position",
        "--lambda-user",
        dest="lambda_user",
        type=float,
        default=100.0,
        help="lambda_1: user position constraint weight (default: 100.0)",
    )
    parser.add_argument(
        "--lambda-hessian",
        "--lambda-hess",
        dest="lambda_hess",
        type=float,
        default=100.0,
        help="lambda_2: Hessian smoothness weight (default: 100.0)",
    )
    parser.add_argument(
        "--check-derivatives",
        action="store_true",
        help=(
            "validate closed-form Jacobian/Hessian against numerical "
            "before showing GUI"
        ),
    )




    args = parser.parse_args()
    app = VariationalAnisotropicApp(
        args.folder,
        np.asarray(args.A, dtype=float).reshape(2, 2),
        args.epsilon,
        args.samples_per_edge,
        lambda_user=args.lambda_user,
        lambda_hess=args.lambda_hess,
    )

    if args.check_derivatives:
        check_points = unique_points(
            np.vstack(
                [
                    x
                    for x in [
                        app.medial_points,
                        app.user_origin,
                        app.boundary_samples,
                    ]
                    if x is not None and len(x)
                ]
            )
        )
        validate_closed_form_derivatives(
            app.V0_math,
            check_points,
            app.A,
            app.B,
            app.inv_B,
        )

    app.show()


if __name__ == "__main__":
    main()