from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def sample(N: int, task: Literal["trigonometry", "mine"] = "trigonometry"):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)

    if task == "trigonometry":
        return _sample_trigonometry(x, y)
    elif task == "mine":
        return _sample_mine(x, y)


main_tasks = np.array([[[9.932413069162699, 4.519830328319423],
  [-0.8168586112347911, 3.0652076063204268]],
 [[6.813502941134584, 4.535151023517105],
  [-0.9787248160055277, -0.7607226180909064]],
 [[5.072345710065466, 8.097182426210178],
  [0.13002298321802952, -0.9435734256299448]],
 [[9.888464240328421, 6.869020754016834],
  [1.1026867568371523, 0.13987090946831815]]])


def generate_main_tasks(n_tasks: int = 4, per_task: int = 2):

    fs = np.random.uniform(2.0, 10.0, (n_tasks, 1, per_task))
    os = np.random.uniform(-np.pi, np.pi, (n_tasks, 1, per_task))

    tasks = np.concatenate((fs, os), axis=1)

    return tasks

def _sample_trigonometry(x: np.ndarray, y: np.ndarray, tasks: np.ndarray | None = None) -> np.ndarray:

    if tasks is None:
        tasks = main_tasks.copy()

    F = 0
    O = 1
    quantiles = [1/3, 2/3]

    z = np.full_like(x, -1)
    c = np.full_like(x, -1, dtype=int)

    mask_ul = (x < 0) & (y > 0)
    mask_ur = (x >= 0) & (y > 0)
    mask_ll = (x < 0) & (y <= 0)
    mask_lr = (x >= 0) & (y <= 0)

    if (not mask_ul.any()) or (not mask_ur.any()) or (not mask_ll.any()) or (not mask_lr.any()):
        raise ValueError("Some quadrant has 0 values")

    def fill_classes(mask: np.ndarray):

        if not (mask.any()):
            return c

        t1, t2 = np.quantile(z[mask], quantiles)
        c_masked = c[mask]
        c_masked[:] = 0
        c_masked[z[mask] > t1] = 1
        c_masked[z[mask] > t2] = 2

        c[mask] = c_masked

        return c

    # Upper Left quadrant (x < 0, y > 0)
    dx = x[mask_ul] + 0.5
    dy = y[mask_ul] - 0.5
    z[mask_ul] = np.cos(dx * tasks[0, F, 0] + tasks[0, O, 0]) \
                + np.tanh(dy * tasks[0, F, 1] + tasks[0, O, 1])
    c = fill_classes(mask_ul)

    # Upper Right quadrant (x >= 0, y > 0)
    dx = x[mask_ur] - 0.5
    dy = y[mask_ur] - 0.5
    z[mask_ur] = np.sin(dx * tasks[1, F, 0] + tasks[1, O, 0]) \
                * np.tanh(dy * tasks[1, F, 1] + tasks[1, O, 1])
    c = fill_classes(mask_ur)

    # Lower Right quadrant (x >= 0, y <= 0)
    dx = x[mask_lr] + 0.5
    dy = y[mask_lr] + 0.5
    z[mask_lr] = np.sin(dx * tasks[2, F, 0] + tasks[2, O, 0] \
                + np.tanh(dy * tasks[2, F, 1] + tasks[2, O, 1]))
    c = fill_classes(mask_lr)

    # Lower Left quadrant (x < 0, y <= 0)
    dx = x[mask_ll] + 0.5
    dy = y[mask_ll] + 0.5
    z[mask_ll] = np.tanh(
        np.cos(dx * tasks[3, F, 0] + tasks[3, O, 0]) \
        * np.sin(dy * tasks[3, F, 1] + tasks[3, O, 1])
    )
    c = fill_classes(mask_ll)

    if x.ndim > 1:
        return None, c

    X = np.empty((x.size, 2))
    X[:, 0] = x
    X[:, 1] = y

    return X, c


def _sample_mine(x: np.ndarray, y: np.ndarray):

    z = np.zeros_like(x, dtype=int)

    mask_ul = (x < 0) & (y > 0)
    mask_ur = (x >= 0) & (y > 0)
    mask_ll = (x < 0) & (y <= 0)
    mask_lr = (x >= 0) & (y <= 0)

    # ------------------------
    # Upper Left: Spiral Pattern
    # ------------------------
    # For this quadrant, use the center (-0.5, 0.5)
    dx = x[mask_ul] + 0.5
    dy = y[mask_ul] - 0.5
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    # Add a twist proportional to the radius to create a spiral effect
    total_angle = np.mod(theta + 4 * r, 2 * np.pi)
    # Divide the circle into three equal segments
    classes_ul = np.floor(total_angle / (2 * np.pi / 3)).astype(int)
    z[mask_ul] = classes_ul

    # ------------------------
    # Upper Right: Grid (Squares) Pattern
    # ------------------------
    # Here, the quadrant spans x in [0,1] and y in [0,1]
    dx = x[mask_ur]
    dy = y[mask_ur]
    # Clip to avoid potential index overflow at the boundary
    dx_clipped = np.clip(dx, 0, 0.9999)
    dy_clipped = np.clip(dy, 0, 0.9999)
    # Divide the quadrant into a 4x4 grid and assign class based on grid indices
    i = np.floor(dx_clipped * 4).astype(int)
    j = np.floor(dy_clipped * 4).astype(int)
    classes_ur = (i + j) % 3
    z[mask_ur] = classes_ur

    # ------------------------
    # Lower Left: Sinusoidal Wave Pattern
    # ------------------------
    # Use local coordinates with center (-0.5, -0.5)
    dx = x[mask_ll] + 0.5
    dy = y[mask_ll] + 0.5
    # Combine two sine waves
    f = np.sin(4 * np.pi * dx) + np.sin(4 * np.pi * dy)
    # Map the combined sine value (which lies between -2 and 2) into 3 classes
    classes_ll = np.zeros_like(f, dtype=int)
    classes_ll[f < -0.67] = 0
    classes_ll[(f >= -0.67) & (f < 0.67)] = 1
    classes_ll[f >= 0.67] = 2
    z[mask_ll] = classes_ll

    # ------------------------
    # Lower Right: Concentric Circles Pattern
    # ------------------------
    # Center at (0.5, -0.5)
    dx = x[mask_lr] - 0.5
    dy = y[mask_lr] + 0.5
    r = np.sqrt(dx**2 + dy**2)
    # Multiply the radius to set the frequency of the rings,
    # then assign classes using floor and modulo 3 to cycle through classes.
    classes_lr = (np.floor(r * 10).astype(int)) % 3
    z[mask_lr] = classes_lr

    X = np.empty((x.size, 2))
    X[:, 0] = x
    X[:, 1] = y

    return X, z

def plot(task: Literal["trigonometry", "mine"] = "trigonometry", N: int = 300):

    from scipy import interpolate

    cmap = ListedColormap(['#1446A0', '#DB3069', '#F5D547'])

    fig, ax = plt.subplots(figsize=(6, 6))

    x, v = sample(N * N, task=task)

    X, Y = np.meshgrid(
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, N)
    )


    interp = interpolate.LinearNDInterpolator(x, v)
    Z = interp(X, Y)

    ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap, interpolation='bicubic')
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    # plt.colorbar(ticks=[0, 1, 2], label='Class')
    plt.show()

    return fig


if __name__ == "__main__":

    plot()