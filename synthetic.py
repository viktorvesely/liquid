import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def sample(N: int):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    return _sample(x, y)

def _sample2(x: np.ndarray, y: np.ndarray):

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


def _sample(x: np.ndarray, y: np.ndarray):

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

def plot():

    from scipy import interpolate

    cmap = ListedColormap(['#1446A0', '#DB3069', '#F5D547'])

    fig, ax = plt.subplots(figsize=(6, 6))

    N = 300

    x, v = sample(N * N)

    X, Y = np.meshgrid(
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, N)
    )


    interp = interpolate.LinearNDInterpolator(x, v)
    Z = interp(X, Y)

    ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap, interpolation='nearest')
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    # plt.colorbar(ticks=[0, 1, 2], label='Class')
    plt.show()


if __name__ == "__main__":

    plot()