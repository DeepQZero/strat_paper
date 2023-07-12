from numba import njit
import numpy as np
## NEED TO INSTALL SCIPY TO USE NP LINALG NORM

GEO = 42.164e6
BASE_VEL_Y = 3.0746e3
MU = 3.9860e14


@njit()
def norm(x1, y1, x2=0, y2=0):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

@njit()
def integrand(pv):
    # MU = 3.9860e14
    return np.concatenate(
        (pv[2:4], (-3.9860e14 / np.linalg.norm(pv[0:2]) ** 3) * pv[0:2]))


@njit()
def runge_kutta(pv, up_len):
    k1 = up_len * integrand(pv)
    k1_2 = k1 / 2.0
    k2 = up_len * integrand(pv + k1_2)
    k2_2 = k2 / 2.0
    k3 = up_len * integrand(pv + k2_2)
    k4 = up_len * integrand(pv + k2)
    return pv + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit()
def propagate(pv, times, update):
    for _ in range(times):
        pv = runge_kutta(pv, update)
    return pv


@njit()
def distance(pos1, pos2):
    """Calculates euclidean distance between two positions."""
    return np.linalg.norm(pos1 - pos2)


@njit()
def rotate(px, py, angle):
    """counterclockwise rotation in radians."""
    return [np.cos(angle) * px - np.sin(angle) * py,
            np.sin(angle) * px + np.cos(angle) * py]


@njit()
def mod_ang(ang, modder):
    return ((ang + modder/2) % modder) - modder / 2



@njit()
def angle_diff(x1, y1, x2, y2):
    """Calculates absolute value difference between two angles in radians."""
    x = np.arctan2(y1, x1)
    y = np.arctan2(y2, x2)
    abs_diff = abs(x - y)
    return min((2 * np.pi) - abs_diff, abs_diff)


# if __name__ == "__main__":
#     arr = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
#     print(propagate(arr, 100, 60))
