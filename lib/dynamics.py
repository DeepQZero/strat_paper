from numba import njit
import numpy as np

GEO = 42.164e6  # altitude of GEO from center of Earth
BASE_VEL_Y = 3.0746e3  # absolute value of velocity of base unit at geo
MU = 3.9860e14  # gravitational constant
# GEO_BOUND = 30e6  # bound from GEO used for episode termination


@njit()
def norm(x1: float, y1: float, x2: float = 0.0, y2: float = 0.0) -> float:
    """Calculates Euclidean norm of one vector or between two vectors."""  # TODO REMOVE
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@njit()
def vec_norm(vec: np.ndarray) -> float:
    """Calculates the norm between two vectors."""
    return np.sum(vec ** 2) ** 0.5


@njit()
def integrand(pv: np.ndarray) -> np.ndarray:
    """Calculates integrand for use Runge-Kutta"""  # TODO reword
    return np.concatenate((pv[2:4], (-MU / vec_norm(pv[0:2]) ** 3) * pv[0:2]))


@njit()
def runge_kutta(pv: np.ndarray, up_len: float) -> np.ndarray:
    """Calculates Runge-Kutta."""  # TODO reword
    k1 = up_len * integrand(pv)
    k1_2 = k1 / 2.0
    k2 = up_len * integrand(pv + k1_2)
    k2_2 = k2 / 2.0
    k3 = up_len * integrand(pv + k2_2)
    k4 = up_len * integrand(pv + k2)
    return pv + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit()
def propagate(pv: np.ndarray, up_times: int, up_len: float):
    """Computes and returns multiple Runge-Kutta updates."""
    for _ in range(up_times):
        pv = runge_kutta(pv, up_len)
    return pv


@njit()
def distance(pos1, pos2):  # TODO REMOVE!!!
    """Calculates Euclidean distance between two positions."""
    return np.linalg.norm(pos1 - pos2)


@njit()
def rotate(px: float, py: float, angle: float) -> np.ndarray:  # TODO REMOVE!!
    """Rotates 2D vector counterclockwise in radians."""
    return [np.cos(angle) * px - np.sin(angle) * py,
            np.sin(angle) * px + np.cos(angle) * py]


@njit()
def vec_rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotates 2D vector counterclockwise in radians."""
    return [np.cos(angle)* vec[0] - np.sin(angle) * vec[1],
            np.sin(angle) * vec[0] + np.cos(angle) * vec[1]]


@njit()
def mod_ang(angle: float, modulus: float = 2*np.pi) -> float:
    """Computes the residue of an angle in range (-modulus, modulus)."""
    return ((angle + modulus/2) % modulus) - modulus/2


@njit()
def angle_diff(x1, y1, x2, y2):  # TODO Remove!!!
    """Calculates absolute value difference between two angles in radians."""
    x = np.arctan2(y1, x1)
    y = np.arctan2(y2, x2)
    abs_diff = abs(x - y)
    return min((2 * np.pi) - abs_diff, abs_diff)


@njit()
def abs_angle_diff(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates smallest angle between two vectors in radians."""
    abs_diff = abs(np.arctan2(vec1[1], vec1[0]) -
                   np.arctan2(vec2[1], vec2[0]))
    return min(2*np.pi - abs_diff, abs_diff)


# if __name__ == "__main__":
#     arr = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
#     print(propagate(arr, 100, 60))
