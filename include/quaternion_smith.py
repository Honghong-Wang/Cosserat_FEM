import numpy as np

eTOL = 1e-8


def clamp(x, minimum, maximum):
    return np.clip(x, minimum, maximum)


def quaternion_inv(q):
    q[0] = -q[0]
    return q


def quat_mul(q1, q2):
    """
    Product of two quaternion
    :param q1: q1
    :param q2: q1
    :return: q1 o q2
    """
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])


def quat_inv_mul(q1, q2):
    """
    Product of two quaternion
    :param q1: q1
    :param q2: q2
    :return: inv(q1) o q2
    """
    return quat_mul(quaternion_inv(q1), q2)


def quat_mul_inv(q1, q2):
    """
    Product of two quaternion
    :param q1: q1
    :param q2: q2
    :return: q1 o inv(q2)
    """
    return quat_mul(q1, quaternion_inv(q2))


def quat_abs(q):
    return q if q[0] < 0.0 else -q


def quat_from_rotation_vector(x):
    """
    EXP
    :param x: rotation vector
    :return: associated quaternion
    """
    x = x.reshape((3,))
    half_angle = np.linalg.norm(x)
    if half_angle < eTOL:
        q = np.array([1, x[0], x[1], x[2]])
        return q / np.linalg.norm(q)
    else:
        cosine_half_angle = np.cos(half_angle)
        sin_half_angle = np.sin(half_angle) / half_angle
        return np.array([cosine_half_angle, sin_half_angle * x[0], sin_half_angle * x[1],  sin_half_angle * x[2]])


def quat_to_rotation_vector(q):
    """
    LOG
    :param q: quaternion
    :return: associated rotation
    """
    length = np.linalg.norm(q[1:])
    if length < eTOL:
        return q[1:]
    else:
        half_angle = np.arccos(clamp(q[0], -1.0, 1.0))
        return half_angle * q[1:] / length


def quat_to_scaled_rotation(q):
    """
    :param q: q
    :return: rotation vector
    """
    return 2.0 * quat_to_rotation_vector(q)


def quat_from_scaled_rotation(x):
    """
    :param x: v
    :return: rotation quaternion
    """
    return quat_from_rotation_vector(x / 2.0)
