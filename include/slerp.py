"""
Function to do spherical linear interpolation also known as slerp
for linear element only that is slerp(X, Q1, Q2)
"""
import numpy as np


def skew(x):
    """
    :param x: vector
    :return: skew symmetric tensor for which x is axial
    """
    x = np.reshape(x, (3,))
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]]
                    )


def slerp(q1, q2, n):
    """
    spherical linear interpolation of two quaternions representing rotation
    :param q1: quaternion 1
    :param q2: quaternion 2
    :param n:  shape function
    :return: q interpolated quaterion
    """
    # TODO: Vectorize it and possible generalize it for n-element interpolation
    # Done due to the fact -q and q represents same rotation
    sign = 1
    if np.dot(q1, q2) < 0:
        sign = -1
    omega = np.arccos(np.dot(q1, sign * q2) / np.linalg.norm(q1) / np.linalg.norm(q2))
    if np.isclose(abs(omega), 0, atol=1e-6):
        return q1 * n[0] + q2 * n[1]
    return (np.sin(n[0] * omega) / np.sin(omega)) * q1 + (np.sin(n[1] * omega) / np.sin(omega)) * q2


def diff_slerp(q1, q2, nx, n):
    """
    derivative of output of :slerp function
    :param q1: quaterion 1
    :param q2: quaterion 2
    :param nx: derivative of lagrange function
    :param n: lagrange function
    :return: dq interpolated derivative
    """
    # TODO: Vectorize it and possible generalize it for n-element interpolation
    # Done due to the fact -q and q represents same rotation
    sign = 1
    if np.dot(q1, q2) < 0:
        sign = -1
    omega = np.arccos(np.dot(q1, sign * q2) / np.linalg.norm(q1) / np.linalg.norm(q2))
    if np.isclose(omega, 0, atol=1e-7):
        return nx[0] * q1 + nx[1] * q2
    return np.cos(n[0] * omega) / np.sin(omega) * omega * nx[0] * q1 + np.cos(n[1] * omega) / np.sin(omega) * omega * nx[1] * q2


def get_rotation_from_quaterion(q):
    """
    Mostly accepts interpolated quaterion
    :param q: quaternion
    :return: r rotation
    """
    q = np.reshape(q, (4,))
    return 2 * np.array([
        [q[0] ** 2 + q[1] ** 2 - 0.5, q[1] * q[2] - q[3] * q[0], q[1] * q[3] + q[2] * q[0]],
        [q[2] * q[1] + q[3] * q[0], q[0] ** 2 + q[2] ** 2 - 0.5, q[2] * q[3] - q[1] * q[0]],
        [q[3] * q[1] - q[2] * q[0], q[3] * q[2] + q[1] * q[0], q[0] ** 2 + q[3] ** 2 - 0.5]
    ])


def get_rot_from_q(q):
    return 1 / np.linalg.norm(q) * (2 * q[0] ** 2 * np.eye(3) + 2 * q[0] * skew(q[1:]) + 2 * np.tensordot(q[1:], q[1:], 0)) - np.eye(3)


def get_rotation_derivative_from_quaterion_deprecated(q, dq, r):
    # TODO: Figure out what should be done if vector part of dq -> 0, for now a better alternative is available (Darboux, G. [1972])
    """
    Mostly accepts interpolated quaterion and its derivative
    :param dq: quaterion derivative
    :param q: quaternion
    :param r: rotation corresponding to q
    :return: rh rotation derivative
    """
    return -2 * np.dot(dq, q) / np.linalg.norm(q) * (np.eye(3) + r) + 0 if np.isclose(np.linalg.norm(dq[1:]), 0,
                                                                                      atol=1e-6) else \
        (2 / np.linalg.norm(dq[1:]) * (2 * q[0] * dq[0] * np.eye(3) + np.tensordot(dq[1:], q[1:], 0)) +
         np.tensordot(q[1:], dq[1:], 0) + dq[0] * skew(q[1:]) + q[0] * skew(dq[1:])) / np.linalg.norm(dq[1:])


def rotation_vector_to_quaterion(x):
    """
    :param x: rotation vector
    :return: quaterion
    """
    theta = np.linalg.norm(x)
    if np.isclose(theta, 0, atol=1e-8):
        return np.array([1, 0, 0, 0])
    e = np.sin(theta / 2) * x / theta
    q = np.array([np.cos(theta / 2), e[0], e[1], e[2]])
    return q


def quatmul(q1, q2):
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


def get_theta_from_rotation(rq):
    """
    Algorithm proposed by Spurrier
    :param rq: rotation matrix
    :return: theta
    """
    rmat = get_rotation_from_quaterion(rq)
    q = np.zeros(4)
    trq = np.trace(rmat)
    v = np.array([trq, rmat[0, 0], rmat[1, 1], rmat[2, 2]])
    m = v.argmax()
    maxval = v[m]
    if m == 0:
        q[0] = 0.5 * np.sqrt(1 + maxval)
        q[1] = 0.25 * (rmat[2, 1] - rmat[1, 2]) / q[0]
        q[2] = 0.25 * (rmat[0, 2] - rmat[2, 0]) / q[0]
        q[3] = 0.25 * (rmat[1, 0] - rmat[0, 1]) / q[0]
    elif m == 1:
        q[1] = np.sqrt(0.5 * maxval + 0.25 * (1 - trq))
        q[0] = 0.25 * (rmat[2, 1] - rmat[1, 2]) / q[1]
        q[2] = 0.25 * (rmat[0, 1] + rmat[1, 0]) / q[1]
        q[3] = 0.25 * (rmat[2, 0] + rmat[0, 2]) / q[1]
    elif m == 2:
        q[2] = np.sqrt(0.5 * maxval + 0.25 * (1 - trq))
        q[1] = 0.25 * (rmat[0, 1] + rmat[1, 0]) / q[2]
        q[0] = 0.25 * (rmat[0, 2] - rmat[2, 0]) / q[2]
        q[3] = 0.25 * (rmat[1, 2] + rmat[2, 1]) / q[2]
    elif m == 3:
        q[3] = np.sqrt(0.5 * maxval + 0.25 * (1 - trq))
        q[1] = 0.25 * (rmat[2, 0] + rmat[0, 2]) / q[3]
        q[2] = 0.25 * (rmat[1, 2] + rmat[2, 1]) / q[3]
        q[0] = 0.25 * (rmat[1, 0] - rmat[0, 1]) / q[3]
    else:
        raise Exception("not max index")
    return q / np.linalg.norm(q)


def rotate_vec(q, v):
    """
    :param q: quaternion
    :param v: vector
    :return: rotated vector
    """
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    v = np.array([0, v[0], v[1], v[2]])
    vc = quatmul(q, quatmul(v, qc))
    return vc[1:]


def quaterion_to_rotation_vec(q):
    q = q / np.linalg.norm(q)
    normt = np.linalg.norm(q[1:])
    if np.isclose(normt, 0, atol=1e-10):
        return q[1:] * 2
    else:
        return 2 * np.arcsin(normt) / normt * q[1:]
