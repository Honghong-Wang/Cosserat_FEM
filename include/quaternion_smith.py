import numpy as np

eTOL = 1e-8


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


def axial(x):
    """
    x better be skew symmetric tensor
    :param x: skew symmetric tensor
    :return: axial vector
    """
    return np.array([x[2, 1], x[0, 2], x[1, 0]])


def clamp(x, minimum, maximum):
    return np.clip(x, minimum, maximum)


def quat_inv(q):
    a = np.zeros_like(q)
    a[1:] = -q[1:]
    a[0] = q[0]
    return a[:]


def quat_mul(q1, q2):
    """
    Product of two quaternion
    :param q1: q1
    :param q2: q1
    :return: q1 o q2
    """
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                     a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])


def quat_inv_mul(q1, q2):
    """
    Product of two quaternion
    :param q1: q1
    :param q2: q2
    :return: inv(q1) o q2
    """
    return quat_mul(quat_inv(q1), q2)


def quat_mul_inv(q1, q2):
    """
    Product of two quaternion
    :param q1: q1
    :param q2: q2
    :return: q1 o inv(q2)
    """
    return quat_mul(q1, quat_inv(q2))


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


def log(theta):
    """
    :param theta: total rotation
    :return: quaterion
    """
    return quat_from_scaled_rotation(theta)


def exp(q):
    """
    :param q: quaternion
    :return: rotation
    """
    return quat_to_scaled_rotation(q)


def kurvature_from_total_rotation(t, tds):
    """
    According to Crisfield & Jelenic
    :param t: theta
    :param tds: theta_prime
    :return: kappa
    """
    norm_t = np.linalg.norm(t)
    tensor_t = skew(t)
    if np.isclose(norm_t, 0, atol=1e-6):
        return tds
    x = np.sin(norm_t) / norm_t
    y = (1 - np.cos(norm_t)) / norm_t
    return (1 / norm_t ** 2 * (1 - x) * t @ t.T + x * np.eye(3) - y * tensor_t) @ tds


def quat_hermite(r0, v0, r1, v1, h_, hx_, hxx_):
    """
    :param r0: node 1 theta
    :param v0: node 1 theta'
    :param r1: node 2 theta
    :param v1: node 2 theta'
    :param h_: hermite shape fn
    :param hx_: derivative of h
    :param hxx_: double derivative of h
    :return: interpolated quaterion and curvature and acceleration
    """
    qr0 = quat_from_scaled_rotation(r0)
    qr1 = quat_from_scaled_rotation(r1)
    v0 = kurvature_from_total_rotation(r0, v0)
    v1 = kurvature_from_total_rotation(r1, v1)
    qr1_sub_qr0 = quat_to_scaled_rotation(quat_abs(quat_mul_inv(qr1, qr0)))
    rot = quat_mul(quat_from_scaled_rotation(h_[2][0] * qr1_sub_qr0 + h_[1][0] * v0 + h_[3][0] * v1), qr0)
    vel = hx_[2][0] * qr1_sub_qr0 + hx_[1][0] * v0 + hx_[3][0] * v1
    acc = hx_[2][0] * qr1_sub_qr0 + hxx_[1][0] * v0 + hx_[3][0] * v1
    return rot, vel, acc


if __name__ == "__main__":
    import os
    print(os.path.basename(__file__))
    q1 = np.array([0.70162935, -0.71254211, 0., 0.])
    q1 = q1 / np.linalg.norm(q1)
    q2 = quat_inv(q1)
    print(q2)
    print(q1)
    print(quat_mul(q1, q2))
