import numpy as np


def init_gauss_points(n=3):
    """
    Gauss Quadrature
    :param n: number of gauss points
    :return: (weights of gp,Gauss points)
    """
    if n == 1:
        wgp = np.array([2])
        egp = np.array([0])
    elif n == 2:
        wgp = np.array([1, 1])
        egp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif n == 3:
        wgp = np.array([5 / 9, 8 / 9, 5 / 9])
        egp = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    else:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 gauss points here")
    return wgp, egp


def impose_boundary_condition(k, f, ibc, bc):
    """
    Elimination of variables, modifies incoming stiffness and force vector
    :param k: Stiffness matrix / Tangent stiffness
    :param f: force vector / residue
    :param ibc: node at with BC is prescribed
    :param bc: boundary condition
    """
    f -= (k[:, ibc] * bc)[:, None]
    f[ibc] = bc
    k[:, ibc] = 0
    k[ibc, :] = 0
    k[ibc, ibc] = 1


def get_displacement_vector(k, f):
    """
    :param k: Non-singular stiffness matrix
    :param f: force vector
    :return: nodal displacement
    """
    return np.linalg.solve(k, f)


def get_hermite_fn(gp, j, element_type=2):
    """
    :param element_type: 2
    :param gp: eta or gauss points or natural coordinate
    :param j: jacobian
    :return: (H,H',H")
    """
    if element_type != 2:
        raise Exception("only linear element for hermite fn")
    Nmat = np.array([.25 * (gp + 2) * (1 - gp) ** 2, j * .25 * (gp + 1) * (1 - gp) ** 2,
                     .25 * (-gp + 2) * (1 + gp) ** 2, j * .25 * (gp - 1) * (1 + gp) ** 2])
    Nmat_ = (1 / j) * np.array([0.75 * (gp ** 2 - 1), j * 0.25 * (3 * gp ** 2 - 2 * gp - 1),
                                0.75 * (1 - gp ** 2), j * 0.25 * (3 * gp ** 2 + 2 * gp - 1)])
    Nmat__ = (1 / j ** 2) * np.array([1.5 * gp, (-.5 + 1.5 * gp) * j,
                                      -1.5 * gp, (.5 + 1.5 * gp) * j])

    return Nmat, Nmat_, Nmat__


def get_lagrange_fn(gp, element_type=2):
    """
    Linear Lagrange shape functions
    :param element_type: element type
    :param gp: gauss point
    :return: (L, L')
    """
    if element_type == 2:
        nmat = np.array([.5 * (1 - gp), .5 * (1 + gp)])
        bmat = np.array([-.5, .5])
    elif element_type == 3:
        nmat = np.array([0.5 * (-1 + gp) * gp, (-gp + 1) * (gp + 1), 0.5 * gp * (1 + gp)])
        bmat = np.array([0.5 * (-1 + 2 * gp), -2 * gp, 0.5 * (1 + 2 * gp)])
    else:
        raise Exception("Sir, This is Wendy's we only do cubic elements here !")
    return nmat[:, None], bmat[:, None]


def get_connectivity_matrix(n, length, element_type=2):
    """
    :param element_type: element type
    :param length: length
    :param n: number of 1d elements
    :return: connectivity vector, nodal_data
    """
    node_data = np.linspace(0, length, (element_type - 1) * n + 1)
    icon = np.zeros((element_type + 1, n), dtype=np.int32)
    icon[0, :] = np.arange(0, (element_type - 1) * n, element_type - 1)
    if element_type == 3:
        icon[1, :] = icon[0, :]
        icon[2, :] = icon[1, :] + 1
        icon[3, :] = icon[2, :] + 1
    elif element_type == 2:
        icon[1, :] = icon[0, :]
        icon[2, :] = icon[1, :] + 1
    else:
        raise Exception("Sir, This is Wendy's we only do cubic elements here !")
    return icon.T, node_data


def init_stiffness_force(nnod, dof):
    """
    :param nnod: number of nodes
    :param dof: Dof
    :return: zero stiffness n force
    """
    return np.zeros((nnod * dof, nnod * dof)), np.zeros((nnod * dof, 1))


def get_theta_from_rotation(rmat):
    """
    Algorithm proposed by Spurrier
    :param rmat: rotation matrix
    :return: theta
    """
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
    if q[0] >= 0:
        normt = 2 * np.arcsin(np.linalg.norm(q[1:]))
    else:
        normt = 2 * (np.pi - np.arcsin(np.linalg.norm(q[1:])))
    if np.isclose(normt, 0, atol=1e-6):
        return np.zeros((3,))
    else:
        return normt / np.linalg.norm(q[1:]) * q[1:]


def get_theta_from_rotation_deprecated(rmat):
    """
    Lie group log map
    :param rmat: rotation matrix
    :return: theta vector
    """
    t = np.arccos((np.trace(rmat) - 1) / 2)
    if np.isclose(t, 0):
        return np.zeros((3,))
    # print(get_axial_from_skew_symmetric_tensor(t * 0.5 / np.sin(t) * (rmat - rmat.T)))
    return axial(t * 0.5 / np.sin(t) * (rmat - rmat.T))


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


def get_rotation_from_theta_tensor_deprecated(x):
    """
    :param x: theta vector
    :return: rotation tensor
    """
    x = skew(x)
    t = np.sqrt(0.5 * np.trace(x.T @ x))
    if np.isclose(abs(t), 0, atol=1e-6):
        return np.eye(3)
    else:
        return np.eye(3) + np.sin(t) / t * x + (1 - np.cos(t)) / t ** 2 * x @ x


def get_rotation_from_theta_tensor(x):
    """
    using quaternions
    :param x: skew symmetric tensor
    :return: rotation tensor
    """
    x = np.reshape(x, (3,))
    normt = np.linalg.norm(x)
    if np.isclose(normt, 0, atol=1e-8):
        return np.eye(3)
    else:
        q = np.zeros(4)
        q[0] = np.cos(normt / 2)
        q[1:] = np.sin(normt / 2) / normt * x
        return 2 * np.array([
            [q[0] ** 2 + q[1] ** 2 - 0.5, q[1] * q[2] - q[3] * q[0], q[1] * q[3] + q[2] * q[0]],
            [q[2] * q[1] + q[3] * q[0], q[0] ** 2 + q[2] ** 2 - 0.5, q[2] * q[3] - q[1] * q[0]],
            [q[3] * q[1] - q[2] * q[0], q[3] * q[2] + q[1] * q[0], q[0] ** 2 + q[3] ** 2 - 0.5]
        ])


def get_assembly_vector(dof, n):
    """
    :param dof: dof
    :param n: nodes
    :return: assembly points
    """
    iv = []
    for i in n:
        for j in range(dof):
            iv.append(dof * i + j)
    return iv


def get_incremental_k(dt, dtds, rot):
    """
    According to Simo
    :param dt: delta_theta
    :param dtds: delta_theta'
    :param rot: rotation matrix
    :return: delta_kappa
    """
    norm_dt = np.linalg.norm(dt)
    if np.isclose(norm_dt, 0, atol=1e-6):
        return rot.T @ (dtds + 0.5 * np.cross(dt.reshape(3, ), dtds.reshape(3, ))[:, None])
    x = np.sin(norm_dt) / norm_dt
    x2 = np.sin(norm_dt * 0.5) / (norm_dt * 0.5)
    return rot.T @ (x * dtds + (1 - x) * (dt.T @ dtds)[0][0] / norm_dt * dt / norm_dt + 0.5 * (x2 ** 2) * np.cross(
        dt.reshape(3, ), dtds.reshape(3, ))[:, None])


def get_incremental_k_path_independent(t, tds):
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


def get_e(dof, n, n_, rds):
    e0 = np.zeros((dof, dof))
    e0[0: 3, 0: 3] = n_ * np.eye(3)
    e0[3: 6, 3: 6] = n_ * np.eye(3)
    e0[3: 6, 0: 3] = -n * rds
    return e0


def get_tangent_stiffness_residue(n_tensor, m_tensor, n, nx, dof, pi_i, c, rds, gloc, ncforce=None):
    """
    :param gloc: gloc
    :param rds: rds
    :param dof: dof
    :param c: elasticity
    :param pi_i: pi_
    :param n_tensor: axial of n
    :param m_tensor: axial of m
    :param n: shape function
    :param nx: derivative of shape function
    :param ncforce: non-conservative force body force
    :return: geometric stiffness matrix
    """
    nmmat = np.zeros((6, 6))
    nmat = np.zeros((6, 6))

    f = np.zeros((6, 6))
    fn, fnx_ = n, nx
    if ncforce:
        fn, fnx_ = get_lagrange_fn(ncforce[1], len(n))
        f[0: 3, 3: 6] = -skew(ncforce[0])

    nmmat[0: 3, 3: 6] = -n_tensor
    nmmat[3: 6, 3: 6] = -m_tensor
    nmat[3: 6, 0: 3] = n_tensor
    k = np.zeros((dof * len(n), dof * len(n)))
    r = np.zeros((dof * len(n), 1))
    for i in range(len(n)):
        r[6 * i: 6 * (i + 1)] += get_e(dof, n[i][0], nx[i][0], rds) @ gloc
        for j in range(len(n)):
            k[6 * i: (i + 1) * 6, 6 * j: (j + 1) * 6] += (
                    get_e(dof, n[i][0], nx[i][0], rds) @ pi_i @ c @ pi_i.T @ get_e(dof, n[j][0], nx[j][0], rds).T + n[j][0]
                    * get_e(dof, n[i][0], nx[i][0], rds) @ nmmat + n[i][0] * nx[j][0] * nmat + fn[i][0] * fn[j][0] * f)

    return k.T, r


def get_pi(rot):
    """
    :param rot: rotation
    :return: pi matrix
    """
    piI = np.zeros((6, 6))
    piI[0: 3, 0: 3] = rot
    piI[3: 6, 3: 6] = rot
    return piI


"""
STRAIN GRADIENT
------------------------------------------------------------------------------------------------------------------
"""


def get_h(n_, nx_):
    """
    :param n_: hermite fn
    :param nx_: hermite derivative
    :return: consolidated shape function
    """
    c = np.zeros((12, 12))
    i = np.eye(3)
    c[0: 3, 0: 3] = i * n_[0]
    c[0: 3, 3: 6] = i * n_[1]
    c[3: 6, 0: 3] = i * nx_[0]
    c[3: 6, 3: 6] = i * nx_[1]
    c[6: 9, 6: 9] = i * n_[0]
    c[6: 9, 9: 12] = i * n_[1]
    c[9: 12, 6: 9] = i * nx_[0]
    c[9: 12, 9: 12] = i * nx_[1]
    return c


"""
***********************************************************************************************
MATERIAL STIFFNESS
"""


def c_full(es, eb, hes, heb, coupler=None):
    """
    :param es: standard stretch stiffness
    :param eb: standard bending stiffness
    :param hes: higher order stretch stiffness
    :param heb: higher order bending stiffness
    :param coupler: coupling
    :return: c_full (refer to notes)
    """
    c = np.zeros((12, 12))
    c[0: 3, 0: 3] = es
    c[3: 6, 3: 6] = hes
    c[6: 9, 6: 9] = eb
    c[9: 12, 9: 12] = heb
    if not coupler:
        return c
    return c + coupler


def d_u(hes, heb):
    """
    :param hes: higher order stretch stiffness
    :param heb: higher order bending stiffness
    :return: d_u (refer to notes)
    """
    c = np.zeros((12, 12))
    c[0: 3, 0: 3] = hes
    c[6: 9, 6: 9] = heb
    return c


def d_l(hes, heb):
    """
    :param hes: higher order stretch stiffness
    :param heb: higher order bending stiffness
    :return: c_l (refer to notes)
    """
    c = np.zeros((12, 12))
    c[3: 6, 3: 6] = hes
    c[9: 12, 9: 12] = heb
    return c


"""
*******************************************************************
"""

"""
***********************************************************************************
ROTATIONS
"""


def pi(r):
    """
    :param r: rotation tensor
    :return: pi
    """
    c = np.zeros((12, 12))
    c[0: 3, 0: 3] = r
    c[3: 6, 3: 6] = r
    c[6: 9, 6: 9] = r
    c[9: 12, 9: 12] = r
    return c


def pi_l(r):
    """
    :param r: rotation tensor
    :return: pi_l
    """
    c = np.zeros((12, 12))
    c[3: 6, 3: 6] = r
    c[9: 12, 9: 12] = r
    return c


def pi_u(r):
    """
    :param r: rotation tensor
    :return: pi_u
    """
    c = np.zeros((12, 12))
    c[0: 3, 0: 3] = r
    c[6: 9, 6: 9] = r
    return c


def pi_uds(rds):
    """
    :param rds: rotation tensor derivative
    :return: pi_uds
    """
    c = np.zeros((12, 12))
    c[0: 3, 0: 3] = rds
    c[6: 9, 6: 9] = rds
    return c


def pi_lds(rds):
    """
    :param rds: rotation tensor derivative
    :return: pi_lds
    """
    c = np.zeros((12, 12))
    c[3: 6, 3: 6] = rds
    c[9: 12, 9: 12] = rds
    return c


def k_u(k):
    """
    :param k: kappa vector
    :return: k_u
    """
    c = np.zeros((12, 12))
    c[0: 3, 0: 3] = skew(k)
    c[6: 9, 6: 9] = skew(k)
    return c


"""
***************************************************************************
OPERATORS
"""


def e(n_, nx_, nxx_, rds, rdsds):
    """
    :param rds: rds
    :param n_: hermite fn
    :param nx_: hermite derivative
    :param nxx_: hermite double derivative
    :param rdsds: rdsds
    :return: e
    """
    i = np.eye(3)
    c = np.zeros((12, 12))
    rds = skew(rds)
    rdsds = skew(rdsds)
    c[0: 3, 0: 3] = i * nx_[0]
    c[0: 3, 3: 6] = i * nx_[1]
    c[0: 3, 6: 9] = rds * n_[0]
    c[0: 3, 9: 12] = rds * n_[1]
    c[3: 6, 0: 3] = i * nxx_[0]
    c[3: 6, 3: 6] = i * nxx_[1]
    c[3: 6, 6: 9] = rdsds * n_[0] + rds * nx_[0]
    c[3: 6, 9: 12] = rdsds * n_[1] + rds * nx_[1]
    c[6: 9, 6: 9] = i * nx_[0]
    c[6: 9, 9: 12] = i * nx_[1]
    c[9: 12, 6: 9] = i * nxx_[0]
    c[9: 12, 9: 12] = i * nxx_[1]
    return c


def e_l(rds):
    c = np.zeros((12, 12))
    c[3: 6, 3: 6] = np.eye(3)
    c[3: 6, 6: 9] = skew(rds)
    c[9: 12, 9: 12] = np.eye(3)
    return c


def e_u(rds):
    c = np.zeros((12, 12))
    c[0: 3, 3: 6] = np.eye(3)
    c[0: 3, 6: 9] = skew(rds)
    c[6: 9, 9: 12] = np.eye(3)
    return c


def e_g(n_, nx_, nxx_, rds, rdsds):
    """
    :param rds: rds
    :param n_: hermite fn
    :param nx_: hermite derivative
    :param nxx_: hermite double derivative
    :param rdsds: rdsds
    :return: e_g
    """
    i = np.eye(3)
    c = np.zeros((12, 12))
    rds = skew(rds)
    rdsds = skew(rdsds)
    c[0: 3, 0: 3] = i * nxx_[0]
    c[0: 3, 3: 6] = i * nxx_[1]
    c[0: 3, 6: 9] = rdsds * n_[0] + rds * nx_[0]
    c[0: 3, 9: 12] = rdsds * n_[1] + rds * nx_[1]
    c[6: 9, 6: 9] = i * nxx_[0]
    c[6: 9, 9: 12] = i * nxx_[1]
    return c


def e_f(nx_, nxx_):
    """
    :param nxx_: hermite double derivative
    :param nx_: hermite derivative
    :return: e_f
    """
    c = np.zeros((12, 12))
    i = np.eye(3)
    c[6: 9, 0: 3] = (nx_[0] + nxx_[0]) * i
    c[6: 9, 3: 6] = (nx_[1] + nxx_[1]) * i
    c[9: 12, 0: 3] = nx_[0] * i
    c[9: 12, 3: 6] = nx_[1] * i
    return c


"""
*******************************************************************
"""

"""
******************************************************************************************
FORCES/STRESSES
"""


def matn(n, nb, nx_, nxx_):
    c = np.zeros((12, 12))
    c[6: 9, 0: 3] = skew(n) * nx_[0] + skew(nb) * nxx_[0]
    c[6: 9, 3: 6] = skew(n) * nx_[1] + skew(nb) * nxx_[1]
    c[9: 12, 0: 3] = skew(nb) * nx_[0]
    c[9: 12, 3: 6] = skew(nb) * nx_[1]
    return c


def matnm(n, nb, m, mb):
    c = np.zeros((12, 12))
    n = skew(n)
    nb = skew(nb)
    m = skew(m)
    mb = skew(mb)
    c[0: 3, 6: 9] = -n
    c[3: 6, 6: 9] = -nb
    c[0: 3, 9: 12] = -nb
    c[6: 9, 6: 9] = -m
    c[9: 12, 6: 9] = -mb
    c[6: 9, 9: 12] = -mb
    return c


"""
*********************************************************************
"""


def get_higher_order_tangent_residue(n_, nx_, nxx_, rds, rdsds, rmat, rmatds, cs, cb, ds, db, kvec, dof, gloc, element_type=2, coupler=None):
    nmat, nbmat, mmat, mbmat = gloc[0: 3], gloc[3: 6], gloc[6: 9], gloc[9: 12]
    f = dof * element_type
    k = np.zeros((f, f))
    r = np.zeros((f, 1))
    for i in range(element_type):
        hi, hi_, hi__ = n_[2 * i: 2 * (i + 1), 0], nx_[2 * i: 2 * (i + 1), 0], nxx_[2 * i: 2 * (i + 1), 0]
        Ei = e(hi, hi_, hi__, rds, rdsds).T
        Hmati = get_h(hi, hi_)
        r[dof * i: dof * (i + 1)] += Ei @ gloc
        for j in range(element_type):
            hj, hj_, hj__ = n_[2 * j: 2 * (j + 1), 0], nx_[2 * j: 2 * (j + 1), 0], nxx_[2 * j: 2 * (j + 1), 0]
            Hmatj = get_h(hj, hj_)
            Ej = e(hj, hj_, hj__, rds, rdsds).T
            E_lj = e_l(rds) @ Hmatj
            E_uj = e_u(rds) @ Hmatj
            E_gj = e_g(hj, hj_, hj__, rds, rdsds)
            E_fj = e_f(hj_, hj__)
            A1 = Ei @ matnm(nmat, nbmat, mmat, mbmat) @ Hmatj
            A2 = Ei @ pi(rmat) @ c_full(cs, cb, ds, db, coupler) @ pi(rmat).T @ Ej.T
            A3 = Ei @ pi_l(rmat) @ d_l(ds, db) @ pi_lds(rmatds).T @ E_lj
            A4 = Ei @ pi_u(rmat) @ k_u(kvec) @ d_u(ds, db) @ pi_uds(rmatds).T @ E_uj
            A5 = Ei @ pi_u(rmat) @ k_u(kvec) @ d_u(ds, db) @ pi_u(rmat).T @ E_gj
            A6 = Hmati.T @ matn(nmat, nbmat, hj_, hj__)
            k[dof * i: dof * (i + 1), dof * j: dof * (j + 1)] += (A1 + A2 + A3 + A4 + A5 + A6)
    return k, r


def beizer_curve():
    pass


if __name__ == "__main__":
    icon_m, i_m = get_connectivity_matrix(10, 1)
    # print(icon_m)
    # print(i_m)
    # b = get_incremental_k(np.array([1, 1, 1]), np.array([1, 1, 2]), np.eye(3))
    # print(b)
