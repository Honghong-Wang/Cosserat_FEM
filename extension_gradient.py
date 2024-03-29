"""
ONLY LINEAR ELEMENTS
classical_rod.py works using quaternions for rotation interpolations,
go to etc. for other methods to interpolate rotations
"""
import numpy as np
from gradientsolver import extension_solver as sol
from include import slerp as slerpsol, quaternion_smith as quat_sol
import matplotlib.pyplot as plt
from include.AnimationController import ControlledAnimation
import pandas as pd
try:
    import scienceplots

    plt.style.use(['science'])
except ImportError as e:
    pass

np.set_printoptions(linewidth=250)

"""
Main loop
"""


def fea(load_iter_, is_halt=False):
    """
    :param load_iter_: Load index
    :param is_halt: signals animator if user requested a pause
    :return: use input , True if user want to stop animation
    """
    global u
    global du
    global residue_norm
    global increments_norm
    global is_log_residue
    for iter_ in range(MAX_ITER):
        KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
        # Follower load
        # s = sol.get_rotation_from_theta_tensor(u[-6: -3]) @ np.array([0, fapp__[load_iter_], 0])[:, None]
        # FG[-12: -9] = s
        # Pure Bending
        FG[-4, 0] = fapp__[load_iter_]
        for elm in range(numberOfElements):
            n = icon[elm][1:]
            xloc = node_data[n][:, None]
            rloc = np.zeros((3, 4))
            rloc[:, [0, 2]] = np.array([u[DOF * n, 0], u[DOF * n + 1, 0], u[DOF * n + 2, 0]])
            rloc[:, [1, 3]] = np.array([u[DOF * n + 3, 0], u[DOF * n + 4, 0], u[DOF * n + 5, 0]])
            kloc, floc = sol.init_stiffness_force(nodesPerElement, DOF)
            gloc = np.zeros((DOF, 1))
            for xgp in range(len(wgp)):
                # N_, Bmat = sol.get_lagrange_fn(gp[xgp], element_type)
                le = xloc[-1][0] - xloc[0][0]
                Jac = le / 2
                N_, Nx_, Nxx_ = sol.get_hermite_fn(gp[xgp], Jac, element_type)
                N_, Nx_, Nxx_ = N_[:, None], Nx_[:, None], Nxx_[:, None]
                # Nx_ = 1 / Jac * Bmat
                rds = rloc @ Nx_
                rdsds = rloc @ Nxx_
                v = rds
                vp = rdsds
                gloc[0: 3] = ElasticityExtension @ (v - np.array([0, 0, 1])[:, None])
                gloc[3: 6] = ElasticityExtensionH @ vp
                tangent, res = sol.get_tangent_stiffness_residue_ext(gloc, N_, Nx_, Nxx_, ElasticityExtension, ElasticityExtensionH, DOF, element_type)
                floc += res * wgp[xgp] * Jac
                kloc += tangent * wgp[xgp] * Jac

            iv = np.array(sol.get_assembly_vector(DOF, n))

            FG[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc
        # TODO: Make a generalized function for application of point as well as body loads
        # f = np.zeros((6, 6))
        # f[0: 3, 3: 6] = -sol.skew(s)
        # KG[-6:, -6:] += f
        for ibc in range(6):
            sol.impose_boundary_condition(KG, FG, ibc, 0)
        sol.impose_boundary_condition(KG, FG, -1, 0)
        du = -sol.get_displacement_vector(KG, FG)
        residue_norm = np.linalg.norm(FG)

        increments_norm = np.linalg.norm(du)
        if increments_norm > 1:
            du = du / increments_norm
        if increments_norm < 1e-6 and residue_norm < 1e-3:
            break
        """
        Configuration update (not working as of now) for angles greater than 360 deg, Make this work for multi-axis rotations
        """
        # for i in range(numberOfNodes):
        #     q = slerpsol.rotation_vector_to_quaterion(u[6 * i + 3: 6 * i + 6, 0])
        #     dq = slerpsol.rotation_vector_to_quaterion(du[6 * i + 3: 6 * i + 6, 0])
        #     q = slerpsol.quatmul(q, dq)
        #     u[6 * i + 3: 6 * i + 6, 0] = slerpsol.quaterion_to_rotation_vec(q)
        #     u[6 * i + 0: 6 * i + 3] += du[6 * i + 0: 6 * i + 3]
        """
        Approx. configuration update
        """
        # TODO: Change this, it works perfectly if two rotations are about one axis (R_(i+1) = exp(dtheta_i) * exp(theta_i))
        u += du

    if is_log_residue:
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------",
            fapp__[load_iter_], load_iter_)
        print(residue_norm, increments_norm)
    return is_halt


if __name__ == "__main__":
    """
    Set Finite Element Parameters
    """
    DIMENSIONS = 1
    DOF = 6

    MAX_ITER = 100  # Max newton raphson iteration
    element_type = 2
    L = 1
    numberOfElements = 50

    icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
    numberOfNodes = len(node_data)
    ngpt = 3
    wgp, gp = sol.init_gauss_points(ngpt)

    # Setting up displacement vectors
    u = np.zeros((numberOfNodes * DOF, 1))
    du = np.zeros((numberOfNodes * DOF, 1))
    nodesPerElement = element_type ** DIMENSIONS

    """
    SET MATERIAL PROPERTIES
    -----------------------------------------------------------------------------------------------------------------------
    """
    l0 = 0.5
    E0 = 10 ** 8
    G0 = E0 / 2.0
    d = 1 / 1000 * 25.0
    A = np.pi * d ** 2 * 0.25
    i0 = np.pi * d ** 4 / 64
    J = i0 * 2
    EI = 3.5 * 10 ** 7
    GA = 1.6 * 10 ** 8
    ElasticityExtension = np.array([[G0 * A, 0, 0],
                                    [0, G0 * A, 0],
                                    [0, 0, E0 * A]])
    ElasticityBending = np.array([[E0 * i0 + l0 ** 2 * E0 * A, 0, 0],
                                  [0, E0 * i0 + l0 ** 2 * E0 * A, 0],
                                  [0, 0, G0 * J + 2 * l0 ** 2 * G0 * A]])

    ElasticityExtensionH = l0 ** 2 * np.array([[G0 * A, 0, 0],
                                               [0, G0 * A, 0],
                                               [0, 0, E0 * A]])
    ElasticityBendingH = np.array([[E0 * i0 * l0 ** 2, 0, 0],
                                   [0, E0 * i0 * l0 ** 2, 0],
                                   [0, 0, G0 * J * l0 ** 2]])

    # ElasticityExtension = np.array([[GA, 0, 0],
    #                                 [0, GA, 0],
    #                                 [0, 0, 2 * GA]])
    # ElasticityBending = np.array([[EI, 0, 0],
    #                               [0, EI, 0],
    #                               [0, 0, 0.5 * EI]])

    """
    Markers
    """
    vi = np.array([i for i in range(numberOfNodes)])
    vii = np.array([i for i in range(numberOfNodes) if i & 1 == 0])

    """
    Starting point
    """
    residue_norm = 0
    increments_norm = 0
    # since rod is lying straight in E3 direction it's centerline will have these coordinates

    # Thetas are zero

    r1 = np.zeros(numberOfNodes)
    r2 = np.zeros(numberOfNodes)
    r3 = np.zeros(numberOfNodes)
    for i in range(numberOfNodes):
        r1[i] = u[DOF * i][0]
        r2[i] = u[DOF * i + 1][0]
        r3[i] = u[DOF * i + 2][0]

    """
    Initialize Graph
    """
    fig, (ax, ay) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[1, 2])
    ax.set_xlim(0, L)
    ax.plot(r3, r2, label="un-deformed", marker="o")

    """
    Set load and load steps
    """
    max_load = 2 * np.pi * E0 * i0 / L
    # max_load = 30 * E0 * i0
    LOAD_INCREMENTS = 2  # Follower load usually needs more steps compared to dead or pure bending
    fapp__ = -np.linspace(0, max_load, LOAD_INCREMENTS)

    u = np.zeros((numberOfNodes * DOF, 1))
    u[DOF * vi + 2, 0] = node_data
    u[DOF * vi + 5, 0] = 1

    marker_ = np.linspace(0, max_load, LOAD_INCREMENTS)
    # marker_ = np.insert(marker_, 0, [2000, 6000, 12000], axis=0)
    """
    ------------------------------------------------------------------------------------------------------------------------------------
    Post Processing
    ------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    Graph limits defaults
    """
    xmax = 1e-7
    ymax = 1e-7
    xmin = 0
    ymin = 0

    video_request = False
    is_log_residue = True  # Prints residue to console after every load iteration if set true


    def act(i):
        global u
        global xmax
        global ymax
        global xmin
        global ymin
        global video_request
        halt = fea(i)
        if halt:
            controlled_animation.stop()
            return
        y0 = u[DOF * vi + 1, 0]
        x0 = u[DOF * vi + 2, 0]
        if np.isclose(abs(fapp__[i]), marker_).any():
            xmax, ymax = max(xmax, np.max(x0)), max(np.max(y0), ymax)
            xmin, ymin = min(xmin, np.min(x0)), min(np.min(y0), ymin)
            ax.axis('equal')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            line1.set_ydata(y0)
            line1.set_xdata(x0)
            ax.set_title("Centerline displacement, Applied Load : " + str(round(fapp__[i], 5)))
            if not video_request:
                ax.plot(x0, y0)

        ay.scatter(abs(fapp__[i]), -L + u[-4, 0], marker=".")
        ay.scatter(abs(fapp__[i]), u[-3, 0], marker="+")
        if i == LOAD_INCREMENTS - 1:
            controlled_animation.disconnect()


    ay.scatter(0, 0, marker=".", label="horizontal tip displacement")
    ay.scatter(0, 0, marker="+", label="vertical tip displacement")
    ay.legend()
    ay.axhline(y=0)
    ay.set_xlabel(r"LOAD", fontsize=16)
    ay.set_ylabel(r"Tip Displacement", fontsize=16)
    ax.set_xlabel(r"$r_3$", fontsize=25)
    ax.set_ylabel(r"$r_2$", fontsize=25)
    ax.set_ylim(-85, 41)
    y = u[DOF * vi + 1, 0]
    x = u[DOF * vi + 2, 0]
    line1, = ax.plot(x, y)
    ax.set_title("Centerline displacement")
    ay.set_title("Tip Displacement vs Load")

    controlled_animation = ControlledAnimation(fig, act, frames=len(fapp__), video_request=video_request, repeat=False)
    controlled_animation.start()
    print(max_load * L / (ElasticityExtension[2, 2]))
    print(u[-6:, 0])
    l0 = l0 / L
    fig2, (a0, a1) = plt.subplots(1, 2, figsize=(12, 6))
    print(node_data)
    node_data = node_data / L
    a1.plot(node_data, u[DOF * vi + 5, 0] - 1, label="FEM")
    a1.plot(node_data, max_load / (ElasticityExtension[2, 2]) * (1. - (1. / (np.cosh(1. / 2. / l0)) * (np.cosh((1. - 2 * node_data) / 2. / l0)))), label="ANALYTICAL")
    a0.plot(node_data, u[DOF * vi + 2, 0] - node_data, label="FEM")
    a0.plot(node_data, max_load / (ElasticityExtension[2, 2]) * (node_data - l0 * np.tanh(1 / 2 / l0) + l0 * (1. / (np.cosh(1. / 2. / l0)) * (np.sinh((1. - 2 * node_data) / 2. / l0)))), label="ANALYTICAL")
    a1.legend()
    a0.legend()
    a0.set_title("DISPLACEMENT")
    a1.set_title("STRAIN")

    # df1 = pd.DataFrame([node_data])
    # df1.loc[len(df1)] = u[DOF * vi + 5, 0] - 1
    # df2 = pd.DataFrame([node_data])
    # df2.loc[len(df2)] = u[DOF * vi + 2, 0] - node_data
    # df1.to_csv('GFG1.csv', index=False, header=False)
    # df2.to_csv('GFG2.csv', index=False, header=False)
    plt.show()
