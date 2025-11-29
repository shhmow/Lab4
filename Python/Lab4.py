import matplotlib.pyplot as plt
import numpy as np
import math

'''
*** BASIC HELPER FUNCTIONS ***
'''

def ECE569_NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero

    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise

    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6

def ECE569_Normalize(V):
    """ECE569_Normalizes a vector

    :param V: A vector
    :return: A unit vector pointing in the same direction as z

    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / np.linalg.norm(V)

'''
*** CHAPTER 3: RIGID-BODY MOTIONS ***
'''

def ECE569_RotInv(R):
    """Inverts a rotation matrix

    :param R: A rotation matrix
    :return: The inverse of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    """
    return np.array(R).T

def ECE569_VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def ECE569_so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def ECE569_AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (ECE569_Normalize(expc3), np.linalg.norm(expc3))

def ECE569_MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = ECE569_so3ToVec(so3mat)
    if ECE569_NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = ECE569_AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def ECE569_MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not ECE569_NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not ECE569_NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return ECE569_VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def ECE569_RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def ECE569_TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def ECE569_TransInv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.

    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = ECE569_TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def ECE569_VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3

    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[ECE569_VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 [[0, 0, 0, 0]]]

def ECE569_se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.array([se3mat[2][1], se3mat[0][2], se3mat[1][0],
                     se3mat[0][3], se3mat[1][3], se3mat[2][3]])

def ECE569_Adjoint(T):
    """Computes the ECE569_adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 ECE569_adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = ECE569_TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(ECE569_VecToso3(p), R), R]]

def ECE569_MatrixExp6(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates

    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat

    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """
    se3mat = np.array(se3mat)
    omgtheta = ECE569_so3ToVec(se3mat[0: 3, 0: 3])
    if ECE569_NearZero(np.linalg.norm(omgtheta)):
        # Pure translation case
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = ECE569_AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        # G = I*theta + (1 - cos(theta))*[omega] + (theta - sin(theta))*[omega]^2
        G = np.eye(3) * theta + (1 - np.cos(theta)) * omgmat \
            + (theta - np.sin(theta)) * np.dot(omgmat, omgmat)
        return np.r_[np.c_[ECE569_MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(G, se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def ECE569_MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix

    :param R: A matrix in SE3
    :return: The matrix logarithm of R

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """
    R, p = ECE569_TransToRp(T)
    omgmat = ECE569_MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        # Pure translation case
        return np.r_[np.c_[np.zeros((3, 3)), p], [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        omgmat_normalized = omgmat / theta
        # G^-1 = I/theta - [omega_hat]/2 + (1/theta - cot(theta/2)/2)*[omega_hat]^2
        Ginv = np.eye(3) / theta - omgmat_normalized / 2.0 \
               + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2.0) \
               * np.dot(omgmat_normalized, omgmat_normalized)
        return np.r_[np.c_[omgmat, np.dot(Ginv, p) * theta], [[0, 0, 0, 0]]]


'''
*** CHAPTER 4: FORWARD KINEMATICS ***
'''

def ECE569_FKinBody(M, Blist, thetalist):
    """Computes forward kinematics in the body frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Body Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(Blist[:, i] * thetalist[i])))
    return T

def ECE569_FKinSpace(M, Slist, thetalist):
    """Computes forward kinematics in the space frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Space Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        T = np.dot(ECE569_MatrixExp6(ECE569_VecTose3(Slist[:, i] * thetalist[i])), T)
    return T

'''
*** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***
'''

def ECE569_JacobianBody(Blist, thetalist):
    """Computes the body Jacobian for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The body Jacobian corresponding to the inputs (6xn real
             numbers)

    Example Input:
        Blist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[-0.04528405, 0.99500417,           0,   1]
                  [ 0.74359313, 0.09304865,  0.36235775,   0]
                  [-0.66709716, 0.03617541, -0.93203909,   0]
                  [ 2.32586047,    1.66809,  0.56410831, 0.2]
                  [-1.44321167, 2.94561275,  1.43306521, 0.3]
                  [-2.06639565, 1.82881722, -1.58868628, 0.4]])
    """
    Jb = np.array(Blist).copy().astype(float)
    T = np.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        # T_i = e^{-B_{i+1} θ_{i+1}} * ... * e^{-B_n θ_n}
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(-Blist[:, i + 1] * thetalist[i + 1])))
        Jb[:, i] = np.dot(ECE569_Adjoint(T), Blist[:, i])
    return Jb

'''
*** CHAPTER 6: INVERSE KINEMATICS ***
'''

def ECE569_IKinBody(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    # TODO: calculate Vb
    # Hint: use four of the ECE569 functions from earlier
    Vb = ECE569_se3ToVec(ECE569_MatrixLog6(np.dot(ECE569_TransInv(ECE569_FKinBody(M, Blist, thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        # TODO: update thetalist
        # Hint: pseudinverse is given by np.linalg.pinv
        thetalist = thetalist + np.dot(np.linalg.pinv(ECE569_JacobianBody(Blist, thetalist)), Vb)
        i += 1
        Vb = ECE569_se3ToVec(ECE569_MatrixLog6(np.dot(ECE569_TransInv(ECE569_FKinBody(M, Blist, thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)

# the ECE569_normalized trapezoid function
def g(t, T, ta):
    if t < 0 or t > T:
        return 0
    
    if t < ta:
        return (T/(T-ta))* t/ta
    elif t > T - ta:
        return (T/(T-ta))*(T - t)/ta
    else:
        return (T/(T-ta))
    
def trapezoid(t, T, ta):
    return g(t, T, ta)

def main():

    ### Step 1: Trajectory Generation

    T = 2*np.pi
    A = 0.15
    B = 0.15
    a = 1
    b = 2

    t_arc = np.linspace(0, T, 1000)
    xd = A*np.sin(a*t_arc)
    yd = B*np.sin(b*t_arc)

    # calculate the arc length
    d = 0
    for i in range(1, len(t_arc)):
        d += np.sqrt((xd[i] - xd[i-1])**2 + (yd[i] - yd[i-1])**2)

    min_tfinal = d / 0.25
    tfinal = max(min_tfinal + 1, 8.0)
    if tfinal > 15:
        tfinal = 15
    # calculate average velocity
    c = d/tfinal

    ta = 0.15 * tfinal

    # forward euler to calculate alpha
    dt = 0.002
    t = np.arange(0, tfinal, dt)
    alpha = np.zeros(len(t))
    for i in range(1, len(t)):
        xdot = A * a * np.cos(a * alpha[i-1])
        ydot = B * b * np.cos(b * alpha[i-1])
        speed = np.sqrt(xdot**2 + ydot**2)
        g_val = g(t[i-1], tfinal, ta)
        if speed > 1e-10:
            alpha_dot = (c * g_val) / speed
        else:
            alpha_dot = 0
        alpha[i] = alpha[i-1] + alpha_dot * dt

    # plot alpha vs t
    plt.plot(t, alpha,'b-',label='alpha')
    plt.plot(t, np.ones(len(t))*T, 'k--',label='T (period)')
    plt.xlabel('t')
    plt.ylabel('alpha')
    plt.title('alpha vs t')
    plt.legend()
    plt.grid()
    plt.show()

    # rescale our trajectory with alpha
    x = A*np.sin(a*alpha)
    y = B*np.sin(b*alpha)

    # calculate velocity
    xdot = np.diff(x)/dt
    ydot = np.diff(y)/dt
    v = np.sqrt(xdot**2 + ydot**2)

    # plot velocity vs t
    plt.plot(t[1:], v, 'b-',label='velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*c, 'k--',label='average velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*0.25, 'r--',label='velocity limit')
    plt.xlabel('t')
    plt.ylabel('velocity')
    plt.title('velocity vs t')
    plt.legend()
    plt.grid()
    plt.show()

    ### Step 2: Forward Kinematics
    L1 = 0.2435
    L2 = 0.2132
    W1 = 0.1311
    W2 = 0.0921
    H1 = 0.1519
    H2 = 0.0854

    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]])

    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -H1, 0, 0])
    S3 = np.array([0, 1, 0, -H1, 0, L1])
    S4 = np.array([0, 1, 0, -H1, 0, L1 + L2])
    S5 = np.array([0, 0, -1, -W1, L1+L2, 0])
    S6 = np.array([0, 1, 0, H2-H1, 0, L1+L2])
    S = np.array([S1, S2, S3, S4, S5, S6]).T

    B1 = np.linalg.inv(ECE569_Adjoint(M))@S1
    B2 = np.linalg.inv(ECE569_Adjoint(M))@S2
    B3 = np.linalg.inv(ECE569_Adjoint(M))@S3
    B4 = np.linalg.inv(ECE569_Adjoint(M))@S4
    B5 = np.linalg.inv(ECE569_Adjoint(M))@S5
    B6 = np.linalg.inv(ECE569_Adjoint(M))@S6
    B = np.array([B1, B2, B3, B4, B5, B6]).T

    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])

    # perform forward kinematics using ECE569_FKinSpace and ECE569_FKinBody
    T0_space = ECE569_FKinSpace(M, S, theta0)
    print(f'T0_space: {T0_space}')
    T0_body = ECE569_FKinBody(M, B, theta0)
    print(f'T0_body: {T0_body}')
    T0_diff = T0_space - T0_body
    print(f'T0_diff: {T0_diff}')
    T0 = T0_body

    # calculate Tsd for each time step
    Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        Td = np.eye(4)
        Td[0, 3] = x[i]
        Td[1, 3] = y[i]
        Td[2, 3] = 0
        Tsd[:, :, i] = T0 @ Td

    # plot p(t) vs t in the {s} frame
    xs = Tsd[0, 3, :]
    ys = Tsd[1, 3, :]
    zs = Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_title('Trajectory in {s} Frame')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    plt.show()

    ### Step 3: Inverse Kinematics

    # when i=0
    thetaAll = np.zeros((6, len(t)))

    initialguess = theta0
    eomg = 1e-6
    ev = 1e-6

    thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,0], initialguess, eomg, ev)
    if not success:
        raise Exception(f'Failed to find a solution at index {0}')
    thetaAll[:, 0] = thetaSol

    # when i=1...,N-1
    for i in range(1, len(t)):
        initialguess = thetaAll[:, i-1]
        thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,i], initialguess, eomg, ev)
        if not success:
            raise Exception(f'Failed to find a solution at index {i}')
        thetaAll[:, i] = thetaSol

    # verify that the joint angles don't change much
    dj = np.diff(thetaAll, axis=1)
    plt.plot(t[1:], dj[0], 'b-',label='joint 1')
    plt.plot(t[1:], dj[1], 'g-',label='joint 2')
    plt.plot(t[1:], dj[2], 'r-',label='joint 3')
    plt.plot(t[1:], dj[3], 'c-',label='joint 4')
    plt.plot(t[1:], dj[4], 'm-',label='joint 5')
    plt.plot(t[1:], dj[5], 'y-',label='joint 6')
    plt.xlabel('t (seconds)')
    plt.ylabel('first order difference')
    plt.title('First Order Difference in Joint Angles')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # verify that the joint angles will trace out our trajectory
    actual_Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        actual_Tsd[:,:,i] = ECE569_FKinBody(M, B, thetaAll[:, i])

    xs = actual_Tsd[0, 3, :]
    ys = actual_Tsd[1, 3, :]
    zs = actual_Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Verified Trajectory in {s} Frame')
    ax.legend()
    plt.show()

    # (3e) verify the robot does not enter kinematic singularity
    # by plotting the determinant of the body jacobian
    body_dets = np.zeros(len(t))
    for i in range(len(t)):
        body_dets[i] = np.linalg.det(ECE569_JacobianBody(B, thetaAll[:, i]))
    plt.plot(t, body_dets, '-')
    plt.xlabel('t (seconds)')
    plt.ylabel('det of J_B')
    plt.title('Manipulability')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # save to csv file (you can modify the led column to control the led)
    # led = 1 means the led is on, led = 0 means the led is off
    led = np.ones_like(t)
    data = np.column_stack((t, thetaAll.T, led))
    np.savetxt('jlove.csv', data, delimiter=',')


if __name__ == "__main__":
    main()