import numpy as np


from IK_utils import IK_utils
from rr_control import check_joint_limits, check_table_clearance
from control import urFwdKin

theta_6x8 = IK_utils.urInvKin(gd, robot_type)


def select_best_solutions(theta_6x8, q_prev):
    """
    Selects the best IK solution.

    Args:
        theta_6x8 (numpy.ndarray): A 6x8 array where each column represents a candidate IK solution.
        q_prev (numpy.ndarray): A 6-element array representing the previous joint configuration.

    Returns:
        numpy.ndarray: The selected IK solution as a 6-element array.
    """
    


    for j in range(theta_6x8.shape[1]):
        q = np.array(theta_6x8[:, j])

        try:
            check_joint_limits(q)
        except Exception:
            continue

        try:
            g = urFwdKin(q, robot_type)
            check_table_clearance(g)
        except Exception:
            continue

        dp = np.abs(q - q_prev)
        W = np.diag([1, 1, 1, 2, 2, 3])
        cost = dp.T @ W @ dp

        