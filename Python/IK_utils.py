import numpy as np
import time
from RDKDC_Final_Project.Python.control import urFwdKin

#==============================================================================
# Helper functions for DH parameter calculations
#==============================================================================

def ROTZ(theta):
    """
    Creates a 3x3 rotation matrix for a rotation around the Z-axis.
    
    Args:
        theta (float): The rotation angle in radians.
        
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def ROTX(alpha):
    """
    Creates a 3x3 rotation matrix for a rotation around the X-axis.
    
    Args:
        alpha (float): The rotation angle in radians.
        
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    return np.array([
        [1, 0,               0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])

#==============================================================================
# Denavit-Hartenberg (DH) Transformation
#==============================================================================

def DH(a, alpha, d, theta):
    """
    Calculates the transformation matrix from Denavit-Hartenberg (DH) parameters.
    This function follows the traditional DH parameter definition.

    Args:
        a (float): Link length.
        alpha (float): Link twist (in radians).
        d (float): Link offset.
        theta (float): Joint angle (in radians).

    Returns:
        numpy.ndarray: The 4x4 homogeneous transformation matrix.
    """
    Td = np.eye(4)
    Td[2, 3] = d
    Rtheta = np.eye(4)
    Rtheta[0:3, 0:3] = ROTZ(theta)
    Ta = np.eye(4)
    Ta[0, 3] = a
    Ralpha = np.eye(4)
    Ralpha[0:3, 0:3] = ROTX(alpha)
    G = Td @ Rtheta @ Ta @ Ralpha
    return G

#==============================================================================
# UR5/UR5e Inverse Kinematics
#==============================================================================

def urInvKin(gd, robot_type='ur5'):
    """
    Analytical Inverse kinematics for a UR5 or UR5e robot.
    """
    if gd.shape != (4, 4):
        raise ValueError("urInvKin function: Input 'gd' must be a 4x4 matrix.")

    theta = np.full((6, 8), np.nan) # Initialize with NaNs

    # DH Parameters for UR5
    d1, d2, d3, d4, d5, d6 = 0.089159, 0, 0, 0.10915, 0.09465, 0.0823
    a1, a2, a3, a4, a5, a6 = 0, -0.425, -0.39225, 0, 0, 0
    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0

    if robot_type == 'ur5e':
        d1, d4, d5, d6 = 0.1625, 0.1333, 0.0997, 0.0996
        a2, a3 = -0.425, -0.3922

    # --- Calculations ---
    p05 = gd @ np.array([[0], [0], [-d6], [1]]) - np.array([[0], [0], [0], [1]])
    p05_norm = np.linalg.norm(p05[0:2])
    
    if p05_norm < 1e-6:
        print("Singularity warning: Wrist is at the base origin.")
    
    # Solve for theta1
    phi_arg = d4 / p05_norm if p05_norm > 1e-6 else 0
    phi = np.arccos(np.clip(phi_arg, -1.0, 1.0))
    psi = np.arctan2(p05[1, 0], p05[0, 0])

    theta[0, 0:4] = np.pi/2 + psi + phi
    theta[0, 4:8] = np.pi/2 + psi - phi

    # Solve for theta5
    for c in [0, 4]:
        T01 = DH(a1, alpha1, d1, theta[0, c])
        T16 = np.linalg.inv(T01) @ gd
        p16z = T16[2, 3]
        
        t5_arg = (p16z - d4) / d6
        t5 = np.arccos(np.clip(t5_arg, -1.0, 1.0))
        theta[4, c:c+2] = t5
        theta[4, c+2:c+4] = -t5

    # Solve for theta6
    for c in [0, 2, 4, 6]:
        if np.isnan(theta[0,c]): continue
        T01 = DH(a1, alpha1, d1, theta[0, c])
        T61 = np.linalg.inv(gd) @ T01
        t5 = theta[4, c]
        sin_t5 = np.sin(t5)
        if abs(sin_t5) < 1e-6:
            theta[5, c:c+2] = 0 # At singularity, choose one solution
        else:
            theta[5, c:c+2] = np.arctan2(-T61[1, 2] / sin_t5, T61[0, 2] / sin_t5)

    # Solve for theta3
    for c in [0, 2, 4, 6]:
        if np.isnan(theta[0,c]): continue
        T01 = DH(a1, alpha1, d1, theta[0, c])
        T56 = DH(a6, alpha6, d6, theta[5, c])
        T45 = DH(a5, alpha5, d5, theta[4, c])
        
        T14 = np.linalg.inv(T01) @ gd @ np.linalg.inv(T45 @ T56)
        p13 = T14 @ np.array([[0], [-d4], [0], [1]]) - np.array([[0], [0], [0], [1]])
        p13norm2 = np.linalg.norm(p13)**2
        
        t3_arg = (p13norm2 - a2**2 - a3**2) / (2 * a2 * a3)
        t3 = np.arccos(np.clip(t3_arg, -1.0, 1.0))
        theta[2, c] = t3
        theta[2, c+1] = -t3

    # Solve for theta2 and theta4
    for c in range(8):
        if np.isnan(theta[0, c]): continue
        T01 = DH(a1, alpha1, d1, theta[0, c])
        T56 = DH(a6, alpha6, d6, theta[5, c])
        T45 = DH(a5, alpha5, d5, theta[4, c])
        
        T14 = np.linalg.inv(T01) @ gd @ np.linalg.inv(T45 @ T56)
        p13 = T14 @ np.array([[0], [-d4], [0], [1]]) - np.array([[0], [0], [0], [1]])
        p13norm = np.linalg.norm(p13)

        asin_arg = a3 * np.sin(theta[2, c]) / p13norm if p13norm > 1e-6 else 0
        theta[1, c] = -np.arctan2(p13[1,0], -p13[0,0]) + np.arcsin(np.clip(asin_arg, -1.0, 1.0))
        
        T23 = DH(a3, alpha3, d3, theta[2, c])
        T12 = DH(a2, alpha2, d2, theta[1, c])
        T34 = np.linalg.inv(T12 @ T23) @ T14
        theta[3, c] = np.arctan2(T34[1, 0], T34[0, 0])

    # Final adjustments and filtering
    theta = np.real(theta)
    theta[0,:] = theta[0,:] - np.pi
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    
    valid_cols = ~np.isnan(theta).any(axis=0)
    return theta[:, valid_cols]

#==============================================================================
# --- Inverse Kinematics Test Script ---
#==============================================================================
if __name__ == '__main__':
    
    errors_5 = np.zeros(6)
    errors_5e = np.zeros(6)

    # Select the number of tests
    n_tests = 1000
    print(f"Running {n_tests} tests...")
    
    start_time = time.time()

    for i in range(n_tests):
        # Generate a random set of joint angles between -pi and pi
        q = (np.random.rand(6) * 2 * np.pi) - np.pi
        
        # --- Test UR5 ---
        gBaseTool = urFwdKin(q, 'ur5')
        q_sol = urInvKin(gBaseTool, 'ur5')
        
        if q_sol.size > 0:
            # Find the closest matching kinematic configuration
            error_norms = np.linalg.norm(q[:, np.newaxis] - q_sol, axis=0)
            min_error_i = np.argmin(error_norms)
            
            # Add the absolute error of the best solution to the total
            errors_5 += np.abs(q_sol[:, min_error_i] - q)

        # --- Test UR5e ---
        gBaseTool_e = urFwdKin(q, 'ur5e')
        q_sol_e = urInvKin(gBaseTool_e, 'ur5e')
        
        if q_sol_e.size > 0:
            # Find the closest matching kinematic configuration for UR5e
            error_norms_e = np.linalg.norm(q[:, np.newaxis] - q_sol_e, axis=0)
            min_error_i_e = np.argmin(error_norms_e)
            
            # Add the absolute error of the best solution to the total
            errors_5e += np.abs(q_sol_e[:, min_error_i_e] - q)

    end_time = time.time()
    print(f"Testing completed in {end_time - start_time:.2f} seconds.")

    # Calculate and print the average errors
    avg_errors_5 = errors_5 / n_tests
    avg_errors_5e = errors_5e / n_tests

    # The f-string formatting provides a clean way to print the results
    print("\n--- Average Joint Errors (radians) ---")
    print(f"UR5 : [{', '.join([f'{err:.3f}' for err in avg_errors_5])}]")
    print(f"UR5e: [{', '.join([f'{err:.3f}' for err in avg_errors_5e])}]")
