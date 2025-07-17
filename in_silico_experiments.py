# This script contains the in silico experiments for the paper "A Projection-Based Inverse Kinematic Model for
# Extensible Continuum Robots and Hyper-Redundant Robots with an Elbow Joint"

import numpy as np
from visualization_functions import *
import time
import time
from scipy.spatial.transform import Rotation as R
from solve_set_equations import *
import cvxpy as cp


def inverse_differential_kinematics_LM_3D(X_d, k, q, error_threshold): #uses adaptive damping
    '''
    Return q, convergence_time, fk_list, error_list.
    In case of no convergence, return q=[0,0,0,0]
    '''
    def levenberg_marquardt_update(J, error, damping_lambda):
        J_T = J.T
        identity = np.eye(J.shape[1])  # (n x n)
        H = J_T @ J + damping_lambda * identity
        delta_q = np.linalg.solve(H, J_T @ error)  # numerically more stable than inv()
        return delta_q

    fk_start = forward_kinematics_3D(q)
    R_actual = fk_start[-1][:3, :3]
    X_e = get_pose_from_T_matrix(fk_start[-1])

    # Construct desired orientation
    psi_x_des, psi_y_des, psi_z_des = X_d[3:6]
    R_x_des = np.array([[1, 0, 0],
                        [0, np.cos(psi_x_des), -np.sin(psi_x_des)],
                        [0, np.sin(psi_x_des),  np.cos(psi_x_des)]])
    R_y_des = np.array([[np.cos(psi_y_des), 0, np.sin(psi_y_des)],
                        [0, 1, 0],
                        [-np.sin(psi_y_des), 0, np.cos(psi_y_des)]])
    R_z_des = np.array([[np.cos(psi_z_des), -np.sin(psi_z_des), 0],
                        [np.sin(psi_z_des),  np.cos(psi_z_des), 0],
                        [0, 0, 1]])
    R_des = R_x_des @ R_y_des @ R_z_des
    delta_R = R_des @ R_actual.T
    delta_angle = retrieve_xyz_euler_angles(delta_R)

    parameterized_pose_error = X_d - X_e
    parameterized_pose_error[3:] = delta_angle

    error_list = [np.linalg.norm(parameterized_pose_error)]
    fk_list = [fk_start]
    counter = 1
    start_time = time.time()

    damping_lambda = 0.01  # Initial LM damping factor

    while np.linalg.norm(parameterized_pose_error) > error_threshold:
        J_eA = analytical_jacobian_3D(q)

        delta_q = levenberg_marquardt_update(J_eA, parameterized_pose_error, damping_lambda)
        q_new = q + k * delta_q

        fk_new = forward_kinematics_3D(q_new)
        X_e_new = get_pose_from_T_matrix(fk_new[-1])
        R_actual_new = fk_new[-1][:3, :3]

        delta_R_new = R_des @ R_actual_new.T
        delta_angle_new = retrieve_xyz_euler_angles(delta_R_new)

        new_error = X_d - X_e_new
        new_error[3:] = delta_angle_new
        new_error_norm = np.linalg.norm(new_error)
        prev_error_norm = np.linalg.norm(parameterized_pose_error)

        # Accept update only if error decreased
        if new_error_norm < prev_error_norm:
            q = q_new
            parameterized_pose_error = new_error
            damping_lambda *= 0.2  # reduce lambda to get closer to Gauss-Newton
            fk_list.append(fk_new)
        else:
            damping_lambda *= 2.0  # increase lambda to stabilize

        error_list.append(np.linalg.norm(parameterized_pose_error))
        counter += 1

        if counter == 1000:
            print("Counter reached 1000 -> exit inverse differential kinematics")
            plot_error(error_list, error_threshold)
            convergence_time = round(time.time() - start_time, 4)
            q = np.array([0, 0, 0, 0])
            return q, convergence_time, fk_list, error_list

    convergence_time = round(time.time() - start_time, 4)
    return q, convergence_time, fk_list, error_list


def inverse_differential_kinematics_3D_QP(X_d, k, q, error_threshold): #uses quadratic programming

    fk_start = forward_kinematics_3D(q)
    R_actual = fk_start[-1][:3, :3]
    X_e = get_pose_from_T_matrix(fk_start[-1])

    psi_x_des, psi_y_des, psi_z_des = X_d[3:]

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(psi_x_des), -np.sin(psi_x_des)],
                    [0, np.sin(psi_x_des), np.cos(psi_x_des)]])
    R_y = np.array([[np.cos(psi_y_des), 0, np.sin(psi_y_des)],
                    [0, 1, 0],
                    [-np.sin(psi_y_des), 0, np.cos(psi_y_des)]])
    R_z = np.array([[np.cos(psi_z_des), -np.sin(psi_z_des), 0],
                    [np.sin(psi_z_des),  np.cos(psi_z_des), 0],
                    [0, 0, 1]])
    
    R_des = R_x @ R_y @ R_z
    delta_R = R_des @ R_actual.T
    delta_angle = retrieve_xyz_euler_angles(delta_R)

    parameterized_pose_error = X_d - X_e
    parameterized_pose_error[3:] = delta_angle
    error_list = [np.linalg.norm(parameterized_pose_error)]
    fk_list = [fk_start]
    counter = 0
    start_time = time.time()

    while np.linalg.norm(parameterized_pose_error) > error_threshold:
        J = analytical_jacobian_3D(q)
        n = J.shape[1]

        dq = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(J @ dq - parameterized_pose_error))
        prob = cp.Problem(objective)
        prob.solve()

        if dq.value is None:
            print("QP failed to solve")
            return np.array([0, 0, 0, 0]), round(time.time() - start_time, 4), fk_list, error_list

        q = q + k * dq.value
        fk_new = forward_kinematics_3D(q)
        X_e = get_pose_from_T_matrix(fk_new[-1])
        R_actual = fk_new[-1][:3, :3]
        delta_R = R_des @ R_actual.T
        delta_angle = retrieve_xyz_euler_angles(delta_R)

        parameterized_pose_error = X_d - X_e
        parameterized_pose_error[3:] = delta_angle

        error_list.append(np.linalg.norm(parameterized_pose_error))
        fk_list.append(fk_new)
        counter += 1

        if counter >= 1000:
            print("Max iterations reached")
            return np.array([0, 0, 0, 0]), round(time.time() - start_time, 4), fk_list, error_list

    convergence_time = round(time.time() - start_time, 4)
    return q, convergence_time, fk_list, error_list


def retrieve_xyz_euler_angles(R):
    """
    Extract intrinsic XYZ Euler angles from a rotation matrix R.
    That is, R = Rx(psi_x) @ Ry(psi_y) @ Rz(psi_z)
    """
    if abs(R[2, 0]) < 1.0:
        psi_y = -np.arcsin(R[2, 0])
        psi_x = np.arctan2(R[2, 1] / np.cos(psi_y), R[2, 2] / np.cos(psi_y))
        psi_z = np.arctan2(R[1, 0] / np.cos(psi_y), R[0, 0] / np.cos(psi_y))
    else:
        # Gimbal lock
        psi_y = np.pi / 2 if R[2, 0] <= -1.0 else -np.pi / 2
        psi_x = 0
        psi_z = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([psi_x, psi_y, psi_z])


def get_pose_from_T_matrix(T):
    """
    Extracts the end effector pose from a transformation matrix T.
    Returns pose as [x, y, z, psi_x, psi_y, psi_z] according to XYZ intrinsic euler angles.
    T is a 4x4 transformation matrix.
    """
    position = T[:3, 3]
    R = T[:3, :3]
    psi_x, psi_y, psi_z = retrieve_xyz_euler_angles(R)
    return np.array([position[0], position[1], position[2], psi_x, psi_y, psi_z]
)


def forward_kinematics_3D(q):
    alpha, robotic_length, theta, delta_l_niTi = q #all anlges are in rad
    
    # Design parameters
    len_distal_part = 130 #mm
    len_enddisk = 46 #mm
    len_wrist = 42 #mm
    dist_tendons= 9.7 #mm
    max_num_of_ridig_links_in_the_bending_section = 10 #mm

    T_list = []

     #it always does the translation first, and then the last point of the translation is rotated about the angle.

    T_list.append(np.array([[1, 0            , 0             , 0      ],
                            [0, np.cos(alpha), -np.sin(alpha), 0      ],
                            [0, np.sin(alpha), np.cos(alpha) , 0      ],
                            [0, 0            , 0             , 1      ]])) 
  
    # transformation matrix for rotation about the z-axis
    def calculate_segment_transform(angle, length):
        return np.array([[np.cos(angle), -np.sin(angle), 0, length],
                         [np.sin(angle), np.cos(angle) , 0, 0     ],
                         [0            , 0             , 1, 0     ],
                         [0            , 0             , 0, 1     ]])
    
    
    T_list.append(calculate_segment_transform(theta, 0))

    T_list.append(calculate_segment_transform(np.deg2rad(-90), len_distal_part)) #-90deg to make the robot come out in a perpendicular fashion
    
    def forward_kinematics_CR_3D(l_right, l_left, dist_tendons):
        gamma = ((l_right-l_left) * 180)/(np.pi * dist_tendons) #in deg
        l = (l_right+l_left)/2 
        r = l * 180/(np.pi * gamma)

        kappa = 1/r

        T_CR = np.array([[np.cos(kappa*l),-np.sin(kappa*l), 0, 1/kappa * (np.cos(kappa * l)-1) ],
                        [np.sin(kappa*l) , np.cos(kappa*l), 0, 1/kappa * np.sin(kappa * l)     ],
                        [0               , 0              , 1, 0                               ],
                        [0               , 0              , 0, 1                               ]])
        
        return T_CR

    
    for i in range (max_num_of_ridig_links_in_the_bending_section):
        T_list.append(forward_kinematics_CR_3D(robotic_length/max_num_of_ridig_links_in_the_bending_section, (robotic_length+delta_l_niTi)/max_num_of_ridig_links_in_the_bending_section, dist_tendons))

   
    T_list.append(calculate_segment_transform(np.deg2rad(90), 0))


    T_list.append(np.array([[1, 0, 0, len_enddisk + len_wrist],
                            [0, 1, 0, 0                      ],
                            [0, 0, 1, 0                      ],
                            [0, 0, 0, 1                      ]])) 
        



    T_sum_list = [T_list[0]]
    for i in range(1,len(T_list)):
        T_sum_list.append(np.dot(T_sum_list[-1],T_list[i]))

    return T_sum_list


def analytical_jacobian_3D(q):
    h = 0.001
    J_eA = np.zeros((6, 4)) 
    fk = forward_kinematics_3D(q) # returns a list of  4x4 transformation matrices, where the last one is the end effector pose
    X_e_q = get_pose_from_T_matrix(fk[-1]) # X_e_q = current pose parameterization of the end effector

    for i in range(len(q)):
        q_h = q.copy()
        q_h[i] = q[i]+h #i=0 -> derivation w.r.t. q_0, i=1 -> derivation w.r.t. q_1, etc.
        fk_q_h = forward_kinematics_3D(q_h)
        X_e_q_h = get_pose_from_T_matrix(fk_q_h[-1]) # X_e_q_h = current pose parameterization of the end effector with perturbed q
        J_eA[:,i] = (X_e_q_h - X_e_q) / h

    return J_eA


def inverse_differential_kinematics_3D(X_d, k, q, error_threshold):  #difference is that the orientation error is calculated correctly 
    ''' return q, convergence_time, fk_list, error_list
    in case of no convergence, return q=[0,0,0,0]'''

    #fk_start is the fk of the start configuration of the robot, i.e. the fk belonging to the original q 
    fk_start = forward_kinematics_3D(q)
    R_actual = fk_start[-1][:3,:3] # R_actual = rotation matrix of the end effector in the start configuration
    X_e = get_pose_from_T_matrix(fk_start[-1]) # X_e = [x,y,z, psi_x, psi_y, psi_z] of the end effector in the start configuration 
    
    psi_x_des = X_d[3]
    psi_y_des = X_d[4]
    psi_z_des = X_d[5]

    R_x_des = np.array([[1, 0                ,                  0],
                        [0, np.cos(psi_x_des), -np.sin(psi_x_des)],
                        [0, np.sin(psi_x_des), np.cos(psi_x_des)]])
    
    R_y_des = np.array([[np.cos(psi_y_des) , 0, np.sin(psi_y_des)],
                        [0                 , 1,                 0],
                        [-np.sin(psi_y_des), 0, np.cos(psi_y_des)]])
    
    R_z_des = np.array([[np.cos(psi_z_des), -np.sin(psi_z_des), 0],
                        [np.sin(psi_z_des),  np.cos(psi_z_des), 0],
                        [0                ,  0                 ,1]])
    
    R_des = R_x_des @ R_y_des @ R_z_des #this order is right for intrinsic (i.e., local) xyz euler angles
    delta_R = R_des @ R_actual.T
    delta_angle = retrieve_xyz_euler_angles(delta_R)

    parameterized_pose_error = X_d-X_e
    parameterized_pose_error[3:] = delta_angle
    error_list = [np.linalg.norm(parameterized_pose_error)]
    counter = 1
    start_time = time.time()

    fk_list = [fk_start]


    while np.linalg.norm(parameterized_pose_error) > error_threshold:
        J_eA = analytical_jacobian_3D(q)
        q = q + k* np.dot(np.linalg.pinv(J_eA), parameterized_pose_error)

        # Calculate new endeffector pose X_e
        fk_new = forward_kinematics_3D(q)

        X_e = get_pose_from_T_matrix(fk_new[-1]) # X_e = current pose parameterization of the end effector
        R_actual = fk_new[-1][:3, :3]
        delta_R = R_des @ R_actual.T
        delta_angle = retrieve_xyz_euler_angles(delta_R)

        parameterized_pose_error = X_d-X_e
        parameterized_pose_error[3:] = delta_angle
        error_list.append(np.linalg.norm(parameterized_pose_error))
        
        counter = counter + 1
        fk_list.append(fk_new)
        
        if counter == 1000:
            print("Counter reached 1000 -> exit inverse differential kinematics")
            plot_error(error_list, error_threshold)
            convergence_time = round(time.time()-start_time, 4)
            q = np.array([0,0,0,0])
                 
            return q, convergence_time, fk_list, error_list
    
    convergence_time = round(time.time()-start_time, 4)
    #print(f"Minimization of inverse differential kinematics took {convergence_time} seconds")
    #plot_error(error_list, error_threshold, k)

    #q_final = interpret_results(q)

    return q, convergence_time, fk_list, error_list


def forward_kinematics_2D(q):
    
    robotic_length, theta, delta_l_niTi = q #all anlges are in rad

    # Design parameters
    #off_set = 0 #312.8 #mm of set from base of the robot to the centerline of the robot
    len_distal_part = 130 #mm
    len_wrist = 88 #mm
    dist_tendons= 9.7 #mm

    T_list = []


    #it always does the translation first, and then the last point of the translation is rotated about the angle. 
    def calculate_segment_transform_2D(angle, length):
        return np.array([[np.cos(angle), -np.sin(angle), length],
                         [np.sin(angle),  np.cos(angle), 0],
                         [0,              0,             1]])
       
        
    T_list.append(calculate_segment_transform_2D(theta, 0))

    T_list.append(calculate_segment_transform_2D(np.deg2rad(-90), len_distal_part)) #-90deg to make the robot come out in a perpendicular fashion
    

    def forward_kinematics_CR_2D(l_connectors, l_NiTi, dist_tendons):
        #this is in accordance with the prototype where the NiTi is on top and the connectors are on the bottom, i.e. if l_Niti > l_connectors --> bending downwards
        gamma = ((l_connectors-l_NiTi) * 180)/(np.pi * dist_tendons) #in deg
        l = (l_connectors+l_NiTi)/2 
        r = l * 180/(np.pi * gamma)

        kappa = 1/r

        T_CR = np.array([[np.cos(kappa*l),-np.sin(kappa*l), 1/kappa * (np.cos(kappa * l)-1) ],
                         [np.sin(kappa*l), np.cos(kappa*l), 1/kappa * np.sin(kappa * l)     ],
                         [0              , 0              , 1                               ]])
        
        return T_CR

    resolution = 10
    for i in range (resolution):
        T_list.append(forward_kinematics_CR_2D(robotic_length/resolution, (robotic_length+delta_l_niTi)/resolution, dist_tendons))

   
    T_list.append(calculate_segment_transform_2D(np.deg2rad(90), 0))


    T_list.append(calculate_segment_transform_2D(0, len_wrist))


    T_sum_list = [T_list[0]]
    for i in range(1,len(T_list)):
        T_sum_list.append(np.dot(T_sum_list[-1],T_list[i]))


    #each T matrix is= [[R11, R12, x, i.e. horizontal -> we call z   ],
    #                   [R21, R22, y, i.e. vertical -> we call x_hat ],
    #                   [0  , 0  , 1                                 ]]

    return T_sum_list


def compute_analytical_Jacobian_2D(q):
        h = 1e-5 #for numeric differentiation
        #Compute analytical Jacobian
        J_eA = np.zeros((3,3))

        fk = forward_kinematics_2D(q)[-1]
        fk_robotic_length = forward_kinematics_2D([q[0]+h, q[1], q[2]])[-1]
        fk_theta = forward_kinematics_2D([q[0], q[1]+h, q[2]])[-1]
        fk_delta_l_niti  = forward_kinematics_2D([q[0], q[1], q[2]+h])[-1]
        
        # x position
        J_eA[0,0] = fk_robotic_length[0,2]-fk[0,2]
        J_eA[0,1] = fk_theta[0,2]-fk[0,2]
        J_eA[0,2] = fk_delta_l_niti [0,2]-fk[0,2]

        # y position
        J_eA[1,0] = fk_robotic_length[1,2]-fk[1,2]
        J_eA[1,1] = fk_theta[1,2]-fk[1,2]
        J_eA[1,2] = fk_delta_l_niti [1,2]-fk[1,2]

        # pitch
        R = fk[:2,:2]
        psi = np.arctan2(-R[0,1], R[0,0])

        R_robotic_length = fk_robotic_length[:2,:2]
        psi_robotic_length = np.arctan2(-R_robotic_length[0,1], R_robotic_length[0,0])

        R_theta = fk_theta[:2,:2]
        psi_theta = np.arctan2(-R_theta[0,1], R_theta[0,0])

        R_delta_l_niti = fk_delta_l_niti[:2,:2]
        psi_delta_l_niti = np.arctan2(-R_delta_l_niti[0,1], R_delta_l_niti[0,0])

        J_eA[2,0] = psi_robotic_length - psi
        J_eA[2,1] = psi_theta - psi
        J_eA[2,2] = psi_delta_l_niti  - psi

        return J_eA/h


def inverse_differential_kinematics_2D(q, k, desired_pose):
    
    #desired_pose_compute = desired_pose * np.array([1,1,1]) # to make the clockwise rotation angle from the desired position into a counterclockwise rotation angle so that the inverse kinematic model can work with it
    
    error_threshold = 1e-6
    counter = 0
    
    fk = forward_kinematics_2D(q)[-1]
    R = fk[:2,:2]
    psi = np.arctan2(-R[0,1], R[0,0])
    actual_pose = np.array([fk[0,2], fk[1,2], psi])
    error_list = []

    

    while np.linalg.norm(actual_pose - desired_pose) > error_threshold:
        counter += 1
        error_list.append(np.linalg.norm(actual_pose - desired_pose))

        J_eA = compute_analytical_Jacobian_2D(q)
        pose_error = desired_pose - actual_pose
        update_vec = np.dot(np.linalg.pinv(J_eA), pose_error)*k
        q[0] += update_vec[0] #robotic_length
        q[1] += update_vec[1] #theta
        q[2] += update_vec[2] # delta_l_niTi

    
        fk = forward_kinematics_2D(q)[-1]
        R = fk[:2,:2]
        psi = np.arctan2(-R[0,1], R[0,0])
        actual_pose = np.array([fk[0,2], fk[1,2], psi])
        

        if counter == 1000:
            #print("Counter reached 1000 -> exit inverse differential kinematics")
            #plot_error(error_list, error_threshold)
            return [0,0,0]


    if q[1]<0 or q[1]>np.deg2rad(60):
        #print("Theta out of bounds")
        return [0,0,0]
        
    #plot_error(error_list, error_threshold)

    return q
  

def IK_2D(X_d, k=.1):
    #start configuration
    #proximal_part_travel = 207
    #alpha = 30 #in deg, rotation of the robot along the shaft's center point
    robotic_length = 100# 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
    theta  = 30 #30 # in deg, positive difference will cause rotation in couter clockwise direction
    delta_l_niTi = 10 #25 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis

    
    q_initial = [robotic_length, np.deg2rad(theta), delta_l_niTi]
    
    start_time = time.time()
    q = inverse_differential_kinematics_2D(q_initial, k, desired_pose=X_d)#
    
    time_needed = round(time.time()-start_time, 4)

    if q == [0,0,0]: #Inverse differential kinematics did not converge
        return [0,0,0], time_needed

    fk = forward_kinematics_2D(q)
    #plot_func_2D(fk, desired_pose=X_d)
    z_e = fk[-1][0,2]
    x_e = fk[-1][1,2]
    psi_e = np.arctan2(-fk[-1][0,1], fk[-1][0,0])


    return [z_e, x_e, psi_e], time_needed


def get_motor_commands_from_2D_IK_model(X_d):
    #start configuration
    robotic_length = 100# 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
    theta  = 30 #30 # in deg, positive difference will cause rotation in couter clockwise direction
    delta_l_niTi = 10 #25 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
    dist_rot_axis_2_niti = 5.05 #mm
    dist_rot_axis_2_connectors = 14.75 #mm
    
    q_initial = [robotic_length, np.deg2rad(theta), delta_l_niTi]
    
    
    q = inverse_differential_kinematics_2D(q_initial, desired_pose=X_d)#
    
    
    #fk = forward_kinematics_2D(q)
    #plot_func_2D(fk, desired_pose=X_d)
    
    compensated_robotic_length = q[0] + (q[1]*dist_rot_axis_2_connectors)
    compensated_delta_l_niTi = q[2] + (q[1]*dist_rot_axis_2_niti)

    print("Minimization of inverse differential kinematics took ", time_needed, "seconds")
    print("robotic_length compensated: ", compensated_robotic_length, "mm")
    print("theta: ", np.rad2deg(q[1]), "deg")
    print("delta_l_niTi compensated: ", compensated_delta_l_niTi, "mm")
    print("-------------------")
    return [compensated_robotic_length, q[1], compensated_delta_l_niTi]



if __name__ == "__main__":

    plot_forwards_kinematics_2D = 0
    plot_forwards_kinematics_3D = 0
    plot_Ik_3D = 0
    plot_Ik_2D = 0
    get_motor_commands = 0
    plot_WS = 0
    comparison = 1


    if plot_forwards_kinematics_2D:
        plot_func_2D(forward_kinematics_2D([100, np.deg2rad(30), 0.1]))


    if plot_forwards_kinematics_3D:

        alpha = np.deg2rad(150) #in deg, rotation of the robot about the main axis of the trocar
        robotic_length = 100 #238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta = np.deg2rad(30) #in deg, deflection of the AAU, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = 5 #25 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        q = [alpha, robotic_length, theta, delta_l_niTi] #delta_l_niTi, rho_proximal, rho_distal

        x_t = 400 # in mm
        y_t = 40 # in mm
        z_t = 100 # in mm
        psi_x_desired = np.arctan2(z_t, y_t) # psi_x_desired = alpha
        psi_y_desired = 0 # always!, no rotation about the y-axis
        psi_z_desired = np.deg2rad(-30)
        X_d = np.array([x_t, y_t, z_t, psi_x_desired, 0, psi_z_desired]) 
        
        plot_func(forward_kinematics_3D(q), desired_pose=X_d)


    if plot_Ik_3D:
        alpha = np.deg2rad(30) #in deg, rotation of the robot along the shaft's center point
        robotic_length = 200 # in mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = np.deg2rad(20) # in deg, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = 5 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis

        q_initial = [alpha, robotic_length, theta, delta_l_niTi]
        fk_inital = forward_kinematics_3D(q_initial) #initial forward kinematics to plot the initial pose of the robot

        # compute the transformed end effector orientation at the defined position
        # euler xyz rotation is correct (i.e., the rotation is applied in the order of angle about x=alpha, angle about y=0, angle about z=psi_z_desired) 
        x_t = 400 # in mm
        y_t = 280 # in mm
        z_t = 100 # in mm
        psi_x_desired = np.arctan2(z_t, y_t) # psi_x_desired = alpha
        psi_y_desired = 0 # always!, no rotation about the y-axis
        psi_z_desired = np.deg2rad(-20)
        X_d = np.array([x_t, y_t, z_t, psi_x_desired, 0, psi_z_desired])  # Desired pose, xyz position and xyz euler angles


        #plot_func(fk_inital, desired_pose=X_d)
        k = 0.5  # Gain
        error_threshold = 1e-4 

        q, _,_,_ = inverse_differential_kinematics_3D(X_d, k, q_initial, error_threshold)
        # print("alpha ", np.rad2deg(q[0]), "in deg")
        # print("robotic_length_travel ", q[1], "in mm") #use as is
        # print("theta ", np.rad2deg(q[2]), "in deg")
        # print("delta_l_niTi ", q[3], "in mm")

        fk = forward_kinematics_3D(q)
        print("X_d: ", X_d)

        # get the end effector pose from the forward kinematics
        print(fk[-1])
        
        print("X_e: ", get_pose_from_T_matrix(fk[-1]))
        plot_func(fk, fk_inital, desired_pose=X_d)


    if plot_Ik_2D:

        z_target = 343
        x_target = 120
        psi = np.deg2rad(21)
        X_d = np.array([z_target, x_target, psi])
        [z_e, x_e, psi_e], time_needed = IK_2D(X_d)
        print("it took: ", time_needed, "seconds")
        print([z_e, x_e, np.rad2deg(psi_e)])
        
    if get_motor_commands:
        X_d = [400, 180, np.deg2rad(40)]
        get_motor_commands_from_2D_IK_model(X_d)

    if plot_WS:
        alpha_min = np.deg2rad(-179)
        alpha_max = np.deg2rad(180)
        robotic_length_min = 0
        robotic_length_max = 160
        theta_min = np.deg2rad(0)
        theta_max = np.deg2rad(60)
        delta_l_niTi_min = -21
        delta_l_niTi_max = 20

        joint_bounds = [(robotic_length_min, robotic_length_max), (theta_min, theta_max), (delta_l_niTi_min, delta_l_niTi_max)] 

        plot_workspace(joint_bounds)

    if comparison: 

        y_lower_bound = 0 # in mm
        y_upper_bound = 50 # in mm
        z_lower_bound =-25 # in mm
        z_upper_bound = 25 # in mm
        X_d_list = []
  
        # 3d helix trajectory
        helix_radius = 25 # in mm
        helix_height = 100 # in mm
        helix_turns = 2
        helix_steps = 50
        for i in range(helix_steps):
            t = i / helix_steps * helix_turns * 2 * np.pi
            x_t = 300 + helix_radius * np.cos(t) # in mm   
            y_t = helix_radius * np.sin(t) # in mm
            z_t = helix_height * t / (2 * np.pi * helix_turns) # in mm
            psi_x_desired = np.arctan2(z_t, y_t) # psi_x_desired = alpha
            psi_y_desired = 0 # always!, no rotation about the y-axis   
            psi_z_desired = np.deg2rad(-30)
            X_d_list.append([x_t, y_t, z_t, psi_x_desired, 0, psi_z_desired])
        
    
        X_d_array = np.array(X_d_list)
        #print(X_d_array)

        alpha_list = []
        X_e_IK_big_k_list = []
        X_e_IK_small_k_list = []
        X_e_IK_damped_list = []
        X_e_IK_QP_list = []
        X_e_NSGC_list = []
        

        time_needed_IK_big_k_convergence_list = []
        time_needed_IK_small_k_convergence_list = []
        time_needed_IK_damped_convergence_list = []
        time_needed_IK_QP_convergence_list = []
        time_needed_NSGC_convergence_list = []

        error_list_big_k = []
        error_list_small_k = []
        error_list_damped = []
        error_list_QP = []
        error_list_NSGC = []

        start_time = time.time()

        # column 0: 1 if the IK big k pose was not achieved
        # column 1: 1 if the IK big k pose was achieved
        # column 2: 1 if the IK small k pose was not achieved
        # column 3: 1 if the IK small k pose achieved
        # column 4: 1 if the IK LM  k pose was not achieved
        # column 5: 1 if the IK LM k pose achieved
        # column 6: 1 if the IK QP k pose was not achieved
        # column 7: 1 if the IK QP k pose achieved
        # column 8: 1 if the NSGC pose was not achieved
        # column 9: 1 if the NSGC pose was achieved
 
        convergence_overview_array = np.zeros((len(X_d_array), 10)) 


        # call inverse kinematics models (IK and set of equations) for each desired pose
        for i, row in enumerate(X_d_array):
            print(f"Target Pose {i+1}/{len(X_d_array)}")
            X_t = row
            x_t =   X_t[0]
            y_t =   X_t[1]
            z_t =   X_t[2]
            angle_x_t = X_t[3]
            angle_y_t = X_t[4]
            angle_z_t = X_t[5]
            
            q_initial = [np.deg2rad(0), 150, np.deg2rad(30), 6] # [alpha, theta, l_b, delta_l_NiTi], initial guess for q for the inverse differential kinematics
            error_threshold=1e-5

            #basic jacobian
            k=0.5
            q_IK_big_k, time_needed_IK_big_k, fk_list_big_k, error_list_big_k = inverse_differential_kinematics_3D(X_t, k, q_initial, error_threshold) #returns q, convergence_time, fk_list, error_list
            # returns q = [0,0,0,0] if the inverse kinematics did not converge
            # fk_list is a list of all fk_lists which were computed during the convergence of the inverse kinematics
            time_needed_IK_big_k_convergence_list.append(time_needed_IK_big_k)
            if np.array_equal(q_IK_big_k, [0,0,0,0]): #if the inverse kinematics did not converge
                convergence_overview_array[i,0] = 1
                X_e_IK_big_k_list.append([0,0,0,0,0,0]) # append a dummy pose
            else: #successful convergence
                X_e_IK_big_k_list.append(get_pose_from_T_matrix(fk_list_big_k[-1][-1])) # append only the final end effector pose
                #print("IKM with big k converged successfully!")


            #basic jacobian
            k=0.4
            q_IK_small_k, time_needed_IK_small_k, fk_list_small_k, error_list_small_k = inverse_differential_kinematics_3D(X_t, k, q_initial, error_threshold)
            time_needed_IK_small_k_convergence_list.append(time_needed_IK_small_k)
            if np.array_equal(q_IK_small_k, [0,0,0,0]):
                convergence_overview_array[i,2] = 1
                X_e_IK_small_k_list.append([0,0,0,0,0,0]) # append a dummy pose
            else:
                X_e_IK_small_k_list.append(get_pose_from_T_matrix(fk_list_small_k[-1][-1]))
                #print("IKM with small k converged successfully!")

            
            #jacobian with adaptive damping (Levenberg-Marquardt)
            k=0.5
            q_IK_damped, time_needed_IK_damped, fk_list_IK_damped, error_list_damped = inverse_differential_kinematics_LM_3D(X_t, k, q_initial, error_threshold) #returns q, convergence_time, fk_list, error_list
            # returns q = [0,0,0,0] if the inverse kinematics did not converge
            # fk_list is a list of all fk_lists which were computed during the convergence of the inverse kinematics
            time_needed_IK_damped_convergence_list.append(time_needed_IK_damped)
            if np.array_equal(q_IK_damped, [0,0,0,0]): #if the inverse kinematics did not converge
                convergence_overview_array[i,4] = 1
                X_e_IK_damped_list.append([0,0,0,0,0,0]) # append a dummy pose
            else: #successful convergence
                X_e_IK_damped_list.append(get_pose_from_T_matrix(fk_list_IK_damped[-1][-1])) # append only the final end effector pose
                #print("IKM with adaptive damping converged successfully!")


            #Quadratic Programming
            k=0.5
            q_IK_QP, time_needed_IK_QP, fk_list_IK_QP, error_list_QP = inverse_differential_kinematics_3D_QP(X_t, k, q_initial, error_threshold) #returns q, convergence_time, fk_list, error_list
            # returns q = [0,0,0,0] if the inverse kinematics did not converge
            # fk_list is a list of all fk_lists which were computed during the convergence of the inverse kinematics
            time_needed_IK_QP_convergence_list.append(time_needed_IK_QP)
            if np.array_equal(q_IK_QP, [0,0,0,0]): #if the inverse kinematics did not converge
                convergence_overview_array[i,6] = 1
                X_e_IK_QP_list.append([0,0,0,0,0,0]) # append a dummy pose
            else: #successful convergence
                X_e_IK_QP_list.append(get_pose_from_T_matrix(fk_list_IK_QP[-1][-1])) # append only the final end effector pose
                #print("IKM with QP converged successfully!")


            alpha = np.arctan2(z_t, y_t) # >>>check if that is the same result as the XYZ euler angles from the forward kinematics
            alpha_list.append(alpha)
            x_t_hat_NSGC = np.sqrt(y_t**2 + z_t**2)

            # for alignement with naming convention in the paper
            z_t_NSGC = x_t
            psi_t_NSGC = psi_z_desired

            X_e_hat_NSGC_first_try, time_needed_NSGC_first_try, num_sol_first_try  = solve_NSGC(z_t_NSGC, x_t_hat_NSGC, psi_t_NSGC) # returns X_e_hat, time_needed, numerical solution 
            # X_e_hat = [z_e_hat, x_e_hat, psi]
            # if the set of equations did not find solution -> return X_e_hat=[0,0,0]
            # if the configuration is straight, the numerical solution will be [0,0,0,0]
            if np.array_equal(X_e_hat_NSGC_first_try, [0,0,0]): #if there in no numerical solution
                X_e_hat_NSGC_second_try, time_needed_NSGC_second_try, num_sol_second_try = solve_NSGC(z_t, -x_t_hat_NSGC, psi_t_NSGC)
                
                if np.array_equal(X_e_hat_NSGC_second_try, [0,0,0]):
                    convergence_overview_array[i,8] = 1
                    print("x_t_hat was inverted and not reached by NSGC")
                    time_needed_NSGC_convergence_list.append(time_needed_NSGC_first_try + time_needed_NSGC_second_try)
                    X_e_NSGC_list.append([0,0,0,0,0,0]) # append a dummy pose

                else:
                    #print("x_t_hat was inverted and solution found by NSGC")
                    #realign with the naming convention in the paper
                    x_t = X_e_hat_NSGC_second_try[0] # z_t_NSGC
                    x_t_hat_NSGC = X_e_hat_NSGC_second_try[1]
                    psi_t = X_e_hat_NSGC_second_try[2] # psi_t_NSGC

                    X_e_NSGC_second_try = [x_t, x_t_hat_NSGC*np.cos(alpha), x_t_hat_NSGC*np.sin(alpha), alpha, 0, psi_t]
                    time_needed_NSGC_convergence_list.append(time_needed_NSGC_first_try + time_needed_NSGC_second_try)
                    X_e_NSGC_list.append(X_e_NSGC_second_try)

            else:
                #realign with the naming convention in the paper
                x_t = X_e_hat_NSGC_first_try[0] # z_t_NSGC
                x_t_hat_NSGC = X_e_hat_NSGC_first_try[1]
                psi_t = X_e_hat_NSGC_first_try[2] # psi_t_NSGC

                X_e_NSGC_first_try = [x_t, x_t_hat_NSGC*np.cos(alpha), x_t_hat_NSGC*np.sin(alpha), alpha, 0, psi_t]
                time_needed_NSGC_convergence_list.append(time_needed_NSGC_first_try)
                X_e_NSGC_list.append(X_e_NSGC_first_try)
                #print("Frist try NSGC worked!!!!!")
            
        X_e_IK_big_k_array = np.array(X_e_IK_big_k_list)
        X_e_IK_small_k_array = np.array(X_e_IK_small_k_list)
        X_e_IK_damped_array = np.array(X_e_IK_damped_list)
        X_e_IK_QP_array = np.array(X_e_IK_QP_list)
        X_e_NSGC_array = np.array(X_e_NSGC_list)

        # check if all poses at which the IKM converged are actually the desired poses (exclude false positives)
        for i in range(len(X_d_array)): 
            if np.linalg.norm(X_e_IK_big_k_array[i] - X_d_array[i]) < 1 and not np.array_equal(X_e_IK_big_k_array[i], [0,0,0,0,0,0]):
            # X_e_IK_array[i] != [0,0,0,0] for the case that the IK did not converge -> needs to be excluded
            # otherwise the distance between [0,0,0,0] and X_d_array[i] would be computed which is always greater than 1
            # thus convergence_overview_array would not have a clear overview of which poses were not achieved (no convergence) and which poses were too far away from the desired pose
                convergence_overview_array[i,1] = 1

        for i in range(len(X_d_array)):
            if np.linalg.norm(X_e_IK_small_k_array[i] - X_d_array[i]) < 1 and not np.array_equal(X_e_IK_small_k_array[i], [0,0,0,0,0,0]):
                convergence_overview_array[i,3] = 1

        for i in range(len(X_d_array)):    
            if np.linalg.norm(X_e_IK_damped_array[i] - X_d_array[i]) < 1 and not np.array_equal(X_e_IK_damped_array[i], [0,0,0,0,0,0]):
                 convergence_overview_array[i,5] = 1

        for i in range(len(X_d_array)):    
            if np.linalg.norm(X_e_IK_QP_array[i] - X_d_array[i]) < 1 and not np.array_equal(X_e_IK_QP_array[i], [0,0,0,0,0,0]):
                 convergence_overview_array[i,7] = 1

        for i in range(len(X_d_array)):    
            if np.linalg.norm(X_e_NSGC_array[i] - X_d_array[i]) < 1 and not np.array_equal(X_e_NSGC_array[i], [0,0,0,0,0,0]):
                 convergence_overview_array[i,9] = 1
        

        print("Percentage of successfully achieved poses with Jacobian based IK big k: ", (np.sum(convergence_overview_array[:,1]))/len(X_d_array)*100, "%")
        print("Percentage of successfully achieved poses with Jacobian based IK small k: ", (np.sum(convergence_overview_array[:,3]))/len(X_d_array)*100, "%")
        print("Percentage of successfully achieved poses with damped Jacobian based IK: ", (np.sum(convergence_overview_array[:,5]))/len(X_d_array)*100, "%")
        print("Percentage of successfully achieved poses with QP based IK: ", (np.sum(convergence_overview_array[:,7]))/len(X_d_array)*100, "%")
        print("Percentage of successfully achieved poses with NSGC: ", (np.sum(convergence_overview_array[:,9]))/len(X_d_array)*100, "%")
        
        # plot_error(error_list_big_k, error_threshold, title="Error of Basic Jacobian k=0.5")
        # plot_error(error_list_small_k, error_threshold, title="Error of Basic Jacobian k=0.4")
        # plot_error(error_list_damped, error_threshold, title="Error of Jacobian with LM k=0.5")
        # plot_error(error_list_QP, error_threshold, title="Error of Jacobian with QP k=0.5")
        # plot_error(error_list_NSGC, error_threshold, title="Error of NSGC")

        plot_time_boxplot(time_needed_IK_big_k_convergence_list, time_needed_IK_small_k_convergence_list, time_needed_IK_damped_convergence_list, time_needed_IK_QP_convergence_list, time_needed_NSGC_convergence_list)
        

        
