from scipy.optimize import fsolve
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def plot_func(fk, desired_pose=None, plot_cube=False, image_number=None):
    #fk is a list of transformation matices starting at the base

    x_list = [mat[0,3] for mat in fk]
    y_list = [mat[1,3] for mat in fk]
    z_list = [mat[2,3] for mat in fk]
  
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(0,0,0, color = "green", marker="^", label = "robot base", s=50)
    ax.plot(x_list, y_list, z_list, color = "chocolate", marker=".", label = "NSGC result")

    if desired_pose is not None :
        ax.scatter3D(desired_pose[0], desired_pose[1], desired_pose[2], color="red", marker="x", label="target pose", s=50)
    
    if plot_cube == True:
        z = [-25, 25]
        y = [-25, 25]
        x = [250, 280]

        # Create cube vertices
        vertices = [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ]

        # Define cube faces using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[4], vertices[7], vertices[3], vertices[0]],  # left
        ]

        cube = Poly3DCollection(faces, linewidths=1, edgecolors='black', alpha=0.2)
        cube.set_facecolor('cyan')
        ax.add_collection3d(cube)

    # horizontal line along X axis
    x_line = np.linspace(0, 500, 100)
    y_line = np.full_like(x_line, 0)
    z_line = np.full_like(x_line, 0)
    ax.plot(x_line, y_line, z_line, color='black', linestyle='--', linewidth=1, label="robot's main axis")


    ax.axes.set_xlim3d(left=0, right=600) 
    ax.axes.set_ylim3d(bottom=-300, top=300) 
    ax.axes.set_zlim3d(bottom=-300, top=300)
    ax.set_xlabel('Z Axis [mm]')
    ax.set_ylabel('Y Axis [mm]')
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"{-int(tick):d}" for tick in y_ticks])
    ax.set_zlabel('X Axis [mm]')
    ax.view_init(elev=20, azim=-45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Gallbladder endoscopy with NSGC in 3D', fontsize=16)

    
    
    ax.legend()
    #plt.show()
    image_directory = r"C:\Dateien\3_ProTUTech_WiMi\Paper 2\Python_code\simulated_use_plot_folder"
    file_name = f"plot_{image_number}.png"
    full_file_path = os.path.join(image_directory, file_name)
    dpi = 300
    fig.savefig(full_file_path, dpi=dpi)
    plt.close()


def forward_kinematics_3D(robot_parameters):
    alpha, theta, robotic_length, delta_l_niTi = robot_parameters
    
    # Design parameters
    len_distal_part = 130 #mm
    len_enddisk = 46 #mm
    len_wrist = 42 #mm
    dist_tendons= 9.7 #mm

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

    resolution = 100
    for i in range (resolution):
        T_list.append(forward_kinematics_CR_3D(robotic_length/resolution, (robotic_length+delta_l_niTi)/resolution, dist_tendons))

   
    T_list.append(calculate_segment_transform(np.deg2rad(90), 0))
    T_list.append(calculate_segment_transform(np.deg2rad(0), len_enddisk + len_wrist)) 


    T_sum_list = [T_list[0]]
    for i in range(1,len(T_list)):
        T_sum_list.append(np.dot(T_sum_list[-1],T_list[i]))

    return T_sum_list


def plot_3D(X_t, robot_parameters,image_number): 
    #robot_parameters = np.array([alpha, theta, l_r, delta_l_niti])
    fk = forward_kinematics_3D(robot_parameters)
    X_t_rearranged = np.array([X_t[2], X_t[0], X_t[1]])  # rearranging to match the forward kinematics output
    plot_func(fk, desired_pose=X_t_rearranged, plot_cube=True, image_number=image_number)


def NSGC(z_T, x_T_hat, psi, comments = False):

    
    # Check if the bending is upwards or downwards
    theta_straight_config = np.arctan(x_T_hat/z_T)  # for straight configuration
   
    # Design parameters
    l_w = 88 # length of wrist
    l_d = 130 # length of distal part
    MAX_THETA = np.deg2rad(60)  # Upper limit for theta
    MIN_THETA = 0               # Lower limit for theta
    epsilon = .1 #initialisation of initial guesses
    start_time = time.time()
    angle_threshold_straight = 1  #deg
    angle_threshold_slight_bending = 15 #deg

    if abs(np.rad2deg(psi) - np.rad2deg(theta_straight_config)) < angle_threshold_straight: #the difference is less than 3deg
        #straight configuration (no bending)
        if comments: print("Straight configuration")
        theta = psi
        r = 0 #representing infinity
        z_e_hat = z_T
        x_e_hat = x_T_hat
        X_e_hat = [z_e_hat, x_e_hat, psi-np.deg2rad(angle_threshold_straight)] #worst case: psi is 4deg smaller than theta_straight_config
        end_time = time.time() 
        numerical_solution = [0,0,0,0]
        return X_e_hat , end_time - start_time, numerical_solution 


    if psi > theta_straight_config: 
        #upwards bending
        if comments: print("Upwards bending")
        def equations(vars):
            z_0, x_0_hat, theta, r = vars
            eq1 = r**2 -(z_T-l_w*np.cos(psi)-z_0)**2 - (x_0_hat - x_T_hat + np.sin(psi)*l_w)**2
            eq2 = r**2 -(l_d*np.cos(theta)-z_0)**2 - (x_0_hat - np.sin(theta)*l_d)**2
            eq3 = np.tan(psi) - (z_T - z_0 - np.cos(psi)*l_w)/(x_0_hat - x_T_hat + np.sin(psi)*l_w)
            eq4 = np.tan(theta) - (np.cos(theta)*l_d-z_0)/(x_0_hat-np.sin(theta)*l_d)
            return [eq1, eq2, eq3, eq4]

    if psi < theta_straight_config:
        #downwards bending
        if comments: print("Downwards bending")
        def equations(vars):
            z_0, x_0_hat, theta, r = vars
            eq1 = r**2 -(z_0 - z_T + np.cos(psi)*l_w)**2 - (-x_0_hat + x_T_hat - np.sin(psi)*l_w)**2
            eq2 = r**2 -(z_0 - np.cos(theta)*l_d)**2 - (-x_0_hat + np.sin(theta)*l_d)**2
            eq3 = np.tan(psi) - (z_0 - (z_T - np.cos(psi)*l_w))/(-x_0_hat + x_T_hat - np.sin(psi)*l_w)
            eq4 = np.tan(theta) - (z_0 - np.cos(theta)*l_d)/(-x_0_hat + np.sin(theta)*l_d)
            return [eq1, eq2, eq3, eq4]

        
    initial_guesses = []
    # initial guess for small difference between psi and theta_straight_config -> slight upwards bending
    if angle_threshold_straight <= np.rad2deg(psi) - np.rad2deg(theta_straight_config) <= angle_threshold_slight_bending: #the difference is between 4 and 15deg -> big initial guesses
        if comments: print("big initial guesses for upwards bending")
        for z_0_factor in [epsilon,1]:
            for x_0_hat_factor in [3, 6, 10]:
                for theta_factor in [0,np.deg2rad(10)]:
                    for r_factor in [epsilon, 2, 5]:
                        initial_guesses.append([z_T*z_0_factor, x_T_hat*x_0_hat_factor, theta_factor, z_T*r_factor])


# initial guess for small difference between psi and theta_straight_config -> slight downwards bending
    if angle_threshold_straight <= np.rad2deg(theta_straight_config) - np.rad2deg(psi) <= angle_threshold_slight_bending: #the difference is between 4 and 15deg -> big initial guesses
        if comments: print("big initial guesses for downwards bending")
        for z_0_factor in [epsilon,1]:
            for x_0_hat_factor in [-3, -6, -10]:
                for theta_factor in [0, np.deg2rad(10)]:
                    for r_factor in [epsilon, 2, 5]:
                        initial_guesses.append([z_T*z_0_factor, abs(x_T_hat)*x_0_hat_factor, theta_factor, z_T*r_factor])


    if abs(np.rad2deg(psi) - np.rad2deg(theta_straight_config)) > angle_threshold_slight_bending: #the difference is bigger than 15deg
        if comments: print("small initial guesses")
        #initial_guesses.append([-233, 326, np.deg2rad(54), 374]) 
        for z_0_factor in [epsilon, 1,-1]: 
            for x_0_hat_factor in [epsilon,-1, 1]:
                for theta_factor in [0, np.deg2rad(10), np.deg2rad(20), np.deg2rad(55)]:
                    for r_factor in [epsilon, 1]:
                        initial_guesses.append([z_T*z_0_factor, x_T_hat*x_0_hat_factor, theta_factor, z_T*r_factor])


    def sanity_checks_upwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_w):
        """Perform sanity checks for upwards bending."""
        z_0, x_0_hat, theta, r = numerical_solution
        eq1 = round(r**2 - (z_T - l_w * np.cos(psi) - z_0)**2 - (x_0_hat - x_T_hat + np.sin(psi) * l_w)**2, 3)
        eq2 = round(r**2 - (l_d * np.cos(theta) - z_0)**2 - (x_0_hat - np.sin(theta) * l_d)**2, 3)
        eq3 = round(np.tan(psi) - (z_T - z_0 - np.cos(psi) * l_w) / (x_0_hat - x_T_hat + np.sin(psi) * l_w), 3)
        eq4 = round(np.tan(theta) - (np.cos(theta) * l_d - z_0) / (x_0_hat - np.sin(theta) * l_d), 3)
        return eq1, eq2, eq3, eq4

    def sanity_checks_downwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_w):
        """Perform sanity checks for downwards bending."""
        z_0, x_0_hat, theta, r = numerical_solution
        eq1 = round(r**2 - (z_0 - z_T + np.cos(psi) * l_w)**2 - (-x_0_hat + x_T_hat - np.sin(psi) * l_w)**2, 3)
        eq2 = round(r**2 - (z_0 - np.cos(theta) * l_d)**2 - (-x_0_hat + np.sin(theta) * l_d)**2, 3)
        eq3 = round(np.tan(psi) - (z_0 - (z_T - np.cos(psi) * l_w)) / (-x_0_hat + x_T_hat - np.sin(psi) * l_w), 3)
        eq4 = round(np.tan(theta) - (z_0 - np.cos(theta) * l_d) / (-x_0_hat + np.sin(theta) * l_d), 3)
        return eq1, eq2, eq3, eq4

    error_list = []

    it_counter = 0
    for guess in initial_guesses:
        it_counter += 1

        try:
    # Attempt to solve the equations
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")  # Capture all warnings
                
                numerical_solution = fsolve(equations, guess)  # Try to solve
                #numerical_solution = least_squares(equations, guess).x #works converges, but very slowly 
                #numerical_solution = root(equations, guess, method= 'hybr').x #works well, slower than fsolve
                
                z_0, x_0_hat, theta, r = numerical_solution

                # Check constraints
                if not (MIN_THETA <= theta <= MAX_THETA):
                    if comments: print(f"Theta = {round(np.rad2deg(theta))} deg out of bounds --> next try")
                    continue

                if psi < theta_straight_config and theta < theta_straight_config:  # Downwards bending
                    if comments: print("Theta too small for downwards bending --> next try")
                    continue

                if psi > theta_straight_config and np.sin(theta) * l_d > x_T_hat - np.sin(psi) * l_w:  # Upwards bending
                    if comments: print("Distal part too high --> next try")
                    continue

                # Sanity checks
                if psi > theta_straight_config:  # Upwards bending
                    eqs = sanity_checks_upwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_w)
                else:  # Downwards bending
                    eqs = sanity_checks_downwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_w)

                if all(eq == 0 for eq in eqs):
                    if comments: print("Sanity checks passed!")
                    break
                else:
                    if comments: print("Sanity checks failed:", eqs)
                    continue

            # Check for warnings from fsolve
            if caught_warnings:
                for warning in caught_warnings:
                    if "fsolve" in str(warning.message):  # Look specifically for fsolve-related warnings
                        if comments: print(f"fsolve warning: {warning.message} for guess: {guess}")
                        continue

        except RuntimeWarning:
            if comments: print("RuntimeWarning occurred!")
            continue
   
    time_needed = time.time() - start_time

    if comments: print("Number of iterations until solution: ", it_counter, "/", len(initial_guesses))
    if it_counter == len(initial_guesses):
        if comments: print("Iterated through all initial guesses but no solution found!")
        X_e_hat = [0, 0, 0]  # No solution found
        return X_e_hat, time_needed, numerical_solution
                
    
    numerical_solution[2] = numerical_solution[2]%(2*np.pi) # angle in the range of 0 to 2pi
    theta = numerical_solution[2]
    numerical_solution[3] = abs(numerical_solution[3])  # # numerical solution for r has to be always positive  
    r = numerical_solution[3]



    z_e_hat = 0
    x_e_hat = 0
            
    if 0 < psi < theta_straight_config: #downward bending & in this case z_0 is to the right of the circle end point where it connects to the wrist
        z_e_hat = np.cos(theta)*l_d + np.sin(theta)*r - np.sin(abs(psi))*r + np.cos(psi)*l_w
        x_e_hat = np.sin(theta)*l_d - np.cos(theta)*r + np.cos(psi)*r + np.sin(psi)*l_w # for psi < 0 --> sin(psi) = - sin(abs(psi))
    if psi < 0: #downward bending & in this case z_0 is to the left of the circle end point where it connects to the wrist
        z_e_hat = np.cos(theta)*l_d + np.sin(theta)*r + np.sin(abs(psi))*r + np.cos(psi)*l_w
        x_e_hat = np.sin(theta)*l_d - np.cos(theta)*r + np.cos(psi)*r + np.sin(psi)*l_w # for psi < 0 --> sin(psi) = - sin(abs(psi))
    if psi > theta_straight_config: # upwards bending
        z_e_hat = np.cos(theta)*l_d - np.sin(theta)*r + np.sin(psi)*r + np.cos(psi)*l_w
        x_e_hat = np.sin(theta)*l_d + np.cos(theta)*r - np.cos(psi)*r + np.sin(psi)*l_w
        
        #there is no way to determine psi from the numerical solution

    X_e_hat = [z_e_hat, x_e_hat, psi]
    
    return X_e_hat, time_needed, numerical_solution
        

def get_robot_length_from_NSGC_solution(numerical_solution, X_d_hat_array):

    psi_t = X_d_hat_array[2]
    theta_straight_config = np.arctan(X_d_hat_array[1]/X_d_hat_array[0])  # for straight configuration

    # Design parameters
    dist_niti_2_neutral_axis = dist_connectors_2_neutral_axis = 9.7/2 #mm
    dist_rot_axis_2_niti = 5.05 #mm
    dist_rot_axis_2_connectors = 14.75 #mm

    # this is already calculated in the function solve_equ
    theta = numerical_solution[2]%(2*np.pi) # angle in the range of 0 to 2pi
    r = abs(numerical_solution[3]) # radius is always positive

    if psi_t > theta_straight_config: # upwards bending
        l_r = np.abs((r+dist_connectors_2_neutral_axis)*(psi_t-theta) )
        delta_l_niti = (r-dist_niti_2_neutral_axis)*(psi_t-theta)-l_r
                    
    if psi_t < theta_straight_config and psi_t<0: # downwards bending with psi < 0	
        l_r = (r-dist_connectors_2_neutral_axis)*(theta+abs(psi_t))
        delta_l_niti = (r+dist_niti_2_neutral_axis)*(theta+abs(psi_t))-l_r

    if psi_t < theta_straight_config and psi_t >= 0: # downwards bending with 0 < psi < psi_t	
        l_r = (r-dist_connectors_2_neutral_axis)*(theta-psi_t) #it is impossible that psi_t>theta for downwards bending
        delta_l_niti = (r+dist_niti_2_neutral_axis)*(theta-psi_t)-l_r

    return l_r, delta_l_niti #compensated_robotic_length, compensated_delta_l_niTi


def NSGC_API (X_t, image_number): #X_t is the target pose in the form of [x, y, z, psi] in mm and rad       
        #X_t = np.array([10  , 20 , 300  , np.deg2rad(-20)]) #coordinates as defined in the paper!
        x_t =   round(X_t[0],2) # target coordinate along the x axis in mm
        y_t =   round(X_t[1],2) # target coordinate along the y axis in mm
        z_t =   round(X_t[2],2) # target coordinate along the z axis in mm
        psi_t = round(X_t[3],3) # orientation at the target position about y_rotated axis in rad

        x_t_hat = np.sqrt(x_t**2 + y_t**2)  # target position in the rotated coordinate system
        alpha = np.arctan2(y_t, x_t) 
        print("alpha: ", alpha)

        X_e_hat, _, numerical_solution = NSGC(z_t, x_t_hat, psi_t, comments = True)
        
        if X_e_hat[0] == 0 and X_e_hat[1] == 0 and X_e_hat[2] == 0: #if there in no numerical solution
            print("x_t_hat is made negative ---------------------------------------------------------------")
            x_t_hat = -x_t_hat
            X_e_hat, _, numerical_solution = NSGC(z_t, x_t_hat, psi_t, comments = True)
            X_actual = np.array([X_e_hat[1]*np.cos(alpha+np.pi), X_e_hat[1]*np.sin(alpha+np.pi), X_e_hat[0], X_e_hat[2]]) #rotate back to the original coordinate system

        else:
            X_actual = np.array([X_e_hat[1]*np.cos(alpha), X_e_hat[1]*np.sin(alpha), X_e_hat[0], X_e_hat[2]])
        
        z_0 = numerical_solution[0]
        x_0_hat = numerical_solution[1]
        theta = numerical_solution[2] # angle in the range of 0 to 2pi
        r = numerical_solution[3] 

        X_t_hat = np.array([z_t, x_t_hat, psi_t])
        # print("X_t_hat: ", z_t, x_t_hat, np.rad2deg(psi_t))
        # print("X_e_hat: ", X_e_hat[0], X_e_hat[1], np.rad2deg(X_e_hat[2]))
        # print("X_t: ", X_t[0], X_t[1], X_t[2], np.rad2deg(X_t[3]))
        # print("X_actual: ", X_actual[0], X_actual[1], X_actual[2], np.rad2deg(X_actual[3]))
        # print("Distance: ", np.linalg.norm(np.array(X_t- X_actual)))
        # print("numerical_solution: z_0:", round(z_0, 3), ", x_0_hat:", round(x_0_hat, 3), ", theta in deg:", round(np.rad2deg(theta), 3), ", r:", round(r, 3 ))
    
        if X_e_hat[0] == 0 and X_e_hat[1] == 0 and X_e_hat[2] == 0: #if there in no numerical solution
            print("No solution found for the given target pose.")
            return
        
        l_r, delta_l_niti = get_robot_length_from_NSGC_solution(numerical_solution, X_t_hat)
        robot_parameters = np.array([alpha, theta, l_r, delta_l_niti])
        plot_3D(X_t, robot_parameters, image_number)


if __name__ == "__main__":
   
    #Define path for use case
    step_size = 1
    
    trajectory_list = []
    for i in range(-25, 25, step_size):
        trajectory_list.append(np.array([i, -25, 250, np.deg2rad(-30)]))

    for i in range(-25, 25, step_size):
        trajectory_list.append(np.array([25, i, 250, np.deg2rad(-50)]))

    for i in range(-50, -30, step_size):
        trajectory_list.append(np.array([25, 25, 250, np.deg2rad(i)]))

    for i in range(250, 280, step_size):
        trajectory_list.append(np.array([25, 25, i, np.deg2rad(-30)]))

    for i in range(-30, -20, step_size):
        trajectory_list.append(np.array([25, 25, 280, np.deg2rad(i)]))


    image_number = 0
    for i in range(len(trajectory_list)):
        image_number +=1
        print(i, "of", len(trajectory_list))
        NSGC_API(trajectory_list[i], image_number)
    
    
  
