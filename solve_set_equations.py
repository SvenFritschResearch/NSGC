from scipy.optimize import fsolve, least_squares, root, broyden1, broyden2, anderson, newton_krylov, anderson, linearmixing, diagbroyden, excitingmixing
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt


def solve_psi_limit(z_t, x_t_hat):
    '''Solve for the limit angle psi_limit using numerical methods. z_t is the target z-coordinate, x_t_hat is the target x-coordinate in the rotated coordinate system.
    Returns a list containing the numerical solution [x_0_hat, r, psi_limit]'''

    # Design parameters
    l_w = 88 # length of wrist
    l_d = 130 # length of distal part

    initial_guesses = []

    def equations(vars):
        x_0_hat, r, psi_limit = vars
        eq1 = r - x_0_hat
        eq2 = r**2 -(r - x_t_hat + np.sin(psi_limit)*l_w)**2 - (z_t - l_d -np.cos(psi_limit) * l_w)**2
        eq3 = psi_limit - np.arctan2((z_t - l_d -np.cos(psi_limit) * l_w),(r - x_t_hat + np.sin(psi_limit)*l_w))
        return [eq1, eq2, eq3]


    for x_0_hat_factor in [0, 1, 5]:
        for r_factor in [0, 1, 5]:
            for psi_factor in [0, np.deg2rad(20)]:
                initial_guesses.append([x_t_hat * x_0_hat_factor, (z_t - l_d) * r_factor, psi_factor])

    counter_it_fsolve = 0
    
    for guess in initial_guesses:
        counter_it_fsolve += 1

        # Attempt to solve the system of equations and catch warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Trigger all warnings

            numerical_solution = fsolve(equations, guess) # <----------------------------------------------------
            
            

            # Check for the specific warning
            if any(issubclass(w[i].category, RuntimeWarning) for i in range(len(w))):
                #print("Warning: fsolve did not converge to a solution with initial guess:", guess)
                continue

            if np.rad2deg(numerical_solution[2]) < 0:
                continue

        break

    #sanity check
    x_0_hat = numerical_solution[0]
    r = numerical_solution[1]
    psi_limit = numerical_solution[2]

    eq1_sanity_check = round(r - x_0_hat)
    eq2_sanity_check = round(r**2 -(r - x_t_hat + np.sin(psi_limit)*l_w)**2 - (z_t - l_d -np.cos(psi_limit) * l_w)**2)
    eq3_sanity_check = round(np.tan(psi_limit) - (z_t - l_d -np.cos(psi_limit) * l_w)/(r - x_t_hat + np.sin(psi_limit)*l_w))

        
    if (eq1_sanity_check == 0) and (eq2_sanity_check == 0) and (eq3_sanity_check == 0):
        print("sanity checks for psi_limit passed!")
        return numerical_solution
    else:
        print("sanity checks for psi_limit did NOT pass!")
        return [0,0,0]
    #print("---------------------")
    #print("numerical_solution for psi_limit: x_0_hat:", round(numerical_solution[0], 3), ", r:", round(numerical_solution[1], 3), ", psi_limit in deg:", round(np.rad2deg(numerical_solution[2]), 3))



def solve_NSGC(z_T, x_T_hat, psi,  comments = False):

    
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

    if abs(np.rad2deg(psi) - np.rad2deg(theta_straight_config)) < angle_threshold_straight: #the difference is less than 1deg
        #straight configuration (no bending)
        if comments: print("Straight configuration")
        theta = psi
        r = 0 #representing infinity
        z_e_hat = z_T
        x_e_hat = x_T_hat
        X_e_hat = [z_e_hat, x_e_hat, psi-np.deg2rad(angle_threshold_straight)]
        end_time = time.time()  # Zeitmessung beenden
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
        return [0,0,0], time_needed, numerical_solution
                
    
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
        


def get_motor_commands_from_geo_numerical_solution(numerical_solution, X_d_hat_array):

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
        

    #compensated_robotic_length = l_r + (theta * dist_rot_axis_2_connectors)
    #compensated_delta_l_niTi = delta_l_niti + (theta*dist_rot_axis_2_niti)

    #print("robotic_length compensated: ", compensated_robotic_length, "mm")
    #print("theta: ", np.rad2deg(theta), "deg")
    #print("delta_l_niTi compensated: ", compensated_delta_l_niTi, "mm")

    return l_r, delta_l_niti #compensated_robotic_length, compensated_delta_l_niTi



if __name__ == "__main__":
    
    x_0_hat, r, psi_limit = solve_psi_limit(300,110)
    print(np.rad2deg(psi_limit))

    # User input    
    # X_d = np.array([-27.77  , 90.48 , 344.94  , -0.84 ]) # target position in the original coordinate system
    # x_t =   round(X_d[0],2) # target coordinate in x
    # y_t =   round(X_d[1],2) 
    # z_t =   round(X_d[2],2) # target coordinate in z
    # psi_t = round(X_d[3],3) #-44 deg  # orientation at the target position about y_rotated axis

    # x_t_hat = np.sqrt(x_t**2 + y_t**2)  # target position in the rotated coordinate system
    # alpha = np.arctan2(y_t, x_t) 
    # print("alpha: ", alpha)

    # X_e_hat, _, numerical_solution = solve_NSGC(z_t, x_t_hat, psi_t, comments = True)
    # if X_e_hat[0] == 0 and X_e_hat[1] == 0 and X_e_hat[2] == 0: #if there in no numerical solution
    #     print("x_t_hat is made negative ---------------------------------------------------------------")
    #     x_t_hat = -x_t_hat
    #     X_e_hat, _, numerical_solution = solve_NSGC(z_t, x_t_hat, psi_t, comments = True)
    #     X_e_geo = np.array([X_e_hat[1]*np.cos(alpha+np.pi), X_e_hat[1]*np.sin(alpha+np.pi), X_e_hat[0], X_e_hat[2]]) #rotate back to the original coordinate system
    # #X_d = np.array([x_t_hat * np.cos(alpha), x_t_hat * np.sin(alpha), z_t, psi_t])
    # else:
    #     X_e_geo = np.array([X_e_hat[1]*np.cos(alpha), X_e_hat[1]*np.sin(alpha), X_e_hat[0], X_e_hat[2]])
    

    # print("X_d_hat: ", z_t, x_t_hat, np.rad2deg(psi_t))
    # print("X_e_hat: ", X_e_hat[0], X_e_hat[1], np.rad2deg(X_e_hat[2]))
    # print("X_d: ", X_d[0], X_d[1], X_d[2], np.rad2deg(X_d[3]))
    # print("X_e_geo: ", X_e_geo[0], X_e_geo[1], X_e_geo[2], np.rad2deg(X_e_geo[3]))
    # print("Distance: ", np.linalg.norm(np.array(X_d- X_e_geo)))
    # print("numerical_solution: z_0:", round(numerical_solution[0], 3), ", x_0_hat:", round(numerical_solution[1], 3), ", theta in deg:", round(np.rad2deg(numerical_solution[2]), 3), ", r:", round(numerical_solution[3], 3 ))
    
    
    #theta_straight_config = np.arctan2(x_t_hat,z_t)
    #get_motor_commands_from_geo_numerical_solution(numerical_solution, theta_straight_config)
    



