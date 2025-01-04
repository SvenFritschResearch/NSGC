from scipy.optimize import fsolve
import numpy as np
import time
import warnings


def solve_equ(z_T, x_T_hat, psi,  comments = False):

    
    theta_straight_config = np.arctan(x_T_hat/z_T)  # for straight configuration
   
    # Design parameters
    l_e = 88 # length of wrist
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
        end_time = time.time()  # Zeitmessung beenden
        numerical_solution = [0,0,0,0]
        return X_e_hat , end_time - start_time, numerical_solution 


    if psi > theta_straight_config: 
        #upwards bending
        if comments: print("Upwards bending")
        def equations(vars):
            z_0, x_0_hat, theta, r = vars
            eq1 = r**2 -(z_T-l_e*np.cos(psi)-z_0)**2 - (x_0_hat - x_T_hat + np.sin(psi)*l_e)**2
            eq2 = r**2 -(l_d*np.cos(theta)-z_0)**2 - (x_0_hat - np.sin(theta)*l_d)**2
            eq3 = np.tan(psi) - (z_T - z_0 - np.cos(psi)*l_e)/(x_0_hat - x_T_hat + np.sin(psi)*l_e)
            eq4 = np.tan(theta) - (np.cos(theta)*l_d-z_0)/(x_0_hat-np.sin(theta)*l_d)
            return [eq1, eq2, eq3, eq4]

    if psi < theta_straight_config:
        #downwards bending
        if comments: print("Downwards bending")
        def equations(vars):
            z_0, x_0_hat, theta, r = vars
            eq1 = r**2 -(z_0 - z_T + np.cos(psi)*l_e)**2 - (-x_0_hat + x_T_hat - np.sin(psi)*l_e)**2
            eq2 = r**2 -(z_0 - np.cos(theta)*l_d)**2 - (-x_0_hat + np.sin(theta)*l_d)**2
            eq3 = np.tan(psi) - (z_0 - (z_T - np.cos(psi)*l_e))/(-x_0_hat + x_T_hat - np.sin(psi)*l_e)
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


    def sanity_checks_upwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_e):
        """Perform sanity checks for upwards bending."""
        z_0, x_0_hat, theta, r = numerical_solution
        eq1 = round(r**2 - (z_T - l_e * np.cos(psi) - z_0)**2 - (x_0_hat - x_T_hat + np.sin(psi) * l_e)**2, 3)
        eq2 = round(r**2 - (l_d * np.cos(theta) - z_0)**2 - (x_0_hat - np.sin(theta) * l_d)**2, 3)
        eq3 = round(np.tan(psi) - (z_T - z_0 - np.cos(psi) * l_e) / (x_0_hat - x_T_hat + np.sin(psi) * l_e), 3)
        eq4 = round(np.tan(theta) - (np.cos(theta) * l_d - z_0) / (x_0_hat - np.sin(theta) * l_d), 3)
        return eq1, eq2, eq3, eq4

    def sanity_checks_downwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_e):
        """Perform sanity checks for downwards bending."""
        z_0, x_0_hat, theta, r = numerical_solution
        eq1 = round(r**2 - (z_0 - z_T + np.cos(psi) * l_e)**2 - (-x_0_hat + x_T_hat - np.sin(psi) * l_e)**2, 3)
        eq2 = round(r**2 - (z_0 - np.cos(theta) * l_d)**2 - (-x_0_hat + np.sin(theta) * l_d)**2, 3)
        eq3 = round(np.tan(psi) - (z_0 - (z_T - np.cos(psi) * l_e)) / (-x_0_hat + x_T_hat - np.sin(psi) * l_e), 3)
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

                # Check conditions I-III
                # condition I
                if not (MIN_THETA <= theta <= MAX_THETA):
                    if comments: print(f"Theta = {round(np.rad2deg(theta))} deg out of bounds --> next try")
                    continue

                # condition II
                if psi < theta_straight_config and theta < theta_straight_config:  # Downwards bending
                    if comments: print("Theta too small for downwards bending --> next try")
                    continue

                # condition III
                if psi > theta_straight_config and np.sin(theta) * l_d > x_T_hat - np.sin(psi) * l_e:  # Upwards bending
                    if comments: print("Distal part too high --> next try")
                    continue

                # Sanity checks
                if psi > theta_straight_config:  # Upwards bending
                    eqs = sanity_checks_upwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_e)
                else:  # Downwards bending
                    eqs = sanity_checks_downwards(numerical_solution, psi, x_T_hat, z_T, l_d, l_e)

                if all(abs(eq) < 1e-6 for eq in eqs):
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
        z_e_hat = np.cos(theta)*l_d + np.sin(theta)*r - np.sin(abs(psi))*r + np.cos(psi)*l_e
        x_e_hat = np.sin(theta)*l_d - np.cos(theta)*r + np.cos(psi)*r + np.sin(psi)*l_e # for psi < 0 --> sin(psi) = - sin(abs(psi))
    if psi < 0: #downward bending & in this case z_0 is to the left of the circle end point where it connects to the wrist
        z_e_hat = np.cos(theta)*l_d + np.sin(theta)*r + np.sin(abs(psi))*r + np.cos(psi)*l_e
        x_e_hat = np.sin(theta)*l_d - np.cos(theta)*r + np.cos(psi)*r + np.sin(psi)*l_e # for psi < 0 --> sin(psi) = - sin(abs(psi))
    if psi > theta_straight_config: # upwards bending
        z_e_hat = np.cos(theta)*l_d - np.sin(theta)*r + np.sin(psi)*r + np.cos(psi)*l_e
        x_e_hat = np.sin(theta)*l_d + np.cos(theta)*r - np.cos(psi)*r + np.sin(psi)*l_e
        
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
        

    compensated_robotic_length = l_r + (theta * dist_rot_axis_2_connectors)
    compensated_delta_l_niTi = delta_l_niti + (theta*dist_rot_axis_2_niti)

    #print("robotic_length compensated: ", compensated_robotic_length, "mm")
    #print("theta: ", np.rad2deg(theta), "deg")
    #print("delta_l_niTi compensated: ", compensated_delta_l_niTi, "mm")

    return l_r, delta_l_niti #compensated_robotic_length, compensated_delta_l_niTi



if __name__ == "__main__":
    
    # User input  X_d = [x_t, y_t, z_t, psi_t] with x_t, y_t, z_t in mm and psi_t in deg 
    X_d = np.array([50, 70 ,400  , np.deg2rad(65) ]) # target position in the original coordinate system
    x_t =   round(X_d[0],2) # target coordinate in global x
    y_t =   round(X_d[1],2) # target coordinate in global y
    z_t =   round(X_d[2],2) # target coordinate in global z
    psi_t = round(X_d[3],3) # orientation at the target position about y_rotated axis (local)

    x_t_hat = np.sqrt(x_t**2 + y_t**2)  # target position in the rotated coordinate system
    print("x_t_hat: ", x_t_hat)
    alpha = np.arctan2(y_t, x_t) 
    print("alpha: ", np.rad2deg(alpha))

    X_e_hat, _, numerical_solution = solve_equ(z_t, x_t_hat, psi_t, comments = False)
    if X_e_hat[0] == 0 and X_e_hat[1] == 0 and X_e_hat[2] == 0: #if there in no numerical solution
        print("x_t_hat is made negative ---------------------------------------------------------------")
        x_t_hat = -x_t_hat
        psi_t = -psi_t
        X_e_hat, _, numerical_solution = solve_equ(z_t, x_t_hat, psi_t, comments = False)
        X_e_geo = np.array([X_e_hat[1]*np.cos(alpha+np.pi), X_e_hat[1]*np.sin(alpha+np.pi), X_e_hat[0], X_e_hat[2]]) #rotate back to the original coordinate system

    else:
        X_e_geo = np.array([X_e_hat[1]*np.cos(alpha), X_e_hat[1]*np.sin(alpha), X_e_hat[0], X_e_hat[2]])
    

    print("X_d_hat: ", z_t, x_t_hat, np.rad2deg(psi_t))
    print("X_e_hat: ", X_e_hat[0], X_e_hat[1], np.rad2deg(X_e_hat[2]))
    print("X_d: ", X_d[0], X_d[1], X_d[2], np.rad2deg(X_d[3]))
    print("X_e_geo: ", X_e_geo[0], X_e_geo[1], X_e_geo[2], np.rad2deg(X_e_geo[3]))
    print("Distance: ", np.linalg.norm(np.array(X_d- X_e_geo)))
    print("numerical_solution: z_0:", round(numerical_solution[0], 3), ", x_0_hat:", round(numerical_solution[1], 3), ", theta in deg:", round(np.rad2deg(numerical_solution[2]), 3), ", r:", round(numerical_solution[3], 3 ))
    
    
    #theta_straight_config = np.arctan2(x_t_hat,z_t)
    #get_motor_commands_from_geo_numerical_solution(numerical_solution, theta_straight_config)
    



