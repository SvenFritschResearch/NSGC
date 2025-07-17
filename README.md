the "NSGC.py" contains the code for the NSGC algorithm: Input the target trajectory in 3D space [x_t, y_t, z_t, psi_t] the main function. Keep in mind, that psi_t is defined in the local end-effector frame. 
If you have questions, please feel free to reach out to me: research@sven-fritsch.de

The "in_silico_experiments.py" file contains the comparison of different IKMs (basic Jacobian with k=0.5 and k=0.4, adaptive damping and quadratic programming) and it calls "NSGC.py" to get the NSGC result and it calls "visualization_functions.py" to visualize the results.
