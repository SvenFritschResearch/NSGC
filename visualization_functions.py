from in_silico_experiments import *
import matplotlib.pyplot as plt
import os
import re
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import iqr


def numerically_sorted(input_list):
    def extract_number(s):
        match = re.search(r'\d+', s)
        if match:
            return int(match.group())
        return float('inf')  # Return infinity if no number found

    sorted_list = sorted(input_list, key=extract_number)
    return sorted_list


def images_to_video(image_folder, num_moving_pics):

    images = numerically_sorted([filename for filename in os.listdir(image_folder) if filename.endswith('.png')])
    
    if not images:
        raise ValueError("No PNG images found in the specified folder.")


    beginning_pics = images[:num_moving_pics]
    middle_pics = images[num_moving_pics:-num_moving_pics]
    end_pics = images[-num_moving_pics:]

    video_name_1 = r"C:\Dateien\4_Avatera_Medical\Paper_2\Python_code\numeric_computation\inverse_kinematic_model_1.mp4"
    frame_rate_1 = 30
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    video1 = cv2.VideoWriter(video_name_1, fourcc, frame_rate_1, (width, height))

    for image in beginning_pics:
        video1.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video1.release()

    video_name_2 = r"C:\Dateien\4_Avatera_Medical\Paper_2\Python_code\numeric_computation\inverse_kinematic_model_2.mp4"
    frame_rate_2 = 5 #for minimization 
    #frame_rate_2 = 30 #for inverse differential kinematics
    video2 = cv2.VideoWriter(video_name_2, fourcc, frame_rate_2, (width, height))

    for image in middle_pics:
        video2.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video2.release()


    video_name_3 = r"C:\Dateien\4_Avatera_Medical\Paper_2\Python_code\numeric_computation\inverse_kinematic_model_3.mp4"
    frame_rate_3 = 30
    video3 = cv2.VideoWriter(video_name_3, fourcc, frame_rate_3, (width, height))

    for image in end_pics:
        video3.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video3.release()

    # Load the video clips with different frame rates
    video1 = VideoFileClip(video_name_1)
    video2 = VideoFileClip(video_name_2)
    video3 = VideoFileClip(video_name_3)

    # Combine the videos horizontally
    clips = [video1, video2, video3]
    # Write the combined video to a file
    combined_video_name = r"C:\Dateien\4_Avatera_Medical\Paper_2\Python_code\numeric_computation\inverse_kinematic_model.mp4"
    final_video = concatenate_videoclips(clips, method="compose")

    # Write the final video to the specified file
    final_video.write_videofile(combined_video_name, codec="libx264")

    # Close the video objects when you're done
    video1.close()
    video2.close()
    video3.close()
    final_video.close()
    
    print("Videos were created successfully.")


def camera_rotation(fk_list, desired_pose, num_moving_pics, i, convergence_time):
    #this is the beginning and end part of the vid (moving camera and static robot)
    #i gives the number of iterations, i.e. #frames -> helps with numbering the pics 
    #j is a couter to give frames with camera rotation for the start and end config

    for j in range(num_moving_pics):
            
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)

        ax3d = fig.add_subplot(111, projection='3d')  # Subplot 1

        ax3d.set_xlim(0, 800)
        ax3d.set_ylim(-400, 400)
        ax3d.set_zlim(-400, 400)
        ax3d.set_xlabel('X Axis')
        ax3d.set_ylabel('Y Axis')
        ax3d.set_zlabel('Z Axis')

        x_list = [mat[0, 3] for mat in fk_list[0]]
        y_list = [mat[1, 3] for mat in fk_list[0]]
        z_list = [mat[2, 3] for mat in fk_list[0]] 

        if i == 0:
            ax3d.plot(x_list, y_list, z_list, color="blue", marker=".", label=f"robot configuration at iteration {0}")
            # add coordinate frame to visualize orientation of end-effector
            R_endeffector = fk_list[0][-1][:3, :3]
            translation_endeffctor = fk_list[0][-1][:3, 3]
            scale= 60
            ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 0], color='orange', label='X-Axis end', length=scale, linewidth=3)
            ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 1], color='green', label='Y-Axis end', length=scale, linewidth=3)
            ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 2], color='purple', label='Z-Axis end', length=scale, linewidth=3)

        if i == len(fk_list)-1:
            ax3d.plot(x_list, y_list, z_list, color="lightblue", marker=".", label=f"robot configuration at iteration {0}")
            x_list_start = [mat[0, 3] for mat in fk_list[-1]]
            y_list_start = [mat[1, 3] for mat in fk_list[-1]]
            z_list_start = [mat[2, 3] for mat in fk_list[-1]]
            ax3d.plot(x_list_start, y_list_start, z_list_start, color="blue", marker=".", label="robot start configuration")

            # add coordinate frame to visualize orientation of end-effector
            R_endeffector = fk_list[-1][-1][:3, :3]
            translation_endeffctor = fk_list[-1][-1][:3, 3] 
            scale= 60
            ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 0], color='orange', label='X-Axis end', length=scale, linewidth=3)
            ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 1], color='green', label='Y-Axis end', length=scale, linewidth=3)
            ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 2], color='purple', label='Z-Axis end', length=scale, linewidth=3)

        ax3d.scatter(0, 0, 0, color="orange", marker=".", label= "trocar center point") #origin
        ax3d.scatter(desired_pose[0], desired_pose[1], desired_pose[2], color="green", marker="o", 
                        label=f"desired end-effector pose with \n x={desired_pose[0]}mm, y={desired_pose[1]}mm, z={desired_pose[2]}mm, \n roll={round(np.rad2deg(desired_pose[3]),2)}{chr(176)}, pitch={round(np.rad2deg(desired_pose[4]),2)}{chr(176)}, yaw={round(np.rad2deg(desired_pose[5]),2)}{chr(176)}")
        
        
        ax3d.legend()
        ax3d.set_title(f'Inverse kinematics using non-linear minimization and quaternions for position \n and orientation control of a 2 segment hyper-redundant robot with {round(convergence_time,3)} s convergence time')
        #ax3d.set_title(f'Inverse differential kinematics and euler angles for position and orientation \n control of a 2 segment hyper-redundant robot with {round(convergence_time,3)} s convergence time')
        

        elevation_total = 30
        azimuthal_total = -60
        if i == 0:
            elevation = j/num_moving_pics *elevation_total
            azimuthal = j/num_moving_pics * azimuthal_total
        if i == len(fk_list)-1:
            elevation = elevation_total-j/num_moving_pics *elevation_total
            azimuthal = azimuthal_total - j/num_moving_pics * azimuthal_total
        ax3d.view_init(elevation, azimuthal)
        

        # add coordinate frame to visualize orientation of desired pose
        position = desired_pose[:3]
        orientation_euler = desired_pose[3:]
        rotation_matrix = Rotation.from_euler('xyz', orientation_euler, degrees=False).as_matrix()
        scale = 120  # Adjust the scale factor as needed
        ax3d.quiver(*position, *rotation_matrix[:, 0], color='orange', label='X-Axis des', length=scale, linewidth=.9)
        ax3d.quiver(*position, *rotation_matrix[:, 1], color='green', label='Y-Axis des', length=scale, linewidth=.9)
        ax3d.quiver(*position, *rotation_matrix[:, 2], color='purple', label='Z-Axis des', length=scale, linewidth=.9)


        image_directory = r"C:\Dateien\4_Avatera_Medical\Paper_2\Python_code\numeric_computation\plot_folder"
        if i == 0:
            file_name = f"plot_{j}.png"
        else:
            file_name = f"plot_{j+i+num_moving_pics}.png"
        full_file_path = os.path.join(image_directory, file_name)
        dpi = 300
        fig.savefig(full_file_path, dpi=dpi)
        plt.close()
    

def mp4_func_with_error(fk_list, desired_pose, error_list, error_threshold, convergence_time):
    #this is the middle part of the vid (the moving CR and error plot)
    # fk_list is a list of sublists, each sublist includes all matrices from first to last robot segment at iteration i
    # fk_list[0] gives the fk_list at itertation 0 (ie. start config), fk_list[-1] gives the fk_list at the last iteration (ie. the end config after termination)

    num_moving_pics = 120 # number of pics that are done with moving camera in the beginning and end of the clip

    for i in range(len(fk_list)):
        if i == 0:
            camera_rotation(fk_list, desired_pose, num_moving_pics, i, convergence_time)

        fig = plt.figure(figsize=(12, 6), constrained_layout=True)

        ax3d = fig.add_subplot(121, projection='3d')  # Subplot 1

        ax3d.set_xlim(0, 800)
        ax3d.set_ylim(-400, 400)
        ax3d.set_zlim(-400, 400)
        ax3d.set_xlabel('X Axis')
        ax3d.set_ylabel('Y Axis')
        ax3d.set_zlabel('Z Axis')

        x_list = [mat[0, 3] for mat in fk_list[i]]
        y_list = [mat[1, 3] for mat in fk_list[i]]
        z_list = [mat[2, 3] for mat in fk_list[i]]
        x_list_start = [mat[0, 3] for mat in fk_list[0]]
        y_list_start = [mat[1, 3] for mat in fk_list[0]]
        z_list_start = [mat[2, 3] for mat in fk_list[0]]
        ax3d.scatter(0, 0, 0, color="orange", marker=".", label= "trocar center point") #origin
        ax3d.scatter(desired_pose[0], desired_pose[1], desired_pose[2], color="green", marker="o", 
                     label=f"desired end-effector pose with \n x={desired_pose[0]}mm, y={desired_pose[1]}mm, z={desired_pose[2]}mm, \n roll={round(np.rad2deg(desired_pose[3]),2)}{chr(176)}, pitch={round(np.rad2deg(desired_pose[4]),2)}{chr(176)}, yaw={round(np.rad2deg(desired_pose[5]),2)}{chr(176)}")
        ax3d.plot(x_list_start, y_list_start, z_list_start, color="lightblue", marker=".", label="robot start configuration")
        ax3d.plot(x_list, y_list, z_list, color="blue", marker=".", label=f"robot configuration at iteration {i}")
        ax3d.legend()
        
        # add coordinate frame to visualize orientation of end-effector
        R_endeffector = fk_list[i][-1][:3, :3]
        translation_endeffctor = fk_list[i][-1][:3, 3]
        scale= 60
        ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 0], color='orange', label='X-Axis end', length=scale, linewidth=3)
        ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 1], color='green', label='Y-Axis end', length=scale, linewidth=3)
        ax3d.quiver(*translation_endeffctor, *R_endeffector[:, 2], color='purple', label='Z-Axis end', length=scale, linewidth=3)

        # add coordinate frame to visualize orientation of desired pose
        position = desired_pose[:3]
        orientation_euler = desired_pose[3:]
        rotation_matrix = Rotation.from_euler('xyz', orientation_euler, degrees=False).as_matrix()
        scale = 120  # Adjust the scale factor as needed
        ax3d.quiver(*position, *rotation_matrix[:, 0], color='orange', label='X-Axis des', length=scale, linewidth=.9)
        ax3d.quiver(*position, *rotation_matrix[:, 1], color='green', label='Y-Axis des', length=scale, linewidth=.9)
        ax3d.quiver(*position, *rotation_matrix[:, 2], color='purple', label='Z-Axis des', length=scale, linewidth=.9)
               
        ax2d = fig.add_subplot(122)
        ax2d.set_xlim(0, len(error_list))
        ax2d.set_ylim(0, max(error_list) + 10)  # Adjust the y-limit as needed
        ax2d.set_xlabel('Iteration')
        ax2d.set_ylabel('Error')
        ax2d.set_title(f'Mean Squared Error was {round(error_list[i],3)} at iteration {i}')

        # 2D Error Plot
        
        ax2d.plot(range(i), error_list[:i], color="red", marker=".", label="error")
        ax2d.axhline(y=error_threshold, color="orange", linestyle="--", label="error threshold")
        ax2d.legend()
        
        plt.suptitle(f"Inverse kinematics using non-linear minimization and quaternions for position \n and orientation control of a 2 segment hyper-redundant robot with {round(convergence_time,3)} s convergence time")
        #plt.suptitle(f'Inverse differential kinematics and euler angles for position and orientation \n control of a 2 segment hyper-redundant robot with {round(convergence_time,3)} s convergence time')
        

        image_directory = r"C:\Dateien\4_Avatera_Medical\Paper_2\Python_code\numeric_computation\plot_folder"
        file_name = f"plot_{num_moving_pics + i}.png"
        full_file_path = os.path.join(image_directory, file_name)
        dpi = 300
        fig.savefig(full_file_path, dpi=dpi)
        plt.close()




        if i == len(fk_list)-1:
            camera_rotation(fk_list, desired_pose, num_moving_pics, i, convergence_time)
            #last frame
            images_to_video(image_directory, num_moving_pics)


def plot_func(fk, fk1=None, fk2=None, desired_pose=None):
    #fk is a list of transformation matices starting at the base

    print("------------------------------------------------------")

    x_list = [mat[0,3] for mat in fk]
    y_list = [mat[1,3] for mat in fk]
    z_list = [mat[2,3] for mat in fk]

   
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    #ax.scatter3D(x_list, y_list, z_list, color = "green", marker=".")
    #ax.scatter3D(0,0,0, color = "red", marker="o")

    
   

    ax.plot(x_list, y_list, z_list, color = "chocolate", marker=".", label = "IKM Jacobian Inverse")
    
    if fk1 is not None:
        x_list_1 = [mat[0,3] for mat in fk1]
        y_list_1 = [mat[1,3] for mat in fk1]
        z_list_1 = [mat[2,3] for mat in fk1]
        ax.plot(x_list_1, y_list_1, z_list_1, color = "purple", marker=".", label = "Start configuration")

    if fk2 is not None:
        x_list_2 = [mat[0,3] for mat in fk2]
        y_list_2 = [mat[1,3] for mat in fk2]
        z_list_2 = [mat[2,3] for mat in fk2]
        ax.plot(x_list_2, y_list_2, z_list_2, color = "black", marker=".", label = "fk_minimize")

    if desired_pose is not None :
        ax.scatter3D(desired_pose[0], desired_pose[1], desired_pose[2], color="red", marker="o", label="target pose")
    
    ax.axes.set_xlim3d(left=0, right=700) 
    ax.axes.set_ylim3d(bottom=-350, top=350) 
    ax.axes.set_zlim3d(bottom=-350, top=350)
    

    # add coordinate frame to visualize orientation of the distal end of the distal wrist (not taking into account rho_distal)
    # R_2 = fk[-2][:3, :3]
    # translation_2 = fk[-2][:3, 3]
    # scale= 50
    # ax.quiver(*translation_2, *R_2[:, 0], color='coral', label='X-Axis dw', length=scale)
    # ax.quiver(*translation_2, *R_2[:, 1], color='yellow', label='Y-Axis dw', length=scale)
    # ax.quiver(*translation_2, *R_2[:, 2], color='purple', label='Z-Axis dw', length=scale)

    
    # add coordinate frame to visualize orientation of end-effector 
    R = fk[-1][:3, :3]
    translation = fk[-1][:3, 3]
    scale = 70
    ax.quiver(*translation, *R[:, 0], color='blue', label='X-Axis EE', length=scale, linewidth=3)
    ax.quiver(*translation, *R[:, 1], color='green', label='Y-Axis EE', length=scale, linewidth=3)
    ax.quiver(*translation, *R[:, 2], color='black', label='Z-Axis EE', length=scale, linewidth=3)

    if desired_pose is not None:

        ax.plot([0, 0], [0, desired_pose[1]], [0, 0], color='grey', linestyle='--')
        ax.plot([0, desired_pose[0]], [desired_pose[1], desired_pose[1]], [0,0], color='grey', linestyle='--')
        ax.plot([desired_pose[0], desired_pose[0]], [desired_pose[1], desired_pose[1]], [0, desired_pose[2]], color='grey', linestyle='--')
            
        # add coordinate frame to visualize orientation of desired pose
        position = desired_pose[:3]
        # orientation_euler = desired_pose[3:]
        # rotation_matrix = Rotation.from_euler('XYZ', orientation_euler, degrees=False).as_matrix() #works as well
        psi_x_des = desired_pose[3]
        psi_y_des = desired_pose[4]
        psi_z_des = desired_pose[5]


        R_z_des = np.array([[np.cos(psi_z_des), -np.sin(psi_z_des), 0],
                            [np.sin(psi_z_des),  np.cos(psi_z_des), 0],
                            [0                ,  0                 ,1]])
        R_y_des = np.array([[np.cos(psi_y_des) , 0, np.sin(psi_y_des)],
                            [0                 , 1,                 0],
                            [-np.sin(psi_y_des), 0, np.cos(psi_y_des)]])
        R_x_des = np.array([[1, 0                ,                  0],
                            [0, np.cos(psi_x_des), -np.sin(psi_x_des)],
                            [0, np.sin(psi_x_des), np.cos(psi_x_des)]])
        
        rotation_matrix = R_x_des @ R_y_des @ R_z_des 


        scale = 120  # Adjust the scale factor as needed
        ax.quiver(*position, *rotation_matrix[:, 0], color='lightblue', label='X-Axis des', length=scale)
        ax.quiver(*position, *rotation_matrix[:, 1], color='lightgreen', label='Y-Axis des', length=scale)
        ax.quiver(*position, *rotation_matrix[:, 2], color='grey', label='Z-Axis des', length=scale)


    
    # Label the axes (optional)
    ax.set_xlabel('X Axis') #('Z Axis')
    ax.set_ylabel('Y Axis') #('X Axis')
    ax.set_zlabel('Z Axis') #('Y Axis')
    ax.view_init(elev=122, azim=-90)
    
    ax.legend()
    plt.show()





def plot_func_2D(fk, fk1=None, fk2=None, desired_pose=None):
    # fk is a list of transformation matrices starting at the base

    print("------------------------------------------------------")

    z_list = [mat[0, 2] for mat in fk]
    x_list = [mat[1, 2] for mat in fk]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(0, 0, color="red", marker="o")
    ax.plot(z_list, x_list, color="green", marker=".", label="fk_differential")

    R = fk[-1][:2, :2]
    translation = fk[-1][:2, 2]
    ax.quiver(*translation, *R[:, 0], color='coral', label='psi')


    if fk1 is not None:
        z_list_1 = [mat[0, 2] for mat in fk1]
        x_list_1 = [mat[1, 2] for mat in fk1]
        ax.plot(z_list_1, x_list_1, color="blue", marker=".", label="fk_minimize_quat")

    if fk2 is not None:
        z_list_2 = [mat[0, 2] for mat in fk2]
        x_list_2 = [mat[1, 2] for mat in fk2]
        ax.plot(z_list_2, x_list_2, color="black", marker=".", label="fk_minimize")

    if desired_pose is not None:
        ax.scatter(desired_pose[0], desired_pose[1], color="orange", marker="o", label="target pose")

    # ax.set_xlim(left=0, right=700)
    # ax.set_ylim(bottom=-350, top=350)

    # Label the axes (optional)
    ax.set_xlabel('z Axis')
    ax.set_ylabel('x Axis')
    ax.axis("equal")

    ax.legend()
    # ax.set_xlim(-200, 300)
    # ax.set_ylim(-400, 400)
    plt.show()
    #plt.pause(2)
    #plt.close()


def plot_func_1(fk_start, start_pose, fk_end, desired_pose):
    #fk is a list of transformation matices starting at the base


    x_list_start = [mat[0,3] for mat in fk_start]
    y_list_start = [mat[1,3] for mat in fk_start]
    z_list_start = [mat[2,3] for mat in fk_start]

   
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    #ax.scatter3D(x_list, y_list, z_list, color = "green", marker=".")
    ax.scatter3D(0,0,0, color = "red", marker="o")
    ax.plot(x_list_start, y_list_start, z_list_start, color = "green", marker=".", label = "Start configuration")
        
    x_list_end = [mat[0,3] for mat in fk_end]
    y_list_end = [mat[1,3] for mat in fk_end]
    z_list_end = [mat[2,3] for mat in fk_end]
    ax.plot(x_list_end, y_list_end, z_list_end, color = "blue", marker=".", label = "End configuration")

    ax.scatter3D(desired_pose[0], desired_pose[1], desired_pose[2], color="orange", marker=".", label="target pose")
    
    ax.axes.set_xlim3d(left=0, right=700) 
    ax.axes.set_ylim3d(bottom=-350, top=350) 
    ax.axes.set_zlim3d(bottom=-350, top=350)
    
    
    # add coordinate frame to visualize orientation of end-effector
    R = fk_end[-1][:3, :3]
    translation = fk_end[-1][:3, 3]
    scale= 50
    ax.quiver(*translation, *R[:, 0], color='blue', label='X-Axis end', length=scale)
    ax.quiver(*translation, *R[:, 1], color='green', label='Y-Axis end', length=scale)
    ax.quiver(*translation, *R[:, 2], color='purple', label='Z-Axis end', length=scale)

    # add coordinate frame to visualize orientation of desired pose
    position = desired_pose[:3]
    orientation_euler = desired_pose[3:]
    rotation_matrix = Rotation.from_euler('xyz', orientation_euler, degrees=False).as_matrix()
    scale = 100  # Adjust the scale factor as needed
    ax.quiver(*position, *rotation_matrix[:, 0], color='lightblue', label='X-Axis des', length=scale)
    ax.quiver(*position, *rotation_matrix[:, 1], color='lightgreen', label='Y-Axis des', length=scale)
    ax.quiver(*position, *rotation_matrix[:, 2], color='black', label='Z-Axis des', length=scale)

    position = start_pose[:3]
    orientation_euler = start_pose[3:]
    rotation_matrix = Rotation.from_euler('xyz', orientation_euler, degrees=False).as_matrix()
    scale = 100  # Adjust the scale factor as needed
    ax.quiver(*position, *rotation_matrix[:, 0], color='lightblue', label='X-Axis des', length=scale)
    ax.quiver(*position, *rotation_matrix[:, 1], color='lightgreen', label='Y-Axis des', length=scale)
    ax.quiver(*position, *rotation_matrix[:, 2], color='black', label='Z-Axis des', length=scale)
    
    # Label the axes (optional)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    ax.legend()
    plt.show()


def plot_poses_as_scatter(X_d_array, X_e_IK_big_k_only_correct_convergence, X_e_IK_big_k_no_convergence, X_e_IK_small_k_only_correct_convergence, 
                              X_e_IK_small_k_no_convergence, X_e_NSGC_only_correct_convergence, X_e_NSGC_no_convergence):


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_d_array[:, 0], X_d_array[:, 1], X_d_array[:, 2], c='blue', s=15, marker='o', label='Target Pose')
        
    ax.scatter(X_e_IK_big_k_only_correct_convergence[:, 0], X_e_IK_big_k_only_correct_convergence[:, 1], X_e_IK_big_k_only_correct_convergence[:, 2], c='red', s=120, marker='x', label='Jacobian based IKM success (k=0.5)', alpha=0.5)
    ax.scatter(X_e_IK_small_k_only_correct_convergence[:, 0], X_e_IK_small_k_only_correct_convergence[:, 1], X_e_IK_small_k_only_correct_convergence[:, 2], c='orange', s=120, marker='x', label='Jacobian based IKM success (k=0.4)', alpha=0.5)
    ax.scatter(X_e_NSGC_only_correct_convergence[:, 0], X_e_NSGC_only_correct_convergence[:, 1], X_e_NSGC_only_correct_convergence[:, 2], c='green', s=120, marker='x', label='NSGC success', alpha=0.5)


    if X_e_IK_big_k_no_convergence is not None and X_e_IK_small_k_no_convergence.size > 0:
        ax.scatter(X_e_IK_big_k_no_convergence[:, 0], X_e_IK_big_k_no_convergence[:, 1], X_e_IK_big_k_no_convergence[:, 2], c='black', s=160, marker='+', label='Jacobian based IKM failure (k=0.5)', alpha=0.5)
    if X_e_IK_small_k_no_convergence is not None and X_e_IK_small_k_no_convergence.size > 0:
        ax.scatter(X_e_IK_small_k_no_convergence[:, 0], X_e_IK_small_k_no_convergence[:, 1], X_e_IK_small_k_no_convergence[:, 2], c='orange', s=160, marker='+', label='Jacobian based IKM failure (k=0.4)', alpha=0.5)
    if X_e_NSGC_no_convergence is not None and X_e_IK_small_k_no_convergence.size > 0:
        ax.scatter(X_e_NSGC_no_convergence[:, 0], X_e_NSGC_no_convergence[:, 1], X_e_NSGC_no_convergence[:, 2], c='yellow', s=160, marker='+', label='NSGC failure', alpha=0.5)
    #ax.scatter(0, 0, 0, c='green', marker='^', s=50, label='Origin')

    # Customize the plot (labels, title, etc.)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=18, framealpha=0)
    #ax.set_title('Comparison of Jacobian based IKM and NSGC', fontsize=16)
    ax.axis("equal")
    p = 200
    ax.set_xlim( 0, 2*p)
    ax.set_ylim(-p, p  )
    ax.set_zlim(-p, p  )
    plt.legend()
    ax.set_xlabel('Z [mm]', fontsize=14)
    ax.set_ylabel('Y [mm]', fontsize=14) 
    ax.invert_yaxis()
    ax.set_zlabel('X [mm]', fontsize=14)

    #plt.show()


def plot_error(error_list, error_threshold, title="Error Plot"):
    counter = range(0, len(error_list))
    plt.plot(counter, error_list, marker=".", label="Error")
    plt.plot(counter, np.full(len(error_list), error_threshold), color="red", label="Error Threshold")
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='upper center')
    plt.show()


def plot_two_positions(desired_position, actual_position):

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    ax.scatter3D(desired_position[0], desired_position[1], desired_position[2], color="green", marker=".")
    ax.scatter3D(actual_position[0], actual_position[1], actual_position[2], color="red", marker=".")
    plt.show()


def plot_nth_joint(fk, n):
    # nth joint counting from the end-effector

    x_list = [mat[0,3] for mat in fk]
    y_list = [mat[1,3] for mat in fk]
    z_list = [mat[2,3] for mat in fk]

    nth_joint = fk[-n]

   
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    ax.scatter3D(0,0,0, color = "red", marker="o")
    ax.plot(x_list, y_list, z_list, color = "green", marker=".", label = "fk_differential")

    ax.scatter3D(nth_joint[0,3], nth_joint[1,3], nth_joint[2,3], color="red", marker="o")
    plt.show()


def plot_convex_hull(combined_points_1, combined_vertices_1, combined_simplices_1):

    # Create a 3D plot to visualize the convex hull mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original point cloud
    #ax.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2], c='blue', marker='.', label='Point Cloud 1')
    #ax.scatter(position_list_2[:, 0], position_list_2[:, 1], position_list_2[:, 2], c='red', marker='.', label='Point Cloud 2')
    # ax.scatter(position_list_3[:, 0], position_list_3[:, 1], position_list_3[:, 2], c='green', marker='.', label='Point Cloud 3')
    # ax.scatter(position_list_4[:, 0], position_list_4[:, 1], position_list_4[:, 2], c='black', marker='.', label='Point Cloud 4')

    

    for simplex in combined_simplices_1:
        ax.plot(combined_points_1[simplex][:,0],combined_points_1[simplex][:,1],combined_points_1[simplex][:,2], 'r-')

    for simplex in combined_simplices_1:
        ax.plot(combined_vertices_1[simplex][:,0],combined_vertices_1[simplex][:,1],combined_vertices_1[simplex][:,2], 'g-')

    # for simplex in combined_simplices_3:
    #     ax.plot(combined_points_3[simplex][:,0],combined_points_3[simplex][:,1],combined_points_3[simplex][:,2], 'b-')

    # for simplex in combined_simplices_4:
    #     ax.plot(combined_points_4[simplex][:,0],combined_points_4[simplex][:,1],combined_points_4[simplex][:,2], 'y-')

    #print(combined_simplices)
    #ax.scatter(combined_points[combined_vertices][:,0], combined_points[combined_vertices][:,1], combined_points[combined_vertices][:,2])
    

    # Customize the plot (labels, title, etc.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title(f"Volume of the convex hull is {round((convex_hull_pos_volume+convex_hull_neg_volume)/1000**3,2)} m^3")


    plt.show()


def plot_pointcloud(points):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], marker=".")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_workspace(joint_bounds):
    
    fk_list = []
    [(robotic_length_min, robotic_length_max), (theta_min, theta_max), (delta_l_niTi_min, delta_l_niTi_max)] = joint_bounds

    for robotic_length in range(robotic_length_min, robotic_length_max, 1):
        for theta in np.arange(np.rad2deg(theta_min), np.rad2deg(theta_max), 2): # [0, 2, 4, 6, 54, 56, 58, 60]:
            for delta_l_niTi in range(delta_l_niTi_min, delta_l_niTi_max, 2):
                q = [robotic_length, np.deg2rad(theta), delta_l_niTi]
                fk = forward_kinematics_2D(q)
                fk_list.append(fk[-1])
                #plot_func_2D(fk)
        
    fk_array = np.array(fk_list)
    
    #  colors = plt.cm.viridis(np.linspace(0, 1, len(fk_array)//25 + 1))
    # for i in range(0, len(fk_array), 25):
    #     plt.scatter(fk_array[i:i+25, 0], fk_array[i:i+25, 1], color=colors[i//25], marker=".")
    
    
    plt.scatter(fk_array[:,0], fk_array[:,1], marker=".")
    #plt.scatter(fk_array[:,0], -fk_array[:,1], marker=".")

    plt.xlabel('z-axis [mm]')
    plt.ylabel('x-axis [mm]')
    plt.axis("equal")
    plt.show()



def plot_time_boxplot(time_needed_IK_big_k_convergence_list,
                      time_needed_IK_small_k_convergence_list,
                      time_needed_IK_damped_convergence_list,
                      time_needed_IK_QP_convergence_list,
                      time_needed_NSGC_convergence_list):

    convergence_dict = {
        "Jacobian k=0.5": time_needed_IK_big_k_convergence_list,
        "Jacobian k=0.4": time_needed_IK_small_k_convergence_list,
        "Damped Jacobian": time_needed_IK_damped_convergence_list,
        "QP Jacobian": time_needed_IK_QP_convergence_list,
        "NSGC": time_needed_NSGC_convergence_list
    }

    for name, data in convergence_dict.items():
        median = np.median(data)
        mean = np.mean(data)
        data_range = np.max(data) - np.min(data)
        data_iqr = iqr(data)
        data_var = np.var(data)
        data_std = np.std(data)

        print(f"\n{name}:")
        print(f"  Median time: {median:.3f} s")
        print(f"  Mean time: {mean:.3f} s")
        print(f"  Range: {data_range:.3f} s")
        print(f"  IQR: {data_iqr:.3f} s")
        print(f"  Variance: {data_var:.6f}")
        print(f"  Standard Deviation: {data_std:.6f}")

    # Create a box plot for the distribution of times
    plt.figure(figsize=(4, 6))
    box = plt.boxplot([time_needed_IK_big_k_convergence_list,
                       time_needed_IK_small_k_convergence_list,
                       time_needed_IK_damped_convergence_list,
                       time_needed_IK_QP_convergence_list,
                       time_needed_NSGC_convergence_list],
                      patch_artist=True)
    colors = ['green', 'red', 'blue', 'orange', 'purple']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Convergence Time', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks([1, 2, 3, 4, 5],
               ['Basic Jacobian\n k=0.5', 'Basic Jacobian\n k=0.4',
                'Adaptive damping\n k=0.5', 'Quadratic Programming\n k=0.5',
                'NSGC'],
               fontsize=12, rotation=30, ha='right')
    plt.ylabel('Time [s]', fontsize=14)
    plt.tight_layout()
    plt.show()
