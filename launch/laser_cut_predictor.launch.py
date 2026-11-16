"""
Launch the cut predictor: P(firing the laser now cuts) from the endoscope and the tip pose.

    ros2 launch visual_proprioception laser_cut_predictor.launch.py
    ros2 launch visual_proprioception laser_cut_predictor.launch.py confirm_threshold:=0.9

Outputs /laser/cut/probability, /laser/cut/close, /laser/cut/ready and /laser/cut/overlay.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model = os.path.join(get_package_share_directory('visual_proprioception'), 'models',
                         'cut_depth_combined', 'model.json')
    args = {
        'model_path': (model, 'model.json of the cut-depth model'),
        'image_topic': ('/ves_camera/image', 'endoscope images, raw or rectified'),
        'camera_info_topic': ('/ves_camera/camera_info', 'endoscope calibration'),
        'tip_pose_topic': ('/ves_smoother/right/tip_pose',
                           'laser probe tip pose (PoseWithCovarianceStamped, camera frame)'),
        'suspect_threshold': ('-1.0', 'P for /laser/cut/close; -1 = the model value (0.6)'),
        'confirm_threshold': ('-1.0', 'P for /laser/cut/ready; -1 = the model value (0.8)'),
        'smooth_sec': ('-1.0', 'mean P over this window; -1 = the model value (0.3 s), 0 = off'),
        'use_sim_time': ('false', 'true when playing a bag with --clock'),
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value, description=text)
          for name, (value, text) in args.items()),
        Node(
            package='visual_proprioception',
            executable='laser_cut_predictor_node',
            name='laser_cut_predictor_node',
            output='screen',
            parameters=[{name: LaunchConfiguration(name) for name in args}],
        ),
    ])
