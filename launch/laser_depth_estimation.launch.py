from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    area_threshold = LaunchConfiguration('area_threshold_px2')
    cut_predictor = LaunchConfiguration('cut_predictor')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false'
        ),

        DeclareLaunchArgument(
            'area_threshold_px2',
            default_value='1000.0'
        ),

        DeclareLaunchArgument(
            'cut_predictor',
            default_value='true',
            description='also run laser_cut_predictor_node (cut prediction shown in the GUI)'
        ),

        Node(
            package='visual_proprioception',
            executable='laser_cut_predictor_node',
            name='laser_cut_predictor_node',
            output='screen',
            condition=IfCondition(cut_predictor),
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        Node(
            package='visual_proprioception',
            executable='laser_segmentation_node',
            name='laser_segmentation_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        Node(
            package='visual_proprioception',
            executable='laser_spot_analysis_node',
            name='laser_spot_analysis_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        Node(
            package='visual_proprioception',
            executable='laser_tissue_touch_node',
            name='laser_tissue_touch_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'area_threshold_px2': area_threshold
            }]
        ),

        Node(
            package='visual_proprioception',
            executable='laser_tissue_triangulation_node',
            name='laser_tissue_triangulation_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        Node(
            package='visual_proprioception',
            executable='laser_spot_monitor_gui',
            name='laser_spot_monitor_gui',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
    ])
