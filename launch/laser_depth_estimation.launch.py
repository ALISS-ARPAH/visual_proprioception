from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    area_threshold = LaunchConfiguration('area_threshold_px2')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false'
        ),

        DeclareLaunchArgument(
            'area_threshold_px2',
            default_value='1000.0'
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
