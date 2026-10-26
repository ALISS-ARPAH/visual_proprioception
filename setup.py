from setuptools import setup

package_name = 'visual_proprioception'



setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            'share/' + package_name + '/launch',
            ['launch/laser_depth_estimation.launch.py']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Visual proprioception for CTRs using laser imaging',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'laser_segmentation_node = visual_proprioception.laser_segmentation_node:main',
            'laser_tip_fusion_node = visual_proprioception.laser_tip_fusion_node:main',
            'laser_tissue_triangulation_node = visual_proprioception.laser_tissue_triangulation_node:main',
            'laser_spot_analysis_node = visual_proprioception.laser_spot_analysis_node:main',
            'laser_depth_area_logger = visual_proprioception.laser_depth_area_logger:main',
            'laser_tissue_touch_node = visual_proprioception.laser_tissue_touch_node:main',
            'laser_spot_monitor_gui = visual_proprioception.laser_spot_monitor_gui:main',
        ],
    },
)
