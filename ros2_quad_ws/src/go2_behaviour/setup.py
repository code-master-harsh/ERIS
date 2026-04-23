from setuptools import setup
import os
from glob import glob

package_name = 'go2_behaviour'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teaching',
    maintainer_email='teaching@todo.todo',
    description='Emotion-driven behaviour layer for Go2 + CHAMP',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gz_ground_truth_node = go2_behaviour.gz_ground_truth_node:main',
            'emotion_motion_node = go2_behaviour.emotion_motion_node:main',
            'emotion_test_publisher = go2_behaviour.emotion_test_publisher:main',
            'behaviour_commander = go2_behaviour.behaviour_commander:main',
             'emotion_bridge_node = go2_behaviour.emotion_bridge_node:main', # <-- ADD THIS
        ],
    },
)



# from setuptools import find_packages, setup

# package_name = 'go2_behaviour'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='teaching',
#     maintainer_email='teaching@todo.todo',
#     description='TODO: Package description',
#     license='MIT',
#     extras_require={
#         'test': [
#             'pytest',
#         ],
#     },
#     entry_points={
#         'console_scripts': [
#         ],
#     },
# )
