from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # 1. Bridge the massive Gazebo pose array into ROS
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/world/default/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V'
            ],
            output='screen'
        ),
        
        # 2. Run our custom filter to isolate the Go2's ground truth pose
        Node(
            package='go2_behaviour',
            executable='gz_ground_truth_node',
            name='gz_ground_truth_node',
            output='screen',
        ),

        # 3. Run the behaviour node, pointing it to our new clean topic
        Node(
            package='go2_behaviour',
            executable='emotion_motion_node',
            name='emotion_motion_node',
            output='screen',
            parameters=[{
                'd_fixed': 2.0,
                'start_delay': 2.0,
                'confidence_threshold': 0.40,
                'tie_margin': 0.10,
                'stop_tolerance': 0.05,
                'control_rate_hz': 50.0,
                'settle_time': 0.5,
                'motion_timeout_factor': 3.0,
                'odom_scale': 1.0,           # Keep at 1.0 since ground truth is perfect
                'odom_topic': '/odom_truth', # Subscribing to our custom ground-truth topic
                'cmd_vel_topic': '/cmd_vel',
                'emotion_topic': '/emotion',
            }],
        )
    ])