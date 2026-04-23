#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry

class GzGroundTruthNode(Node):
    def __init__(self):
        super().__init__('gz_ground_truth_node')
        self.sub = self.create_subscription(
            TFMessage, '/world/default/pose/info', self.pose_cb, 10)
        self.pub = self.create_publisher(Odometry, '/odom_truth', 10)
        self.printed_frames = False

    def pose_cb(self, msg: TFMessage):
        if not self.printed_frames:
            frames = [t.child_frame_id for t in msg.transforms]
            self.get_logger().info(f"Available Gazebo frames: {frames}")
            self.printed_frames = True

        for t in msg.transforms:
            # Look explicitly for the go2_robot frame, ignoring the parent frame name
            if t.child_frame_id == 'go2_robot':
                odom = Odometry()
                odom.header.stamp = self.get_clock().now().to_msg()
                odom.header.frame_id = 'world'
                odom.child_frame_id = t.child_frame_id
                odom.pose.pose.position.x = t.transform.translation.x
                odom.pose.pose.position.y = t.transform.translation.y
                odom.pose.pose.position.z = t.transform.translation.z
                odom.pose.pose.orientation = t.transform.rotation
                self.pub.publish(odom)
                return

def main(args=None):
    rclpy.init(args=args)
    node = GzGroundTruthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()