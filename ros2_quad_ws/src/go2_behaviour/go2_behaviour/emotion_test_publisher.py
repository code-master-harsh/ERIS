#!/usr/bin/env python3
import argparse
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

PRESETS = {
    'happy':   [0.80, 0.05, 0.10, 0.05],
    'sad':     [0.05, 0.80, 0.05, 0.10],
    'angry':   [0.10, 0.05, 0.80, 0.05],
    'neutral': [0.10, 0.10, 0.10, 0.70],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', choices=list(PRESETS.keys()))
    parser.add_argument('--probs', nargs=4, type=float)
    parser.add_argument('--topic', default='/emotion')
    args, ros_args = parser.parse_known_args()

    if not args.emotion and not args.probs:
        sys.exit(1)

    probs = args.probs if args.probs else PRESETS[args.emotion]
    total = sum(probs)
    probs = [p / total for p in probs]

    rclpy.init(args=ros_args)
    node = Node('emotion_test_publisher')
    pub = node.create_publisher(Float32MultiArray, args.topic, 10)

    for _ in range(30):
        if pub.get_subscription_count() > 0: break
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)

    msg = Float32MultiArray()
    msg.data = [float(p) for p in probs]
    pub.publish(msg)
    node.get_logger().info(f'Published: {probs}')
    
    time.sleep(0.3)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()