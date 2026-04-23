#!/usr/bin/env python3
"""
behaviour_commander.py
CLI tool to send dynamic multi-parameter commands to the Go2.
"""
import argparse
import sys
import time
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def main():
    parser = argparse.ArgumentParser(description="Send dynamic emotion commands to Go2.")
    parser.add_argument('--action', choices=['move', 'turn'], required=True, help="Type of motion.")
    parser.add_argument('--value', type=float, required=True, help="Meters (if move) or Degrees (if turn, +CCW/-CW).")
    parser.add_argument('--emotion', choices=['happy', 'sad', 'angry', 'neutral'], help="Directly set emotion.")
    parser.add_argument('--probs', nargs=4, type=float, metavar=('H', 'S', 'A', 'N'), help="Use probability array instead.")
    parser.add_argument('--topic', default='/behaviour_command')
    
    args, ros_args = parser.parse_known_args()

    if not args.emotion and not args.probs:
        print('Error: Specify either --emotion NAME or --probs h s a n', file=sys.stderr)
        sys.exit(1)

    # Build the payload
    payload = {
        "action": args.action,
        "value": args.value
    }

    if args.emotion:
        payload["emotion"] = args.emotion
    elif args.probs:
        total = sum(args.probs)
        if total <= 0:
            print('Error: Probabilities must sum to > 0', file=sys.stderr)
            sys.exit(1)
        payload["probs"] = [p / total for p in args.probs]

    rclpy.init(args=ros_args)
    node = Node('behaviour_commander')
    pub = node.create_publisher(String, args.topic, 10)

    # Wait for discovery
    for _ in range(30):
        if pub.get_subscription_count() > 0: break
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)

    msg = String()
    msg.data = json.dumps(payload)
    pub.publish(msg)
    
    node.get_logger().info(f'Published Command: {msg.data}')
    
    time.sleep(0.3)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()