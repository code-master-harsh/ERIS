#!/usr/bin/env python3
"""
emotion_bridge_node.py
Continuously polls the emotion_state.json from the Gradio app.
When the robot is IDLE, it pulls the latest emotion and sends a new command.
"""
import os
import time
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Define what task to do repeatedly for each emotion.
# You can change these to 'turn' or adjust values as needed.
TASK_MAPPING = {
    'happy':   {"action": "move", "value": 2.0},
    'sad':     {"action": "move", "value": 1.5},
    'angry':   {"action": "move", "value": 2.0},
    'neutral': {"action": "move", "value": 2.0}
}

class EmotionBridgeNode(Node):
    def __init__(self):
        super().__init__('emotion_bridge_node')
        
        # ── Parameters ──
        # IMPORTANT: Default path assumes the Gradio app is running in the workspace root.
        # You can override this path via command line if needed.
        default_json_path = os.path.expanduser('~/Desktop/emotion_v4-copy/emotion_state.json')
        
        self.declare_parameter('json_path', default_json_path)
        self.declare_parameter('stale_timeout', 15.0) # Stop moving if no fresh data for 15s
        
        self.json_path = self.get_parameter('json_path').value
        self.stale_timeout = self.get_parameter('stale_timeout').value
        
        # ── ROS Interfaces ──
        self.pub_cmd = self.create_publisher(String, '/behaviour_command', 10)
        self.sub_status = self.create_subscription(String, '/behaviour/status', self.status_cb, 10)
        
        # ── State ──
        self.is_idle = True
        
        # Poll the file at 2 Hz
        self.timer = self.create_timer(0.5, self.tick)
        self.get_logger().info(f'Emotion Bridge ready. Watching for AI data at: {self.json_path}')

    def status_cb(self, msg: String):
        """Listens to the behavior node to know when it is safe to send the next command."""
        status = msg.data
        if "DONE" in status or "ABORT" in status:
            self.is_idle = True
        elif "HOLDING" in status or "MOVING" in status or "FINALE" in status:
            self.is_idle = False

    def tick(self):
        # Only issue a command if the robot has finished its previous sequence
        if not self.is_idle:
            return
            
        if not os.path.exists(self.json_path):
            return
            
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return # File might be mid-write by the Gradio app; ignore and try next tick
            
        emotion = data.get('emotion')
        timestamp = data.get('timestamp', 0.0)
        
        # 1. Safety Check: Is the data fresh? 
        # If the Gradio app crashes or pauses, we stop sending commands.
        if time.time() - timestamp > self.stale_timeout:
            return
            
        if emotion not in TASK_MAPPING:
            return
            
        # 2. Build the Payload
        task = TASK_MAPPING[emotion]
        payload = {
            "action": task["action"],
            "value": task["value"],
            "emotion": emotion
        }
        
        # 3. Dispatch the Command
        cmd_msg = String()
        cmd_msg.data = json.dumps(payload)
        self.pub_cmd.publish(cmd_msg)
        
        self.get_logger().info(f'Triggering loop sequence: {emotion.upper()}')
        
        # Lock the state so we don't spam commands before the behavior node processes it
        self.is_idle = False 

def main(args=None):
    rclpy.init(args=args)
    node = EmotionBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()