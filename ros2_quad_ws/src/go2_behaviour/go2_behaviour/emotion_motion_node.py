#!/usr/bin/env python3
"""
emotion_motion_node.py  –  v6 (Dynamic Commands & Heading Correction)
"""
import math
import json
from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rcl_interfaces.msg import Parameter as RclParam, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters

CHAMP_CTRL_NODE = '/champ_controller'
PARAM_STANCE    = 'stance_duration'
PARAM_SWING     = 'swing_height'

NOMINAL_Z      =  0.0
NOMINAL_PITCH  =  0.0
Z_MIN, Z_MAX   = -0.06, 0.10  
P_MAX          =  0.40        
BASE_STANCE    =  0.25
BASE_SWING     =  0.028   

EMOTION_LABELS = ['happy', 'sad', 'angry', 'neutral']

@dataclass(frozen=True)
class Gait:
    stance: float   
    swing:  float   

@dataclass(frozen=True)
class Profile:
    v:     float    # linear target
    acc:   float    # linear accel
    dec:   float    # linear decel
    w:     float    # angular target (rad/s)
    w_acc: float    # angular accel
    w_dec: float    # angular decel
    z:     float    
    pitch: float    
    gait:  Gait
    burst: bool = False  

PROFILES: dict[str, Profile] = {
    'happy': Profile(
        v=0.25, acc=0.30, dec=0.40,           
        w=0.80, w_acc=1.00, w_dec=1.00, # Bouncy fast turn
        z= 0.03, pitch=-0.08,                 
        gait=Gait(stance=0.40, swing=0.085),  
    ),
    'sad': Profile(
        v=0.18, acc=0.14, dec=0.18,
        w=0.30, w_acc=0.20, w_dec=0.30, # Slow dragging turn
        z=-0.04, pitch= 0.10,
        gait=Gait(stance=0.40, swing=0.014),
    ),
    'angry': Profile(
        v=0.80, acc=1.80, dec=2.20,
        w=1.20, w_acc=2.50, w_dec=2.50, # Sharp aggressive turn
        z=-0.01, pitch= 0.05,
        gait=Gait(stance=0.17, swing=0.052),
        burst=True,
    ),
    'neutral': Profile(
        v=0.40, acc=0.60, dec=0.60,
        w=0.60, w_acc=0.80, w_dec=0.80, # Smooth standard turn
        z= 0.00, pitch= 0.00,
        gait=Gait(stance=BASE_STANCE, swing=BASE_SWING),
    ),
}

BURST_WALK_S = 0.55   
BURST_STOP_S = 0.22   

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

def get_yaw_from_quat(q):
    """Convert quaternion to euler yaw."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class EmotionMotionNode(Node):
    def __init__(self):
        super().__init__('emotion_motion_node')

        self.declare_parameter('start_delay',          2.0)
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('tie_margin',           0.10)
        self.declare_parameter('control_rate_hz',     50.0)
        self.declare_parameter('settle_time',          1.0)
        self.declare_parameter('odom_scale',           1.0)
        self.declare_parameter('kp_yaw',               1.5) # Heading correction gain

        def _p(n): return self.get_parameter(n).value

        self.start_delay = float(_p('start_delay'))
        self.conf_thresh = float(_p('confidence_threshold'))
        self.tie_margin  = float(_p('tie_margin'))
        self.ctrl_rate   = float(_p('control_rate_hz'))
        self.settle_time = float(_p('settle_time'))
        self.odom_scale  = float(_p('odom_scale'))
        self.kp_yaw      = float(_p('kp_yaw'))
        self.dt          = 1.0 / self.ctrl_rate
        self.profiles    = dict(PROFILES)

        self._gait_cli = self.create_client(SetParameters, f'{CHAMP_CTRL_NODE}/set_parameters')

        s_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.sub_cmd  = self.create_subscription(String, '/behaviour_command', self.on_command, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom_truth', self.on_odom, s_qos)
        
        self.pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',   10)
        self.pub_pose   = self.create_publisher(Pose,   '/body_pose', 10)
        self.pub_status = self.create_publisher(String, '/behaviour/status',   10)

        self.state          = 'IDLE'
        self.have_odom      = False
        self.pose_x = self.pose_y = self.pose_yaw = 0.0
        
        # Dynamic targets
        self.target_action  = 'move' # 'move' or 'turn'
        self.target_value   = 0.0
        self.start_x = self.start_y = self.start_yaw = 0.0
        
        self.v_cmd = self.w_cmd = 0.0
        
        self.active_profile: Profile | None = None
        self.active_emotion: str    | None = None
        
        self.current_z     = NOMINAL_Z
        self.current_pitch = NOMINAL_PITCH
        self.reset_z       = 0.0
        self.reset_pitch   = 0.0

        self._burst_phase = 'WALK'
        self._burst_t     = 0.0
        self.phase_start  = None
        self._warmup      = int(self.ctrl_rate * 0.6)  

        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info('emotion_motion_node v6 (Dynamic JSON Commands & Heading Fix) ready.')

    def on_odom(self, msg: Odometry):
        self.pose_x   = msg.pose.pose.position.x * self.odom_scale
        self.pose_y   = msg.pose.pose.position.y * self.odom_scale
        self.pose_yaw = get_yaw_from_quat(msg.pose.pose.orientation)
        self.have_odom = True

    def on_command(self, msg: String):
        if self.state != 'IDLE' or not self.have_odom:
            self.get_logger().warn('Ignoring command. Not IDLE or missing Odometry.')
            return
            
        try:
            data = json.loads(msg.data)
            self.target_action = data.get('action', 'move')
            raw_val = float(data.get('value', 0.0))
            
            if self.target_action == 'turn':
                # Convert degrees to radians for internal math
                self.target_value = math.radians(raw_val)
            else:
                self.target_value = raw_val

            # Emotion selection
            chosen = 'neutral'
            if 'emotion' in data:
                chosen = data['emotion']
            elif 'probs' in data:
                probs = data['probs']
                indexed = sorted(enumerate(probs), key=lambda kv: -kv[1])
                top_idx, top_p = indexed[0]
                _, second_p = indexed[1]
                if (top_p - second_p) >= self.tie_margin and top_p >= self.conf_thresh:
                    chosen = EMOTION_LABELS[top_idx]

            self.active_emotion = chosen
            self.active_profile = self.profiles[chosen]
            
            self.start_x = self.pose_x
            self.start_y = self.pose_y
            self.start_yaw = self.pose_yaw
            
            self.v_cmd = self.w_cmd = 0.0
            self._set_gait(self.active_profile.gait)

            self.phase_start = self.get_clock().now()
            self.state       = 'HOLDING'
            self._publish_status(f'HOLDING:{chosen}')
            
            val_str = f"{raw_val}deg" if self.target_action == 'turn' else f"{raw_val}m"
            self.get_logger().info(f'[START] {self.target_action} {val_str} as {chosen.upper()}')
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse JSON command: {e}")

    def tick(self):
        if self._warmup > 0:
            self._send_posture(NOMINAL_Z, NOMINAL_PITCH)
            self._send_cmd(0.0)
            self._warmup -= 1
            return

        if self.state == 'IDLE':
            return

        now     = self.get_clock().now()
        elapsed = (now - self.phase_start).nanoseconds * 1e-9

        # ── 1. HOLDING ───────────────────────────────────────────────────────────
        if self.state == 'HOLDING':
            self._send_cmd(0.0)
            prog = min(1.0, elapsed / self.start_delay)

            bz = self._lerp(NOMINAL_Z,     self.active_profile.z,     prog)
            bp = self._lerp(NOMINAL_PITCH, self.active_profile.pitch,  prog)

            if prog >= 0.6:
                anim_scale = min(1.0, (prog - 0.6) / 0.4) 
                bz, bp = self._idle_anim(self.active_emotion, elapsed, bz, bp, anim_scale)

            self.current_z, self.current_pitch = bz, bp
            self._send_posture(bz, bp)

            if elapsed >= self.start_delay:
                self.state       = 'MOVING'
                self.phase_start = now
                self._burst_phase = 'WALK'
                self._burst_t    = 0.0
                # Snap start yaw here to ignore minor twitches during HOLDING
                self.start_yaw   = self.pose_yaw 
                self._publish_status(f'MOVING:{self.active_emotion}')
            return

        # ── 2. MOVING ────────────────────────────────────────────────────────────
        if self.state == 'MOVING':
            self._send_posture(self.active_profile.z, self.active_profile.pitch)

            if self.target_action == 'turn':
                self._tick_turn()
            else:
                self._tick_move()
            return

        # ── 3. STOPPING ─────────────────────────────────────────────────────────
        if self.state == 'STOPPING':
            self._send_cmd(0.0, 0.0)
            self._send_posture(self.active_profile.z, self.active_profile.pitch)

            if elapsed >= self.settle_time:
                self.state = 'FINALE'
                self.phase_start = now
            return

        # ── 4. FINALE (Extra Actions) ────────────────────────────────────────────
        if self.state == 'FINALE':
            if self.active_emotion == 'happy':
                duration = 1.2
                z_jump, p_jump = 0.07, -0.32
                if elapsed <= 0.6:
                    prog = elapsed / 0.6
                    bz = self._lerp(self.active_profile.z, z_jump, prog)
                    bp = self._lerp(self.active_profile.pitch, p_jump, prog)
                elif elapsed <= 1.2:
                    prog = (elapsed - 0.6) / 0.6
                    bz = self._lerp(z_jump, self.active_profile.z, prog)
                    bp = self._lerp(p_jump, self.active_profile.pitch, prog)
                else:
                    bz, bp = self.active_profile.z, self.active_profile.pitch

                self.current_z, self.current_pitch = bz, bp
                self._send_posture(bz, bp)
                if elapsed >= duration: self._start_reset(now)

            elif self.active_emotion == 'angry':
                duration = 2.6
                self._send_cmd(0.0, 1.2)
                self._send_posture(self.active_profile.z, self.active_profile.pitch)
                if elapsed >= duration:
                    self._send_cmd(0.0, 0.0)
                    self._start_reset(now)

            elif self.active_emotion == 'sad':
                duration = 2.5
                self._send_posture(self.active_profile.z, self.active_profile.pitch)
                if elapsed >= duration: self._start_reset(now)
            else:
                self._start_reset(now)
            return

        # ── 5. RESETTING (Smooth restore) ─────────────────────────────────────────
        if self.state == 'RESETTING':
            self._send_cmd(0.0, 0.0)
            duration = 1.5
            
            prog = min(1.0, elapsed / duration)
            self.current_z     = self._lerp(self.reset_z, NOMINAL_Z, prog)
            self.current_pitch = self._lerp(self.reset_pitch, NOMINAL_PITCH, prog)
            
            self._send_posture(self.current_z, self.current_pitch)

            if elapsed >= duration:
                self.get_logger().info(f'[DONE] task complete.')
                self._publish_status(f'DONE:{self.active_emotion}')
                self.state = 'IDLE'
            return

    # ── Action Ticks ────────────────────────────────────────────────────────

    def _tick_move(self):
        """Moves linear distance while actively correcting heading drift."""
        disp = math.hypot(self.pose_x - self.start_x, self.pose_y - self.start_y)
        remaining = self.target_value - disp

        if remaining <= 0.05: # Stop tolerance
            self._trigger_stop()
            return

        if self.active_profile.burst:
            self._tick_burst_logic(remaining)
        else:
            self._tick_velocity_logic(remaining)

    def _tick_turn(self):
        """Rotates on the spot to target angle."""
        target_yaw = normalize_angle(self.start_yaw + self.target_value)
        yaw_err = normalize_angle(target_yaw - self.pose_yaw)
        
        if abs(yaw_err) <= 0.05: # Stop tolerance (approx 3 degrees)
            self._trigger_stop()
            return
            
        w = self.w_cmd
        w_dec = self.active_profile.w_dec
        braking = (w * w) / (2.0 * w_dec) if w_dec > 0 else 0.0
        
        direction = 1.0 if yaw_err > 0 else -1.0
        
        if abs(yaw_err) <= braking + 0.02:
            new_w = max(0.0, abs(w) - w_dec * self.dt)
        else:
            new_w = min(self.active_profile.w, abs(w) + self.active_profile.w_acc * self.dt)
            
        self.w_cmd = new_w * direction
        self._send_cmd(0.0, self.w_cmd)

    def _tick_velocity_logic(self, remaining: float):
        v = self.v_cmd
        dec = self.active_profile.dec
        braking = (v * v) / (2.0 * dec) if dec > 0 else 0.0
        
        if remaining <= braking + 0.02:
            self.v_cmd = max(0.0, v - dec * self.dt)
        else:
            self.v_cmd = min(self.active_profile.v, v + self.active_profile.acc * self.dt)
            
        # P-Controller for Heading Correction
        drift_err = normalize_angle(self.start_yaw - self.pose_yaw)
        w_corr = self.kp_yaw * drift_err
        
        self._send_cmd(self.v_cmd, w_corr)

    def _tick_burst_logic(self, remaining: float):
        self._burst_t += self.dt
        if self._burst_phase == 'WALK':
            if self._burst_t >= BURST_WALK_S:
                self._burst_phase, self._burst_t = 'STOP', 0.0
                self.v_cmd = 0.0
                self._send_cmd(0.0, 0.0)
            else:
                self._tick_velocity_logic(remaining)
        else:
            self.v_cmd = 0.0
            # Keep heading correction active even while stopped in burst
            drift_err = normalize_angle(self.start_yaw - self.pose_yaw)
            self._send_cmd(0.0, self.kp_yaw * drift_err)
            
            if self._burst_t >= BURST_STOP_S:
                self._burst_phase, self._burst_t = 'WALK', 0.0

    def _trigger_stop(self):
        self.v_cmd = self.w_cmd = 0.0
        self._send_cmd(0.0, 0.0)
        self._set_gait(Gait(BASE_STANCE, BASE_SWING))   
        self.state = 'STOPPING'
        self.phase_start = self.get_clock().now()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _start_reset(self, now):
        self.state = 'RESETTING'
        self.phase_start = now
        self.reset_z = self.current_z
        self.reset_pitch = self.current_pitch

    def _idle_anim(self, emotion: str, t: float, base_z: float, base_pitch: float, scale: float) -> tuple:
        if emotion == 'happy':
            bob = scale * 0.028 * math.sin(2 * math.pi * 1.5 * t)
            zbob = scale * 0.008 * math.sin(2 * math.pi * 1.5 * t - math.pi / 2)
            return base_z + zbob, base_pitch + bob
        elif emotion == 'angry':
            tremor_p = scale * 0.007 * math.sin(2 * math.pi * 9.0 * t)
            tremor_z = scale * 0.003 * math.sin(2 * math.pi * 9.0 * t + 1.0)
            pulse = scale * 0.025 * math.sin(math.pi * min(1.0, (t - self.start_delay * 0.8) / (self.start_delay * 0.2))) if t > self.start_delay * 0.8 else 0.0
            return base_z + tremor_z, base_pitch + tremor_p + pulse
        elif emotion == 'sad':
            return base_z, base_pitch + (scale * 0.005 * math.sin(2 * math.pi * 0.30 * t))
        return base_z, base_pitch

    def _set_gait(self, g: Gait):
        if not self._gait_cli.service_is_ready(): return
        def _dp(name: str, val: float):
            p = RclParam()
            p.name = name
            p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(val))
            return p
        req = SetParameters.Request()
        req.parameters = [_dp(PARAM_STANCE, g.stance), _dp(PARAM_SWING, g.swing)]
        fut = self._gait_cli.call_async(req)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _send_cmd(self, vx: float, wz: float = 0.0):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.pub_cmd.publish(msg)

    def _send_posture(self, z_delta: float, pitch: float):
        msg = Pose()
        msg.position.x = msg.position.y = 0.0
        msg.position.z = max(Z_MIN, min(Z_MAX, float(z_delta)))
        p = max(-P_MAX, min(P_MAX, float(pitch)))
        msg.orientation.x = msg.orientation.z = 0.0
        msg.orientation.y = math.sin(p / 2.0)
        msg.orientation.w = math.cos(p / 2.0)
        self.pub_pose.publish(msg)

    def _publish_status(self, s: str):
        m = String()
        m.data = s
        self.pub_status.publish(m)

def main(args=None):
    rclpy.init(args=args)
    node = EmotionMotionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        try:
            node.pub_cmd.publish(Twist())
            node.pub_pose.publish(Pose()) 
        except Exception: pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()