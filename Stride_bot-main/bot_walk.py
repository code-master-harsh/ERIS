import time
import numpy as np
import pybullet as p
import pybullet_data
import csv
import json
import os

from src.kinematic_model import robotKinematics
from src.pybullet_debugger import pybulletDebug
from src.gaitPlanner import trotGait
from src.emotion_gait import get_emotion_params

# Path to the shared emotion state file written by app_copy.py
_EMOTION_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "emotion_state.json")
_FALLBACK_EMOTION = "neutral"

def get_current_emotion():
    """Read the latest predicted emotion from the shared state file."""
    try:
        with open(_EMOTION_STATE_FILE, "r") as f:
            data = json.load(f)
        emotion = data.get("emotion", _FALLBACK_EMOTION)
        # Validate it's one of the known emotions
        if emotion in ("happy", "sad", "neutral", "angry"):
            return emotion
        return _FALLBACK_EMOTION
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return _FALLBACK_EMOTION


def rendering(render):
    """Enable/disable rendering"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, render)

def robot_init( dt, body_pos, fixed = False ):
    physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
    # turn off rendering while loading the models
    rendering(0)

    p.setGravity(0,0,-10)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(
        fixedTimeStep=dt,
        numSolverIterations=100,
        enableFileCaching=0,
        numSubSteps=1,
        solverResidualThreshold=1e-10,
        erp=0.2,
        contactERP=0.2,
        frictionERP=0.2,
    )
    # add floor
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=1.0, restitution=0.0,
                     contactStiffness=1e6, contactDamping=1e4)
    # add robot
    body_id = p.loadURDF("stridebot.urdf", body_pos, useFixedBase=fixed)
    joint_ids = []

    #robot properties
    maxVel = 3.703 #rad/s
    foot_links = {3, 7, 11, 15}  # tip link of each leg
    for j in range(p.getNumJoints(body_id)):
        p.changeDynamics(body_id, j, linearDamping=0, angularDamping=0,
                         maxJointVelocity=maxVel, restitution=0.0)
        if j in foot_links:
            p.changeDynamics(body_id, j, lateralFriction=1.0,
                             spinningFriction=0.1, rollingFriction=0.1,
                             contactStiffness=1e6, contactDamping=1e4)
        else:
            p.changeDynamics(body_id, j, lateralFriction=0.4)
        joint_ids.append(p.getJointInfo(body_id, j))
    rendering(1)
    return body_id, joint_ids

def robot_stepsim( body_id, body_pos, body_orn, body2feet ):
    #robot properties
    fr_index, fl_index, br_index, bl_index = 3, 7, 11, 15
    maxForce = 9 #N/m
    
    #####################################################################################
    #####   kinematics Model: Input body orientation, deviation and foot position    ####
    #####   and get the angles, neccesary to reach that position, for every joint    ####
    fr_angles, fl_angles, br_angles, bl_angles , body2feet_ = robotKinematics.solve( body_orn , body_pos , body2feet )
    #move movable joints
    for i in range(3):
        p.setJointMotorControl2(body_id, i, p.POSITION_CONTROL, targetPosition = fr_angles[i] , force = maxForce)
        p.setJointMotorControl2(body_id, 4 + i, p.POSITION_CONTROL, targetPosition = fl_angles[i] , force = maxForce)
        p.setJointMotorControl2(body_id, 8 + i, p.POSITION_CONTROL, targetPosition = br_angles[i] , force = maxForce) 
        p.setJointMotorControl2(body_id, 12 + i, p.POSITION_CONTROL, targetPosition = bl_angles[i] , force = maxForce)

    # Safety: clamp velocity to prevent flying
    lin_vel, ang_vel = p.getBaseVelocity(body_id)
    speed = (lin_vel[0]**2 + lin_vel[1]**2 + lin_vel[2]**2) ** 0.5
    if speed > 1.0:
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])

    p.stepSimulation()
    
    return body2feet_

def robot_quit():
    p.disconnect()
        
        
if __name__ == '__main__':
    dT = 0.005
    bodyId, jointIds = robot_init( dt = dT, body_pos = [0,0,0.18], fixed = False )
    pybulletDebug = pybulletDebug()
    robotKinematics = robotKinematics()
    trot = trotGait()

    """initial foot position"""
    #foot separation (Ydist = 0.16 -> theta =0) and distance to floor
    Xdist, Ydist, height = 0.18, 0.15, 0.10
    #body frame to foot frame vector
    bodytoFeet0 = np.matrix([[ Xdist/2. , -Ydist/2. , height],
                            [ Xdist/2. ,  Ydist/2. , height],
                            [-Xdist/2. , -Ydist/2. , height],
                            [-Xdist/2. ,  Ydist/2. , height]])

    offset = np.array([0.5 , 0.5 , 0. , 0.]) #defines the offset between each foot step in this order (FR,FL,BR,BL)
    footFR_index, footFL_index, footBR_index, footBL_index = 3, 7, 11, 15
    T = 0.5 #period of time (in seconds) of every step
    
    N_steps = 50000

    # ── Spawn (default) gait params — the "rest" pose ─────────────────
    SPAWN_PARAMS = {
        "L": 0.0, "angle": 0, "Lrot": 0, "T": 0.5, "sda": 0,
        "offset": [0.5, 0.5, 0.0, 0.0],
        "pos": np.array([0.0, 0.0, 0.0]),
        "orn": np.array([0.0, 0.0, 0.0]),
        "bodytoFeet_override": None,
    }

    def _params_for(emotion, t):
        """Get gait params dict for an emotion (or spawn defaults)."""
        if emotion is None:
            return dict(SPAWN_PARAMS)
        return get_emotion_params(emotion, t=t)

    def _lerp_pose(p_a, p_b, alpha, bTF_default):
        """Interpolate only the body pose and foot geometry (NOT gait planner params)."""
        bTF_a = p_a["bodytoFeet_override"] if p_a["bodytoFeet_override"] is not None else bTF_default
        bTF_b = p_b["bodytoFeet_override"] if p_b["bodytoFeet_override"] is not None else bTF_default
        return {
            "pos":       p_a["pos"] * (1 - alpha) + p_b["pos"] * alpha,
            "orn":       p_a["orn"] * (1 - alpha) + p_b["orn"] * alpha,
            "bodytoFeet": np.array(bTF_a) * (1 - alpha) + np.array(bTF_b) * alpha,
        }

    def _smooth_alpha(t_now, t_start, duration):
        """Cosine ease-in-out: 0→1 over duration, no jerk at boundaries."""
        raw = max(0.0, min(1.0, (t_now - t_start) / duration))
        return 0.5 * (1.0 - np.cos(np.pi * raw))   # smooth S-curve

    # ── Emotion transition state ──────────────────────────────────────
    current_emotion  = get_current_emotion()
    previous_emotion = current_emotion
    last_poll_time   = time.time()
    POLL_INTERVAL    = 0.5
    BLEND_SECS       = 3.0         # total blend time old → spawn → new
    HALF_BLEND       = BLEND_SECS / 2.0
    blend_start      = 0.0
    blending         = False
    frozen_old_params = None       # snapshot of old emotion params at moment of change

    for k_ in range(0, N_steps):
        now = time.time()
        sim_t = k_ * dT

        # ── Poll the emotion file periodically ────────────────────────
        if now - last_poll_time >= POLL_INTERVAL:
            last_poll_time = now
            new_emotion = get_current_emotion()

            if new_emotion != current_emotion and not blending:
                # Freeze the old emotion's current params (stops oscillations from being a moving target)
                frozen_old_params = _params_for(previous_emotion, sim_t)
                previous_emotion = current_emotion
                current_emotion  = new_emotion
                blend_start      = now
                blending         = True

        # ── Camera follows robot ──────────────────────────────────────
        camInfo = p.getBasePositionAndOrientation(bodyId)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=camInfo[0]
        )

        # ── Determine current gait params ─────────────────────────────
        if blending:
            elapsed = now - blend_start
            if elapsed < HALF_BLEND:
                # Phase 1: old emotion → spawn pose (using frozen params)
                alpha = _smooth_alpha(now, blend_start, HALF_BLEND)
                blended = _lerp_pose(frozen_old_params, SPAWN_PARAMS, alpha, bodytoFeet0)
            elif elapsed < BLEND_SECS:
                # Phase 2: spawn pose → new emotion
                alpha = _smooth_alpha(now, blend_start + HALF_BLEND, HALF_BLEND)
                new_p = _params_for(current_emotion, sim_t)
                blended = _lerp_pose(SPAWN_PARAMS, new_p, alpha, bodytoFeet0)
            else:
                # Blend complete
                blending = False
                blended = None

            if blended is not None:
                # BYPASS trot.loop — feed interpolated foot positions directly
                bTF = np.matrix(blended["bodytoFeet"])
                robot_stepsim(bodyId, blended["pos"], blended["orn"], bTF)
                continue
            else:
                # Blend just finished — reset trot for clean gait start
                trot = trotGait()

        # ── Normal emotion gait (no blend active) ─────────────────────
        EMOTION = current_emotion
        if EMOTION is None:
            pos, orn, L, angle, Lrot, T, sda, offset = pybulletDebug.cam_and_robotstates(bodyId)
            bTF_base = bodytoFeet0
        else:
            params  = get_emotion_params(EMOTION, t=sim_t)
            pos     = params["pos"]
            orn     = params["orn"]
            L       = params["L"]
            angle   = params["angle"]
            Lrot    = params["Lrot"]
            T       = params["T"]
            sda     = params["sda"]
            offset  = params["offset"]
            bTF_base = params["bodytoFeet_override"] if params["bodytoFeet_override"] is not None else bodytoFeet0

            bodytoFeet = trot.loop(L, angle, Lrot, T, offset, bTF_base, sda)
            robot_stepsim(bodyId, pos, orn, bodytoFeet)
        
    robot_quit()
