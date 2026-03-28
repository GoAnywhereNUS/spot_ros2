from sensor_msgs.msg import Joy
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from enum import IntEnum
from threading import Lock
import time
import numpy as np

from rclpy.duration import Duration

from bosdyn.api import arm_command_pb2
from bosdyn_msgs.msg import ArmVelocityCommandRequest
from bosdyn_msgs.conversions import convert

class LocomotionHint(IntEnum):
    HINT_AUTO = 1
    HINT_TROT = 2
    HINT_SPEED_SELECT_TROT = 3
    HINT_CRAWL = 4
    HINT_SPEED_SELECT_CRAWL = 10
    HINT_AMBLE = 5
    HINT_SPEED_SELECT_AMBLE = 6

    @classmethod
    def _missing_(cls, value):
        return cls.HINT_AUTO

class StairsMode(IntEnum):
    STAIRS_MODE_OFF = 1
    STAIRS_MODE_ON = 2
    STAIRS_MODE_AUTO = 3

    @classmethod
    def _missing_(cls, value):
        return cls.STAIRS_MODE_AUTO

class GripperPoseType(IntEnum):
    NO_POSE = 1
    LOOK_FORWARD = 2
    LOOK_LEFT = 3
    LOOK_RIGHT = 4
    LOOK_FORWARD_HIGH = 5

    @classmethod
    def _missing_(cls, value):
        return cls.NO_POSE

gripper_poses = {
    GripperPoseType.LOOK_FORWARD : (np.array([0.5, 0.0, 0.6]), np.array([0., 0., 0., 1.])),
    GripperPoseType.LOOK_LEFT : (np.array([0.5, 0.25, 0.6]), np.array([0., 0., 0.5, 0.866])),
    GripperPoseType.LOOK_RIGHT : (np.array([0.5, -0.25, 0.6]), np.array([0., 0., -0.5, 0.866])),
    GripperPoseType.LOOK_FORWARD_HIGH : (np.array([0.5, 0.0, 0.9]), np.array([0., 0., 0., 1.])),
}

class TeleopFuncs:
    """
    Handles commands to execute discrete services (e.g. power-on/sit/stand 
    etc.) from joystick commands. Complementary to teleop_twist_joy, which 
    handles control (velocity) commands.
    """

    def __init__(
        self, 
        spot_wrapper, 
        movement_query_fn,
        power_on_pause_secs=3,
        sit_stand_pause_secs=5,
        toggle_pause_secs=0.5,
    ) -> None:
        self.spot_wrapper = spot_wrapper
        self.movement_query_fn = movement_query_fn
        self.power_on_pause_secs = power_on_pause_secs
        self.sit_stand_pause_secs = sit_stand_pause_secs
        self.toggle_pause_secs = toggle_pause_secs
        self.pause = False
        self.lock = Lock()

        self.valid_locomotion_hints = [
            (LocomotionHint.HINT_AUTO, "AUTO"),
            (LocomotionHint.HINT_TROT, "TROT"),
            (LocomotionHint.HINT_SPEED_SELECT_TROT, "TROT WITH STOP"),
            (LocomotionHint.HINT_CRAWL, "CRAWL"),
            (LocomotionHint.HINT_SPEED_SELECT_CRAWL, "CRAWL WITH STOP"),
            (LocomotionHint.HINT_AMBLE, "AMBLE"),
            (LocomotionHint.HINT_SPEED_SELECT_AMBLE, "AMBLE WITH STOP"),
        ]
        self.locomotion_mode_idx = 0

        self.valid_stair_hints = [
            (StairsMode.STAIRS_MODE_OFF, "OFF"),
            (StairsMode.STAIRS_MODE_ON,  "ON"),
            (StairsMode.STAIRS_MODE_AUTO, "AUTOSELECT"),
        ]
        self.stair_mode_idx = 0

    def handle_joy(self, joy_msg, node_time):
        enable_hand = joy_msg.buttons[5] == 1
        if enable_hand:
            self._handle_arm_movement(joy_msg, node_time)

        enable = joy_msg.axes[2] < -0.99
        if enable:
            with self.lock:
                print("Enabled:", self.pause)
            trigger_check = False
            toggle_check = False

            toggle_power = False
            toggle_sit_stand = False
            toggle_locomotion_mode = False
            toggle_stairs_mode = False
            set_gripper_pose_mode = False

            with self.lock:
                # Check requests that will trigger robot actions with refractory period.
                if not self.pause:
                    toggle_power = joy_msg.axes[5] < -0.9 # Power on/off request
                    toggle_sit_stand = joy_msg.axes[7] > 0.9 # Sit/stand request
                    toggle_locomotion_mode = joy_msg.axes[6] > 0.9 # Locomotion mode change request
                    toggle_stairs_mode = joy_msg.axes[6] < -0.9 # Stairs mode change request

                    set_gripper_pose_mode = any(joy_msg.buttons[:4])
                    if joy_msg.buttons[3] and joy_msg.buttons[0]:
                        gripper_pose = GripperPoseType.LOOK_FORWARD_HIGH
                    elif joy_msg.buttons[3]:
                        gripper_pose = GripperPoseType.LOOK_FORWARD
                    elif joy_msg.buttons[0]:
                        gripper_pose = GripperPoseType.NO_POSE

                self.pause = (
                    toggle_power 
                    or toggle_sit_stand 
                    or toggle_locomotion_mode 
                    or toggle_stairs_mode
                    or set_gripper_pose_mode
                )

            # Handle mode change requests (only highest priority request)
            if toggle_power:
                self._handle_toggle_power()
            elif toggle_sit_stand:
                self._handle_toggle_sit_stand()
            elif toggle_locomotion_mode:
                self._handle_toggle_locomotion_mode()
            elif toggle_stairs_mode:
                self._handle_toggle_stairs_mode()
            elif set_gripper_pose_mode:
                self._handle_gripper_pose(gripper_pose)

    def _handle_toggle_power(self):
        print("Received power on/off command")
        if self.spot_wrapper.check_is_powered_on():
            resp = self.spot_wrapper.safe_power_off()
            print("ON --> OFF", resp[0], resp[1])
        else:
            resp = self.spot_wrapper.power_on()
            print("OFF --> ON", resp[0], resp[1])

        # BD API is non-blocking, so we add a little wait afterward
        time.sleep(self.power_on_pause_secs)

        with self.lock:
            self.pause = False

    def _handle_toggle_sit_stand(self):
        print("Received sit/stand command")
        if self.movement_query_fn is not None and not self.movement_query_fn(autonomous_command=False):
            return "Not changing sit/stand. Robot motion not allowed!"
        
        # We check if it is sitting first. There can be occasions
        # where it is registered as both in sitting and standing
        # states by the wrapper. In these ambiguous situations, it
        # is safer to try and stand (from a known sitting position,
        # or a standing position) than to try and sit into a position
        # of unknown stability.
        if self.spot_wrapper.is_sitting:
            resp = self.spot_wrapper.stand()
            print("SIT --> STAND:", resp[0], resp[1])
        else:
            resp = self.spot_wrapper.sit()
            print("STAND --> SIT", resp[0], resp[1])

        # BD API is non-blocking, so we add a little wait afterward
        time.sleep(self.sit_stand_pause_secs)

        with self.lock:
            self.pause = False

    def _handle_toggle_locomotion_mode(self):
        self.locomotion_mode_idx = (self.locomotion_mode_idx + 1) % len(self.valid_locomotion_hints)
        locomotion_hint, msg = self.valid_locomotion_hints[self.locomotion_mode_idx]
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.locomotion_hint = locomotion_hint
            self.spot_wrapper.set_mobility_params(mobility_params)
            print("Set locomotion mode to: ", msg)
        except Exception as e:
            print("Error setting locomotion mode:{}".format(e))
        
        # BD API is non-blocking, so we add a little wait afterward
        time.sleep(self.toggle_pause_secs)

        with self.lock:
            self.pause = False

    def _handle_toggle_stairs_mode(self):
        self.stair_mode_idx = (self.stair_mode_idx + 1) % len(self.valid_stair_hints)
        stair_hint, msg = self.valid_stair_hints[self.stair_mode_idx]
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.stair_hint = stair_hint
            self.spot_wrapper.set_mobility_params(mobility_params)
            print("Set stair mode to: ", msg)
        except Exception as e:
            print("Error setting stair mode:{}".format(e))

        # BD API is non-blocking, so we add a little wait afterward
        time.sleep(self.toggle_pause_secs)

        with self.lock:
            self.pause = False

    def _handle_gripper_pose(self, gripper_pose_type):
        if gripper_pose_type is GripperPoseType.NO_POSE:
            self.spot_wrapper.spot_arm.arm_stow()
        else:
            pos, ori = gripper_poses[gripper_pose_type]
            data = np.concatenate([pos, ori])
            self.spot_wrapper.spot_arm.hand_pose(
                x=pos[0], y=pos[1], z=pos[2],
                qx=ori[0], qy=ori[1], qz=ori[2], qw=ori[3],
            )
            self.spot_wrapper.spot_arm.gripper_open()

    def _handle_arm_movement(self, joy_msg, node_time):
        move_in = joy_msg.axes[1] < -0.8
        move_out = joy_msg.axes[1] > 0.8
        move_down = joy_msg.axes[4] < -0.8
        move_up = joy_msg.axes[4] > 0.8
        turn_ccw = joy_msg.axes[3] > 0.8
        turn_cw = joy_msg.axes[3] < -0.8

        v_r = 0.0
        v_r = -0.7 if move_in else v_r
        v_r = 0.7 if move_out else v_r

        v_z = 0.0
        v_z = -0.7 if move_down else v_z
        v_z = 0.7 if move_up else v_z

        v_theta = 0.0
        v_theta = 0.7 if turn_ccw else v_theta
        v_theta = -0.7 if turn_cw else v_theta

        msg = ArmVelocityCommandRequest()
        msg.end_time = (node_time + Duration(seconds=0.4)).to_msg()
        msg.command.cylindrical_velocity.linear_velocity.r = v_r
        msg.command.cylindrical_velocity.linear_velocity.z = v_z
        msg.command.cylindrical_velocity.linear_velocity.theta = v_theta
        msg.command.cylindrical_velocity.max_linear_velocity.data = 0.5
        msg.command.command_choice = 1
        msg.has_field = 16

        try:
            proto_command = arm_command_pb2.ArmVelocityCommand.Request()
            convert(msg, proto_command)
            result, message = self.spot_wrapper.spot_arm.handle_arm_velocity(
                arm_velocity_command=proto_command, cmd_duration=0.2
            )
            if not result:
                print(f"Failed to execute arm velocity command: {message}")
        except Exception as e:
            print(str(e))
