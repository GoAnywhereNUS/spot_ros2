from sensor_msgs.msg import Joy
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from enum import IntEnum
from threading import Lock
import time

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

    def handle_joy(self, joy_msg):
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

            with self.lock:
                # Check requests that will trigger robot actions with refractory period.
                if not self.pause:
                    toggle_power = joy_msg.axes[5] < -0.9 # Power on/off request
                    toggle_sit_stand = joy_msg.axes[7] > 0.9 # Sit/stand request
                    toggle_locomotion_mode = joy_msg.axes[6] > 0.9 # Locomotion mode change request
                    toggle_stairs_mode = joy_msg.axes[6] < -0.9 # Stairs mode change request
                self.pause = (
                    toggle_power 
                    or toggle_sit_stand 
                    or toggle_locomotion_mode 
                    or toggle_stairs_mode
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
