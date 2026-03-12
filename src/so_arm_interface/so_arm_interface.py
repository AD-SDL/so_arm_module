#!/usr/bin/env python3
"""
SOArm Robot Interface
=====================
Clean interface layer for the SO-Arm robot over LeRobot.

Connection model
----------------
The interface operates in one of two modes:

    manual  - A background hold thread continuously sends the current target
              to the robot at control_fps, keeping the arm locked in position.

              On connect(), the arm's current position is read and set as the
              initial hold target so the arm never drops on startup.

              moveJ() updates the hold target and blocks until all joints
              converge within tolerance (or timeout), then returns. The hold
              thread keeps sending the target after moveJ returns so the arm
              stays locked at the new position.

              getJ() and home() are also available.

    policy  - The hold thread is stopped and the robot connection is handed
              off exclusively to the policy loop for the duration of
              run_episode(). Manual actions raise PolicyModeError while active.
              On exit, connect() re-seeds the hold target from the arm's
              post-episode position and the hold thread resumes.

Mode switches automatically on run_episode() entry and exit, even on error.

Supported methods:
    connect()                             - Connect and start hold thread
    disconnect()                          - Stop hold thread and disconnect
    getJ()                                - Read current joint positions (degrees)
    home(timeout, tolerance)              - Move all joints to zero
    moveJ(location, timeout, tolerance)   - Move to joint positions via LocationArgument,
                                            block until converged, keep holding
    load_policy(path)                     - Load an ACT policy checkpoint
    run_episode(episode_length)           - Deploy loaded policy for one episode

Note: All joint positions are in degrees, matching the LeRobot SO-Arm schema.
"""

import contextlib
import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from madsci.common.types.location_types import LocationArgument
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint schema — must match LeRobot SO-Arm observation/action schema
# ---------------------------------------------------------------------------

STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

NUM_JOINTS = len(STATE_KEYS)

# Control loop
DEFAULT_CONTROL_FPS = 60
DEFAULT_EPISODE_LENGTH = 15.0  # seconds
DEFAULT_MOVE_TIMEOUT = 10.0  # seconds
DEFAULT_TOLERANCE = 3.0  # degrees — loosened to match SO-Arm positional noise

# Policy paths
POLICY_PICK_AND_PLACE_CUBE = (
    "/home/vision/humanoids/lerobot_workspace/outputs"
    "/cube_bowl_training_150ep/checkpoints/last/pretrained_model"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PolicyModeError(RuntimeError):
    """Raised when a manual action is attempted during policy execution."""


class PolicyNotLoadedError(RuntimeError):
    """Raised when run_episode is called before a policy has been loaded."""


class NotConnectedError(RuntimeError):
    """Raised when a robot action is attempted before connecting."""


# ---------------------------------------------------------------------------
# Interface mode
# ---------------------------------------------------------------------------


class InterfaceMode(str, Enum):
    """Operating mode of the SOArmInterface."""

    MANUAL = "manual"
    POLICY = "policy"


# ---------------------------------------------------------------------------
# Normalizer helpers
# ---------------------------------------------------------------------------


def _load_normalizers(
    policy_path: Path, device: str
) -> tuple[Optional[dict], Optional[dict]]:
    """Load pre/post normalizer weights from a policy checkpoint directory."""
    pre_files = list(policy_path.glob("*normalizer_processor.safetensors"))
    post_files = list(policy_path.glob("*unnormalizer_processor.safetensors"))

    if not pre_files or not post_files:
        return None, None

    pre_weights = load_file(str(pre_files[0]), device=device)
    post_weights = load_file(str(post_files[0]), device=device)
    return pre_weights, post_weights


def _normalize_obs(batch: dict, pre_weights: Optional[dict], device: str) -> dict:
    if pre_weights is None:
        return batch
    normalized = {}
    for key, value in batch.items():
        mean_key = f"{key}.mean"
        std_key = f"{key}.std"
        if mean_key in pre_weights and std_key in pre_weights:
            mean = pre_weights[mean_key].to(device)
            std = pre_weights[std_key].to(device)
            while mean.dim() < value.dim():
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            normalized[key] = (value - mean) / (std + 1e-8)
        else:
            normalized[key] = value
    return normalized


def _denormalize_action(
    action_tensor: torch.Tensor,
    post_weights: Optional[dict],
    device: str,
) -> torch.Tensor:
    if post_weights is None:
        return action_tensor
    mean_key = "action.mean"
    std_key = "action.std"
    if mean_key in post_weights and std_key in post_weights:
        mean = post_weights[mean_key].to(device)
        std = post_weights[std_key].to(device)
        while mean.dim() < action_tensor.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return action_tensor * std + mean
    return action_tensor


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class SOArmInterface:
    """
    Interface layer for the SO-Arm robot.

    A background hold thread starts on connect() and continuously sends the
    current hold target to the robot at control_fps. moveJ() updates the
    target and waits for convergence, then returns while the hold thread
    keeps the arm locked at that position.

    Args:
        robot_port:    Serial port for the SO-Arm follower (e.g. /dev/ttyACM0).
        camera_serial: RealSense camera serial number or name.
        device:        Torch device for policy inference ('cuda' or 'cpu').
        control_fps:   Control loop frequency in Hz.
        camera_fps:    Camera capture FPS.
        camera_width:  Camera frame width in pixels.
        camera_height: Camera frame height in pixels.
    """

    def __init__(  # noqa: ANN204
        self,
        robot_port: str = "/dev/ttyACM0",
        camera_serial: str = "327122076093",
        device: str = "cuda",
        control_fps: int = DEFAULT_CONTROL_FPS,
        camera_fps: int = 60,
        camera_width: int = 640,
        camera_height: int = 480,
    ):
        """
        Initialize the SO-Arm interface without connecting to the robot.

        Call connect() to establish the serial connection and start the
        background hold thread.

        Args:
            robot_port:    Serial port for the SO-Arm follower. Default /dev/ttyACM0.
            camera_serial: RealSense camera serial number or name.
            device:        Torch device for policy inference. Default 'cuda'.
            control_fps:   Control loop frequency in Hz for the hold thread
                        and policy execution. Default 60.
            camera_fps:    Camera capture FPS. Default 60.
            camera_width:  Camera frame width in pixels. Default 640.
            camera_height: Camera frame height in pixels. Default 480.
        """
        self.robot_port = robot_port
        self.camera_serial = camera_serial
        self.device = device
        self.control_fps = control_fps
        self.camera_fps = camera_fps
        self.camera_width = camera_width
        self.camera_height = camera_height

        self._robot = None
        self._policy: Optional[ACTPolicy] = None
        self._pre_weights: Optional[dict] = None
        self._post_weights: Optional[dict] = None
        self._policy_path: Optional[str] = None

        self._mode = InterfaceMode.MANUAL
        self._mode_lock = threading.Lock()

        # Hold thread state
        self._hold_target = dict.fromkeys(STATE_KEYS, 0.0)
        self._hold_target_lock = threading.Lock()
        self._hold_thread: Optional[threading.Thread] = None
        self._hold_stop_event = threading.Event()

        # Last observation cache — written exclusively by the hold thread,
        # read by getJ() and moveJ() to avoid concurrent serial access.
        self._last_obs: Optional[dict] = None
        self._last_obs_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> InterfaceMode:
        """Current operating mode of the interface."""
        with self._mode_lock:
            return self._mode

    @property
    def is_connected(self) -> bool:
        """True if the robot is currently connected."""
        return self._robot is not None

    @property
    def policy_loaded(self) -> bool:
        """True if a policy is currently loaded."""
        return self._policy is not None

    @property
    def policy_path(self) -> Optional[str]:
        """Path to the currently loaded policy checkpoint, or None."""
        return self._policy_path

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Connect to the robot and start the background hold thread.

        Reads the current arm position immediately and uses it as the initial
        hold target so the arm is locked in place from the moment of connection.
        No-op if already connected.
        """
        if self._robot is not None:
            return

        self._robot = make_robot_from_config(self._build_robot_config())
        self._robot.connect()

        # Seed hold target and obs cache from current position before the thread
        # starts — safe to call get_observation() directly here since the hold
        # thread is not running yet, so there is no concurrent serial access.
        obs = self._robot.get_observation()
        joint_obs = {k: obs[k] for k in STATE_KEYS}
        with self._hold_target_lock:
            self._hold_target = joint_obs.copy()
        with self._last_obs_lock:
            self._last_obs = joint_obs.copy()

        self._hold_stop_event.clear()
        self._hold_thread = threading.Thread(
            target=self._hold_loop, daemon=True, name="soarm-hold"
        )
        self._hold_thread.start()

    def disconnect(self) -> None:
        """
        Stop the hold thread and disconnect from the robot.
        No-op if not connected. Safe to call from any mode for cleanup.
        """
        self._stop_hold_thread()

        if self._robot is None:
            return
        try:
            self._robot.disconnect()
        finally:
            self._robot = None

    # ------------------------------------------------------------------
    # Manual mode actions
    # ------------------------------------------------------------------

    def getJ(self) -> dict:  # noqa: N802
        """
        Read current joint positions from the robot.

        Returns from the hold thread's observation cache — no direct serial
        access, so it is safe to call at any time without port contention.

        Returns:
            Dict mapping joint name → position (degrees) for each STATE_KEY.

        Raises:
            PolicyModeError:   If called during policy execution.
            NotConnectedError: If the robot is not connected.
        """
        self._require_manual()
        with self._last_obs_lock:
            if self._last_obs is None:
                raise NotConnectedError(
                    "No observation available yet — hold thread may not have ticked."
                )
            return self._last_obs.copy()

    def home(self, timeout: float = DEFAULT_MOVE_TIMEOUT) -> None:
        """
        Move all joints to zero. Blocks until the arm is close to zero or
        timeout expires, then returns while the hold thread keeps holding.

        Args:
            timeout: Max seconds to wait before returning. Default 10.0s.

        Raises:
            PolicyModeError:   If called during policy execution.
            NotConnectedError: If the robot is not connected.
        """
        self.moveJ(
            LocationArgument(location=dict.fromkeys(STATE_KEYS, 0.0)), timeout=timeout
        )

    def moveJ(  # noqa: N802
        self,
        location: LocationArgument,
        timeout: float = DEFAULT_MOVE_TIMEOUT,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        """
        Move to the joint configuration specified by a LocationArgument, block
        until the arm is within tolerance (or timeout expires), then return
        while the hold thread continues holding that position.

        Args:
            location:  LocationArgument whose location field is a dict mapping
                       joint name → target position (degrees). Keys must be a
                       subset of STATE_KEYS. Missing joints default to 0.0.
            timeout:   Max seconds to wait before returning. Default 10.0s.
            tolerance: Per-joint threshold in degrees to consider arrived.
                       Default 3.0°.

        Raises:
            PolicyModeError:   If called during policy execution.
            NotConnectedError: If the robot is not connected.
            ValueError:        If location.location is not a dict, or contains
                               unknown joint names.
        """
        self._require_manual()
        positions = location.location
        if not isinstance(positions, dict):
            raise ValueError(
                f"location.location must be a dict mapping joint names to positions in degrees, got {type(positions).__name__}."
            )
        invalid = set(positions) - set(STATE_KEYS)
        if invalid:
            raise ValueError(
                f"Unknown joint names: {invalid}. Valid joints: {STATE_KEYS}"
            )

        targets = {k: positions.get(k, 0.0) for k in STATE_KEYS}

        # Update hold target — background thread picks this up on next tick
        with self._hold_target_lock:
            self._hold_target = targets.copy()

        # Wait until all joints are within tolerance or timeout expires.
        # Reads from the obs cache — hold thread owns all serial access.
        dt = 1.0 / self.control_fps
        start = time.perf_counter()

        while time.perf_counter() - start < timeout:
            with self._last_obs_lock:
                obs = self._last_obs.copy() if self._last_obs else None
            if obs and all(abs(obs[k] - targets[k]) < tolerance for k in STATE_KEYS):
                return
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load_policy(self, policy_path: str) -> None:
        """
        Load (or reload) an ACT policy from a checkpoint directory.
        Safe to call at any time in manual mode.

        Args:
            policy_path: Path to the pretrained_model checkpoint directory.

        Raises:
            PolicyModeError:   If called during policy execution.
            FileNotFoundError: If the checkpoint directory does not exist.
        """
        self._require_manual()
        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {path}")

        self._policy = ACTPolicy.from_pretrained(str(path))
        self._policy.to(self.device)
        self._policy.eval()
        self._pre_weights, self._post_weights = _load_normalizers(path, self.device)
        self._policy_path = str(path)

    # ------------------------------------------------------------------
    # Policy execution
    # ------------------------------------------------------------------

    def run_episode(self, episode_length: float = DEFAULT_EPISODE_LENGTH) -> None:
        """
        Deploy the loaded policy for one episode. Blocking.

        Stops the hold thread and hands the robot connection exclusively to the
        policy loop. On exit, connect() re-seeds the hold target from the arm's
        post-episode position and the hold thread resumes from there.

        Args:
            episode_length: Duration of the episode in seconds.

        Raises:
            PolicyModeError:      If an episode is already running.
            PolicyNotLoadedError: If no policy has been loaded.
        """
        with self._mode_lock:
            if self._mode == InterfaceMode.POLICY:
                raise PolicyModeError("An episode is already running.")
            if self._policy is None:
                raise PolicyNotLoadedError(
                    "No policy loaded. Call load_policy() first."
                )
            self._mode = InterfaceMode.POLICY

        try:
            # Release hold thread and connection so the policy loop owns the port
            self.disconnect()
            self._run_policy_loop(episode_length)
        finally:
            # Reconnect — seeds hold target from post-episode position automatically
            self.connect()
            with self._mode_lock:
                self._mode = InterfaceMode.MANUAL

    def pick_and_place_cube(
        self, episode_length: float = DEFAULT_EPISODE_LENGTH
    ) -> None:
        """
        Run the cube-to-bowl pick and place policy for one episode.

        Loads the policy from POLICY_PICK_AND_PLACE_CUBE if not already loaded,
        then runs run_episode(). Arm returns to holding post-episode position.

        Args:
            episode_length: Duration of the episode in seconds.

        Raises:
            PolicyModeError: If an episode is already running.
            FileNotFoundError: If the policy checkpoint is not found.
        """
        if self._policy_path != POLICY_PICK_AND_PLACE_CUBE or self._policy is None:
            self.load_policy(POLICY_PICK_AND_PLACE_CUBE)
        self.run_episode(episode_length=episode_length)

    # ------------------------------------------------------------------
    # Internal — hold thread
    # ------------------------------------------------------------------

    def _hold_loop(self) -> None:
        """
        Background thread: the sole owner of serial communication in manual mode.
        Each tick:
          1. Reads get_observation() and caches it in _last_obs.
          2. Sends the current hold target via send_action().
        All serial access is serialized here — getJ() and moveJ() read from
        _last_obs rather than calling get_observation() directly.
        """
        dt = 1.0 / self.control_fps

        while not self._hold_stop_event.is_set():
            loop_start = time.perf_counter()

            if self._robot is not None:
                with self._hold_target_lock:
                    target = self._hold_target.copy()
                with contextlib.suppress(Exception):
                    obs = self._robot.get_observation()
                    with self._last_obs_lock:
                        self._last_obs = {k: obs[k] for k in STATE_KEYS if k in obs}
                    self._robot.send_action(target)

            sleep_time = dt - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _stop_hold_thread(self) -> None:
        """Signal the hold thread to stop and wait for it to exit."""
        self._hold_stop_event.set()
        if self._hold_thread is not None:
            self._hold_thread.join(timeout=2.0)
            self._hold_thread = None

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _build_robot_config(self) -> SOFollowerRobotConfig:
        return SOFollowerRobotConfig(
            port=self.robot_port,
            id="follower_arm1",
            cameras={
                "front": RealSenseCameraConfig(
                    serial_number_or_name=self.camera_serial,
                    fps=self.camera_fps,
                    width=self.camera_width,
                    height=self.camera_height,
                )
            },
        )

    def _require_manual(self) -> None:
        with self._mode_lock:
            if self._mode != InterfaceMode.MANUAL:
                raise PolicyModeError(
                    f"Action unavailable in '{self._mode}' mode. "
                    "Wait for the current episode to finish."
                )
        if self._robot is None:
            raise NotConnectedError("Robot is not connected. Call connect() first.")

    def _run_policy_loop(self, episode_length: float) -> None:
        """
        Core policy control loop. Opens its own robot connection so the
        serial port is exclusively owned for the episode duration.
        """
        dt = 1.0 / self.control_fps
        robot = None

        try:
            robot = make_robot_from_config(self._build_robot_config())
            robot.connect()

            self._policy.reset()
            start_time = time.perf_counter()

            while time.perf_counter() - start_time < episode_length:
                loop_start = time.perf_counter()

                obs = robot.get_observation()
                state = np.array([obs[k] for k in STATE_KEYS], dtype=np.float32)
                image = obs["front"].astype(np.float32) / 255.0

                batch = {
                    "observation.state": (
                        torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    ),
                    "observation.images.front": (
                        torch.from_numpy(image)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(self.device)
                    ),
                }

                batch = _normalize_obs(batch, self._pre_weights, self.device)

                with torch.inference_mode():
                    action_tensor = self._policy.select_action(batch)

                action_tensor = _denormalize_action(
                    action_tensor, self._post_weights, self.device
                )

                action_np = action_tensor.squeeze().cpu().numpy()
                action_dict = {k: float(v) for k, v in zip(STATE_KEYS, action_np)}
                robot.send_action(action_dict)

                sleep_time = dt - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            if robot is not None:
                with contextlib.suppress(Exception):
                    robot.disconnect()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arm = SOArmInterface(robot_port="/dev/ttyACM0", camera_serial="327122076093")

    try:
        arm.connect()
        logger.info("Connected — arm locked at current position.")

        state = arm.getJ()
        logger.info("Joint positions: %s", state)

        arm.home()
        logger.info("Homed.")

        target = LocationArgument(
            location={
                "shoulder_pan.pos": 0.48,
                "shoulder_lift.pos": -81.45,
                "elbow_flex.pos": 74.81,
                "wrist_flex.pos": 46.77,
                "wrist_roll.pos": 8.66,
                "gripper.pos": 0.0,
            }
        )
        arm.moveJ(target)
        logger.info("Moved.")

        arm.load_policy(POLICY_PICK_AND_PLACE_CUBE)
        logger.info("Policy loaded.")

        logger.info("Running pick and place episode...")
        arm.pick_and_place_cube()
        logger.info("Episode complete.")

    finally:
        arm.disconnect()
        logger.info("Disconnected.")
