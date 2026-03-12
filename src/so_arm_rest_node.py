#!/usr/bin/env python3
"""REST node for the SO-Arm robot (LeRobot-based).

Example moveJ call:

    from madsci.client.node.rest_node_client import RestNodeClient
    from madsci.common.types.action_types import ActionRequest
    from madsci.common.types.location_types import LocationArgument

    client = RestNodeClient(url="http://localhost:3000")

    request = ActionRequest(
        action_name="moveJ",
        args={
            "location": LocationArgument(
                location={
                    "shoulder_pan.pos":  0.48,
                    "shoulder_lift.pos": -81.45,
                    "elbow_flex.pos":    74.81,
                    "wrist_flex.pos":    46.77,
                    "wrist_roll.pos":    8.66,
                    "gripper.pos":       0.0,
                },
            ).model_dump(mode="json"),
        },
    )
    client.send_action(request)
"""

from typing import Annotated, Optional

from madsci.common.types.action_types import ActionFailed
from madsci.common.types.location_types import LocationArgument
from madsci.common.types.node_types import RestNodeConfig
from madsci.node_module.helpers import action
from madsci.node_module.rest_node_module import RestNode

from so_arm_interface.so_arm_interface import (
    NotConnectedError,
    PolicyModeError,
    PolicyNotLoadedError,
    SOArmInterface,
)


class SOArmNodeConfig(RestNodeConfig):
    """Configuration for the SO-Arm node."""

    robot_port: str = "/dev/ttyACM0"
    """Serial port for the SO-Arm follower."""

    camera_serial: str = "327122076093"
    """RealSense camera serial number or name."""

    default_episode_length: float = 15.0
    """Default episode duration in seconds."""

    control_fps: int = 60
    """Control loop frequency in Hz."""


class SOArmNode(RestNode):
    """MADSci REST node for the SO-Arm robot."""

    robot: Optional[SOArmInterface] = None
    config: SOArmNodeConfig = SOArmNodeConfig()
    config_model = SOArmNodeConfig

    # ------------------------------------------------------------------
    # Lifecycle handlers
    # ------------------------------------------------------------------

    def startup_handler(self) -> None:
        """Connect to the robot."""
        self.robot = SOArmInterface(
            robot_port=self.config.robot_port,
            camera_serial=self.config.camera_serial,
            control_fps=self.config.control_fps,
        )
        self.robot.connect()
        self.logger.log_info("SOArm node initialized.")

    def shutdown_handler(self) -> None:
        """Disconnect the robot cleanly."""
        try:
            if self.robot is not None:
                self.robot.disconnect()
                self.robot = None
        except Exception as err:
            self.logger.log_error(f"Error during shutdown: {err}")

    def state_handler(self) -> None:
        """
        Periodically update node state with current joint positions.
        Backs off silently during policy execution to avoid port contention.
        """
        if self.robot is None:
            return

        if self.robot.mode.value == "policy":
            self.node_state["mode"] = "policy"
            return

        try:
            joint_state = self.robot.getJ()
            self.node_state = {
                "mode": "manual",
                "joint_positions": joint_state,
                "policy_loaded": self.robot.policy_loaded,
                "policy_path": self.robot.policy_path,
            }
        except (PolicyModeError, NotConnectedError):
            pass
        except Exception as err:
            self.logger.log_error(f"state_handler error: {err}")

    # ------------------------------------------------------------------
    # Actions — manual mode
    # ------------------------------------------------------------------

    @action(name="getJ", description="Read current joint positions from the SO-Arm.")
    def getJ(self) -> dict:  # noqa: N802
        """Return current joint positions keyed by joint name."""
        try:
            return self.robot.getJ()
        except (PolicyModeError, NotConnectedError) as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"getJ failed: {err}"])

    @action(
        name="home",
        description="Move all joints to zero. Blocks until arm arrives or timeout.",
    )
    def home(
        self,
        timeout: Annotated[float, "Max seconds to wait before returning."] = 10.0,
    ) -> Optional[ActionFailed]:
        """Move all joints to zero. Arm keeps holding after return."""
        try:
            self.robot.home(timeout=timeout)
            return None
        except NotConnectedError as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"home failed: {err}"])

    @action(name="moveJ", description="Move SO-Arm to a target joint configuration.")
    def moveJ(  # noqa: N802
        self,
        location: Annotated[
            LocationArgument,
            "Target joint positions as a LocationArgument. "
            "location field should be a dict mapping joint names to positions in degrees: "
            "{shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos}.",
        ],
        timeout: Annotated[float, "Max seconds to wait before returning."] = 10.0,
        tolerance: Annotated[float, "Per-joint arrival threshold in degrees."] = 3.0,
    ) -> Optional[ActionFailed]:
        """Move to the target joint configuration. Arm keeps holding after return."""
        try:
            self.robot.moveJ(
                location=location,
                timeout=timeout,
                tolerance=tolerance,
            )
            return None
        except (NotConnectedError, ValueError) as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"moveJ failed: {err}"])

    # ------------------------------------------------------------------
    # Actions — policy
    # ------------------------------------------------------------------

    @action(
        name="load_policy",
        description="Load (or reload) an ACT policy from a checkpoint directory.",
    )
    def load_policy(
        self,
        policy_path: Annotated[
            str, "Path to the pretrained_model checkpoint directory."
        ],
    ) -> Optional[ActionFailed]:
        """Load policy weights and normalizers. Only available in manual mode."""
        try:
            self.robot.load_policy(policy_path)
            return None
        except (PolicyModeError, FileNotFoundError) as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"load_policy failed: {err}"])

    @action(
        name="pick_and_place_cube",
        description="Run the cube-to-bowl pick and place policy for one episode.",
    )
    def pick_and_place_cube(
        self,
        episode_length: Annotated[
            Optional[float],
            "Episode duration in seconds. Defaults to config.default_episode_length.",
        ] = None,
    ) -> Optional[ActionFailed]:
        """Load the cube pick and place policy (if not already loaded) and run one episode."""
        duration = (
            episode_length
            if episode_length is not None
            else self.config.default_episode_length
        )
        try:
            self.robot.pick_and_place_cube(episode_length=duration)
            return None
        except (
            PolicyModeError,
            PolicyNotLoadedError,
            NotConnectedError,
            FileNotFoundError,
        ) as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"pick_and_place_cube failed: {err}"])

    @action(
        name="run_task",
        description="Load a policy from the given path and run one episode.",
    )
    def run_task(
        self,
        policy_path: Annotated[
            str, "Path to the pretrained_model checkpoint directory."
        ],
        episode_length: Annotated[
            Optional[float],
            "Episode duration in seconds. Defaults to config.default_episode_length.",
        ] = None,
    ) -> Optional[ActionFailed]:
        """Load any policy by path and run one episode. Blocking."""
        duration = (
            episode_length
            if episode_length is not None
            else self.config.default_episode_length
        )
        try:
            self.robot.load_policy(policy_path)
            self.robot.run_episode(episode_length=duration)
            return None
        except (
            PolicyModeError,
            PolicyNotLoadedError,
            NotConnectedError,
            FileNotFoundError,
        ) as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"run_task failed: {err}"])

    @action(
        name="run_episode",
        description="Deploy the already-loaded policy for one episode.",
    )
    def run_episode(
        self,
        episode_length: Annotated[
            Optional[float],
            "Episode duration in seconds. Defaults to config.default_episode_length.",
        ] = None,
    ) -> Optional[ActionFailed]:
        """Run one episode with the currently loaded policy. Blocking."""
        duration = (
            episode_length
            if episode_length is not None
            else self.config.default_episode_length
        )
        try:
            self.robot.run_episode(episode_length=duration)
            return None
        except (PolicyModeError, PolicyNotLoadedError, NotConnectedError) as err:
            return ActionFailed(errors=[str(err)])
        except Exception as err:
            return ActionFailed(errors=[f"run_episode failed: {err}"])

    # ------------------------------------------------------------------
    # Lifecycle passthrough
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the node, halting action execution."""
        self.node_status.paused = True
        return True

    def resume(self) -> None:
        """Resume the node after a pause."""
        self.node_status.paused = False
        return True

    def shutdown(self) -> None:
        """Shut down the node and disconnect the robot."""
        self.shutdown_handler()
        return True

    def reset(self) -> None:
        """Reset the node to its initial state."""
        return super().reset()

    def safety_stop(self) -> None:
        """Emergency stop — disconnect immediately regardless of mode."""
        self.node_status.stopped = True
        try:
            if self.robot is not None:
                self.robot.disconnect()
        except Exception as err:
            self.logger.log_error(f"Error during safety stop: {err}")
        return True

    def cancel(self) -> None:
        """Cancel the currently executing action."""
        self.node_status.cancelled = True
        return True


if __name__ == "__main__":
    node = SOArmNode()
    node.start_node()
