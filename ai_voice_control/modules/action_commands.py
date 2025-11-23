"""
Action Commands Module
Implements Command Pattern for all system actions
Provides better testability, undo support, and extensibility
"""
import time
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of action execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ActionResult:
    """Result of an action execution"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    status: ActionStatus = ActionStatus.PENDING


class Action(ABC):
    """
    Abstract base class for all actions (Command Pattern)
    Each action encapsulates a request as an object
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.result: Optional[ActionResult] = None
        self.start_time: float = 0.0
        self.status = ActionStatus.PENDING

    @abstractmethod
    def execute(self) -> ActionResult:
        """Execute the action"""
        pass

    def can_undo(self) -> bool:
        """Check if this action can be undone"""
        return False

    def undo(self) -> ActionResult:
        """Undo the action if possible"""
        return ActionResult(
            success=False,
            message="Undo not supported for this action",
            status=ActionStatus.FAILED
        )

    def validate(self) -> bool:
        """Validate parameters before execution"""
        return True

    def _start_execution(self):
        """Mark execution start"""
        self.start_time = time.time()
        self.status = ActionStatus.RUNNING

    def _end_execution(self, result: ActionResult) -> ActionResult:
        """Mark execution end and calculate time"""
        result.execution_time = time.time() - self.start_time
        result.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
        self.status = result.status
        self.result = result
        return result


class OpenAppAction(Action):
    """Action to open an application using the app registry"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.app_name = params.get("app_name", "")
        self.launched_app = None

    def validate(self) -> bool:
        return bool(self.app_name)

    def execute(self) -> ActionResult:
        """Execute application launch"""
        self._start_execution()

        try:
            from app_registry import get_app_registry

            registry = get_app_registry()
            app = registry.find_application(self.app_name)

            if not app:
                return self._end_execution(ActionResult(
                    success=False,
                    message=f"Application '{self.app_name}' not found",
                    error="APP_NOT_FOUND"
                ))

            # Launch the application
            success = registry.launch_application(app)

            if success:
                self.launched_app = app
                return self._end_execution(ActionResult(
                    success=True,
                    message=f"Opened {app.display_name}",
                    data={"app_name": app.display_name}
                ))
            else:
                return self._end_execution(ActionResult(
                    success=False,
                    message=f"Failed to launch {app.display_name}",
                    error="LAUNCH_FAILED"
                ))

        except Exception as e:
            logger.error(f"Error opening app: {e}")
            return self._end_execution(ActionResult(
                success=False,
                message=f"Error: {str(e)}",
                error="EXCEPTION"
            ))


class CloseAppAction(Action):
    """Action to close an application"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.app_name = params.get("app_name", "")

    def execute(self) -> ActionResult:
        """Execute application close"""
        self._start_execution()

        try:
            # Use pkill to close the application
            result = subprocess.run(
                ["pkill", "-f", self.app_name],
                capture_output=True,
                text=True
            )

            return self._end_execution(ActionResult(
                success=True,
                message=f"Closed {self.app_name}",
                data={"returncode": result.returncode}
            ))

        except Exception as e:
            logger.error(f"Error closing app: {e}")
            return self._end_execution(ActionResult(
                success=False,
                message=f"Error: {str(e)}",
                error="EXCEPTION"
            ))


class WaitAction(Action):
    """Action to wait for a specified duration"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.seconds = params.get("seconds", 1.0)

    def validate(self) -> bool:
        return isinstance(self.seconds, (int, float)) and 0 < self.seconds <= 60

    def execute(self) -> ActionResult:
        """Execute wait"""
        self._start_execution()

        try:
            time.sleep(self.seconds)
            return self._end_execution(ActionResult(
                success=True,
                message=f"Waited {self.seconds} seconds"
            ))

        except Exception as e:
            return self._end_execution(ActionResult(
                success=False,
                message=f"Error during wait: {str(e)}",
                error="EXCEPTION"
            ))


class ExecuteCommandAction(Action):
    """Action to execute a shell command"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.command = params.get("command", "")
        self.timeout = params.get("timeout", 30)

    def validate(self) -> bool:
        return bool(self.command)

    def execute(self) -> ActionResult:
        """Execute shell command"""
        self._start_execution()

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return self._end_execution(ActionResult(
                success=result.returncode == 0,
                message="Command executed",
                data={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            ))

        except subprocess.TimeoutExpired:
            return self._end_execution(ActionResult(
                success=False,
                message=f"Command timed out after {self.timeout}s",
                error="TIMEOUT"
            ))
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return self._end_execution(ActionResult(
                success=False,
                message=f"Error: {str(e)}",
                error="EXCEPTION"
            ))


class TypeTextAction(Action):
    """Action to type text (using keyboard automation)"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.text = params.get("text", "")

    def execute(self) -> ActionResult:
        """Execute text typing"""
        self._start_execution()

        try:
            import pyautogui
            pyautogui.typewrite(self.text, interval=0.05)

            return self._end_execution(ActionResult(
                success=True,
                message=f"Typed text: {self.text[:30]}..."
            ))

        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return self._end_execution(ActionResult(
                success=False,
                message=f"Error: {str(e)}",
                error="EXCEPTION"
            ))


class PressKeyAction(Action):
    """Action to press a keyboard key"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.key = params.get("key", "")

    def execute(self) -> ActionResult:
        """Execute key press"""
        self._start_execution()

        try:
            import pyautogui
            pyautogui.press(self.key)

            return self._end_execution(ActionResult(
                success=True,
                message=f"Pressed key: {self.key}"
            ))

        except Exception as e:
            logger.error(f"Error pressing key: {e}")
            return self._end_execution(ActionResult(
                success=False,
                message=f"Error: {str(e)}",
                error="EXCEPTION"
            ))


# Factory for creating actions
class ActionFactory:
    """
    Factory Pattern for creating action objects
    Centralizes action creation and makes it easy to add new actions
    """

    _action_types = {
        "open_app": OpenAppAction,
        "close_app": CloseAppAction,
        "wait": WaitAction,
        "execute_command": ExecuteCommandAction,
        "type_text": TypeTextAction,
        "press_key": PressKeyAction,
    }

    @classmethod
    def create_action(cls, action_type: str, params: Dict[str, Any]) -> Optional[Action]:
        """
        Create an action object

        Args:
            action_type: Type of action to create
            params: Parameters for the action

        Returns:
            Action object or None if type not found
        """
        action_class = cls._action_types.get(action_type)

        if not action_class:
            logger.warning(f"Unknown action type: {action_type}")
            return None

        try:
            action = action_class(params)
            return action
        except Exception as e:
            logger.error(f"Error creating action {action_type}: {e}")
            return None

    @classmethod
    def register_action(cls, action_type: str, action_class: type):
        """Register a new action type"""
        cls._action_types[action_type] = action_class
        logger.info(f"Registered action type: {action_type}")

    @classmethod
    def get_available_actions(cls) -> List[str]:
        """Get list of available action types"""
        return list(cls._action_types.keys())


class ActionQueue:
    """
    Queue for managing and executing multiple actions
    Supports sequential execution with error handling
    """

    def __init__(self):
        self.actions: List[Action] = []
        self.results: List[ActionResult] = []
        self.stop_on_error = True

    def add_action(self, action: Action):
        """Add an action to the queue"""
        self.actions.append(action)

    def execute_all(self) -> List[ActionResult]:
        """
        Execute all actions in the queue

        Returns:
            List of ActionResults
        """
        self.results = []

        for i, action in enumerate(self.actions):
            logger.info(f"Executing action {i+1}/{len(self.actions)}: {action.__class__.__name__}")

            # Validate before execution
            if not action.validate():
                result = ActionResult(
                    success=False,
                    message="Action validation failed",
                    error="VALIDATION_FAILED",
                    status=ActionStatus.FAILED
                )
                self.results.append(result)

                if self.stop_on_error:
                    logger.warning("Stopping execution due to validation failure")
                    break
                continue

            # Execute the action
            result = action.execute()
            self.results.append(result)

            if not result.success and self.stop_on_error:
                logger.warning(f"Stopping execution due to error: {result.message}")
                break

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        total_time = sum(r.execution_time for r in self.results)

        return {
            "total_actions": total,
            "successful": successful,
            "failed": failed,
            "total_time": total_time,
            "success_rate": successful / total if total > 0 else 0
        }

    def clear(self):
        """Clear the queue"""
        self.actions.clear()
        self.results.clear()
