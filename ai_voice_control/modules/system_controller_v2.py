"""
System Controller V2 - Refactored with Design Patterns
Uses Command Pattern, Strategy Pattern, and proper error handling
Integrates with AppRegistry for robust application management
"""
import logging
from typing import Dict, Optional, List, Any
from action_commands import ActionFactory, ActionQueue, Action, ActionResult
from app_registry import get_app_registry
import config

logger = logging.getLogger(__name__)


class SystemControllerV2:
    """
    Refactored system controller using proper design patterns
    - Command Pattern for actions
    - Strategy Pattern for different execution strategies
    - Factory Pattern for action creation
    """

    def __init__(self):
        self.app_registry = get_app_registry()
        self.action_factory = ActionFactory()
        self.browser_controller = None  # Lazy initialization
        logger.info(f"SystemControllerV2 initialized with {len(self.app_registry.apps)} apps")

    def execute_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action from dictionary specification

        Args:
            action_dict: Dictionary with 'type' and parameters

        Returns:
            Result dictionary compatible with old interface
        """
        action_type = action_dict.get("type")
        params = {k: v for k, v in action_dict.items() if k != "type"}

        # Create action using factory
        action = self.action_factory.create_action(action_type, params)

        if not action:
            return {
                "success": False,
                "error": f"Unknown action type: {action_type}"
            }

        # Execute the action
        result = action.execute()

        # Convert to old format for compatibility
        return self._convert_result(result)

    def execute_actions(self, actions: List[Dict[str, Any]], stop_on_error: bool = True) -> List[Dict[str, Any]]:
        """
        Execute multiple actions in sequence

        Args:
            actions: List of action dictionaries
            stop_on_error: Whether to stop on first error

        Returns:
            List of results
        """
        queue = ActionQueue()
        queue.stop_on_error = stop_on_error

        # Create and add all actions
        for action_dict in actions:
            action_type = action_dict.get("type")
            params = {k: v for k, v in action_dict.items() if k != "type"}

            action = self.action_factory.create_action(action_type, params)
            if action:
                queue.add_action(action)

        # Execute all actions
        results = queue.execute_all()

        # Log summary
        summary = queue.get_summary()
        logger.info(f"Executed {summary['total_actions']} actions: "
                   f"{summary['successful']} successful, {summary['failed']} failed")

        # Convert to old format
        return [self._convert_result(r) for r in results]

    def _convert_result(self, result: ActionResult) -> Dict[str, Any]:
        """Convert ActionResult to dictionary format"""
        output = {
            "success": result.success,
            "message": result.message,
        }

        if result.data:
            output["data"] = result.data

        if result.error:
            output["error"] = result.error

        return output

    def search_applications(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Search for applications

        Returns:
            List of application dictionaries
        """
        apps = self.app_registry.search_applications(query, limit)

        return [
            {
                "name": app.name,
                "display_name": app.display_name,
                "description": app.description or "",
                "categories": ", ".join(app.categories[:3])
            }
            for app in apps
        ]

    def get_application_info(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific application"""
        app = self.app_registry.find_application(app_name)

        if not app:
            return None

        return {
            "name": app.name,
            "display_name": app.display_name,
            "exec_command": app.exec_command,
            "categories": app.categories,
            "terminal": app.terminal,
            "description": app.description,
            "keywords": app.keywords,
        }

    def list_applications_by_category(self, category: str) -> List[str]:
        """List all applications in a category"""
        apps = self.app_registry.get_applications_by_category(category)
        return [app.display_name for app in apps]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered applications"""
        return self.app_registry.get_stats()

    def refresh_applications(self):
        """Refresh the application registry"""
        self.app_registry.refresh()
        logger.info("Application registry refreshed")

    def cleanup(self):
        """Cleanup resources"""
        if self.browser_controller:
            try:
                self.browser_controller.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        logger.info("System controller cleanup complete")


# Backward compatibility - create alias
SystemController = SystemControllerV2
