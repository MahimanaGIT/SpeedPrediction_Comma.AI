"""
Application Registry Module
Discovers and manages all installed Ubuntu applications using .desktop files
Implements proper design patterns for robustness and extensibility
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import configparser
from difflib import SequenceMatcher
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class Application:
    """Represents an installed application"""
    name: str
    display_name: str
    exec_command: str
    categories: List[str]
    icon: Optional[str] = None
    terminal: bool = False
    description: Optional[str] = None
    keywords: List[str] = None
    desktop_file: Optional[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class AppRegistry:
    """
    Singleton registry for discovering and managing Ubuntu applications
    Uses the XDG Desktop Entry Specification
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not AppRegistry._initialized:
            self.apps: Dict[str, Application] = {}
            self.name_mapping: Dict[str, str] = {}  # common name -> app key
            self.aliases: Dict[str, str] = {}
            self._setup_aliases()
            self.discover_applications()
            AppRegistry._initialized = True
            logger.info(f"App Registry initialized with {len(self.apps)} applications")

    def _setup_aliases(self):
        """Setup common aliases for applications"""
        self.aliases = {
            # Browsers
            "chrome": ["google-chrome", "chrome", "chromium", "google chrome"],
            "firefox": ["firefox", "firefox-esr", "mozilla firefox"],
            "brave": ["brave-browser", "brave"],
            "edge": ["microsoft-edge", "edge"],

            # Terminals
            "terminal": ["gnome-terminal", "terminal", "konsole", "xterm", "terminator"],
            "console": ["gnome-terminal", "terminal"],

            # Editors
            "code": ["code", "vscode", "visual studio code"],
            "cursor": ["cursor", "cursor-editor"],
            "vim": ["vim", "gvim", "nvim", "neovim"],
            "nano": ["nano"],
            "gedit": ["gedit", "text editor", "gnome-text-editor"],

            # IDEs
            "arduino": ["arduino", "arduino ide"],
            "pycharm": ["pycharm", "pycharm-community"],

            # Communication
            "beeper": ["beeper"],
            "slack": ["slack"],
            "discord": ["discord"],
            "zoom": ["zoom"],

            # Notes
            "obsidian": ["obsidian"],
            "notion": ["notion"],

            # Utilities
            "calculator": ["gnome-calculator", "calculator", "calc", "kcalc"],
            "files": ["nautilus", "files", "file manager", "dolphin"],
            "settings": ["gnome-control-center", "settings", "system settings"],

            # Media
            "vlc": ["vlc"],
            "spotify": ["spotify"],
        }

    def discover_applications(self) -> int:
        """
        Discover all installed applications by scanning .desktop files
        Returns number of apps discovered
        """
        desktop_paths = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path.home() / ".local/share/applications",
            Path("/var/lib/snapd/desktop/applications"),  # Snap apps
            Path("/var/lib/flatpak/exports/share/applications"),  # Flatpak apps
        ]

        discovered_count = 0

        for desktop_path in desktop_paths:
            if not desktop_path.exists():
                continue

            for desktop_file in desktop_path.glob("*.desktop"):
                try:
                    app = self._parse_desktop_file(desktop_file)
                    if app:
                        # Use lowercase name as key
                        key = app.name.lower()

                        # Don't override if already exists (prefer system apps)
                        if key not in self.apps:
                            self.apps[key] = app
                            discovered_count += 1

                            # Create name mappings
                            self.name_mapping[app.display_name.lower()] = key

                            # Add keyword mappings
                            for keyword in app.keywords:
                                self.name_mapping[keyword.lower()] = key

                except Exception as e:
                    logger.debug(f"Failed to parse {desktop_file}: {e}")

        logger.info(f"Discovered {discovered_count} applications")
        return discovered_count

    def _parse_desktop_file(self, filepath: Path) -> Optional[Application]:
        """Parse a .desktop file and extract application info"""
        try:
            config = configparser.ConfigParser(interpolation=None)
            config.read(filepath, encoding='utf-8')

            if 'Desktop Entry' not in config:
                return None

            entry = config['Desktop Entry']

            # Skip if NoDisplay or Hidden
            if entry.get('NoDisplay', 'false').lower() == 'true':
                return None
            if entry.get('Hidden', 'false').lower() == 'true':
                return None

            name = entry.get('Name', '')
            if not name:
                return None

            exec_cmd = entry.get('Exec', '')
            if not exec_cmd:
                return None

            # Clean up exec command (remove %u, %f, etc.)
            exec_cmd = re.sub(r'%[a-zA-Z]', '', exec_cmd).strip()

            # Extract categories
            categories = entry.get('Categories', '').split(';')
            categories = [c.strip() for c in categories if c.strip()]

            # Extract keywords
            keywords = entry.get('Keywords', '').split(';')
            keywords = [k.strip() for k in keywords if k.strip()]

            # Check if terminal app
            terminal = entry.get('Terminal', 'false').lower() == 'true'

            app = Application(
                name=name,
                display_name=name,
                exec_command=exec_cmd,
                categories=categories,
                icon=entry.get('Icon'),
                terminal=terminal,
                description=entry.get('Comment'),
                keywords=keywords,
                desktop_file=str(filepath)
            )

            return app

        except Exception as e:
            logger.debug(f"Error parsing {filepath}: {e}")
            return None

    def find_application(self, query: str, threshold: float = 0.6) -> Optional[Application]:
        """
        Find an application by name using fuzzy matching

        Args:
            query: Application name to search for
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            Application object or None
        """
        query_lower = query.lower().strip()

        # 1. Check exact match
        if query_lower in self.apps:
            return self.apps[query_lower]

        # 2. Check name mappings
        if query_lower in self.name_mapping:
            key = self.name_mapping[query_lower]
            return self.apps.get(key)

        # 3. Check aliases
        for alias_name, alias_list in self.aliases.items():
            if query_lower in [a.lower() for a in alias_list]:
                # Try to find app matching this alias
                for app_key in self.apps:
                    if any(a.lower() in app_key for a in alias_list):
                        return self.apps[app_key]

        # 4. Fuzzy matching on display names
        best_match = None
        best_score = threshold

        for app in self.apps.values():
            # Check display name
            score = SequenceMatcher(None, query_lower, app.display_name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = app

            # Check keywords
            for keyword in app.keywords:
                score = SequenceMatcher(None, query_lower, keyword.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = app

            # Check if query is substring of app name
            if query_lower in app.display_name.lower():
                return app

        return best_match

    def search_applications(self, query: str, limit: int = 5) -> List[Application]:
        """
        Search for applications matching query

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching applications sorted by relevance
        """
        query_lower = query.lower().strip()
        results = []

        for app in self.apps.values():
            score = 0.0

            # Exact match bonus
            if query_lower == app.display_name.lower():
                score = 1.0
            else:
                # Fuzzy match on display name
                score = SequenceMatcher(None, query_lower, app.display_name.lower()).ratio()

                # Bonus for substring match
                if query_lower in app.display_name.lower():
                    score += 0.3

                # Check keywords
                for keyword in app.keywords:
                    kw_score = SequenceMatcher(None, query_lower, keyword.lower()).ratio()
                    score = max(score, kw_score)

            if score > 0.3:  # Minimum threshold
                results.append((score, app))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        return [app for score, app in results[:limit]]

    def get_applications_by_category(self, category: str) -> List[Application]:
        """Get all applications in a specific category"""
        category_lower = category.lower()
        return [
            app for app in self.apps.values()
            if any(category_lower in cat.lower() for cat in app.categories)
        ]

    def launch_application(self, app: Application) -> bool:
        """
        Launch an application

        Args:
            app: Application to launch

        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the exec command to handle complex cases
            exec_parts = app.exec_command.split()

            if app.terminal:
                # Launch in terminal
                subprocess.Popen(
                    ["gnome-terminal", "--", "bash", "-c", app.exec_command],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            else:
                # Launch normally
                subprocess.Popen(
                    exec_parts,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )

            logger.info(f"Launched application: {app.display_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to launch {app.display_name}: {e}")
            return False

    def get_all_applications(self) -> List[Application]:
        """Get list of all discovered applications"""
        return list(self.apps.values())

    def refresh(self):
        """Refresh the application registry"""
        self.apps.clear()
        self.name_mapping.clear()
        self.discover_applications()
        logger.info("App registry refreshed")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about discovered applications"""
        categories = {}
        terminal_apps = 0

        for app in self.apps.values():
            if app.terminal:
                terminal_apps += 1

            for cat in app.categories:
                categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_apps": len(self.apps),
            "terminal_apps": terminal_apps,
            "gui_apps": len(self.apps) - terminal_apps,
            "categories": categories
        }


# Singleton instance
_registry = None


def get_app_registry() -> AppRegistry:
    """Get the singleton AppRegistry instance"""
    global _registry
    if _registry is None:
        _registry = AppRegistry()
    return _registry
