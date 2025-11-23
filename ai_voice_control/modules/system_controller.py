"""
System Controller Module
Executes system-level actions like opening apps, controlling mouse/keyboard, browser automation
"""
import subprocess
import time
import logging
import pyautogui
from typing import Dict, Optional, List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import config

logger = logging.getLogger(__name__)

# Configure PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5


class SystemController:
    """Controls system operations and browser automation"""

    def __init__(self):
        self.browser = None
        self.browser_type = config.DEFAULT_BROWSER
        logger.info("System controller initialized")

    def execute_action(self, action: Dict) -> Dict:
        """
        Execute a single action

        Args:
            action: Action dictionary with type and parameters

        Returns:
            Result dictionary with success status and data
        """
        action_type = action.get("type")
        logger.info(f"Executing action: {action_type}")

        try:
            if action_type == "open_app":
                return self._open_app(action.get("app_name"))

            elif action_type == "close_app":
                return self._close_app(action.get("app_name"))

            elif action_type == "search_web":
                return self._search_web(action.get("query"))

            elif action_type == "navigate_to":
                return self._navigate_to(action.get("url"))

            elif action_type == "click":
                return self._click(action.get("x"), action.get("y"), action.get("element"))

            elif action_type == "type_text":
                return self._type_text(action.get("text"))

            elif action_type == "press_key":
                return self._press_key(action.get("key"))

            elif action_type == "extract_info":
                return self._extract_info(action.get("target"))

            elif action_type == "execute_command":
                return self._execute_command(action.get("command"))

            elif action_type == "wait":
                return self._wait(action.get("seconds", 1))

            else:
                logger.warning(f"Unknown action type: {action_type}")
                return {"success": False, "error": f"Unknown action type: {action_type}"}

        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return {"success": False, "error": str(e)}

    def _open_app(self, app_name: str) -> Dict:
        """Open an application"""
        try:
            logger.info(f"Opening app: {app_name}")

            # Map common app names to commands
            app_commands = {
                "chrome": "google-chrome",
                "firefox": "firefox",
                "terminal": "gnome-terminal",
                "files": "nautilus",
                "calculator": "gnome-calculator",
                "text editor": "gedit",
                "settings": "gnome-control-center",
                "vscode": "code",
                "code": "code",
            }

            command = app_commands.get(app_name.lower(), app_name)

            # Launch the application
            subprocess.Popen([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Special handling for browsers
            if app_name.lower() in ["chrome", "firefox"]:
                time.sleep(2)  # Wait for browser to start
                self._init_browser(app_name.lower())

            return {"success": True, "message": f"Opened {app_name}"}

        except Exception as e:
            logger.error(f"Failed to open app {app_name}: {e}")
            return {"success": False, "error": str(e)}

    def _close_app(self, app_name: str) -> Dict:
        """Close an application"""
        try:
            logger.info(f"Closing app: {app_name}")

            # Use pkill to close the app
            subprocess.run(["pkill", "-f", app_name], check=False)

            # Close browser if it's a browser app
            if app_name.lower() in ["chrome", "firefox"] and self.browser:
                self.browser.quit()
                self.browser = None

            return {"success": True, "message": f"Closed {app_name}"}

        except Exception as e:
            logger.error(f"Failed to close app {app_name}: {e}")
            return {"success": False, "error": str(e)}

    def _init_browser(self, browser_type: Optional[str] = None):
        """Initialize browser for automation"""
        if self.browser:
            return

        try:
            browser_type = browser_type or self.browser_type

            if browser_type == "chrome":
                options = webdriver.ChromeOptions()
                if config.HEADLESS_BROWSER:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")

                service = ChromeService(ChromeDriverManager().install())
                self.browser = webdriver.Chrome(service=service, options=options)

            elif browser_type == "firefox":
                options = webdriver.FirefoxOptions()
                if config.HEADLESS_BROWSER:
                    options.add_argument("--headless")

                service = FirefoxService(GeckoDriverManager().install())
                self.browser = webdriver.Firefox(service=service, options=options)

            logger.info(f"Browser initialized: {browser_type}")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise

    def _search_web(self, query: str) -> Dict:
        """Search the web"""
        try:
            logger.info(f"Searching web: {query}")

            if not self.browser:
                self._init_browser()

            # Navigate to Google
            self.browser.get("https://www.google.com")
            time.sleep(1)

            # Find search box and enter query
            search_box = self.browser.find_element(By.NAME, "q")
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results
            time.sleep(2)

            return {"success": True, "message": f"Searched for: {query}"}

        except Exception as e:
            logger.error(f"Failed to search web: {e}")
            return {"success": False, "error": str(e)}

    def _navigate_to(self, url: str) -> Dict:
        """Navigate to a URL"""
        try:
            logger.info(f"Navigating to: {url}")

            if not self.browser:
                self._init_browser()

            self.browser.get(url)
            time.sleep(2)

            return {"success": True, "message": f"Navigated to: {url}"}

        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return {"success": False, "error": str(e)}

    def _click(self, x: Optional[int], y: Optional[int], element: Optional[str]) -> Dict:
        """Click at coordinates or on an element"""
        try:
            if x is not None and y is not None:
                logger.info(f"Clicking at ({x}, {y})")
                pyautogui.click(x, y)
            elif element:
                logger.info(f"Clicking element: {element}")
                # Use browser automation to click element
                if self.browser:
                    elem = self.browser.find_element(By.CSS_SELECTOR, element)
                    elem.click()

            return {"success": True, "message": "Clicked"}

        except Exception as e:
            logger.error(f"Failed to click: {e}")
            return {"success": False, "error": str(e)}

    def _type_text(self, text: str) -> Dict:
        """Type text"""
        try:
            logger.info(f"Typing text: {text}")
            pyautogui.typewrite(text, interval=0.05)

            return {"success": True, "message": f"Typed: {text}"}

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return {"success": False, "error": str(e)}

    def _press_key(self, key: str) -> Dict:
        """Press a keyboard key"""
        try:
            logger.info(f"Pressing key: {key}")
            pyautogui.press(key)

            return {"success": True, "message": f"Pressed: {key}"}

        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            return {"success": False, "error": str(e)}

    def _extract_info(self, target: str) -> Dict:
        """Extract information from current page"""
        try:
            logger.info(f"Extracting info: {target}")

            if not self.browser:
                return {"success": False, "error": "No browser active"}

            # Get page source
            page_source = self.browser.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Extract based on target
            if target == "prices":
                # Look for price patterns
                prices = []
                price_patterns = soup.find_all(text=lambda text: text and ('$' in text or '€' in text or '£' in text))
                for price in price_patterns[:5]:  # Limit to top 5
                    prices.append(price.strip())

                return {"success": True, "data": {"prices": prices}, "message": f"Found {len(prices)} prices"}

            elif target == "text":
                # Extract all visible text
                text = soup.get_text(separator='\n', strip=True)
                return {"success": True, "data": {"text": text[:1000]}, "message": "Extracted text"}  # Limit to 1000 chars

            elif target == "title":
                title = self.browser.title
                return {"success": True, "data": {"title": title}, "message": f"Title: {title}"}

            else:
                # Generic extraction
                elements = soup.find_all(class_=target) or soup.find_all(id=target)
                data = [elem.get_text(strip=True) for elem in elements]
                return {"success": True, "data": {target: data}, "message": f"Extracted {len(data)} elements"}

        except Exception as e:
            logger.error(f"Failed to extract info: {e}")
            return {"success": False, "error": str(e)}

    def _execute_command(self, command: str) -> Dict:
        """Execute a shell command"""
        try:
            logger.info(f"Executing command: {command}")

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            return {
                "success": result.returncode == 0,
                "data": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                "message": "Command executed"
            }

        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return {"success": False, "error": str(e)}

    def _wait(self, seconds: float) -> Dict:
        """Wait for specified seconds"""
        logger.info(f"Waiting {seconds} seconds")
        time.sleep(seconds)
        return {"success": True, "message": f"Waited {seconds} seconds"}

    def cleanup(self):
        """Cleanup resources"""
        if self.browser:
            try:
                self.browser.quit()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
