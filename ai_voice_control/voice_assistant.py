"""
AI Voice Control Assistant - Main Orchestrator
Speech-to-speech AI assistant with computer control capabilities
"""
import sys
import os
import time
import logging
from datetime import datetime
from typing import Optional
from colorama import Fore, Style, init as colorama_init

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from speech_recognition_module import SpeechRecognizer
from text_to_speech_module import TextToSpeech
from ai_processor import AIProcessor
from system_controller import SystemController
from preference_manager import PreferenceManager
import config

# Initialize colorama for colored terminal output
colorama_init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Main voice assistant orchestrator"""

    def __init__(self):
        logger.info("Initializing AI Voice Control Assistant...")
        print(f"{Fore.CYAN}╔═══════════════════════════════════════════════════════════╗")
        print(f"{Fore.CYAN}║     AI Voice Control Assistant for Ubuntu                ║")
        print(f"{Fore.CYAN}║     Speech-to-Speech Computer Control                    ║")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

        # Initialize modules
        print(f"{Fore.YELLOW}[*] Initializing speech recognition...{Style.RESET_ALL}")
        self.speech_recognizer = SpeechRecognizer()

        print(f"{Fore.YELLOW}[*] Initializing text-to-speech...{Style.RESET_ALL}")
        self.tts = TextToSpeech()

        print(f"{Fore.YELLOW}[*] Initializing AI processor...{Style.RESET_ALL}")
        self.ai_processor = AIProcessor()

        print(f"{Fore.YELLOW}[*] Initializing system controller...{Style.RESET_ALL}")
        self.system_controller = SystemController()

        print(f"{Fore.YELLOW}[*] Initializing preference manager...{Style.RESET_ALL}")
        self.preference_manager = PreferenceManager()

        self.running = False
        self.wake_word_mode = config.USE_WAKE_WORD

        print(f"{Fore.GREEN}✓ Initialization complete!{Style.RESET_ALL}\n")
        logger.info("Voice assistant initialized successfully")

    def start(self):
        """Start the voice assistant"""
        self.running = True

        print(f"{Fore.GREEN}╔═══════════════════════════════════════════════════════════╗")
        print(f"{Fore.GREEN}║     Voice Assistant Started                              ║")
        print(f"{Fore.GREEN}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

        if self.wake_word_mode:
            print(f"{Fore.CYAN}Wake word mode enabled. Say '{config.WAKE_WORD}' to activate.{Style.RESET_ALL}")
            self.tts.speak(f"Voice assistant ready. Say {config.WAKE_WORD} to activate.")
        else:
            print(f"{Fore.CYAN}Always listening mode. Say 'exit' or 'quit' to stop.{Style.RESET_ALL}")
            self.tts.speak("Voice assistant ready. I'm listening.")

        print(f"{Fore.YELLOW}Commands: 'exit', 'quit', 'help', 'history', 'preferences'{Style.RESET_ALL}\n")

        try:
            while self.running:
                if self.wake_word_mode:
                    self._wait_for_wake_word()
                else:
                    self._listen_for_command()

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Interrupted by user{Style.RESET_ALL}")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.stop()

    def _wait_for_wake_word(self):
        """Wait for wake word before listening for commands"""
        print(f"{Fore.CYAN}[Listening for wake word...]{Style.RESET_ALL}")

        if self.speech_recognizer.listen_for_wake_word():
            self.tts.speak("Yes, I'm listening.")
            print(f"{Fore.GREEN}✓ Wake word detected! Listening for command...{Style.RESET_ALL}")
            self._listen_for_command()

    def _listen_for_command(self):
        """Listen for and process a command"""
        print(f"{Fore.CYAN}🎤 Listening...{Style.RESET_ALL}")

        # Listen for command
        command = self.speech_recognizer.listen(timeout=10, phrase_time_limit=15)

        if not command:
            return

        print(f"{Fore.BLUE}You: {command}{Style.RESET_ALL}")

        # Handle special commands
        if self._handle_special_command(command):
            return

        # Process command
        self._process_command(command)

    def _handle_special_command(self, command: str) -> bool:
        """
        Handle special system commands

        Returns:
            True if command was handled, False otherwise
        """
        command_lower = command.lower()

        if command_lower in ["exit", "quit", "stop", "goodbye"]:
            self.stop()
            return True

        elif command_lower in ["help", "what can you do"]:
            self._show_help()
            return True

        elif command_lower in ["history", "show history"]:
            self._show_history()
            return True

        elif command_lower in ["preferences", "show preferences"]:
            self._show_preferences()
            return True

        elif "clear history" in command_lower:
            self._clear_history()
            return True

        return False

    def _process_command(self, command: str):
        """Process a user command"""
        start_time = time.time()

        try:
            # Get context from preferences
            context = self.preference_manager.get_context()

            # Process with AI
            print(f"{Fore.YELLOW}[*] Processing command...{Style.RESET_ALL}")
            result = self.ai_processor.process_command(command, context)

            # Speak the response
            response = result.get("response", "Processing your request.")
            print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")
            self.tts.speak(response)

            # Check if confirmation needed
            if result.get("needs_confirmation", False):
                self.tts.speak("Should I proceed? Say yes or no.")
                print(f"{Fore.YELLOW}[?] Confirmation needed. Say 'yes' or 'no'.{Style.RESET_ALL}")

                confirmation = self.speech_recognizer.listen(timeout=10)
                if not confirmation or "yes" not in confirmation.lower():
                    self.tts.speak("Okay, cancelled.")
                    print(f"{Fore.RED}✗ Cancelled by user{Style.RESET_ALL}")
                    return

            # Execute actions
            actions = result.get("actions", [])
            if actions:
                print(f"{Fore.YELLOW}[*] Executing {len(actions)} actions...{Style.RESET_ALL}")
                success = self._execute_actions(actions)
            else:
                success = True

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log command
            self.preference_manager.log_command(
                command=command,
                intent=result.get("intent"),
                actions=actions,
                success=success,
                execution_time=execution_time
            )

            print(f"{Fore.GREEN}✓ Done! (took {execution_time:.2f}s){Style.RESET_ALL}\n")

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.tts.speak(f"I encountered an error: {str(e)}")
            print(f"{Fore.RED}✗ Error: {e}{Style.RESET_ALL}\n")

    def _execute_actions(self, actions: list) -> bool:
        """
        Execute a list of actions

        Returns:
            True if all actions succeeded, False otherwise
        """
        all_success = True

        for idx, action in enumerate(actions, 1):
            action_type = action.get("type")
            reason = action.get("reason", "")

            print(f"{Fore.YELLOW}  [{idx}/{len(actions)}] {action_type}: {reason}{Style.RESET_ALL}")

            # Execute action
            result = self.system_controller.execute_action(action)

            if result.get("success"):
                print(f"{Fore.GREEN}  ✓ {result.get('message', 'Success')}{Style.RESET_ALL}")

                # If action extracted data, speak it
                if "data" in result:
                    self._handle_extracted_data(action_type, result["data"])

            else:
                print(f"{Fore.RED}  ✗ {result.get('error', 'Failed')}{Style.RESET_ALL}")
                all_success = False

            # Small delay between actions
            time.sleep(0.5)

        return all_success

    def _handle_extracted_data(self, action_type: str, data: dict):
        """Handle and speak extracted data"""
        try:
            if action_type == "extract_info":
                if "prices" in data:
                    prices = data["prices"]
                    if prices:
                        response = f"I found these prices: {', '.join(prices[:3])}"
                        print(f"{Fore.CYAN}  ℹ {response}{Style.RESET_ALL}")
                        self.tts.speak(response)

                elif "title" in data:
                    title = data["title"]
                    response = f"The page title is: {title}"
                    print(f"{Fore.CYAN}  ℹ {response}{Style.RESET_ALL}")
                    self.tts.speak(response)

                elif "text" in data:
                    text = data["text"][:200]  # Limit length
                    print(f"{Fore.CYAN}  ℹ Extracted text: {text}...{Style.RESET_ALL}")

        except Exception as e:
            logger.error(f"Error handling extracted data: {e}")

    def _show_help(self):
        """Show help information"""
        help_text = """
I can help you control your Ubuntu computer with voice commands. Here are some examples:

• "Open Chrome and search for Python tutorials"
• "Look at prices of RayBan Meta glasses"
• "Open terminal and run ls command"
• "Close all browser windows"
• "Show me my desktop"

Special commands:
• "exit" or "quit" - Stop the assistant
• "help" - Show this help message
• "history" - Show recent commands
• "preferences" - Show your preferences
• "clear history" - Clear command history
"""
        print(f"{Fore.CYAN}{help_text}{Style.RESET_ALL}")
        self.tts.speak("I can control your computer with voice commands. Say things like 'open Chrome' or 'search the web'.")

    def _show_history(self):
        """Show command history"""
        history = self.preference_manager.get_command_history(limit=5)

        if not history:
            print(f"{Fore.YELLOW}No command history yet.{Style.RESET_ALL}")
            self.tts.speak("No command history yet.")
            return

        print(f"\n{Fore.CYAN}Recent Commands:{Style.RESET_ALL}")
        for idx, entry in enumerate(history, 1):
            status = "✓" if entry["success"] else "✗"
            print(f"{idx}. {status} {entry['command']} ({entry['timestamp']})")

        self.tts.speak(f"You have {len(history)} recent commands in history.")

    def _show_preferences(self):
        """Show user preferences"""
        prefs = self.preference_manager.get_all_preferences()
        fav_apps = self.preference_manager.get_favorite_apps(limit=3)

        print(f"\n{Fore.CYAN}User Preferences:{Style.RESET_ALL}")
        if prefs:
            for key, value in prefs.items():
                print(f"  • {key}: {value}")
        else:
            print("  No preferences set yet.")

        if fav_apps:
            print(f"\n{Fore.CYAN}Favorite Apps:{Style.RESET_ALL}")
            for app in fav_apps:
                print(f"  • {app['app_name']} (opened {app['open_count']} times)")

        self.tts.speak("Your preferences are displayed on screen.")

    def _clear_history(self):
        """Clear command history"""
        self.preference_manager.clear_history()
        self.ai_processor.clear_history()
        print(f"{Fore.GREEN}✓ History cleared{Style.RESET_ALL}")
        self.tts.speak("History cleared.")

    def stop(self):
        """Stop the voice assistant"""
        print(f"\n{Fore.YELLOW}[*] Shutting down...{Style.RESET_ALL}")
        self.tts.speak("Goodbye!")

        self.running = False

        # Cleanup resources
        self.system_controller.cleanup()

        print(f"{Fore.GREEN}✓ Voice assistant stopped{Style.RESET_ALL}")
        logger.info("Voice assistant stopped")


def main():
    """Main entry point"""
    try:
        assistant = VoiceAssistant()
        assistant.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
