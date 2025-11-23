"""
AI Command Processor Module
Uses LLM to understand user intent and generate action plans
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class AIProcessor:
    """Processes user commands using AI to understand intent and generate actions"""

    def __init__(self):
        self.client = None
        self.conversation_history = []
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the AI client based on configuration"""
        try:
            if config.AI_PROVIDER == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")

            elif config.AI_PROVIDER == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")

            elif config.AI_PROVIDER == "ollama":
                import requests
                self.client = requests  # Using requests for Ollama
                logger.info("Ollama client initialized")

            else:
                logger.error(f"Unknown AI provider: {config.AI_PROVIDER}")

        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI"""
        return """You are an AI assistant that controls a Ubuntu computer through voice commands. Your role is to:

1. Understand user intent from natural language commands
2. Generate a structured action plan to fulfill the request
3. Provide clear, concise spoken responses

You can control the computer by generating actions in this JSON format:
{
    "actions": [
        {
            "type": "open_app",
            "app_name": "chrome",
            "reason": "Opening browser to search"
        },
        {
            "type": "search_web",
            "query": "rayban meta glasses price",
            "reason": "Searching for product prices"
        },
        {
            "type": "extract_info",
            "target": "prices",
            "reason": "Extracting price information"
        }
    ],
    "response": "I'll search for RayBan Meta glasses prices for you.",
    "needs_confirmation": false
}

Available action types:
- open_app: Open an application (chrome, firefox, terminal, files, etc.)
- close_app: Close an application
- search_web: Search the web for information
- navigate_to: Navigate to a specific URL
- click: Click at coordinates or element
- type_text: Type text
- press_key: Press keyboard key(s)
- extract_info: Extract specific information from current page
- execute_command: Execute a shell command
- wait: Wait for specified seconds

Always:
- Break complex tasks into simple actions
- Provide clear spoken responses
- Ask for confirmation on destructive actions
- Learn from user preferences and adapt

Current datetime: {datetime}
""".format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def process_command(self, command: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a user command and generate actions

        Args:
            command: User's voice command
            context: Optional context (user preferences, current state, etc.)

        Returns:
            Dictionary with actions and response
        """
        try:
            logger.info(f"Processing command: {command}")

            # Build the prompt
            user_message = f"User command: {command}"

            if context:
                user_message += f"\n\nContext: {json.dumps(context, indent=2)}"

            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Keep history limited
            if len(self.conversation_history) > config.MAX_COMMAND_HISTORY * 2:
                # Keep system message and recent history
                self.conversation_history = self.conversation_history[-20:]

            # Get AI response
            if config.AI_PROVIDER == "openai":
                response = self._process_openai(command, context)
            elif config.AI_PROVIDER == "anthropic":
                response = self._process_anthropic(command, context)
            elif config.AI_PROVIDER == "ollama":
                response = self._process_ollama(command, context)
            else:
                response = self._fallback_response(command)

            logger.info(f"AI Response: {response}")
            return response

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return {
                "actions": [],
                "response": f"I encountered an error processing your request: {str(e)}",
                "needs_confirmation": False
            }

    def _process_openai(self, command: str, context: Optional[Dict]) -> Dict:
        """Process command using OpenAI"""
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                *self.conversation_history
            ]

            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })

            return result

        except Exception as e:
            logger.error(f"OpenAI processing error: {e}")
            return self._fallback_response(command)

    def _process_anthropic(self, command: str, context: Optional[Dict]) -> Dict:
        """Process command using Anthropic Claude"""
        try:
            # Anthropic uses a different message format
            response = self.client.messages.create(
                model=config.MODEL_NAME,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                system=self._get_system_prompt(),
                messages=self.conversation_history
            )

            result = json.loads(response.content[0].text)

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content[0].text
            })

            return result

        except Exception as e:
            logger.error(f"Anthropic processing error: {e}")
            return self._fallback_response(command)

    def _process_ollama(self, command: str, context: Optional[Dict]) -> Dict:
        """Process command using local Ollama"""
        try:
            import requests

            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                *self.conversation_history
            ]

            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": config.MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "format": "json"
                }
            )

            result = json.loads(response.json()["message"]["content"])

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.json()["message"]["content"]
            })

            return result

        except Exception as e:
            logger.error(f"Ollama processing error: {e}")
            return self._fallback_response(command)

    def _fallback_response(self, command: str) -> Dict:
        """Generate a fallback response when AI processing fails"""
        logger.warning("Using fallback response")

        # Simple pattern matching for common commands
        command_lower = command.lower()

        if "open" in command_lower and "chrome" in command_lower:
            return {
                "actions": [
                    {"type": "open_app", "app_name": "chrome", "reason": "Opening Chrome browser"}
                ],
                "response": "Opening Chrome browser.",
                "needs_confirmation": False
            }
        elif "search" in command_lower or "look" in command_lower:
            # Extract search query
            query = command
            return {
                "actions": [
                    {"type": "open_app", "app_name": "chrome", "reason": "Opening browser"},
                    {"type": "search_web", "query": query, "reason": "Searching the web"}
                ],
                "response": f"I'll search for that.",
                "needs_confirmation": False
            }
        else:
            return {
                "actions": [],
                "response": "I'm not sure how to handle that command. Could you rephrase it?",
                "needs_confirmation": False
            }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
