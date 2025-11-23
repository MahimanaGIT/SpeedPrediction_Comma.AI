"""
Text-to-Speech Module
Handles converting text responses to speech
"""
import pyttsx3
import logging
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Handles text-to-speech conversion"""

    def __init__(self):
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the TTS engine"""
        try:
            self.engine = pyttsx3.init()

            # Configure voice properties
            self.engine.setProperty('rate', config.TTS_VOICE_RATE)
            self.engine.setProperty('volume', config.TTS_VOLUME)

            # Try to set a better voice (if available)
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available, otherwise use first available
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        logger.info(f"Using voice: {voice.name}")
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
                    logger.info(f"Using voice: {voices[0].name}")

            logger.info("Text-to-speech engine initialized")

        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None

    def speak(self, text: str, blocking: bool = True):
        """
        Convert text to speech

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete before returning
        """
        if not self.engine:
            logger.error("TTS engine not initialized")
            return

        try:
            logger.info(f"Speaking: {text}")
            self.engine.say(text)

            if blocking:
                self.engine.runAndWait()
            else:
                # Start speech in background
                self.engine.startLoop(False)
                self.engine.iterate()
                self.engine.endLoop()

        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}")

    def speak_async(self, text: str):
        """
        Convert text to speech asynchronously

        Args:
            text: Text to speak
        """
        self.speak(text, blocking=False)

    def stop(self):
        """Stop current speech"""
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error stopping TTS: {e}")

    def set_rate(self, rate: int):
        """
        Set speech rate

        Args:
            rate: Words per minute (typically 100-300)
        """
        if self.engine:
            try:
                self.engine.setProperty('rate', rate)
                logger.info(f"Speech rate set to {rate}")
            except Exception as e:
                logger.error(f"Error setting speech rate: {e}")

    def set_volume(self, volume: float):
        """
        Set speech volume

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine:
            try:
                volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
                self.engine.setProperty('volume', volume)
                logger.info(f"Speech volume set to {volume}")
            except Exception as e:
                logger.error(f"Error setting speech volume: {e}")

    def list_voices(self):
        """List all available voices"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            logger.info("Available voices:")
            for idx, voice in enumerate(voices):
                logger.info(f"  {idx}: {voice.name} ({voice.id})")
            return voices
        return []

    def set_voice(self, voice_id: str):
        """
        Set the voice to use

        Args:
            voice_id: ID of the voice to use
        """
        if self.engine:
            try:
                self.engine.setProperty('voice', voice_id)
                logger.info(f"Voice changed to {voice_id}")
            except Exception as e:
                logger.error(f"Error setting voice: {e}")
