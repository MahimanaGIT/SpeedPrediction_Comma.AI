"""
Speech Recognition Module
Handles voice input and converts it to text
"""
import speech_recognition as sr
import logging
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Handles speech-to-text conversion"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=config.MICROPHONE_INDEX)

        # Configure recognizer
        self.recognizer.energy_threshold = config.ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = config.DYNAMIC_ENERGY_THRESHOLD
        self.recognizer.pause_threshold = config.PAUSE_THRESHOLD

        # Calibrate for ambient noise
        self._calibrate_ambient_noise()

        logger.info("Speech recognizer initialized")

    def _calibrate_ambient_noise(self):
        """Calibrate the recognizer for ambient noise"""
        try:
            with self.microphone as source:
                logger.info("Calibrating for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                logger.info(f"Calibration complete. Energy threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            logger.error(f"Failed to calibrate ambient noise: {e}")

    def listen(self, timeout: Optional[int] = None, phrase_time_limit: Optional[int] = None) -> Optional[str]:
        """
        Listen for voice input and convert to text

        Args:
            timeout: Maximum time to wait for speech to start (seconds)
            phrase_time_limit: Maximum time for a phrase (seconds)

        Returns:
            Recognized text or None if recognition failed
        """
        try:
            with self.microphone as source:
                logger.info("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            # Use configured speech recognition engine
            logger.info("Processing speech...")

            if config.SPEECH_RECOGNITION_ENGINE == "google":
                text = self.recognizer.recognize_google(audio)
            elif config.SPEECH_RECOGNITION_ENGINE == "sphinx":
                text = self.recognizer.recognize_sphinx(audio)
            else:
                text = self.recognizer.recognize_google(audio)  # Fallback

            logger.info(f"Recognized: {text}")
            return text

        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during speech recognition: {e}")
            return None

    def listen_for_wake_word(self) -> bool:
        """
        Listen for the wake word

        Returns:
            True if wake word detected, False otherwise
        """
        text = self.listen(timeout=5, phrase_time_limit=3)

        if text and config.WAKE_WORD.lower() in text.lower():
            logger.info(f"Wake word '{config.WAKE_WORD}' detected!")
            return True

        return False

    def continuous_listen(self, callback):
        """
        Continuously listen for voice input and call callback with recognized text

        Args:
            callback: Function to call with recognized text
        """
        logger.info("Starting continuous listening mode")

        with self.microphone as source:
            while True:
                try:
                    logger.debug("Waiting for speech...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)

                    try:
                        if config.SPEECH_RECOGNITION_ENGINE == "google":
                            text = self.recognizer.recognize_google(audio)
                        else:
                            text = self.recognizer.recognize_google(audio)  # Fallback

                        if text:
                            logger.info(f"Recognized: {text}")
                            callback(text)

                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        logger.error(f"Recognition service error: {e}")
                        continue

                except sr.WaitTimeoutError:
                    continue
                except KeyboardInterrupt:
                    logger.info("Stopping continuous listening")
                    break
                except Exception as e:
                    logger.error(f"Error in continuous listening: {e}")
                    continue
