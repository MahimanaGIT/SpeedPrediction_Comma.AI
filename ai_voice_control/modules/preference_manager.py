"""
Preference Manager Module
Learns and adapts to user preferences over time
"""
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)


class PreferenceManager:
    """Manages user preferences and learning"""

    def __init__(self):
        self.db_path = config.DB_PATH
        self._init_database()
        logger.info("Preference manager initialized")

    def _init_database(self):
        """Initialize the preferences database"""
        try:
            # Create parent directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Commands history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    command TEXT NOT NULL,
                    intent TEXT,
                    actions TEXT,
                    success INTEGER,
                    execution_time REAL
                )
            """)

            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Command patterns table (for learning)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_used TEXT NOT NULL,
                    UNIQUE(pattern, intent)
                )
            """)

            # App usage statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_usage (
                    app_name TEXT PRIMARY KEY,
                    open_count INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    last_opened TEXT,
                    favorite INTEGER DEFAULT 0
                )
            """)

            conn.commit()
            conn.close()

            logger.info("Database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def log_command(self, command: str, intent: Optional[str], actions: List[Dict],
                   success: bool, execution_time: float):
        """
        Log a command execution

        Args:
            command: The user's command
            intent: Detected intent
            actions: List of actions executed
            success: Whether execution was successful
            execution_time: Time taken to execute
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO command_history (timestamp, command, intent, actions, success, execution_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                command,
                intent,
                json.dumps(actions),
                1 if success else 0,
                execution_time
            ))

            conn.commit()
            conn.close()

            # Update command pattern
            if config.LEARNING_MODE:
                self._update_command_pattern(command, intent)

        except Exception as e:
            logger.error(f"Failed to log command: {e}")

    def _update_command_pattern(self, command: str, intent: Optional[str]):
        """Update command pattern frequency"""
        if not intent:
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Normalize command (lowercase, remove extra spaces)
            pattern = " ".join(command.lower().split())

            cursor.execute("""
                INSERT INTO command_patterns (pattern, intent, frequency, last_used)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(pattern, intent) DO UPDATE SET
                    frequency = frequency + 1,
                    last_used = ?
            """, (pattern, intent, datetime.now().isoformat(), datetime.now().isoformat()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update command pattern: {e}")

    def get_command_history(self, limit: int = 10) -> List[Dict]:
        """Get recent command history"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, command, intent, actions, success, execution_time
                FROM command_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            history = []
            for row in rows:
                history.append({
                    "timestamp": row[0],
                    "command": row[1],
                    "intent": row[2],
                    "actions": json.loads(row[3]) if row[3] else [],
                    "success": bool(row[4]),
                    "execution_time": row[5]
                })

            return history

        except Exception as e:
            logger.error(f"Failed to get command history: {e}")
            return []

    def get_similar_commands(self, command: str, limit: int = 5) -> List[Dict]:
        """Get similar commands from history"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Normalize command
            pattern = " ".join(command.lower().split())

            # Simple similarity: match keywords
            keywords = pattern.split()
            where_clause = " OR ".join([f"pattern LIKE '%{kw}%'" for kw in keywords])

            cursor.execute(f"""
                SELECT pattern, intent, frequency, last_used
                FROM command_patterns
                WHERE {where_clause}
                ORDER BY frequency DESC, last_used DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            similar = []
            for row in rows:
                similar.append({
                    "pattern": row[0],
                    "intent": row[1],
                    "frequency": row[2],
                    "last_used": row[3]
                })

            return similar

        except Exception as e:
            logger.error(f"Failed to get similar commands: {e}")
            return []

    def set_preference(self, key: str, value: str):
        """Set a user preference"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = ?,
                    updated_at = ?
            """, (key, value, datetime.now().isoformat(), value, datetime.now().isoformat()))

            conn.commit()
            conn.close()

            logger.info(f"Preference set: {key} = {value}")

        except Exception as e:
            logger.error(f"Failed to set preference: {e}")

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a user preference"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
            row = cursor.fetchone()
            conn.close()

            return row[0] if row else default

        except Exception as e:
            logger.error(f"Failed to get preference: {e}")
            return default

    def get_all_preferences(self) -> Dict[str, str]:
        """Get all user preferences"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT key, value FROM user_preferences")
            rows = cursor.fetchall()
            conn.close()

            return {row[0]: row[1] for row in rows}

        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return {}

    def log_app_usage(self, app_name: str, duration: Optional[float] = None):
        """Log app usage"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO app_usage (app_name, open_count, total_time, last_opened)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(app_name) DO UPDATE SET
                    open_count = open_count + 1,
                    total_time = total_time + ?,
                    last_opened = ?
            """, (
                app_name,
                duration or 0,
                datetime.now().isoformat(),
                duration or 0,
                datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log app usage: {e}")

    def get_favorite_apps(self, limit: int = 5) -> List[Dict]:
        """Get most frequently used apps"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT app_name, open_count, total_time, last_opened
                FROM app_usage
                ORDER BY open_count DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            apps = []
            for row in rows:
                apps.append({
                    "app_name": row[0],
                    "open_count": row[1],
                    "total_time": row[2],
                    "last_opened": row[3]
                })

            return apps

        except Exception as e:
            logger.error(f"Failed to get favorite apps: {e}")
            return []

    def get_context(self) -> Dict:
        """Get context for AI processing"""
        try:
            context = {
                "recent_commands": self.get_command_history(limit=5),
                "favorite_apps": self.get_favorite_apps(limit=3),
                "preferences": self.get_all_preferences()
            }

            return context

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return {}

    def clear_history(self):
        """Clear command history"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("DELETE FROM command_history")
            conn.commit()
            conn.close()

            logger.info("Command history cleared")

        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
