# database/connection.py
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator
from config import config

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.db_path = config.database.name
            self.timeout = config.database.timeout
            self.check_same_thread = config.database.check_same_thread
            self.initialized = True
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=self.check_same_thread
        )
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_script(self, script: str) -> None:
        """Execute a SQL script."""
        with self.get_connection() as conn:
            conn.executescript(script)
            conn.commit()
    
    def execute_query(self, query: str, params=None):
        """Execute a single query and return results."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list):
        """Execute a query with multiple parameter sets."""
        with self.get_connection() as conn:
            conn.executemany(query, params_list)
            conn.commit()

db_manager = DatabaseManager()