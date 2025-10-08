"""
Database Manager for AI Therapist.

Provides SQLite database connection management with connection pooling,
database initialization, schema creation, transaction management, and
health monitoring for HIPAA-compliant data storage.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, ContextManager
from contextlib import contextmanager
import logging

# Note: UserRole and UserStatus are not needed in this module


class DatabaseError(Exception):
    """Database-related errors."""
    pass


class DatabaseConnectionPool:
    """SQLite connection pool for thread-safe database access."""

    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 30.0):
        """Initialize connection pool."""
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self.connections: List[sqlite3.Connection] = []
        self.available: List[sqlite3.Connection] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Initialize pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            for _ in range(self.max_connections):
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=self.timeout,
                    isolation_level=None,  # Enable autocommit mode
                    check_same_thread=False  # Allow sharing across threads
                )
                conn.row_factory = sqlite3.Row  # Enable column access by name
                conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
                conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency

                self.connections.append(conn)
                self.available.append(conn)

            self.logger.info(f"Initialized database connection pool with {self.max_connections} connections")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError(f"Failed to initialize connection pool: {e}")

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        with self.lock:
            if not self.available:
                # Wait for a connection to become available
                timeout_time = time.time() + self.timeout
                while not self.available and time.time() < timeout_time:
                    time.sleep(0.01)

                if not self.available:
                    raise DatabaseError("Connection pool exhausted - no connections available")

            conn = self.available.pop()
            return conn

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        with self.lock:
            if conn in self.connections:
                self.available.append(conn)

    def close_all(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    self.logger.error(f"Error closing connection: {e}")

            self.connections.clear()
            self.available.clear()

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'total_connections': len(self.connections),
                'available_connections': len(self.available),
                'used_connections': len(self.connections) - len(self.available),
                'pool_utilization': (len(self.connections) - len(self.available)) / len(self.connections) * 100
            }


class DatabaseManager:
    """Comprehensive database manager with connection pooling and health monitoring."""

    def __init__(self, db_path: Optional[str] = None, connection_timeout: float = 30.0):
        """Initialize database manager."""
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "./data/ai_therapist.db")

        self.db_path = db_path
        self.connection_timeout = connection_timeout
        self.pool: Optional[DatabaseConnectionPool] = None
        self.logger = logging.getLogger(__name__)
        self._health_check_interval = 60  # Health check every 60 seconds
        self._last_health_check = 0
        self._health_status = "unknown"

        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Initialize connection pool
        self._initialize_connection_pool()

        # Initialize database schema
        self._initialize_schema()

        # Start health monitoring
        self._start_health_monitoring()

    def _initialize_connection_pool(self):
        """Initialize the database connection pool."""
        try:
            max_connections = int(os.getenv("DB_MAX_CONNECTIONS", "10"))
            self.pool = DatabaseConnectionPool(
                self.db_path,
                max_connections=max_connections,
                timeout=self.connection_timeout
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError(f"Failed to initialize connection pool: {e}")

    def _initialize_schema(self):
        """Initialize database schema."""
        try:
            with self.transaction() as conn:
                # Create users table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        role TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        last_login TIMESTAMP,
                        login_attempts INTEGER DEFAULT 0,
                        account_locked_until TIMESTAMP,
                        password_reset_token TEXT,
                        password_reset_expires TIMESTAMP,
                        preferences TEXT,  -- JSON string
                        medical_info TEXT, -- JSON string (HIPAA protected)
                        password_hash TEXT NOT NULL
                    )
                ''')

                # Create sessions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                    )
                ''')

                # Create voice_data table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS voice_data (
                        data_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        data_type TEXT NOT NULL,  -- 'recording', 'transcription', 'analysis'
                        encrypted_data BLOB NOT NULL,
                        metadata TEXT,  -- JSON string with HIPAA-compliant metadata
                        created_at TIMESTAMP NOT NULL,
                        retention_until TIMESTAMP,
                        is_deleted BOOLEAN DEFAULT 0,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE SET NULL
                    )
                ''')

                # Create audit_logs table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        log_id TEXT PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        event_type TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        details TEXT,  -- JSON string
                        severity TEXT DEFAULT 'INFO',
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE SET NULL
                    )
                ''')

                # Create consent_records table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS consent_records (
                        consent_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        consent_type TEXT NOT NULL,
                        granted BOOLEAN NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        version TEXT NOT NULL,
                        details TEXT,  -- JSON string
                        revoked_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                        UNIQUE(user_id, consent_type, version)
                    )
                ''')

                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_users_status ON users(status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_voice_data_user_id ON voice_data(user_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_voice_data_created_at ON voice_data(created_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_consent_records_user_id ON consent_records(user_id)')

                self.logger.info("Database schema initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}")
            raise DatabaseError(f"Failed to initialize database schema: {e}")

    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            raise DatabaseError("Database connection pool not initialized")

        conn = None
        try:
            conn = self.pool.get_connection()
            yield conn
        finally:
            if conn:
                self.pool.return_connection(conn)

    @contextmanager
    def transaction(self):
        """Execute operations within a database transaction."""
        with self.get_connection() as conn:
            try:
                # Begin transaction
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction failed, rolled back: {e}")
                raise DatabaseError(f"Transaction failed: {e}")

    def execute_query(self, query: str, params: Tuple = (), fetch: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Execute a database query."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)

                if fetch:
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                else:
                    return None

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {e}")

    def execute_in_transaction(self, operations: List[Tuple[str, Tuple]]) -> bool:
        """Execute multiple operations in a single transaction."""
        try:
            with self.transaction() as conn:
                for query, params in operations:
                    conn.execute(query, params)
                return True
        except Exception as e:
            self.logger.error(f"Transaction operations failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        health_info = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'connection_pool': {},
            'database_size': 0,
            'table_counts': {},
            'last_backup': None,
            'issues': []
        }

        try:
            # Check connection pool
            if self.pool:
                pool_stats = self.pool.get_pool_stats()
                health_info['connection_pool'] = pool_stats

                # Test connection
                with self.get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                    health_info['status'] = 'healthy'
            else:
                health_info['status'] = 'unhealthy'
                health_info['issues'].append('Connection pool not initialized')
                return health_info

            # Get database file size
            if os.path.exists(self.db_path):
                health_info['database_size'] = os.path.getsize(self.db_path)

            # Get table row counts
            table_counts = {}
            tables = ['users', 'sessions', 'voice_data', 'audit_logs', 'consent_records']

            for table in tables:
                try:
                    result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}", fetch=True)
                    table_counts[table] = result[0]['count'] if result else 0
                except Exception as e:
                    self.logger.warning(f"Could not get count for table {table}: {e}")
                    table_counts[table] = -1

            health_info['table_counts'] = table_counts

            # Check for potential issues
            if pool_stats.get('pool_utilization', 0) > 80:
                health_info['issues'].append('High connection pool utilization')

            # Check for data retention issues
            retention_query = """
                SELECT COUNT(*) as expired_count FROM voice_data
                WHERE retention_until IS NOT NULL AND retention_until < ?
            """
            result = self.execute_query(retention_query, (datetime.now(),), fetch=True)
            if result and result[0]['expired_count'] > 0:
                health_info['issues'].append(f"{result[0]['expired_count']} expired records need cleanup")

        except Exception as e:
            health_info['status'] = 'unhealthy'
            health_info['issues'].append(f"Health check failed: {str(e)}")
            self.logger.error(f"Database health check failed: {e}")

        self._health_status = health_info['status']
        return health_info

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            stats = {
                'database_path': self.db_path,
                'database_size_mb': 0,
                'total_tables': 0,
                'total_rows': 0,
                'connection_pool_stats': {},
                'table_sizes': {},
                'indexes': []
            }

            # Database file size
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

            # Connection pool stats
            if self.pool:
                stats['connection_pool_stats'] = self.pool.get_pool_stats()

            # Get table information
            with self.get_connection() as conn:
                # Get table list
                tables_result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                stats['total_tables'] = len(tables_result)

                # Get row counts and sizes for each table
                for table_row in tables_result:
                    table_name = table_row['name']
                    try:
                        count_result = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                        stats['table_sizes'][table_name] = count_result['count']
                        stats['total_rows'] += count_result['count']
                    except Exception as e:
                        self.logger.warning(f"Could not get stats for table {table_name}: {e}")

                # Get index information
                index_result = conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'").fetchall()
                stats['indexes'] = [{'name': row['name'], 'table': row['tbl_name']} for row in index_result]

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database."""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"

        try:
            # SQLite backup using VACUUM INTO (SQLite 3.27+)
            with self.get_connection() as conn:
                conn.execute(f"VACUUM INTO '{backup_path}'")

            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise DatabaseError(f"Database backup failed: {e}")

    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention policies."""
        try:
            removed_count = 0

            # Clean up expired voice data
            with self.transaction() as conn:
                # Mark expired records as deleted (soft delete for audit trail)
                expired_query = """
                    UPDATE voice_data SET is_deleted = 1
                    WHERE retention_until IS NOT NULL AND retention_until < ? AND is_deleted = 0
                """
                cursor = conn.execute(expired_query, (datetime.now(),))
                removed_count += cursor.rowcount

                # Clean up old audit logs (hard delete after retention period)
                retention_days = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))
                cutoff_date = datetime.now() - timedelta(days=retention_days)

                audit_cleanup_query = "DELETE FROM audit_logs WHERE timestamp < ?"
                cursor = conn.execute(audit_cleanup_query, (cutoff_date,))
                removed_count += cursor.rowcount

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired records")

            return removed_count

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            return 0

    def _start_health_monitoring(self):
        """Start background health monitoring."""
        def monitor_health():
            while True:
                try:
                    self.health_check()
                    time.sleep(self._health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(self._health_check_interval)

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_health, daemon=True)
        monitor_thread.start()

    def close(self):
        """Close the database manager and clean up resources."""
        try:
            if self.pool:
                self.pool.close_all()
                self.pool = None
            self.logger.info("Database manager closed")
        except Exception as e:
            self.logger.error(f"Error closing database manager: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def initialize_database(db_path: Optional[str] = None) -> DatabaseManager:
    """Initialize and return the database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager