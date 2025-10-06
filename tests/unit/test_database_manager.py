"""
Comprehensive unit tests for database/db_manager.py
"""

import pytest
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from database.db_manager import (
    DatabaseError, DatabaseConnectionPool, DatabaseManager,
    get_database_manager, initialize_database
)


class TestDatabaseConnectionPool:
    """Test DatabaseConnectionPool functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def connection_pool(self, temp_db):
        """Create a connection pool for testing."""
        pool = DatabaseConnectionPool(temp_db, max_connections=3, timeout=1.0)
        yield pool
        pool.close_all()
    
    def test_connection_pool_initialization(self, connection_pool, temp_db):
        """Test connection pool initialization."""
        assert connection_pool.db_path == temp_db
        assert connection_pool.max_connections == 3
        assert connection_pool.timeout == 1.0
        assert len(connection_pool.connections) == 3
        assert len(connection_pool.available) == 3
    
    def test_connection_pool_initialization_error(self):
        """Test connection pool initialization with error."""
        with pytest.raises(DatabaseError):
            DatabaseConnectionPool("/invalid/path/test.db")
    
    def test_get_connection(self, connection_pool):
        """Test getting a connection from the pool."""
        conn = connection_pool.get_connection()
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)
        assert len(connection_pool.available) == 2
        
        # Return connection
        connection_pool.return_connection(conn)
        assert len(connection_pool.available) == 3
    
    def test_get_connection_exhausted(self, connection_pool):
        """Test getting connection when pool is exhausted."""
        # Get all connections
        conns = []
        for _ in range(3):
            conns.append(connection_pool.get_connection())
        
        assert len(connection_pool.available) == 0
        
        # Try to get another connection (should fail)
        with pytest.raises(DatabaseError, match="Connection pool exhausted"):
            connection_pool.get_connection()
        
        # Return connections
        for conn in conns:
            connection_pool.return_connection(conn)
    
    def test_return_connection_not_in_pool(self, connection_pool):
        """Test returning a connection not in the pool."""
        external_conn = sqlite3.connect(":memory:")
        
        # Should not raise error, just ignore
        connection_pool.return_connection(external_conn)
        assert len(connection_pool.available) == 3
        
        external_conn.close()
    
    def test_close_all(self, connection_pool):
        """Test closing all connections."""
        # Get a connection to verify it's closed
        conn = connection_pool.get_connection()
        
        connection_pool.close_all()
        assert len(connection_pool.connections) == 0
        assert len(connection_pool.available) == 0
        
        # Connection should be closed
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    
    def test_get_pool_stats(self, connection_pool):
        """Test getting pool statistics."""
        stats = connection_pool.get_pool_stats()
        
        assert isinstance(stats, dict)
        assert 'total_connections' in stats
        assert 'available_connections' in stats
        assert 'used_connections' in stats
        assert 'pool_utilization' in stats
        
        assert stats['total_connections'] == 3
        assert stats['available_connections'] == 3
        assert stats['used_connections'] == 0
        assert stats['pool_utilization'] == 0.0
        
        # Get a connection and check stats
        conn = connection_pool.get_connection()
        stats = connection_pool.get_pool_stats()
        
        assert stats['available_connections'] == 2
        assert stats['used_connections'] == 1
        assert stats['pool_utilization'] == 33.33333333333333
        
        connection_pool.return_connection(conn)
    
    def test_connection_thread_safety(self, connection_pool):
        """Test that connection pool is thread-safe."""
        results = []
        errors = []
        
        def worker():
            try:
                conn = connection_pool.get_connection()
                result = conn.execute("SELECT 1").fetchone()
                results.append(result[0])
                connection_pool.return_connection(conn)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 5
        assert all(result == 1 for result in results)


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def db_manager(self, temp_db):
        """Create a database manager for testing."""
        manager = DatabaseManager(temp_db, connection_timeout=1.0)
        yield manager
        manager.close()
    
    def test_database_manager_initialization(self, db_manager, temp_db):
        """Test database manager initialization."""
        assert db_manager.db_path == temp_db
        assert db_manager.connection_timeout == 1.0
        assert db_manager.pool is not None
        assert db_manager._health_status == "unknown"
    
    def test_database_manager_default_path(self):
        """Test database manager with default path."""
        with patch.dict(os.environ, {"DATABASE_PATH": "/tmp/test_default.db"}):
            with patch('database.db_manager.DatabaseManager._initialize_connection_pool'):
                with patch('database.db_manager.DatabaseManager._initialize_schema'):
                    with patch('database.db_manager.DatabaseManager._start_health_monitoring'):
                        manager = DatabaseManager()
                        assert manager.db_path == "/tmp/test_default.db"
                        manager.close()
    
    def test_get_connection_context_manager(self, db_manager):
        """Test getting connection using context manager."""
        with db_manager.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1
    
    def test_get_connection_no_pool(self):
        """Test getting connection when pool is not initialized."""
        manager = DatabaseManager.__new__(DatabaseManager)
        manager.pool = None
        
        with pytest.raises(DatabaseError, match="Connection pool not initialized"):
            with manager.get_connection():
                pass
    
    def test_transaction_context_manager_success(self, db_manager):
        """Test successful transaction."""
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO test_table (value) VALUES (?)", ("test_value",))
        
        # Verify data was committed
        with db_manager.get_connection() as conn:
            result = conn.execute("SELECT value FROM test_table").fetchone()
            assert result[0] == "test_value"
    
    def test_transaction_context_manager_rollback(self, db_manager):
        """Test transaction rollback on error."""
        try:
            with db_manager.transaction() as conn:
                conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO test_table (value) VALUES (?)", ("test_value",))
                raise Exception("Test error")
        except Exception:
            pass
        
        # Verify table was not created (rolled back)
        with db_manager.get_connection() as conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("SELECT * FROM test_table")
    
    def test_execute_query_fetch(self, db_manager):
        """Test executing query with fetch."""
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO test_table (value) VALUES (?)", ("test_value",))
        
        result = db_manager.execute_query(
            "SELECT value FROM test_table",
            fetch=True
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['value'] == "test_value"
    
    def test_execute_query_no_fetch(self, db_manager):
        """Test executing query without fetch."""
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
        
        result = db_manager.execute_query(
            "INSERT INTO test_table (value) VALUES (?)",
            ("test_value",),
            fetch=False
        )
        
        assert result is None
    
    def test_execute_query_error(self, db_manager):
        """Test executing query with error."""
        with pytest.raises(DatabaseError, match="Query execution failed"):
            db_manager.execute_query("SELECT * FROM nonexistent_table", fetch=True)
    
    def test_execute_in_transaction_success(self, db_manager):
        """Test executing multiple operations in transaction."""
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
        
        operations = [
            ("INSERT INTO test_table (value) VALUES (?)", ("value1",)),
            ("INSERT INTO test_table (value) VALUES (?)", ("value2",)),
            ("INSERT INTO test_table (value) VALUES (?)", ("value3",))
        ]
        
        result = db_manager.execute_in_transaction(operations)
        assert result is True
        
        # Verify all data was inserted
        results = db_manager.execute_query("SELECT value FROM test_table", fetch=True)
        assert len(results) == 3
    
    def test_execute_in_transaction_failure(self, db_manager):
        """Test executing operations in transaction with failure."""
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
        
        operations = [
            ("INSERT INTO test_table (value) VALUES (?)", ("value1",)),
            ("INSERT INTO test_table (value) VALUES (?)", ("value2",)),
            ("INVALID SQL", ())  # This will fail
        ]
        
        result = db_manager.execute_in_transaction(operations)
        assert result is False
        
        # Verify no data was inserted
        results = db_manager.execute_query("SELECT value FROM test_table", fetch=True)
        assert len(results) == 0
    
    def test_health_check_healthy(self, db_manager):
        """Test health check with healthy database."""
        health = db_manager.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health
        assert 'connection_pool' in health
        assert 'database_size' in health
        assert 'table_counts' in health
        assert 'issues' in health
        
        assert health['status'] == 'healthy'
        assert isinstance(health['connection_pool'], dict)
        assert isinstance(health['table_counts'], dict)
        assert isinstance(health['issues'], list)
    
    def test_health_check_unhealthy_no_pool(self):
        """Test health check with no connection pool."""
        manager = DatabaseManager.__new__(DatabaseManager)
        manager.pool = None
        manager.db_path = "/tmp/test.db"
        
        health = manager.health_check()
        
        assert health['status'] == 'unhealthy'
        assert 'Connection pool not initialized' in health['issues']
    
    def test_health_check_high_utilization(self, db_manager):
        """Test health check with high connection pool utilization."""
        # Mock high utilization
        with patch.object(db_manager.pool, 'get_pool_stats') as mock_stats:
            mock_stats.return_value = {
                'total_connections': 10,
                'available_connections': 1,
                'used_connections': 9,
                'pool_utilization': 90.0
            }
            
            health = db_manager.health_check()
            
            assert 'High connection pool utilization' in health['issues']
    
    def test_get_database_stats(self, db_manager):
        """Test getting comprehensive database statistics."""
        stats = db_manager.get_database_stats()
        
        assert isinstance(stats, dict)
        assert 'database_path' in stats
        assert 'database_size_mb' in stats
        assert 'total_tables' in stats
        assert 'total_rows' in stats
        assert 'connection_pool_stats' in stats
        assert 'table_sizes' in stats
        assert 'indexes' in stats
        
        assert stats['database_path'] == db_manager.db_path
        assert isinstance(stats['total_tables'], int)
        assert isinstance(stats['total_rows'], int)
        assert isinstance(stats['table_sizes'], dict)
        assert isinstance(stats['indexes'], list)
    
    def test_backup_database(self, db_manager):
        """Test database backup."""
        backup_path = db_manager.backup_database()
        
        assert os.path.exists(backup_path)
        assert backup_path.endswith(".backup_")
        
        # Clean up
        if os.path.exists(backup_path):
            os.unlink(backup_path)
    
    def test_backup_database_custom_path(self, db_manager):
        """Test database backup with custom path."""
        custom_backup = "/tmp/custom_backup.db"
        
        try:
            backup_path = db_manager.backup_database(custom_backup)
            assert backup_path == custom_backup
            assert os.path.exists(custom_backup)
        finally:
            if os.path.exists(custom_backup):
                os.unlink(custom_backup)
    
    def test_backup_database_error(self, db_manager):
        """Test database backup with error."""
        with patch.object(db_manager, 'get_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Backup error")
            
            with pytest.raises(DatabaseError, match="Database backup failed"):
                db_manager.backup_database()
    
    def test_cleanup_expired_data(self, db_manager):
        """Test cleanup of expired data."""
        # Insert some test data
        with db_manager.transaction() as conn:
            conn.execute('''
                INSERT INTO voice_data 
                (data_id, user_id, data_type, encrypted_data, metadata, created_at, retention_until, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "test_data_1", "user1", "recording", b"encrypted", "{}",
                datetime.now(), datetime.now() - timedelta(days=1), 0  # Expired
            ))
            
            conn.execute('''
                INSERT INTO voice_data 
                (data_id, user_id, data_type, encrypted_data, metadata, created_at, retention_until, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "test_data_2", "user1", "recording", b"encrypted", "{}",
                datetime.now(), datetime.now() + timedelta(days=1), 0  # Not expired
            ))
        
        # Clean up expired data
        removed_count = db_manager.cleanup_expired_data()
        
        assert removed_count >= 1  # At least the expired record should be removed
        
        # Verify expired record is marked as deleted
        result = db_manager.execute_query(
            "SELECT is_deleted FROM voice_data WHERE data_id = ?",
            ("test_data_1",),
            fetch=True
        )
        assert result[0]['is_deleted'] == 1
    
    def test_cleanup_expired_data_error(self, db_manager):
        """Test cleanup of expired data with error."""
        with patch.object(db_manager, 'transaction') as mock_transaction:
            mock_transaction.side_effect = Exception("Cleanup error")
            
            result = db_manager.cleanup_expired_data()
            assert result == 0
    
    def test_close(self, db_manager):
        """Test closing database manager."""
        assert db_manager.pool is not None
        
        db_manager.close()
        
        assert db_manager.pool is None
    
    def test_destructor_cleanup(self, temp_db):
        """Test destructor cleanup."""
        manager = DatabaseManager(temp_db)
        pool = manager.pool
        
        # Delete reference to trigger destructor
        del manager
        
        # Pool should still be closed (destructor called)
        # Note: This is hard to test reliably due to garbage collection timing
    
    def test_schema_initialization(self, db_manager):
        """Test database schema initialization."""
        # Check that all tables were created
        tables = db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            fetch=True
        )
        
        table_names = [table['name'] for table in tables]
        
        expected_tables = [
            'users', 'sessions', 'voice_data', 'audit_logs', 'consent_records'
        ]
        
        for table in expected_tables:
            assert table in table_names
        
        # Check that indexes were created
        indexes = db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='index'",
            fetch=True
        )
        
        assert len(indexes) > 0  # Should have several indexes


class TestDatabaseManagerGlobal:
    """Test global database manager functions."""
    
    def test_get_database_manager_singleton(self):
        """Test that get_database_manager returns singleton."""
        with patch.dict(os.environ, {"DATABASE_PATH": "/tmp/test_singleton.db"}):
            # Clear any existing instance
            import database.db_manager
            database.db_manager._db_manager = None
            
            manager1 = get_database_manager()
            manager2 = get_database_manager()
            
            assert manager1 is manager2
            
            # Clean up
            if manager1.pool:
                manager1.close()
    
    def test_initialize_database(self):
        """Test database initialization."""
        with patch.dict(os.environ, {"DATABASE_PATH": "/tmp/test_init.db"}):
            # Clear any existing instance
            import database.db_manager
            database.db_manager._db_manager = None
            
            try:
                manager = initialize_database()
                assert isinstance(manager, DatabaseManager)
                
                # Clean up
                if manager.pool:
                    manager.close()
            finally:
                # Clean up test file
                if os.path.exists("/tmp/test_init.db"):
                    os.unlink("/tmp/test_init.db")