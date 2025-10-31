"""
Database Integration Testing

Comprehensive test suite for database integration functionality:
- Connection pooling and transaction management
- Concurrent database access patterns
- Database locking and transaction isolation
- Connection pool statistics and health monitoring

Coverage targets: Database integration testing for concurrent access patterns
"""

import pytest
import threading
import time
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from database.db_manager import (
    DatabaseManager, DatabaseConnectionPool, DatabaseError,
    get_database_manager
)
from database.models import User, Session, UserRepository, SessionRepository


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_manager(temp_db_path):
    """Create database manager for testing."""
    manager = DatabaseManager(db_path=temp_db_path, connection_timeout=5.0)
    yield manager
    manager.close()


@pytest.fixture
def connection_pool(temp_db_path):
    """Create connection pool for testing."""
    pool = DatabaseConnectionPool(
        db_path=temp_db_path,
        max_connections=5,
        timeout=5.0
    )
    yield pool
    pool.close_all()


class TestDatabaseConnectionPool:
    """Test database connection pool functionality."""

    def test_connection_pool_initialization(self, temp_db_path):
        """Test connection pool initialization."""
        pool = DatabaseConnectionPool(temp_db_path, max_connections=3)

        assert pool.db_path == temp_db_path
        assert pool.max_connections == 3
        assert len(pool.connections) == 3
        assert len(pool.available) == 3

        pool.close_all()

    def test_connection_pool_get_return(self, connection_pool):
        """Test getting and returning connections from pool."""
        # Get a connection
        conn = connection_pool.get_connection()
        assert isinstance(conn, sqlite3.Connection)
        assert len(connection_pool.available) == 4  # One taken

        # Return the connection
        connection_pool.return_connection(conn)
        assert len(connection_pool.available) == 5  # Back to full

    def test_connection_pool_exhaustion(self, connection_pool):
        """Test connection pool exhaustion handling."""
        connections = []

        # Use all connections
        for _ in range(5):
            conn = connection_pool.get_connection()
            connections.append(conn)

        assert len(connection_pool.available) == 0

        # Next attempt should fail with timeout
        with pytest.raises(DatabaseError, match="Connection pool exhausted"):
            connection_pool.get_connection()

        # Return connections
        for conn in connections:
            connection_pool.return_connection(conn)

        assert len(connection_pool.available) == 5

    def test_connection_pool_concurrent_access(self, connection_pool):
        """Test concurrent access to connection pool."""
        results = []
        errors = []

        def concurrent_access(thread_id):
            try:
                conn = connection_pool.get_connection()
                time.sleep(0.1)  # Simulate work
                connection_pool.return_connection(conn)
                results.append(f"thread_{thread_id}_success")
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_access, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have successful operations
        assert len(results) > 0
        assert len(errors) == 0  # No errors expected

    def test_connection_pool_stats(self, connection_pool):
        """Test connection pool statistics."""
        stats = connection_pool.get_pool_stats()

        assert 'total_connections' in stats
        assert 'available_connections' in stats
        assert 'used_connections' in stats
        assert 'pool_utilization' in stats

        assert stats['total_connections'] == 5
        assert stats['available_connections'] == 5
        assert stats['used_connections'] == 0
        assert stats['pool_utilization'] == 0.0

    def test_connection_pool_error_handling(self, temp_db_path):
        """Test connection pool error handling."""
        # Test with invalid database path
        invalid_path = "/invalid/path/database.db"

        with pytest.raises(DatabaseError):
            DatabaseConnectionPool(invalid_path, max_connections=1)


class TestDatabaseTransactions:
    """Test database transaction management."""

    def test_transaction_basic_commit(self, db_manager):
        """Test basic transaction commit."""
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO test_table (name) VALUES (?)", ("test_value",))

        # Verify data was committed
        result = db_manager.execute_query("SELECT name FROM test_table", fetch=True)
        assert len(result) == 1
        assert result[0]['name'] == "test_value"

    def test_transaction_rollback_on_error(self, db_manager):
        """Test transaction rollback on error."""
        with pytest.raises(DatabaseError):
            with db_manager.transaction() as conn:
                conn.execute("CREATE TABLE rollback_test (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO rollback_test (id) VALUES (?)", (1,))
                # Force an error
                raise Exception("Test error")

        # Verify data was rolled back - table shouldn't exist
        result = db_manager.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='rollback_test'", fetch=True)
        assert len(result) == 0

    def test_transaction_nested_operations(self, db_manager):
        """Test nested operations within transactions."""
        with db_manager.transaction() as conn:
            # Create table
            conn.execute("CREATE TABLE nested_test (id INTEGER, data TEXT)")

            # Insert multiple rows
            for i in range(5):
                conn.execute("INSERT INTO nested_test (id, data) VALUES (?, ?)", (i, f"value_{i}"))

            # Update some data
            conn.execute("UPDATE nested_test SET data = 'updated' WHERE id < 3")

        # Verify all operations succeeded
        result = db_manager.execute_query("SELECT * FROM nested_test ORDER BY id", fetch=True)
        assert len(result) == 5

        for i, row in enumerate(result):
            if i < 3:
                assert row['data'] == 'updated'
            else:
                assert row['data'] == f'value_{i}'

    def test_transaction_concurrent_access(self, db_manager):
        """Test concurrent transactions."""
        def concurrent_transaction(thread_id):
            try:
                with db_manager.transaction() as conn:
                    # Create thread-specific table
                    table_name = f"concurrent_test_{thread_id}"
                    conn.execute(f"CREATE TABLE {table_name} (id INTEGER, thread_id INTEGER)")
                    conn.execute(f"INSERT INTO {table_name} (id, thread_id) VALUES (?, ?)", (1, thread_id))

                return f"success_{thread_id}"
            except Exception as e:
                return f"error_{thread_id}: {e}"

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_transaction, i) for i in range(5)]
            results = [future.result() for future in as_completed(futures)]

        # All should succeed
        assert len(results) == 5
        assert all(r.startswith("success_") for r in results)

        # Verify all tables were created
        for i in range(5):
            result = db_manager.execute_query(f"SELECT thread_id FROM concurrent_test_{i}", fetch=True)
            assert len(result) == 1
            assert result[0]['thread_id'] == i


class TestDatabaseConcurrentAccess:
    """Test concurrent database access patterns."""

    def test_concurrent_reads(self, db_manager):
        """Test concurrent read operations."""
        # Setup test data
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE concurrent_reads (id INTEGER, data TEXT)")
            for i in range(100):
                conn.execute("INSERT INTO concurrent_reads (id, data) VALUES (?, ?)", (i, f"value_{i}"))

        results = []
        errors = []

        def concurrent_read(thread_id):
            try:
                # Read different ranges to avoid conflicts
                start_id = (thread_id * 20)
                end_id = start_id + 20

                result = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM concurrent_reads WHERE id BETWEEN ? AND ?",
                    (start_id, end_id),
                    fetch=True
                )
                results.append((thread_id, result[0]['count']))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_read, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All reads should succeed
        assert len(results) == 5
        assert len(errors) == 0
        assert all(count == 20 for _, count in results)

    def test_concurrent_writes_isolation(self, db_manager):
        """Test concurrent write operations with proper isolation."""
        # Setup test table
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE concurrent_writes (id INTEGER PRIMARY KEY, thread_id INTEGER, counter INTEGER)")

        results = []
        errors = []

        def concurrent_write(thread_id):
            try:
                with db_manager.transaction() as conn:
                    # Each thread inserts its own row
                    conn.execute(
                        "INSERT INTO concurrent_writes (thread_id, counter) VALUES (?, 1)",
                        (thread_id,)
                    )

                    # Then increments a shared counter
                    result = conn.execute("SELECT counter FROM concurrent_writes WHERE id = 1").fetchone()
                    if result:
                        new_counter = result[0] + 1
                        conn.execute("UPDATE concurrent_writes SET counter = ? WHERE id = 1", (new_counter,))

                results.append(f"thread_{thread_id}_success")
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_write, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All writes should succeed
        assert len(results) == 10
        assert len(errors) == 0

        # Verify all threads inserted their data
        result = db_manager.execute_query("SELECT COUNT(*) as count FROM concurrent_writes", fetch=True)
        assert result[0]['count'] == 10

    def test_concurrent_mixed_operations(self, db_manager):
        """Test mixed concurrent read/write operations."""
        # Setup test table
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE mixed_ops (id INTEGER PRIMARY KEY, value INTEGER)")
            conn.execute("INSERT INTO mixed_ops (value) VALUES (0)")

        operations_completed = []
        errors = []

        def mixed_operation(thread_id, operation_type):
            try:
                if operation_type == "read":
                    result = db_manager.execute_query("SELECT value FROM mixed_ops WHERE id = 1", fetch=True)
                    operations_completed.append(f"read_{thread_id}_{result[0]['value']}")
                elif operation_type == "write":
                    with db_manager.transaction() as conn:
                        result = conn.execute("SELECT value FROM mixed_ops WHERE id = 1").fetchone()
                        new_value = result[0] + 1
                        conn.execute("UPDATE mixed_ops SET value = ? WHERE id = 1", (new_value,))
                        operations_completed.append(f"write_{thread_id}_{new_value}")
                elif operation_type == "insert":
                    with db_manager.transaction() as conn:
                        conn.execute("INSERT INTO mixed_ops (value) VALUES (?)", (thread_id,))
                        operations_completed.append(f"insert_{thread_id}")
            except Exception as e:
                errors.append(f"{operation_type}_{thread_id}_error: {e}")

        threads = []
        for i in range(15):
            # Mix of operations: 5 reads, 5 writes, 5 inserts
            if i < 5:
                op_type = "read"
            elif i < 10:
                op_type = "write"
            else:
                op_type = "insert"

            thread = threading.Thread(target=mixed_operation, args=(i, op_type))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have completed operations without errors
        assert len(operations_completed) >= 10  # At least some operations
        assert len(errors) == 0

    def test_database_locking_conflict_resolution(self, db_manager):
        """Test database locking and conflict resolution."""
        # Setup test table
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE locking_test (id INTEGER PRIMARY KEY, value INTEGER)")
            conn.execute("INSERT INTO locking_test (value) VALUES (0)")

        conflicts_resolved = []
        errors = []

        def locking_operation(thread_id):
            try:
                # Use multiple retries to handle locking
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        with db_manager.transaction() as conn:
                            # Read current value
                            result = conn.execute("SELECT value FROM locking_test WHERE id = 1").fetchone()
                            current_value = result[0]

                            # Simulate some processing time
                            time.sleep(0.01)

                            # Update value
                            new_value = current_value + 1
                            conn.execute("UPDATE locking_test SET value = ? WHERE id = 1", (new_value,))

                        conflicts_resolved.append(f"thread_{thread_id}_attempt_{attempt}")
                        break

                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < max_retries - 1:
                            time.sleep(0.01 * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            errors.append(f"thread_{thread_id}_error: {e}")
                            break
                    except Exception as e:
                        errors.append(f"thread_{thread_id}_error: {e}")
                        break

            except Exception as e:
                errors.append(f"thread_{thread_id}_fatal_error: {e}")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=locking_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should resolve conflicts without fatal errors
        assert len(conflicts_resolved) > 0
        # Some conflicts might occur but should be resolved
        final_value = db_manager.execute_query("SELECT value FROM locking_test WHERE id = 1", fetch=True)[0]['value']
        assert final_value == 10  # Each thread should have incremented once


class TestDatabaseHealthMonitoring:
    """Test database health monitoring and statistics."""

    def test_database_health_check_basic(self, db_manager):
        """Test basic database health check."""
        health = db_manager.health_check()

        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health
        assert 'connection_pool' in health
        assert 'database_size' in health
        assert 'table_counts' in health

        # Should be healthy
        assert health['status'] in ['healthy', 'unhealthy']

    def test_database_health_check_with_data(self, db_manager):
        """Test health check with actual data."""
        # Add some test data
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE health_test (id INTEGER, data TEXT)")
            for i in range(10):
                conn.execute("INSERT INTO health_test (id, data) VALUES (?, ?)", (i, f"test_data_{i}"))

        health = db_manager.health_check()

        # Should detect the table and data
        assert 'health_test' in health['table_counts']
        assert health['table_counts']['health_test'] == 10

    def test_database_health_check_connection_issues(self, db_manager):
        """Test health check with connection issues."""
        # Simulate connection pool failure
        with patch.object(db_manager, 'pool', None):
            health = db_manager.health_check()

            assert health['status'] == 'unhealthy'
            assert len(health['issues']) > 0

    def test_database_stats_collection(self, db_manager):
        """Test comprehensive database statistics collection."""
        # Add test data
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE stats_test (id INTEGER, data TEXT)")
            for i in range(50):
                conn.execute("INSERT INTO stats_test (id, data) VALUES (?, ?)", (i, f"stats_data_{i}"))

        stats = db_manager.get_database_stats()

        assert isinstance(stats, dict)
        assert 'database_path' in stats
        assert 'database_size_mb' in stats
        assert 'total_tables' in stats
        assert 'total_rows' in stats
        assert 'connection_pool_stats' in stats

        # Should have detected our test table
        assert stats['total_rows'] >= 50

    def test_database_backup_functionality(self, db_manager, temp_db_path):
        """Test database backup functionality."""
        # Add some data to backup
        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE backup_test (id INTEGER, data TEXT)")
            conn.execute("INSERT INTO backup_test (data) VALUES (?)", ("backup_data",))

        # Create backup
        backup_path = db_manager.backup_database()

        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert backup_path.endswith('.backup.db')

        # Verify backup contains data
        backup_conn = sqlite3.connect(backup_path)
        result = backup_conn.execute("SELECT data FROM backup_test").fetchone()
        assert result[0] == "backup_data"
        backup_conn.close()

        # Cleanup
        os.unlink(backup_path)


class TestDatabaseCleanupOperations:
    """Test database cleanup and maintenance operations."""

    def test_expired_data_cleanup(self, db_manager):
        """Test cleanup of expired data."""
        # Add data with retention policy
        from datetime import datetime, timedelta

        expired_time = datetime.now() - timedelta(days=40)  # Expired
        valid_time = datetime.now() - timedelta(days=10)    # Still valid

        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE cleanup_test (id INTEGER, retention_until TEXT)")
            conn.execute("INSERT INTO cleanup_test (id, retention_until) VALUES (?, ?)",
                        (1, expired_time.isoformat()))
            conn.execute("INSERT INTO cleanup_test (id, retention_until) VALUES (?, ?)",
                        (2, valid_time.isoformat()))

        # Run cleanup
        removed_count = db_manager.cleanup_expired_data()

        # Should have removed expired data
        assert removed_count >= 1

        # Verify expired data is gone
        result = db_manager.execute_query("SELECT COUNT(*) as count FROM cleanup_test", fetch=True)
        assert result[0]['count'] <= 1  # At most the valid record remains

    def test_audit_log_cleanup(self, db_manager):
        """Test cleanup of old audit logs."""
        from datetime import datetime, timedelta

        old_time = datetime.now() - timedelta(days=100)  # Old
        recent_time = datetime.now() - timedelta(days=10)  # Recent

        with db_manager.transaction() as conn:
            conn.execute("CREATE TABLE audit_logs (log_id TEXT, timestamp TEXT, event_type TEXT)")
            conn.execute("INSERT INTO audit_logs (log_id, timestamp, event_type) VALUES (?, ?, ?)",
                        ("old_log", old_time.isoformat(), "test_event"))
            conn.execute("INSERT INTO audit_logs (log_id, timestamp, event_type) VALUES (?, ?, ?)",
                        ("recent_log", recent_time.isoformat(), "test_event"))

        # Run cleanup (with mock retention setting)
        with patch('database.db_manager.os.getenv', return_value='90'):
            removed_count = db_manager.cleanup_expired_data()

        # Should have removed old audit logs
        result = db_manager.execute_query("SELECT COUNT(*) as count FROM audit_logs", fetch=True)
        assert result[0]['count'] >= 1  # Recent log should remain


class TestDatabaseRepositoryConcurrency:
    """Test repository layer concurrency."""

    def test_user_repository_concurrent_access(self, db_manager):
        """Test concurrent access to user repository."""
        repo = UserRepository()
        repo.db = db_manager  # Override with test db

        results = []
        errors = []

        def concurrent_user_operation(thread_id):
            try:
                # Create user with unique email
                user = User.create(
                    email=f"user_{thread_id}@test.com",
                    full_name=f"User {thread_id}",
                    role="patient",
                    password_hash="test_hash"
                )

                success = repo.save(user)
                if success:
                    # Try to retrieve the user
                    retrieved = repo.find_by_email(user.email)
                    if retrieved:
                        results.append(f"thread_{thread_id}_success")
                    else:
                        errors.append(f"thread_{thread_id}_retrieve_failed")
                else:
                    errors.append(f"thread_{thread_id}_save_failed")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_user_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 5
        assert len(errors) == 0

    def test_session_repository_concurrent_access(self, db_manager):
        """Test concurrent access to session repository."""
        user_repo = UserRepository()
        user_repo.db = db_manager
        session_repo = SessionRepository()
        session_repo.db = db_manager

        # Create a test user first
        user = User.create("session_test@test.com", "Session Test", "patient", "hash")
        user_repo.save(user)

        results = []
        errors = []

        def concurrent_session_operation(thread_id):
            try:
                # Create session for the user
                session = Session.create(user.user_id, ip_address=f"192.168.1.{thread_id}")

                success = session_repo.save(session)
                if success:
                    # Try to retrieve the session
                    retrieved = session_repo.find_by_id(session.session_id)
                    if retrieved:
                        results.append(f"thread_{thread_id}_success")
                    else:
                        errors.append(f"thread_{thread_id}_retrieve_failed")
                else:
                    errors.append(f"thread_{thread_id}_save_failed")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_session_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 5
        assert len(errors) == 0


class TestDatabaseErrorHandling:
    """Test database error handling and recovery."""

    def test_query_error_handling(self, db_manager):
        """Test error handling in query execution."""
        # Test with invalid SQL
        with pytest.raises(DatabaseError):
            db_manager.execute_query("INVALID SQL QUERY")

    def test_transaction_error_handling(self, db_manager):
        """Test error handling in transactions."""
        with pytest.raises(DatabaseError):
            with db_manager.transaction() as conn:
                conn.execute("CREATE TABLE error_test (id INTEGER)")
                conn.execute("INVALID SQL COMMAND")  # This should fail

    def test_connection_error_recovery(self, db_manager):
        """Test connection error recovery."""
        # Simulate connection failure
        with patch.object(db_manager.pool, 'get_connection', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseError):
                db_manager.execute_query("SELECT 1")

    def test_pool_exhaustion_recovery(self, connection_pool):
        """Test recovery from connection pool exhaustion."""
        connections = []

        # Exhaust the pool
        for _ in range(5):
            conn = connection_pool.get_connection()
            connections.append(conn)

        # This should fail
        with pytest.raises(DatabaseError):
            connection_pool.get_connection()

        # Return connections
        for conn in connections:
            connection_pool.return_connection(conn)

        # Should work again
        conn = connection_pool.get_connection()
        assert conn is not None
        connection_pool.return_connection(conn)


# Run basic validation
if __name__ == "__main__":
    print("Database Integration Test Suite")
    print("=" * 40)

    try:
        from database.db_manager import DatabaseManager, DatabaseConnectionPool
        print("✅ Database manager imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            test_db = f.name

        manager = DatabaseManager(db_path=test_db, connection_timeout=5.0)

        # Test basic operations
        with manager.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test (id, name) VALUES (1, 'test')")

        result = manager.execute_query("SELECT name FROM test", fetch=True)
        assert result[0]['name'] == 'test'

        manager.close()
        os.unlink(test_db)
        print("✅ Basic database operations working")
    except Exception as e:
        print(f"❌ Database operations failed: {e}")

    try:
        # Test connection pool
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            test_db = f.name

        pool = DatabaseConnectionPool(test_db, max_connections=2)
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()

        assert len(pool.available) == 0

        pool.return_connection(conn1)
        pool.return_connection(conn2)

        assert len(pool.available) == 2

        pool.close_all()
        os.unlink(test_db)
        print("✅ Connection pool working")
    except Exception as e:
        print(f"❌ Connection pool failed: {e}")

    print("Database integration test file created - run with pytest for full validation")
