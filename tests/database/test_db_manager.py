"""
Database Manager Tests for AI Therapist.

Tests for SQLite database connection management with connection pooling,
database initialization, schema creation, transaction management, and
health monitoring for HIPAA-compliant data storage.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from database.db_manager import DatabaseManager, DatabaseConnectionPool, DatabaseError
from database.models import User, Session, VoiceData, AuditLog, ConsentRecord


class TestDatabaseConnectionPool:
    """Test database connection pool functionality."""

    def test_pool_initialization(self):
        """Test connection pool initialization."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            pool = DatabaseConnectionPool(db_path, max_connections=5)
            assert len(pool.connections) == 5
            assert len(pool.available) == 5
            pool.close_all()
        finally:
            os.unlink(db_path)

    def test_connection_acquisition(self):
        """Test getting connections from pool."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            pool = DatabaseConnectionPool(db_path, max_connections=2)

            # Get first connection
            conn1 = pool.get_connection()
            assert len(pool.available) == 1
            assert conn1 in pool.connections

            # Get second connection
            conn2 = pool.get_connection()
            assert len(pool.available) == 0

            # Return connections
            pool.return_connection(conn1)
            assert len(pool.available) == 1

            pool.return_connection(conn2)
            assert len(pool.available) == 2

            pool.close_all()
        finally:
            os.unlink(db_path)

    def test_pool_exhaustion(self):
        """Test pool exhaustion handling."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            pool = DatabaseConnectionPool(db_path, max_connections=1)

            # Get the only connection
            conn = pool.get_connection()
            assert len(pool.available) == 0

            # Try to get another connection (should fail after timeout)
            with pytest.raises(DatabaseError, match="Connection pool exhausted"):
                pool.get_connection()

            pool.close_all()
        finally:
            os.unlink(db_path)


class TestDatabaseManager:
    """Test database manager functionality."""

    @pytest.fixture
    def db_manager(self):
        """Create a temporary database manager for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        manager = DatabaseManager(db_path)
        yield manager

        # Cleanup
        manager.close()
        os.unlink(db_path)

    def test_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager.db_path is not None
        assert db_manager.pool is not None
        assert db_manager.connection_timeout == 30.0

    def test_schema_initialization(self, db_manager):
        """Test database schema initialization."""
        # Check that tables were created
        result = db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            fetch=True
        )
        table_names = [row['name'] for row in result]

        expected_tables = ['users', 'sessions', 'voice_data', 'audit_logs', 'consent_records']
        for table in expected_tables:
            assert table in table_names

    def test_transaction_management(self, db_manager):
        """Test transaction management."""
        # Test successful transaction
        with db_manager.transaction() as conn:
            conn.execute("INSERT INTO users (user_id, email, full_name, role, status, created_at, updated_at, password_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ('test_user', 'test@example.com', 'Test User', 'patient', 'active', '2025-01-01T00:00:00', '2025-01-01T00:00:00', 'hash'))

        # Verify data was inserted
        result = db_manager.execute_query(
            "SELECT * FROM users WHERE user_id = ?",
            ('test_user',),
            fetch=True
        )
        assert len(result) == 1
        assert result[0]['email'] == 'test@example.com'

    def test_transaction_rollback(self, db_manager):
        """Test transaction rollback on error."""
        with pytest.raises(DatabaseError):
            with db_manager.transaction() as conn:
                conn.execute("INSERT INTO users (user_id, email, full_name, role, status, created_at, updated_at, password_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            ('test_user', 'test@example.com', 'Test User', 'patient', 'active', '2025-01-01T00:00:00', '2025-01-01T00:00:00', 'hash'))
                # Force an error
                raise Exception("Test error")

        # Verify data was not inserted
        result = db_manager.execute_query(
            "SELECT * FROM users WHERE user_id = ?",
            ('test_user',),
            fetch=True
        )
        assert len(result) == 0

    def test_health_check(self, db_manager):
        """Test database health check."""
        health = db_manager.health_check()

        assert 'status' in health
        assert 'timestamp' in health
        assert 'connection_pool' in health
        assert 'database_size' in health
        assert 'table_counts' in health

        # Should be healthy
        assert health['status'] == 'healthy'

    def test_database_stats(self, db_manager):
        """Test database statistics retrieval."""
        stats = db_manager.get_database_stats()

        assert 'database_path' in stats
        assert 'database_size_mb' in stats
        assert 'total_tables' in stats
        assert 'total_rows' in stats
        assert 'connection_pool_stats' in stats
        assert 'table_sizes' in stats

    def test_backup_database(self, db_manager):
        """Test database backup functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as backup_tmp:
            backup_path = backup_tmp.name

        try:
            # Create some test data first
            with db_manager.transaction() as conn:
                conn.execute("INSERT INTO users (user_id, email, full_name, role, status, created_at, updated_at, password_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            ('test_user', 'test@example.com', 'Test User', 'patient', 'active', '2025-01-01T00:00:00', '2025-01-01T00:00:00', 'hash'))

            # Create backup
            backup_file = db_manager.backup_database(backup_path)
            assert os.path.exists(backup_file)

            # Verify backup contains data
            backup_manager = DatabaseManager(backup_file)
            result = backup_manager.execute_query(
                "SELECT * FROM users WHERE user_id = ?",
                ('test_user',),
                fetch=True
            )
            assert len(result) == 1
            backup_manager.close()

        finally:
            if os.path.exists(backup_path):
                os.unlink(backup_path)


class TestDatabaseModels:
    """Test database model functionality."""

    @pytest.fixture
    def db_manager(self):
        """Create a temporary database manager for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        manager = DatabaseManager(db_path)
        yield manager

        # Cleanup
        manager.close()
        os.unlink(db_path)

    def test_user_model(self, db_manager):
        """Test User model operations."""
        user_repo = UserRepository()
        user_repo.db = db_manager  # Override with test db

        # Create user
        user = User.create(
            email="test@example.com",
            full_name="Test User",
            role="patient",
            password_hash="hashed_password"
        )

        # Save user
        assert user_repo.save(user)

        # Find user by ID
        found_user = user_repo.find_by_id(user.user_id)
        assert found_user is not None
        assert found_user.email == user.email

        # Find user by email
        found_user = user_repo.find_by_email(user.email)
        assert found_user is not None
        assert found_user.user_id == user.user_id

    def test_session_model(self, db_manager):
        """Test Session model operations."""
        session_repo = SessionRepository()
        session_repo.db = db_manager  # Override with test db

        # Create session
        session = Session.create(
            user_id="test_user",
            session_timeout_minutes=30
        )

        # Save session
        assert session_repo.save(session)

        # Find session by ID
        found_session = session_repo.find_by_id(session.session_id)
        assert found_session is not None
        assert found_session.user_id == session.user_id

        # Find sessions by user ID
        user_sessions = session_repo.find_by_user_id("test_user")
        assert len(user_sessions) == 1
        assert user_sessions[0].session_id == session.session_id

    def test_voice_data_model(self, db_manager):
        """Test VoiceData model operations."""
        voice_repo = VoiceDataRepository()
        voice_repo.db = db_manager  # Override with test db

        # Create voice data
        voice_data = VoiceData.create(
            user_id="test_user",
            data_type="recording",
            encrypted_data=b"encrypted_audio_data"
        )

        # Save voice data
        assert voice_repo.save(voice_data)

        # Find voice data by ID
        found_data = voice_repo.find_by_id(voice_data.data_id)
        assert found_data is not None
        assert found_data.user_id == voice_data.user_id

        # Find voice data by user ID
        user_data = voice_repo.find_by_user_id("test_user")
        assert len(user_data) == 1
        assert user_data[0].data_id == voice_data.data_id

    def test_audit_log_model(self, db_manager):
        """Test AuditLog model operations."""
        audit_repo = AuditLogRepository()
        audit_repo.db = db_manager  # Override with test db

        # Create audit log
        audit_log = AuditLog.create(
            event_type="VOICE_INPUT",
            user_id="test_user",
            details={"action": "test"}
        )

        # Save audit log
        assert audit_repo.save(audit_log)

        # Find audit logs by user ID
        user_logs = audit_repo.find_by_user_id("test_user")
        assert len(user_logs) == 1
        assert user_logs[0].event_type == audit_log.event_type

    def test_consent_model(self, db_manager):
        """Test ConsentRecord model operations."""
        consent_repo = ConsentRepository()
        consent_repo.db = db_manager  # Override with test db

        # Create consent record
        consent = ConsentRecord.create(
            user_id="test_user",
            consent_type="voice_recording",
            granted=True
        )

        # Save consent
        assert consent_repo.save(consent)

        # Find consents by user ID
        user_consents = consent_repo.find_by_user_id("test_user")
        assert len(user_consents) == 1
        assert user_consents[0].consent_type == consent.consent_type

        # Check active consent
        has_consent = consent_repo.has_active_consent("test_user", "voice_recording")
        assert has_consent is True