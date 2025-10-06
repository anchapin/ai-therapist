"""
Corrected unit tests for database/db_manager.py and database/models.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import threading
from datetime import datetime, timedelta
import json
import os

# Mock the auth module to avoid import issues
mock_auth = Mock()
mock_auth.UserProfile = Mock()
mock_auth.UserRole = Mock()

with patch.dict('sys.modules', {'auth': mock_auth}):
    from database.db_manager import DatabaseConnectionPool, DatabaseManager, DatabaseError
    from database.models import (
        User, Session, VoiceData, AuditLog, ConsentRecord,
        UserRepository, SessionRepository, VoiceDataRepository,
        AuditLogRepository, ConsentRepository
    )


class TestDatabaseConnectionPool:
    """Test DatabaseConnectionPool functionality."""
    
    @pytest.fixture
    def db_pool(self):
        """Create a database connection pool for testing."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            pool = DatabaseConnectionPool(":memory:", max_connections=5)
            return pool
    
    def test_pool_initialization(self, db_pool):
        """Test pool initialization."""
        assert db_pool.max_connections == 5
        assert len(db_pool.connections) == 5
        assert len(db_pool.available) == 5
    
    def test_get_connection(self, db_pool):
        """Test getting a connection from the pool."""
        conn = db_pool.get_connection()
        assert conn is not None
        assert len(db_pool.available) == 4
    
    def test_return_connection(self, db_pool):
        """Test returning a connection to the pool."""
        conn = db_pool.get_connection()
        db_pool.return_connection(conn)
        assert len(db_pool.available) == 5
    
    def test_get_connection_from_pool(self, db_pool):
        """Test getting a connection from the available pool."""
        conn = db_pool.get_connection()
        db_pool.return_connection(conn)
        conn2 = db_pool.get_connection()
        assert conn2 is conn
        assert len(db_pool.available) == 4
    
    def test_max_connections_limit(self, db_pool):
        """Test max connections limit."""
        connections = []
        for _ in range(5):
            connections.append(db_pool.get_connection())
        
        # Should have reached max connections
        assert len(db_pool.available) == 0
        
        # Next connection should raise an exception
        with pytest.raises(DatabaseError):
            db_pool.get_connection()
    
    def test_close_all_connections(self, db_pool):
        """Test closing all connections."""
        conn = db_pool.get_connection()
        db_pool.return_connection(conn)
        db_pool.close_all()
        assert len(db_pool.connections) == 0
        assert len(db_pool.available) == 0
    
    def test_get_pool_stats(self, db_pool):
        """Test getting pool statistics."""
        stats = db_pool.get_pool_stats()
        assert 'total_connections' in stats
        assert 'available_connections' in stats
        assert 'used_connections' in stats
        assert 'pool_utilization' in stats
        assert stats['total_connections'] == 5
        assert stats['available_connections'] == 5


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a database manager for testing."""
        with patch('database.db_manager.DatabaseConnectionPool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            with patch.object(DatabaseManager, '_initialize_schema'):
                with patch.object(DatabaseManager, '_start_health_monitoring'):
                    manager = DatabaseManager(":memory:")
                    return manager
    
    def test_manager_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager.pool is not None
        assert db_manager.db_path == ":memory:"
        assert db_manager.connection_timeout == 30.0
    
    def test_get_connection_context_manager(self, db_manager):
        """Test getting a connection through context manager."""
        mock_conn = Mock()
        db_manager.pool.get_connection.return_value = mock_conn
        
        with db_manager.get_connection() as conn:
            assert conn is mock_conn
        
        db_manager.pool.return_connection.assert_called_once_with(mock_conn)
    
    def test_transaction_context_manager(self, db_manager):
        """Test transaction context manager."""
        mock_conn = Mock()
        db_manager.pool.get_connection.return_value = mock_conn
        
        with db_manager.transaction() as conn:
            assert conn is mock_conn
        
        mock_conn.execute.assert_any_call("BEGIN IMMEDIATE")
        mock_conn.commit.assert_called_once()
    
    def test_transaction_context_manager_rollback(self, db_manager):
        """Test transaction context manager rollback on exception."""
        mock_conn = Mock()
        db_manager.pool.get_connection.return_value = mock_conn
        
        with pytest.raises(ValueError):
            with db_manager.transaction() as conn:
                raise ValueError("Test error")
        
        mock_conn.rollback.assert_called_once()
    
    def test_execute_query(self, db_manager):
        """Test executing a query."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        db_manager.pool.get_connection.return_value = mock_conn
        
        # Mock query results
        mock_cursor.fetchall.return_value = [{'id': 1, 'name': 'test'}]
        
        results = db_manager.execute_query("SELECT * FROM test", fetch=True)
        
        assert results == [{'id': 1, 'name': 'test'}]
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test", ())
    
    def test_execute_query_with_params(self, db_manager):
        """Test executing a query with parameters."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        db_manager.pool.get_connection.return_value = mock_conn
        
        results = db_manager.execute_query("SELECT * FROM test WHERE id = ?", (1,), fetch=True)
        
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test WHERE id = ?", (1,))
    
    def test_execute_in_transaction(self, db_manager):
        """Test executing multiple operations in a transaction."""
        mock_conn = Mock()
        db_manager.pool.get_connection.return_value = mock_conn
        
        operations = [
            ("INSERT INTO test (name) VALUES (?)", ("test1",)),
            ("INSERT INTO test (name) VALUES (?)", ("test2",))
        ]
        
        result = db_manager.execute_in_transaction(operations)
        
        assert result is True
        assert mock_conn.execute.call_count == 2
    
    def test_health_check(self, db_manager):
        """Test database health check."""
        mock_pool = Mock()
        mock_pool.get_pool_stats.return_value = {
            'total_connections': 10,
            'available_connections': 8,
            'used_connections': 2,
            'pool_utilization': 20.0
        }
        db_manager.pool = mock_pool
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        db_manager.pool.get_connection.return_value = mock_conn
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                health = db_manager.health_check()
        
        assert health['status'] == 'healthy'
        assert 'connection_pool' in health
        assert 'database_size' in health
        assert 'table_counts' in health
    
    def test_health_check_unhealthy(self, db_manager):
        """Test database health check when unhealthy."""
        db_manager.pool.get_connection.side_effect = Exception("Connection failed")
        
        health = db_manager.health_check()
        
        assert health['status'] == 'unhealthy'
        assert 'issues' in health
    
    def test_get_database_stats(self, db_manager):
        """Test getting database statistics."""
        mock_pool = Mock()
        mock_pool.get_pool_stats.return_value = {
            'total_connections': 10,
            'available_connections': 8
        }
        db_manager.pool = mock_pool
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = [
            [{'name': 'users'}, {'name': 'sessions'}],  # Tables
            [{'count': 5}],  # Users count
            [{'count': 2}]   # Sessions count
        ]
        mock_conn.cursor.return_value = mock_cursor
        db_manager.pool.get_connection.return_value = mock_conn
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                stats = db_manager.get_database_stats()
        
        assert stats['total_tables'] == 2
        assert stats['total_rows'] == 7
        assert 'table_sizes' in stats
    
    def test_backup_database(self, db_manager):
        """Test creating a database backup."""
        mock_conn = Mock()
        db_manager.pool.get_connection.return_value = mock_conn
        
        with patch('os.path.exists', return_value=True):
            backup_path = db_manager.backup_database()
        
        assert backup_path.endswith('.backup_')
        mock_conn.execute.assert_called_once()
    
    def test_cleanup_expired_data(self, db_manager):
        """Test cleaning up expired data."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value = mock_cursor
        db_manager.pool.get_connection.return_value = mock_conn
        
        with patch('os.getenv', return_value='90'):
            result = db_manager.cleanup_expired_data()
        
        assert result == 5
    
    def test_close(self, db_manager):
        """Test closing database manager."""
        db_manager.close()
        db_manager.pool.close_all.assert_called_once()
        assert db_manager.pool is None


class TestUserRepository:
    """Test UserRepository functionality."""
    
    @pytest.fixture
    def user_repo(self):
        """Create a user repository for testing."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            repo = UserRepository()
            return repo
    
    def test_save_user(self, user_repo):
        """Test saving a user."""
        mock_db = user_repo.db
        mock_db.transaction.return_value.__enter__.return_value = Mock()
        
        user = User.create(
            email="test@example.com",
            full_name="Test User",
            role=mock_auth.UserRole.PATIENT,
            password_hash="hashed_password"
        )
        
        result = user_repo.save(user)
        assert result is True
    
    def test_find_user_by_id(self, user_repo):
        """Test finding a user by ID."""
        mock_db = user_repo.db
        mock_db.execute_query.return_value = [{
            'user_id': 'user_123',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'patient',
            'status': 'active',
            'password_hash': 'hashed_password',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'preferences': '{}',
            'medical_info': '{}'
        }]
        
        user = user_repo.find_by_id('user_123')
        assert user is not None
        assert user.email == 'test@example.com'
    
    def test_find_user_by_email(self, user_repo):
        """Test finding a user by email."""
        mock_db = user_repo.db
        mock_db.execute_query.return_value = [{
            'user_id': 'user_123',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'patient',
            'status': 'active',
            'password_hash': 'hashed_password',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'preferences': '{}',
            'medical_info': '{}'
        }]
        
        user = user_repo.find_by_email('test@example.com')
        assert user is not None
        assert user.email == 'test@example.com'
    
    def test_find_all_users(self, user_repo):
        """Test finding all users."""
        mock_db = user_repo.db
        mock_db.execute_query.return_value = [
            {
                'user_id': 'user_123',
                'email': 'test1@example.com',
                'full_name': 'Test User 1',
                'role': 'patient',
                'status': 'active',
                'password_hash': 'hashed_password',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'preferences': '{}',
                'medical_info': '{}'
            },
            {
                'user_id': 'user_456',
                'email': 'test2@example.com',
                'full_name': 'Test User 2',
                'role': 'therapist',
                'status': 'active',
                'password_hash': 'hashed_password',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'preferences': '{}',
                'medical_info': '{}'
            }
        ]
        
        users = user_repo.find_all()
        assert len(users) == 2
        assert users[0].email == 'test1@example.com'
        assert users[1].email == 'test2@example.com'
    
    def test_update_user(self, user_repo):
        """Test updating a user."""
        user = User.create(
            email="test@example.com",
            full_name="Test User",
            role=mock_auth.UserRole.PATIENT,
            password_hash="hashed_password"
        )
        
        with patch.object(user_repo, 'save', return_value=True):
            result = user_repo.update(user)
            assert result is True
    
    def test_delete_user(self, user_repo):
        """Test deleting a user."""
        mock_db = user_repo.db
        mock_db.execute_query.return_value = None
        
        result = user_repo.delete('user_123')
        assert result is True


class TestSessionRepository:
    """Test SessionRepository functionality."""
    
    @pytest.fixture
    def session_repo(self):
        """Create a session repository for testing."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            repo = SessionRepository()
            return repo
    
    def test_save_session(self, session_repo):
        """Test saving a session."""
        mock_db = session_repo.db
        mock_db.transaction.return_value.__enter__.return_value = Mock()
        
        session = Session.create(
            user_id='user_123',
            session_timeout_minutes=30,
            ip_address='127.0.0.1'
        )
        
        result = session_repo.save(session)
        assert result is True
    
    def test_find_session_by_id(self, session_repo):
        """Test finding a session by ID."""
        mock_db = session_repo.db
        mock_db.execute_query.return_value = [{
            'session_id': 'session_123',
            'user_id': 'user_123',
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=1)).isoformat(),
            'ip_address': '127.0.0.1',
            'user_agent': 'test-agent',
            'is_active': True
        }]
        
        session = session_repo.find_by_id('session_123')
        assert session is not None
        assert session.user_id == 'user_123'
    
    def test_find_sessions_by_user_id(self, session_repo):
        """Test finding sessions by user ID."""
        mock_db = session_repo.db
        mock_db.execute_query.return_value = [
            {
                'session_id': 'session_123',
                'user_id': 'user_123',
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=1)).isoformat(),
                'ip_address': '127.0.0.1',
                'user_agent': 'test-agent',
                'is_active': True
            }
        ]
        
        sessions = session_repo.find_by_user_id('user_123')
        assert len(sessions) == 1
        assert sessions[0].user_id == 'user_123'
    
    def test_update_session(self, session_repo):
        """Test updating a session."""
        session = Session.create(
            user_id='user_123',
            session_timeout_minutes=30
        )
        
        with patch.object(session_repo, 'save', return_value=True):
            result = session_repo.update(session)
            assert result is True
    
    def test_delete_expired_sessions(self, session_repo):
        """Test deleting expired sessions."""
        mock_db = session_repo.db
        mock_result = Mock()
        mock_result.rowcount = 5
        mock_db.execute_query.return_value = mock_result
        
        result = session_repo.delete_expired()
        assert result == 5


class TestVoiceDataRepository:
    """Test VoiceDataRepository functionality."""
    
    @pytest.fixture
    def voice_data_repo(self):
        """Create a voice data repository for testing."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            repo = VoiceDataRepository()
            return repo
    
    def test_save_voice_data(self, voice_data_repo):
        """Test saving voice data."""
        mock_db = voice_data_repo.db
        mock_db.transaction.return_value.__enter__.return_value = Mock()
        
        voice_data = VoiceData.create(
            user_id='user_123',
            data_type='recording',
            encrypted_data=b'mock_audio_data'
        )
        
        result = voice_data_repo.save(voice_data)
        assert result is True
    
    def test_find_voice_data_by_id(self, voice_data_repo):
        """Test finding voice data by ID."""
        mock_db = voice_data_repo.db
        mock_db.execute_query.return_value = [{
            'data_id': 'voice_123',
            'user_id': 'user_123',
            'session_id': 'session_123',
            'data_type': 'recording',
            'encrypted_data': b'mock_audio_data',
            'metadata': '{}',
            'created_at': datetime.now().isoformat(),
            'retention_until': None,
            'is_deleted': False
        }]
        
        voice_data = voice_data_repo.find_by_id('voice_123')
        assert voice_data is not None
        assert voice_data.user_id == 'user_123'
    
    def test_find_voice_data_by_user_id(self, voice_data_repo):
        """Test finding voice data by user ID."""
        mock_db = voice_data_repo.db
        mock_db.execute_query.return_value = [
            {
                'data_id': 'voice_123',
                'user_id': 'user_123',
                'session_id': 'session_123',
                'data_type': 'recording',
                'encrypted_data': b'mock_audio_data',
                'metadata': '{}',
                'created_at': datetime.now().isoformat(),
                'retention_until': None,
                'is_deleted': False
            }
        ]
        
        voice_data_list = voice_data_repo.find_by_user_id('user_123')
        assert len(voice_data_list) == 1
        assert voice_data_list[0].user_id == 'user_123'
    
    def test_mark_voice_data_as_deleted(self, voice_data_repo):
        """Test marking voice data as deleted."""
        mock_db = voice_data_repo.db
        mock_db.execute_query.return_value = None
        
        result = voice_data_repo.mark_as_deleted('voice_123')
        assert result is True


class TestAuditLogRepository:
    """Test AuditLogRepository functionality."""
    
    @pytest.fixture
    def audit_log_repo(self):
        """Create an audit log repository for testing."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            repo = AuditLogRepository()
            return repo
    
    def test_save_audit_log(self, audit_log_repo):
        """Test saving an audit log."""
        mock_db = audit_log_repo.db
        mock_db.transaction.return_value.__enter__.return_value = Mock()
        
        audit_log = AuditLog.create(
            event_type='login',
            user_id='user_123',
            details={'ip': '127.0.0.1'}
        )
        
        result = audit_log_repo.save(audit_log)
        assert result is True
    
    def test_find_audit_logs_by_user_id(self, audit_log_repo):
        """Test finding audit logs by user ID."""
        mock_db = audit_log_repo.db
        mock_db.execute_query.return_value = [
            {
                'log_id': 'log_123',
                'timestamp': datetime.now().isoformat(),
                'event_type': 'login',
                'user_id': 'user_123',
                'session_id': 'session_123',
                'details': '{"ip": "127.0.0.1"}',
                'severity': 'INFO'
            }
        ]
        
        logs = audit_log_repo.find_by_user_id('user_123')
        assert len(logs) == 1
        assert logs[0].event_type == 'login'
    
    def test_find_audit_logs_by_date_range(self, audit_log_repo):
        """Test finding audit logs by date range."""
        mock_db = audit_log_repo.db
        mock_db.execute_query.return_value = [
            {
                'log_id': 'log_123',
                'timestamp': datetime.now().isoformat(),
                'event_type': 'login',
                'user_id': 'user_123',
                'session_id': 'session_123',
                'details': '{"ip": "127.0.0.1"}',
                'severity': 'INFO'
            }
        ]
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        logs = audit_log_repo.find_by_date_range(start_date, end_date)
        assert len(logs) == 1
        assert logs[0].event_type == 'login'


class TestConsentRepository:
    """Test ConsentRepository functionality."""
    
    @pytest.fixture
    def consent_repo(self):
        """Create a consent repository for testing."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            repo = ConsentRepository()
            return repo
    
    def test_save_consent_record(self, consent_repo):
        """Test saving a consent record."""
        mock_db = consent_repo.db
        mock_db.transaction.return_value.__enter__.return_value = Mock()
        
        consent = ConsentRecord.create(
            user_id='user_123',
            consent_type='voice_data',
            granted=True
        )
        
        result = consent_repo.save(consent)
        assert result is True
    
    def test_find_consent_records_by_user_id(self, consent_repo):
        """Test finding consent records by user ID."""
        mock_db = consent_repo.db
        mock_db.execute_query.return_value = [
            {
                'consent_id': 'consent_123',
                'user_id': 'user_123',
                'consent_type': 'voice_data',
                'granted': True,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'details': '{}',
                'revoked_at': None
            }
        ]
        
        consents = consent_repo.find_by_user_id('user_123')
        assert len(consents) == 1
        assert consents[0].consent_type == 'voice_data'
    
    def test_has_active_consent(self, consent_repo):
        """Test checking if user has active consent."""
        mock_db = consent_repo.db
        mock_db.execute_query.return_value = [{'granted': True}]
        
        result = consent_repo.has_active_consent('user_123', 'voice_data')
        assert result is True