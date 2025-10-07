"""
Comprehensive unit tests for database/db_manager.py and database/models.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import threading
from datetime import datetime, timedelta
import json

# Mock the auth module to avoid import issues
mock_auth = Mock()
mock_auth.UserProfile = Mock()
mock_auth.UserRole = Mock()

with patch.dict('sys.modules', {'auth': mock_auth}):
    from database.db_manager import DatabaseConnectionPool, DatabaseManager, DatabaseError
    from database.models import (
        UserRepository, SessionRepository, ConversationRepository,
        MessageRepository, VoiceSessionRepository, SecurityAuditRepository,
        PerformanceMetricsRepository, SystemConfigRepository
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
        assert len(db_pool.available) == 5  # All connections start as available
        assert len(db_pool.connections) == 5
    
    def test_get_connection(self, db_pool):
        """Test getting a connection from the pool."""
        initial_available = len(db_pool.available)
        
        conn = db_pool.get_connection()
        
        assert conn is not None
        assert len(db_pool.available) == initial_available - 1
        assert len(db_pool.connections) == 5
    
    def test_return_connection(self, db_pool):
        """Test returning a connection to the pool."""
        initial_available = len(db_pool.available)
        
        conn = db_pool.get_connection()
        db_pool.return_connection(conn)
        
        assert len(db_pool.available) == initial_available
        assert len(db_pool.connections) == 5
    
    def test_get_connection_from_pool(self, db_pool):
        """Test getting a connection from the available pool."""
        # Get connection and return it to pool
        initial_available = len(db_pool.available)
        conn = db_pool.get_connection()
        db_pool.return_connection(conn)
        
        # Get connection from pool
        conn2 = db_pool.get_connection()
        
        assert conn2 is conn
        assert len(db_pool.available) == initial_available - 1
        assert len(db_pool.connections) == 5
    
    def test_max_connections_limit(self, db_pool):
        """Test max connections limit."""
        connections = []
        for _ in range(5):
            connections.append(db_pool.get_connection())
        
        # Should have reached max connections
        assert len(db_pool.available) == 0
        
        # Next connection should raise an exception
        with pytest.raises(Exception):
            db_pool.get_connection()
    
    def test_close_all_connections(self, db_pool):
        """Test closing all connections."""
        # Get and return some connections
        conn1 = db_pool.get_connection()
        conn2 = db_pool.get_connection()
        db_pool.return_connection(conn1)
        
        # Close all connections
        db_pool.close_all()
        
        assert len(db_pool.connections) == 0
        assert len(db_pool.available) == 0


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a database manager for testing."""
        with patch('database.db_manager.DatabaseConnectionPool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            manager = DatabaseManager(":memory:")
            return manager
    
    def test_manager_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager.pool is not None
        assert db_manager.db_path == ":memory:"
    
    def test_execute_query(self, db_manager):
        """Test executing a query."""
        # Mock the connection 
        mock_conn = Mock()
        
        # Mock cursor that behaves like sqlite3.Cursor
        mock_cursor = Mock()
        mock_row1 = {'id': 1}
        mock_row2 = {'id': 2}
        mock_cursor.fetchall.return_value = [mock_row1, mock_row2]
        mock_conn.execute.return_value = mock_cursor
        
        with patch.object(db_manager.pool, 'get_connection', return_value=mock_conn):
            result = db_manager.execute_query("SELECT * FROM users", fetch=True)
            
            assert result is not None
            assert len(result) == 2
            assert result[0]['id'] == 1
            mock_conn.execute.assert_called_with("SELECT * FROM users", ())
    
    def test_execute_query_with_params(self, db_manager):
        """Test executing a query with parameters."""
        # Mock the connection
        mock_conn = Mock()
        
        # Mock cursor 
        mock_cursor = Mock()
        mock_row = {'id': 1}
        mock_cursor.fetchall.return_value = [mock_row]
        mock_conn.execute.return_value = mock_cursor
        
        with patch.object(db_manager.pool, 'get_connection', return_value=mock_conn):
            result = db_manager.execute_query("SELECT * FROM users WHERE id = ?", (1,), fetch=True)
            
            assert result is not None
            assert len(result) == 1
            mock_conn.execute.assert_called_with("SELECT * FROM users WHERE id = ?", (1,))
    
    
    def test_execute_update(self, db_manager):
        """Test executing an update query."""
        # Mock the connection  
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []  # Update queries return no rows
        mock_conn.execute.return_value = mock_cursor
        
        with patch.object(db_manager.pool, 'get_connection', return_value=mock_conn):
            # Test that we can execute update queries through execute_query
            result = db_manager.execute_query("UPDATE users SET name = ? WHERE id = ?", ("new_name", 1))
            
            mock_conn.execute.assert_called_with("UPDATE users SET name = ? WHERE id = ?", ("new_name", 1))
    
    def test_health_check(self, db_manager):
        """Test database health check."""
        # This test just verifies the health_check method runs without error
        health = db_manager.health_check()
        
        assert 'status' in health
        assert 'timestamp' in health
        assert isinstance(health['status'], str)
    
    def test_health_check_unhealthy(self, db_manager):
        """Test database health check when unhealthy."""
        with patch.object(db_manager.pool, 'get_connection', side_effect=Exception("Connection failed")):
            health = db_manager.health_check()
            
            assert health['status'] == 'unhealthy'
    
    def test_transaction_context_manager(self, db_manager):
        """Test transaction context manager."""
        mock_conn = Mock()
        
        with patch.object(db_manager.pool, 'get_connection', return_value=mock_conn):
            with db_manager.transaction() as conn:
                assert conn is mock_conn
            
            mock_conn.commit.assert_called_once()
    
    def test_transaction_context_manager_rollback(self, db_manager):
        """Test transaction context manager rollback on exception."""
        mock_conn = Mock()
        
        with patch.object(db_manager.pool, 'get_connection', return_value=mock_conn):
            try:
                with db_manager.transaction() as conn:
                    raise ValueError("Test error")
            except (ValueError, DatabaseError):
                pass  # Expected exceptions
            
            mock_conn.rollback.assert_called_once()
    
    def test_close(self, db_manager):
        """Test closing database manager."""
        with patch.object(db_manager.pool, 'close_all') as mock_close:
            db_manager.close()
            mock_close.assert_called_once()


class TestUserRepository:
    """Test UserRepository functionality."""
    
    @pytest.fixture
    def user_repo(self):
        """Create a user repository for testing."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db_manager = Mock()
            mock_get_db.return_value = mock_db_manager
            repo = UserRepository()
            return repo
    
    def test_create_user(self, user_repo):
        """Test creating a user."""
        user_repo.db.execute_query.return_value = None
        
        user_data = {
            'email': 'test@example.com',
            'password_hash': 'hashed_password',
            'role': 'patient',
            'created_at': datetime.now()
        }
        
        user_id = user_repo.create(user_data)
        
        assert user_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_user_by_id(self, user_repo):
        """Test getting a user by ID."""
        mock_db_manager = user_repo.db_manager
        mock_db_manager.execute_query.return_value = [(1, 'test@example.com', 'hashed_password', 'patient')]
        
        user = user_repo.get_by_id(1)
        
        assert user is not None
        assert user[0] == 1
        assert user[1] == 'test@example.com'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_get_user_by_email(self, user_repo):
        """Test getting a user by email."""
        mock_db_manager = user_repo.db_manager
        mock_db_manager.execute_query.return_value = [(1, 'test@example.com', 'hashed_password', 'patient')]
        
        user = user_repo.get_by_email('test@example.com')
        
        assert user is not None
        assert user[1] == 'test@example.com'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_update_user(self, user_repo):
        """Test updating a user."""
        mock_db_manager = user_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        update_data = {'email': 'newemail@example.com'}
        result = user_repo.update(1, update_data)
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()
    
    def test_delete_user(self, user_repo):
        """Test deleting a user."""
        mock_db_manager = user_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        result = user_repo.delete(1)
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()
    
    def test_list_users(self, user_repo):
        """Test listing users."""
        mock_db_manager = user_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 'test1@example.com', 'hash1', 'patient'),
            (2, 'test2@example.com', 'hash2', 'therapist')
        ]
        
        users = user_repo.list(limit=10, offset=0)
        
        assert len(users) == 2
        assert users[0][1] == 'test1@example.com'
        assert users[1][1] == 'test2@example.com'
        mock_db_manager.execute_query.assert_called_once()


class TestSessionRepository:
    """Test SessionRepository functionality."""
    
    @pytest.fixture
    def session_repo(self):
        """Create a session repository for testing."""
        mock_db_manager = Mock()
        repo = SessionRepository(mock_db_manager)
        return repo
    
    def test_create_session(self, session_repo):
        """Test creating a session."""
        mock_db_manager = session_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        session_data = {
            'user_id': 1,
            'token': 'test_token',
            'expires_at': datetime.now() + timedelta(hours=1),
            'created_at': datetime.now()
        }
        
        session_id = session_repo.create(session_data)
        
        assert session_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_session_by_token(self, session_repo):
        """Test getting a session by token."""
        mock_db_manager = session_repo.db_manager
        mock_db_manager.execute_query.return_value = [(1, 1, 'test_token', 'active')]
        
        session = session_repo.get_by_token('test_token')
        
        assert session is not None
        assert session[2] == 'test_token'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_update_session_status(self, session_repo):
        """Test updating session status."""
        mock_db_manager = session_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        result = session_repo.update_status(1, 'expired')
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()
    
    def test_cleanup_expired_sessions(self, session_repo):
        """Test cleaning up expired sessions."""
        mock_db_manager = session_repo.db_manager
        mock_db_manager.execute_update.return_value = 5
        
        result = session_repo.cleanup_expired()
        
        assert result == 5
        mock_db_manager.execute_update.assert_called_once()


class TestConversationRepository:
    """Test ConversationRepository functionality."""
    
    @pytest.fixture
    def conversation_repo(self):
        """Create a conversation repository for testing."""
        mock_db_manager = Mock()
        repo = ConversationRepository(mock_db_manager)
        return repo
    
    def test_create_conversation(self, conversation_repo):
        """Test creating a conversation."""
        mock_db_manager = conversation_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        conversation_data = {
            'user_id': 1,
            'title': 'Test Conversation',
            'created_at': datetime.now()
        }
        
        conversation_id = conversation_repo.create(conversation_data)
        
        assert conversation_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_conversations_by_user(self, conversation_repo):
        """Test getting conversations by user."""
        mock_db_manager = conversation_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 1, 'Test Conversation 1'),
            (2, 1, 'Test Conversation 2')
        ]
        
        conversations = conversation_repo.get_by_user(1)
        
        assert len(conversations) == 2
        assert conversations[0][2] == 'Test Conversation 1'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_update_conversation_title(self, conversation_repo):
        """Test updating conversation title."""
        mock_db_manager = conversation_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        result = conversation_repo.update_title(1, 'New Title')
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()


class TestMessageRepository:
    """Test MessageRepository functionality."""
    
    @pytest.fixture
    def message_repo(self):
        """Create a message repository for testing."""
        mock_db_manager = Mock()
        repo = MessageRepository(mock_db_manager)
        return repo
    
    def test_create_message(self, message_repo):
        """Test creating a message."""
        mock_db_manager = message_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        message_data = {
            'conversation_id': 1,
            'role': 'user',
            'content': 'Hello, world!',
            'created_at': datetime.now()
        }
        
        message_id = message_repo.create(message_data)
        
        assert message_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_messages_by_conversation(self, message_repo):
        """Test getting messages by conversation."""
        mock_db_manager = message_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 1, 'user', 'Hello'),
            (2, 1, 'assistant', 'Hi there!')
        ]
        
        messages = message_repo.get_by_conversation(1)
        
        assert len(messages) == 2
        assert messages[0][2] == 'user'
        assert messages[1][2] == 'assistant'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_delete_messages_by_conversation(self, message_repo):
        """Test deleting messages by conversation."""
        mock_db_manager = message_repo.db_manager
        mock_db_manager.execute_update.return_value = 10
        
        result = message_repo.delete_by_conversation(1)
        
        assert result == 10
        mock_db_manager.execute_update.assert_called_once()


class TestVoiceSessionRepository:
    """Test VoiceSessionRepository functionality."""
    
    @pytest.fixture
    def voice_session_repo(self):
        """Create a voice session repository for testing."""
        mock_db_manager = Mock()
        repo = VoiceSessionRepository(mock_db_manager)
        return repo
    
    def test_create_voice_session(self, voice_session_repo):
        """Test creating a voice session."""
        mock_db_manager = voice_session_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        session_data = {
            'user_id': 1,
            'session_id': 'voice_session_123',
            'created_at': datetime.now()
        }
        
        voice_session_id = voice_session_repo.create(session_data)
        
        assert voice_session_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_voice_session_by_session_id(self, voice_session_repo):
        """Test getting a voice session by session ID."""
        mock_db_manager = voice_session_repo.db_manager
        mock_db_manager.execute_query.return_value = [(1, 1, 'voice_session_123', 'active')]
        
        session = voice_session_repo.get_by_session_id('voice_session_123')
        
        assert session is not None
        assert session[2] == 'voice_session_123'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_update_voice_session_status(self, voice_session_repo):
        """Test updating voice session status."""
        mock_db_manager = voice_session_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        result = voice_session_repo.update_status(1, 'completed')
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()


class TestSecurityAuditRepository:
    """Test SecurityAuditRepository functionality."""
    
    @pytest.fixture
    def security_audit_repo(self):
        """Create a security audit repository for testing."""
        mock_db_manager = Mock()
        repo = SecurityAuditRepository(mock_db_manager)
        return repo
    
    def test_create_audit_log(self, security_audit_repo):
        """Test creating an audit log."""
        mock_db_manager = security_audit_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        audit_data = {
            'user_id': 1,
            'action': 'login',
            'resource': 'auth',
            'details': {'ip': '127.0.0.1'},
            'created_at': datetime.now()
        }
        
        audit_id = security_audit_repo.create(audit_data)
        
        assert audit_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_audit_logs_by_user(self, security_audit_repo):
        """Test getting audit logs by user."""
        mock_db_manager = security_audit_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 1, 'login', 'auth', '{"ip": "127.0.0.1"}'),
            (2, 1, 'logout', 'auth', '{"ip": "127.0.0.1"}')
        ]
        
        logs = security_audit_repo.get_by_user(1)
        
        assert len(logs) == 2
        assert logs[0][2] == 'login'
        assert logs[1][2] == 'logout'
        mock_db_manager.execute_query.assert_called_once()
    
    def test_get_audit_logs_by_action(self, security_audit_repo):
        """Test getting audit logs by action."""
        mock_db_manager = security_audit_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 1, 'login', 'auth', '{"ip": "127.0.0.1"}'),
            (2, 2, 'login', 'auth', '{"ip": "127.0.0.2"}')
        ]
        
        logs = security_audit_repo.get_by_action('login')
        
        assert len(logs) == 2
        assert logs[0][2] == 'login'
        assert logs[1][2] == 'login'
        mock_db_manager.execute_query.assert_called_once()


class TestPerformanceMetricsRepository:
    """Test PerformanceMetricsRepository functionality."""
    
    @pytest.fixture
    def performance_repo(self):
        """Create a performance metrics repository for testing."""
        mock_db_manager = Mock()
        repo = PerformanceMetricsRepository(mock_db_manager)
        return repo
    
    def test_create_metric(self, performance_repo):
        """Test creating a performance metric."""
        mock_db_manager = performance_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        metric_data = {
            'metric_name': 'response_time',
            'metric_value': 0.5,
            'tags': {'endpoint': '/api/chat'},
            'created_at': datetime.now()
        }
        
        metric_id = performance_repo.create(metric_data)
        
        assert metric_id == 1
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_metrics_by_name(self, performance_repo):
        """Test getting metrics by name."""
        mock_db_manager = performance_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 'response_time', 0.5, '{"endpoint": "/api/chat"}'),
            (2, 'response_time', 0.7, '{"endpoint": "/api/chat"}')
        ]
        
        metrics = performance_repo.get_by_name('response_time')
        
        assert len(metrics) == 2
        assert metrics[0][1] == 'response_time'
        assert metrics[0][2] == 0.5
        mock_db_manager.execute_query.assert_called_once()
    
    def test_get_metrics_by_time_range(self, performance_repo):
        """Test getting metrics by time range."""
        mock_db_manager = performance_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            (1, 'response_time', 0.5, '{"endpoint": "/api/chat"}'),
            (2, 'memory_usage', 0.8, '{"process": "main"}')
        ]
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        metrics = performance_repo.get_by_time_range(start_time, end_time)
        
        assert len(metrics) == 2
        mock_db_manager.execute_query.assert_called_once()


class TestSystemConfigRepository:
    """Test SystemConfigRepository functionality."""
    
    @pytest.fixture
    def config_repo(self):
        """Create a system config repository for testing."""
        mock_db_manager = Mock()
        repo = SystemConfigRepository(mock_db_manager)
        return repo
    
    def test_set_config(self, config_repo):
        """Test setting a config value."""
        mock_db_manager = config_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        result = config_repo.set('max_sessions', 100)
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()
    
    def test_get_config(self, config_repo):
        """Test getting a config value."""
        mock_db_manager = config_repo.db_manager
        mock_db_manager.execute_query.return_value = [(100,)]
        
        value = config_repo.get('max_sessions')
        
        assert value == 100
        mock_db_manager.execute_query.assert_called_once()
    
    def test_get_all_configs(self, config_repo):
        """Test getting all config values."""
        mock_db_manager = config_repo.db_manager
        mock_db_manager.execute_query.return_value = [
            ('max_sessions', 100),
            ('timeout', 3600),
            ('debug', True)
        ]
        
        configs = config_repo.get_all()
        
        assert len(configs) == 3
        assert configs['max_sessions'] == 100
        assert configs['timeout'] == 3600
        assert configs['debug'] is True
        mock_db_manager.execute_query.assert_called_once()
    
    def test_delete_config(self, config_repo):
        """Test deleting a config value."""
        mock_db_manager = config_repo.db_manager
        mock_db_manager.execute_update.return_value = 1
        
        result = config_repo.delete('max_sessions')
        
        assert result is True
        mock_db_manager.execute_update.assert_called_once()