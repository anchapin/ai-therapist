"""
Unit tests for session management functionality without database dependencies.

Tests session creation, validation, expiration, and cleanup logic in isolation.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import threading
import time

from auth.auth_service import AuthService, AuthSession
from auth.user_model import UserRole, UserStatus


class MockSessionRepository:
    """Mock session repository for testing without database."""
    
    def __init__(self):
        self._sessions = {}
        self._lock = threading.RLock()
    
    def save(self, session) -> bool:
        """Save session to repository."""
        with self._lock:
            self._sessions[session.session_id] = session
            return True
    
    def find_by_id(self, session_id):
        """Find session by ID."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def find_by_user_id(self, user_id, active_only=True):
        """Find sessions by user ID."""
        with self._lock:
            sessions = []
            for session in self._sessions.values():
                if session.user_id == user_id and (not active_only or session.is_active):
                    sessions.append(session)
            return sessions
    
    def delete(self, session_id) -> bool:
        """Delete session by ID."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        with self._lock:
            now = datetime.now()
            expired_sessions = []
            
            for session_id, session in self._sessions.items():
                if session.expires_at < now:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
            
            return len(expired_sessions)
    
    def get_all_sessions(self) -> list:
        """Get all sessions for testing."""
        with self._lock:
            return list(self._sessions.values())
    
    def clear_all(self):
        """Clear all sessions for testing."""
        with self._lock:
            self._sessions.clear()


class MockAuthSession:
    """Mock authentication session for testing."""
    
    def __init__(self, session_id: str, user_id: str, expires_at: datetime = None,
                 ip_address: str = None, user_agent: str = None, is_active: bool = True):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.expires_at = expires_at or (datetime.now() + timedelta(minutes=30))
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.is_active = is_active
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at
    
    def deactivate(self):
        """Deactivate session."""
        self.is_active = False
    
    def extend(self, minutes: int = 30):
        """Extend session expiration."""
        self.expires_at = datetime.now() + timedelta(minutes=minutes)
    
    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'is_active': self.is_active
        }


class TestSessionManagement:
    """Test session management functionality."""
    
    @pytest.fixture
    def session_repo(self):
        """Provide mock session repository."""
        return MockSessionRepository()
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample session for testing."""
        return MockAuthSession(
            session_id="session_123",
            user_id="user_123",
            expires_at=datetime.now() + timedelta(minutes=30),
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
    
    @pytest.fixture
    def expired_session(self):
        """Create an expired session for testing."""
        return MockAuthSession(
            session_id="expired_session",
            user_id="user_123",
            expires_at=datetime.now() - timedelta(minutes=10),
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
    
    def test_session_creation(self, session_repo):
        """Test session creation and storage."""
        session = MockAuthSession(
            session_id="test_session",
            user_id="user_123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Save session
        result = session_repo.save(session)
        
        assert result is True
        
        # Retrieve session
        retrieved = session_repo.find_by_id("test_session")
        
        assert retrieved is not None
        assert retrieved.session_id == "test_session"
        assert retrieved.user_id == "user_123"
        assert retrieved.ip_address == "192.168.1.1"
        assert retrieved.user_agent == "Mozilla/5.0"
        assert retrieved.is_active is True
    
    def test_session_not_found(self, session_repo):
        """Test retrieving non-existent session."""
        result = session_repo.find_by_id("nonexistent_session")
        assert result is None
    
    def test_find_sessions_by_user_id(self, session_repo):
        """Test finding sessions by user ID."""
        # Create multiple sessions for different users
        session1 = MockAuthSession("session_1", "user_123")
        session2 = MockAuthSession("session_2", "user_456")
        session3 = MockAuthSession("session_3", "user_123")
        session4 = MockAuthSession("session_4", "user_123")
        session4.deactivate()  # Deactivate one session
        
        # Save all sessions
        for session in [session1, session2, session3, session4]:
            session_repo.save(session)
        
        # Find all sessions for user_123 (active only)
        active_sessions = session_repo.find_by_user_id("user_123", active_only=True)
        
        assert len(active_sessions) == 2
        session_ids = [s.session_id for s in active_sessions]
        assert "session_1" in session_ids
        assert "session_3" in session_ids
        assert "session_4" not in session_ids  # Inactive session excluded
        
        # Find all sessions including inactive
        all_sessions = session_repo.find_by_user_id("user_123", active_only=False)
        
        assert len(all_sessions) == 3
        session_ids = [s.session_id for s in all_sessions]
        assert "session_4" in session_ids  # Inactive session included
    
    def test_session_deletion(self, session_repo, sample_session):
        """Test session deletion."""
        # Save session
        session_repo.save(sample_session)
        
        # Verify it exists
        assert session_repo.find_by_id("session_123") is not None
        
        # Delete session
        result = session_repo.delete("session_123")
        
        assert result is True
        
        # Verify it's gone
        assert session_repo.find_by_id("session_123") is None
    
    def test_session_deletion_nonexistent(self, session_repo):
        """Test deleting non-existent session."""
        result = session_repo.delete("nonexistent_session")
        assert result is False
    
    def test_session_expiration_check(self, sample_session, expired_session):
        """Test session expiration checking."""
        # Active session should not be expired
        assert sample_session.is_expired() is False
        
        # Expired session should be expired
        assert expired_session.is_expired() is True
    
    def test_session_extension(self, sample_session):
        """Test session expiration extension."""
        original_expires_at = sample_session.expires_at
        
        # Extend session by 30 minutes
        sample_session.extend(30)
        
        assert sample_session.expires_at > original_expires_at
        assert sample_session.is_expired() is False
    
    def test_session_deactivation(self, sample_session):
        """Test session deactivation."""
        assert sample_session.is_active is True
        
        sample_session.deactivate()
        
        assert sample_session.is_active is False
    
    def test_cleanup_expired_sessions(self, session_repo, sample_session, expired_session):
        """Test cleanup of expired sessions."""
        # Save both sessions
        session_repo.save(sample_session)
        session_repo.save(expired_session)
        
        # Add another expired session
        another_expired = MockAuthSession(
            "another_expired",
            "user_456",
            expires_at=datetime.now() - timedelta(hours=1)
        )
        session_repo.save(another_expired)
        
        # Verify all sessions exist
        assert len(session_repo.get_all_sessions()) == 3
        
        # Cleanup expired sessions
        cleaned_count = session_repo.cleanup_expired()
        
        assert cleaned_count == 2
        
        # Verify only active session remains
        remaining_sessions = session_repo.get_all_sessions()
        assert len(remaining_sessions) == 1
        assert remaining_sessions[0].session_id == "session_123"
    
    def test_concurrent_session_access(self, session_repo):
        """Test thread-safe session access."""
        sessions_created = []
        errors = []
        
        def create_sessions(thread_id):
            try:
                for i in range(10):
                    session = MockAuthSession(
                        f"session_{thread_id}_{i}",
                        f"user_{thread_id}",
                        ip_address=f"192.168.1.{thread_id}"
                    )
                    session_repo.save(session)
                    sessions_created.append(session.session_id)
                    
                    # Small delay to simulate real usage
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Create sessions in multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=create_sessions, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
        
        # Verify all sessions were created
        assert len(sessions_created) == 50
        
        # Verify sessions can be retrieved
        for session_id in sessions_created:
            session = session_repo.find_by_id(session_id)
            assert session is not None
            assert session.session_id == session_id


class TestAuthServiceSessionManagement:
    """Test AuthService session management with mock repository."""
    
    @pytest.fixture
    def auth_service_with_mock_repo(self):
        """Create AuthService with mock session repository."""
        with patch('auth.auth_service.SessionRepository') as mock_repo_class:
            mock_repo = MockSessionRepository()
            mock_repo_class.return_value = mock_repo
            
            with patch('auth.auth_service.UserModel') as mock_user_model:
                mock_user_instance = MagicMock()
                mock_user_model.return_value = mock_user_instance
                
                auth_service = AuthService(mock_user_instance)
                auth_service.session_repo = mock_repo
                
                yield auth_service, mock_repo
    
    def test_create_session_success(self, auth_service_with_mock_repo):
        """Test successful session creation."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Mock user
        mock_user = MagicMock()
        mock_user.user_id = "user_123"
        
        # Create session
        session = auth_service._create_session(
            user_id="user_123",
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        assert session is not None
        assert session.user_id == "user_123"
        assert session.ip_address == "127.0.0.1"
        assert session.user_agent == "test-agent"
        assert session.is_active is True
        assert session.expires_at > datetime.now()
        
        # Verify session was saved to repository
        saved_session = mock_repo.find_by_id(session.session_id)
        assert saved_session is not None
        assert saved_session.user_id == "user_123"
    
    def test_session_validation_success(self, auth_service_with_mock_repo):
        """Test successful session validation."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create and save session
        session = MockAuthSession("session_123", "user_123")
        mock_repo.save(session)
        
        # Validate session
        result = auth_service._is_session_valid("session_123", "user_123")
        
        assert result is True
    
    def test_session_validation_invalid_session_id(self, auth_service_with_mock_repo):
        """Test session validation with invalid session ID."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        result = auth_service._is_session_valid("invalid_session", "user_123")
        
        assert result is False
    
    def test_session_validation_wrong_user(self, auth_service_with_mock_repo):
        """Test session validation with wrong user ID."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create and save session
        session = MockAuthSession("session_123", "user_123")
        mock_repo.save(session)
        
        # Try to validate with different user ID
        result = auth_service._is_session_valid("session_123", "user_456")
        
        assert result is False
    
    def test_session_validation_expired(self, auth_service_with_mock_repo):
        """Test session validation with expired session."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create expired session
        expired_session = MockAuthSession(
            "expired_session",
            "user_123",
            expires_at=datetime.now() - timedelta(minutes=10)
        )
        mock_repo.save(expired_session)
        
        # Validate expired session
        result = auth_service._is_session_valid("expired_session", "user_123")
        
        assert result is False
    
    def test_session_validation_inactive(self, auth_service_with_mock_repo):
        """Test session validation with inactive session."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create inactive session
        inactive_session = MockAuthSession("session_123", "user_123")
        inactive_session.deactivate()
        mock_repo.save(inactive_session)
        
        # Validate inactive session
        result = auth_service._is_session_valid("session_123", "user_123")
        
        assert result is False
    
    def test_invalidate_session(self, auth_service_with_mock_repo):
        """Test session invalidation."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create and save session
        session = MockAuthSession("session_123", "user_123")
        mock_repo.save(session)
        
        # Verify session is active
        assert session.is_active is True
        
        # Invalidate session
        auth_service._invalidate_session("session_123", "user_123")
        
        # Verify session is now inactive
        updated_session = mock_repo.find_by_id("session_123")
        assert updated_session is not None
        assert updated_session.is_active is False
    
    def test_concurrent_session_limit_enforcement(self, auth_service_with_mock_repo):
        """Test concurrent session limit enforcement."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Set max concurrent sessions to 3 for testing
        auth_service.max_concurrent_sessions = 3
        
        # Create existing sessions for the user
        existing_sessions = []
        for i in range(3):
            session = MockAuthSession(f"session_{i}", "user_123")
            mock_repo.save(session)
            existing_sessions.append(session)
        
        # Mock repo to return existing sessions
        mock_repo.find_by_user_id = Mock(return_value=existing_sessions)
        
        # Create new session (should invalidate oldest)
        with patch.object(auth_service, '_invalidate_session') as mock_invalidate:
            new_session = auth_service._create_session("user_123")
            
            # Should have invalidated the oldest session
            mock_invalidate.assert_called_once()
            oldest_session_id = mock_invalidate.call_args[0][0]
            assert oldest_session_id in ["session_0", "session_1", "session_2"]
    
    def test_get_user_sessions(self, auth_service_with_mock_repo):
        """Test getting user sessions."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create sessions for user
        sessions = [
            MockAuthSession("session_1", "user_123"),
            MockAuthSession("session_2", "user_123"),
            MockAuthSession("session_3", "user_456")
        ]
        
        for session in sessions:
            mock_repo.save(session)
        
        # Get sessions for user_123
        user_sessions = auth_service.get_user_sessions("user_123")
        
        assert len(user_sessions) == 2
        session_ids = [s.session_id for s in user_sessions]
        assert "session_1" in session_ids
        assert "session_2" in session_ids
        assert "session_3" not in session_ids
    
    def test_invalidate_user_sessions(self, auth_service_with_mock_repo):
        """Test invalidating all user sessions except current."""
        auth_service, mock_repo = auth_service_with_mock_repo
        
        # Create sessions for user
        sessions = [
            MockAuthSession("session_1", "user_123"),
            MockAuthSession("session_2", "user_123"),
            MockAuthSession("session_3", "user_123"),
            MockAuthSession("session_4", "user_456")
        ]
        
        for session in sessions:
            mock_repo.save(session)
        
        # Mock repo methods
        def mock_find_by_user_id(user_id, active_only=True):
            if user_id == "user_123":
                return [s for s in sessions if s.user_id == user_id and (not active_only or s.is_active)]
            return []
        
        mock_repo.find_by_user_id.side_effect = mock_find_by_user_id
        mock_repo.save.return_value = True
        
        # Invalidate sessions keeping session_2 as current
        invalidated_count = auth_service.invalidate_user_sessions("user_123", keep_current="session_2")
        
        assert invalidated_count == 2  # Should invalidate session_1 and session_3
    
    def test_session_serialization(self):
        """Test session serialization to dictionary."""
        # Create a sample session for this test
        sample_session = MockAuthSession(
            session_id="session_123",
            user_id="user_123",
            expires_at=datetime.now() + timedelta(minutes=30),
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        session_dict = sample_session.to_dict()
        
        assert session_dict['session_id'] == "session_123"
        assert session_dict['user_id'] == "user_123"
        assert session_dict['ip_address'] == "127.0.0.1"
        assert session_dict['user_agent'] == "test-agent"
        assert session_dict['is_active'] is True
        assert 'created_at' in session_dict
        assert 'expires_at' in session_dict