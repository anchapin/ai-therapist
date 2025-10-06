"""
Comprehensive unit tests for database/models.py
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, ANY

from database.models import (
    BaseModel, User, Session, VoiceData, AuditLog, ConsentRecord,
    UserRepository, SessionRepository, VoiceDataRepository,
    AuditLogRepository, ConsentRepository
)

# Mock the auth module to avoid import issues
with patch.dict('sys.modules', {'auth.user_model': Mock()}):
    from auth.user_model import UserRole, UserStatus


class TestBaseModel:
    """Test BaseModel functionality."""
    
    def test_to_dict(self):
        """Test converting model to dictionary."""
        # Create a concrete implementation of BaseModel
        class TestModel(BaseModel):
            def __init__(self, name, created_at):
                self.name = name
                self.created_at = created_at
        
        now = datetime.now()
        model = TestModel("test", now)
        
        result = model.to_dict()
        
        assert isinstance(result, dict)
        assert result['name'] == "test"
        assert result['created_at'] == now.isoformat()
    
    def test_from_dict(self):
        """Test creating model from dictionary."""
        # Create a concrete implementation of BaseModel
        class TestModel(BaseModel):
            def __init__(self, name, created_at):
                self.name = name
                self.created_at = created_at
        
        data = {
            'name': 'test',
            'created_at': '2023-01-01T12:00:00',
            'updated_at': '2023-01-01T12:30:00'
        }
        
        model = TestModel.from_dict(data)
        
        assert model.name == 'test'
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)
    
    def test_from_dict_invalid_datetime(self):
        """Test creating model from dictionary with invalid datetime."""
        class TestModel(BaseModel):
            def __init__(self, created_at):
                self.created_at = created_at
        
        data = {'created_at': 'invalid-date'}
        
        model = TestModel.from_dict(data)
        
        # Should keep as string if parsing fails
        assert model.created_at == 'invalid-date'


class TestUser:
    """Test User model functionality."""
    
    def test_user_creation(self):
        """Test user creation with all fields."""
        now = datetime.now()
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            password_hash="hashed_password",
            created_at=now,
            updated_at=now,
            last_login=now + timedelta(hours=1),
            login_attempts=0,
            preferences={"theme": "dark"},
            medical_info={"conditions": ["anxiety"]}
        )
        
        assert user.user_id == "user123"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == UserRole.PATIENT
        assert user.status == UserStatus.ACTIVE
        assert user.password_hash == "hashed_password"
        assert user.created_at == now
        assert user.updated_at == now
        assert user.last_login == now + timedelta(hours=1)
        assert user.login_attempts == 0
        assert user.preferences == {"theme": "dark"}
        assert user.medical_info == {"conditions": ["anxiety"]}
    
    def test_user_create_classmethod(self):
        """Test User.create class method."""
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            user = User.create(
                email="test@example.com",
                full_name="Test User",
                role=UserRole.THERAPIST,
                password_hash="hashed_password"
            )
            
            assert user.email == "test@example.com"
            assert user.full_name == "Test User"
            assert user.role == UserRole.THERAPIST
            assert user.status == UserStatus.ACTIVE
            assert user.password_hash == "hashed_password"
            assert user.created_at == mock_now
            assert user.updated_at == mock_now
            assert user.user_id.startswith("user_20230101_120000_")
    
    def test_user_is_locked_true(self):
        """Test user is locked when lock time is in future."""
        future_time = datetime.now() + timedelta(hours=1)
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.LOCKED,
            password_hash="hash",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=future_time
        )
        
        assert user.is_locked() is True
    
    def test_user_is_locked_false(self):
        """Test user is not locked when lock time is in past."""
        past_time = datetime.now() - timedelta(hours=1)
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            password_hash="hash",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=past_time
        )
        
        assert user.is_locked() is False
    
    def test_user_is_locked_none(self):
        """Test user is not locked when lock time is None."""
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            password_hash="hash",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            account_locked_until=None
        )
        
        assert user.is_locked() is False
    
    def test_user_increment_login_attempts_no_lock(self):
        """Test incrementing login attempts without reaching lock threshold."""
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            password_hash="hash",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=2
        )
        
        user.increment_login_attempts(max_attempts=5)
        
        assert user.login_attempts == 3
        assert user.status == UserStatus.ACTIVE
        assert user.account_locked_until is None
    
    def test_user_increment_login_attempts_with_lock(self):
        """Test incrementing login attempts reaching lock threshold."""
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.ACTIVE,
            password_hash="hash",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=4
        )
        
        user.increment_login_attempts(max_attempts=5, lock_duration_minutes=30)
        
        assert user.login_attempts == 5
        assert user.status == UserStatus.LOCKED
        assert user.account_locked_until is not None
        assert user.account_locked_until > datetime.now()
    
    def test_user_reset_login_attempts(self):
        """Test resetting login attempts."""
        future_time = datetime.now() + timedelta(hours=1)
        user = User(
            user_id="user123",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            status=UserStatus.LOCKED,
            password_hash="hash",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            login_attempts=5,
            account_locked_until=future_time
        )
        
        user.reset_login_attempts()
        
        assert user.login_attempts == 0
        assert user.account_locked_until is None
        assert user.status == UserStatus.ACTIVE


class TestSession:
    """Test Session model functionality."""
    
    def test_session_creation(self):
        """Test session creation with all fields."""
        now = datetime.now()
        expires = now + timedelta(hours=1)
        
        session = Session(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            is_active=True
        )
        
        assert session.session_id == "session123"
        assert session.user_id == "user123"
        assert session.created_at == now
        assert session.expires_at == expires
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.is_active is True
    
    def test_session_create_classmethod(self):
        """Test Session.create class method."""
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            session = Session.create(
                user_id="user123",
                session_timeout_minutes=60,
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0"
            )
            
            assert session.user_id == "user123"
            assert session.created_at == mock_now
            assert session.expires_at == mock_now + timedelta(minutes=60)
            assert session.ip_address == "192.168.1.1"
            assert session.user_agent == "Mozilla/5.0"
            assert session.is_active is True
            assert session.session_id.startswith("session_20230101_120000_")
    
    def test_session_is_expired_true(self):
        """Test session is expired when expiration time is in past."""
        past_time = datetime.now() - timedelta(hours=1)
        session = Session(
            session_id="session123",
            user_id="user123",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=past_time
        )
        
        assert session.is_expired() is True
    
    def test_session_is_expired_false(self):
        """Test session is not expired when expiration time is in future."""
        future_time = datetime.now() + timedelta(hours=1)
        session = Session(
            session_id="session123",
            user_id="user123",
            created_at=datetime.now(),
            expires_at=future_time
        )
        
        assert session.is_expired() is False
    
    def test_session_extend(self):
        """Test extending session expiration."""
        now = datetime.now()
        original_expires = now + timedelta(hours=1)
        
        session = Session(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=original_expires
        )
        
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 13, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            session.extend(minutes=30)
            
            assert session.expires_at == mock_now + timedelta(minutes=30)


class TestVoiceData:
    """Test VoiceData model functionality."""
    
    def test_voice_data_creation(self):
        """Test voice data creation with all fields."""
        now = datetime.now()
        retention = now + timedelta(days=30)
        
        voice_data = VoiceData(
            data_id="voice123",
            user_id="user123",
            session_id="session123",
            data_type="recording",
            encrypted_data=b"encrypted_audio_data",
            metadata={"duration": 120, "format": "wav"},
            created_at=now,
            retention_until=retention,
            is_deleted=False
        )
        
        assert voice_data.data_id == "voice123"
        assert voice_data.user_id == "user123"
        assert voice_data.session_id == "session123"
        assert voice_data.data_type == "recording"
        assert voice_data.encrypted_data == b"encrypted_audio_data"
        assert voice_data.metadata == {"duration": 120, "format": "wav"}
        assert voice_data.created_at == now
        assert voice_data.retention_until == retention
        assert voice_data.is_deleted is False
    
    def test_voice_data_create_classmethod(self):
        """Test VoiceData.create class method."""
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            voice_data = VoiceData.create(
                user_id="user123",
                data_type="transcription",
                encrypted_data=b"encrypted_text",
                session_id="session123",
                metadata={"confidence": 0.95},
                retention_days=30
            )
            
            assert voice_data.user_id == "user123"
            assert voice_data.data_type == "transcription"
            assert voice_data.encrypted_data == b"encrypted_text"
            assert voice_data.session_id == "session123"
            assert voice_data.metadata == {"confidence": 0.95}
            assert voice_data.created_at == mock_now
            assert voice_data.retention_until == mock_now + timedelta(days=30)
            assert voice_data.is_deleted is False
            assert voice_data.data_id.startswith("voice_20230101_120000_")
    
    def test_voice_data_create_no_retention(self):
        """Test VoiceData.create with no retention."""
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            voice_data = VoiceData.create(
                user_id="user123",
                data_type="recording",
                encrypted_data=b"encrypted_audio",
                retention_days=0
            )
            
            assert voice_data.retention_until is None
    
    def test_voice_data_is_expired_true(self):
        """Test voice data is expired when retention time is in past."""
        past_time = datetime.now() - timedelta(days=1)
        voice_data = VoiceData(
            data_id="voice123",
            user_id="user123",
            session_id="session123",
            data_type="recording",
            encrypted_data=b"data",
            created_at=datetime.now() - timedelta(days=2),
            retention_until=past_time
        )
        
        assert voice_data.is_expired() is True
    
    def test_voice_data_is_expired_false(self):
        """Test voice data is not expired when retention time is in future."""
        future_time = datetime.now() + timedelta(days=1)
        voice_data = VoiceData(
            data_id="voice123",
            user_id="user123",
            session_id="session123",
            data_type="recording",
            encrypted_data=b"data",
            created_at=datetime.now(),
            retention_until=future_time
        )
        
        assert voice_data.is_expired() is False
    
    def test_voice_data_is_expired_none(self):
        """Test voice data is not expired when retention time is None."""
        voice_data = VoiceData(
            data_id="voice123",
            user_id="user123",
            session_id="session123",
            data_type="recording",
            encrypted_data=b"data",
            created_at=datetime.now(),
            retention_until=None
        )
        
        assert voice_data.is_expired() is False


class TestAuditLog:
    """Test AuditLog model functionality."""
    
    def test_audit_log_creation(self):
        """Test audit log creation with all fields."""
        now = datetime.now()
        
        audit_log = AuditLog(
            log_id="log123",
            timestamp=now,
            event_type="LOGIN_SUCCESS",
            user_id="user123",
            session_id="session123",
            details={"ip_address": "192.168.1.1"},
            severity="INFO"
        )
        
        assert audit_log.log_id == "log123"
        assert audit_log.timestamp == now
        assert audit_log.event_type == "LOGIN_SUCCESS"
        assert audit_log.user_id == "user123"
        assert audit_log.session_id == "session123"
        assert audit_log.details == {"ip_address": "192.168.1.1"}
        assert audit_log.severity == "INFO"
    
    def test_audit_log_create_classmethod(self):
        """Test AuditLog.create class method."""
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            audit_log = AuditLog.create(
                event_type="VOICE_DATA_ACCESS",
                user_id="user123",
                session_id="session123",
                details={"data_id": "voice123"},
                severity="WARNING"
            )
            
            assert audit_log.event_type == "VOICE_DATA_ACCESS"
            assert audit_log.user_id == "user123"
            assert audit_log.session_id == "session123"
            assert audit_log.details == {"data_id": "voice123"}
            assert audit_log.severity == "WARNING"
            assert audit_log.timestamp == mock_now
            assert audit_log.log_id.startswith("log_20230101_120000_")


class TestConsentRecord:
    """Test ConsentRecord model functionality."""
    
    def test_consent_record_creation(self):
        """Test consent record creation with all fields."""
        now = datetime.now()
        revoked = now + timedelta(days=30)
        
        consent = ConsentRecord(
            consent_id="consent123",
            user_id="user123",
            consent_type="VOICE_RECORDING",
            granted=True,
            timestamp=now,
            version="1.0",
            details={"purpose": "Therapy sessions"},
            revoked_at=revoked
        )
        
        assert consent.consent_id == "consent123"
        assert consent.user_id == "user123"
        assert consent.consent_type == "VOICE_RECORDING"
        assert consent.granted is True
        assert consent.timestamp == now
        assert consent.version == "1.0"
        assert consent.details == {"purpose": "Therapy sessions"}
        assert consent.revoked_at == revoked
    
    def test_consent_record_create_classmethod(self):
        """Test ConsentRecord.create class method."""
        with patch('database.models.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            consent = ConsentRecord.create(
                user_id="user123",
                consent_type="DATA_PROCESSING",
                granted=True,
                version="2.0",
                details={"retention": "90 days"}
            )
            
            assert consent.user_id == "user123"
            assert consent.consent_type == "DATA_PROCESSING"
            assert consent.granted is True
            assert consent.version == "2.0"
            assert consent.details == {"retention": "90 days"}
            assert consent.timestamp == mock_now
            assert consent.revoked_at is None
            assert consent.consent_id.startswith("consent_20230101_120000_")
    
    def test_consent_revoke(self):
        """Test revoking consent."""
        now = datetime.now()
        consent = ConsentRecord(
            consent_id="consent123",
            user_id="user123",
            consent_type="VOICE_RECORDING",
            granted=True,
            timestamp=now,
            version="1.0"
        )
        
        with patch('database.models.datetime') as mock_datetime:
            mock_revoked = datetime(2023, 1, 2, 12, 0, 0)
            mock_datetime.now.return_value = mock_revoked
            
            consent.revoke()
            
            assert consent.granted is False
            assert consent.revoked_at == mock_revoked
    
    def test_consent_is_active_true(self):
        """Test consent is active when granted and not revoked."""
        consent = ConsentRecord(
            consent_id="consent123",
            user_id="user123",
            consent_type="VOICE_RECORDING",
            granted=True,
            timestamp=datetime.now(),
            version="1.0",
            revoked_at=None
        )
        
        assert consent.is_active() is True
    
    def test_consent_is_active_false_granted(self):
        """Test consent is not active when not granted."""
        consent = ConsentRecord(
            consent_id="consent123",
            user_id="user123",
            consent_type="VOICE_RECORDING",
            granted=False,
            timestamp=datetime.now(),
            version="1.0",
            revoked_at=None
        )
        
        assert consent.is_active() is False
    
    def test_consent_is_active_false_revoked(self):
        """Test consent is not active when revoked."""
        consent = ConsentRecord(
            consent_id="consent123",
            user_id="user123",
            consent_type="VOICE_RECORDING",
            granted=True,
            timestamp=datetime.now(),
            version="1.0",
            revoked_at=datetime.now()
        )
        
        assert consent.is_active() is False


class TestUserRepository:
    """Test UserRepository functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            yield mock_db
    
    @pytest.fixture
    def user_repository(self, mock_db_manager):
        """Create user repository for testing."""
        return UserRepository()
    
    def test_save_success(self, user_repository, mock_db_manager):
        """Test successful user save."""
        user = User.create(
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            password_hash="hash"
        )
        
        mock_db_manager.transaction.return_value.__enter__.return_value = Mock()
        
        result = user_repository.save(user)
        
        assert result is True
        mock_db_manager.transaction.assert_called_once()
    
    def test_save_error(self, user_repository, mock_db_manager):
        """Test user save with error."""
        user = User.create(
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            password_hash="hash"
        )
        
        mock_db_manager.transaction.side_effect = Exception("Database error")
        
        result = user_repository.save(user)
        
        assert result is False
    
    def test_find_by_id_success(self, user_repository, mock_db_manager):
        """Test successful find by ID."""
        mock_data = {
            'user_id': 'user123',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'PATIENT',
            'status': 'ACTIVE',
            'password_hash': 'hash',
            'created_at': '2023-01-01T12:00:00',
            'updated_at': '2023-01-01T12:00:00',
            'preferences': '{}',
            'medical_info': '{}'
        }
        
        mock_db_manager.execute_query.return_value = [mock_data]
        
        result = user_repository.find_by_id('user123')
        
        assert result is not None
        assert result.user_id == 'user123'
        assert result.email == 'test@example.com'
        assert result.role == UserRole.PATIENT
        assert result.status == UserStatus.ACTIVE
    
    def test_find_by_id_not_found(self, user_repository, mock_db_manager):
        """Test find by ID when user not found."""
        mock_db_manager.execute_query.return_value = []
        
        result = user_repository.find_by_id('nonexistent')
        
        assert result is None
    
    def test_find_by_id_error(self, user_repository, mock_db_manager):
        """Test find by ID with error."""
        mock_db_manager.execute_query.side_effect = Exception("Database error")
        
        result = user_repository.find_by_id('user123')
        
        assert result is None
    
    def test_find_by_email_success(self, user_repository, mock_db_manager):
        """Test successful find by email."""
        mock_data = {
            'user_id': 'user123',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'PATIENT',
            'status': 'ACTIVE',
            'password_hash': 'hash',
            'created_at': '2023-01-01T12:00:00',
            'updated_at': '2023-01-01T12:00:00',
            'preferences': '{}',
            'medical_info': '{}'
        }
        
        mock_db_manager.execute_query.return_value = [mock_data]
        
        result = user_repository.find_by_email('test@example.com')
        
        assert result is not None
        assert result.user_id == 'user123'
        assert result.email == 'test@example.com'
    
    def test_find_all_with_status(self, user_repository, mock_db_manager):
        """Test find all users with status filter."""
        mock_data = [{
            'user_id': 'user123',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'PATIENT',
            'status': 'ACTIVE',
            'password_hash': 'hash',
            'created_at': '2023-01-01T12:00:00',
            'updated_at': '2023-01-01T12:00:00',
            'preferences': '{}',
            'medical_info': '{}'
        }]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        result = user_repository.find_all(status=UserStatus.ACTIVE, limit=10)
        
        assert len(result) == 1
        assert result[0].user_id == 'user123'
        mock_db_manager.execute_query.assert_called_with(
            "SELECT * FROM users WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (UserStatus.ACTIVE.value, 10),
            fetch=True
        )
    
    def test_find_all_no_status(self, user_repository, mock_db_manager):
        """Test find all users without status filter."""
        mock_data = []
        mock_db_manager.execute_query.return_value = mock_data
        
        result = user_repository.find_all(limit=10)
        
        assert len(result) == 0
        mock_db_manager.execute_query.assert_called_with(
            "SELECT * FROM users ORDER BY created_at DESC LIMIT ?",
            (10,),
            fetch=True
        )
    
    def test_update(self, user_repository, mock_db_manager):
        """Test updating user."""
        user = User.create(
            email="test@example.com",
            full_name="Test User",
            role=UserRole.PATIENT,
            password_hash="hash"
        )
        
        with patch.object(user_repository, 'save') as mock_save:
            mock_save.return_value = True
            
            result = user_repository.update(user)
            
            assert result is True
            mock_save.assert_called_once_with(user)
            assert user.updated_at > user.created_at
    
    def test_delete_success(self, user_repository, mock_db_manager):
        """Test successful user deletion."""
        mock_db_manager.execute_query.return_value = None
        
        result = user_repository.delete('user123')
        
        assert result is True
        mock_db_manager.execute_query.assert_called_with(
            "DELETE FROM users WHERE user_id = ?",
            ('user123',)
        )
    
    def test_delete_error(self, user_repository, mock_db_manager):
        """Test user deletion with error."""
        mock_db_manager.execute_query.side_effect = Exception("Database error")
        
        result = user_repository.delete('user123')
        
        assert result is False


class TestSessionRepository:
    """Test SessionRepository functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            yield mock_db
    
    @pytest.fixture
    def session_repository(self, mock_db_manager):
        """Create session repository for testing."""
        return SessionRepository()
    
    def test_save_success(self, session_repository, mock_db_manager):
        """Test successful session save."""
        session = Session.create(
            user_id="user123",
            session_timeout_minutes=60
        )
        
        mock_db_manager.transaction.return_value.__enter__.return_value = Mock()
        
        result = session_repository.save(session)
        
        assert result is True
        mock_db_manager.transaction.assert_called_once()
    
    def test_find_by_id_success(self, session_repository, mock_db_manager):
        """Test successful find by ID."""
        mock_data = {
            'session_id': 'session123',
            'user_id': 'user123',
            'created_at': '2023-01-01T12:00:00',
            'expires_at': '2023-01-01T13:00:00',
            'ip_address': '192.168.1.1',
            'user_agent': 'Mozilla/5.0',
            'is_active': True
        }
        
        mock_db_manager.execute_query.return_value = [mock_data]
        
        result = session_repository.find_by_id('session123')
        
        assert result is not None
        assert result.session_id == 'session123'
        assert result.user_id == 'user123'
    
    def test_find_by_user_id_active_only(self, session_repository, mock_db_manager):
        """Test find sessions by user ID with active filter."""
        mock_data = [{
            'session_id': 'session123',
            'user_id': 'user123',
            'created_at': '2023-01-01T12:00:00',
            'expires_at': '2023-01-01T13:00:00',
            'ip_address': '192.168.1.1',
            'user_agent': 'Mozilla/5.0',
            'is_active': True
        }]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        result = session_repository.find_by_user_id('user123', active_only=True)
        
        assert len(result) == 1
        assert result[0].session_id == 'session123'
        mock_db_manager.execute_query.assert_called_with(
            "SELECT * FROM sessions WHERE user_id = ? AND is_active = 1 ORDER BY created_at DESC",
            ('user123',),
            fetch=True
        )
    
    def test_delete_expired_success(self, session_repository, mock_db_manager):
        """Test successful deletion of expired sessions."""
        mock_result = Mock()
        mock_result.rowcount = 5
        mock_db_manager.execute_query.return_value = mock_result
        
        result = session_repository.delete_expired()
        
        assert result == 5
        mock_db_manager.execute_query.assert_called_with(
            "DELETE FROM sessions WHERE expires_at < ? OR is_active = 0",
            (mock.ANY,),  # datetime string
        )


class TestVoiceDataRepository:
    """Test VoiceDataRepository functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            yield mock_db
    
    @pytest.fixture
    def voice_data_repository(self, mock_db_manager):
        """Create voice data repository for testing."""
        return VoiceDataRepository()
    
    def test_save_success(self, voice_data_repository, mock_db_manager):
        """Test successful voice data save."""
        voice_data = VoiceData.create(
            user_id="user123",
            data_type="recording",
            encrypted_data=b"audio_data"
        )
        
        mock_db_manager.transaction.return_value.__enter__.return_value = Mock()
        
        result = voice_data_repository.save(voice_data)
        
        assert result is True
        mock_db_manager.transaction.assert_called_once()
    
    def test_find_by_id_success(self, voice_data_repository, mock_db_manager):
        """Test successful find by ID."""
        mock_data = {
            'data_id': 'voice123',
            'user_id': 'user123',
            'session_id': 'session123',
            'data_type': 'recording',
            'encrypted_data': b'audio_data',
            'metadata': '{"duration": 120}',
            'created_at': '2023-01-01T12:00:00',
            'retention_until': '2023-01-31T12:00:00',
            'is_deleted': False
        }
        
        mock_db_manager.execute_query.return_value = [mock_data]
        
        result = voice_data_repository.find_by_id('voice123')
        
        assert result is not None
        assert result.data_id == 'voice123'
        assert result.user_id == 'user123'
        assert result.metadata == {"duration": 120}
    
    def test_find_by_user_id_with_type(self, voice_data_repository, mock_db_manager):
        """Test find voice data by user ID with type filter."""
        mock_data = [{
            'data_id': 'voice123',
            'user_id': 'user123',
            'session_id': 'session123',
            'data_type': 'recording',
            'encrypted_data': b'audio_data',
            'metadata': '{"duration": 120}',
            'created_at': '2023-01-01T12:00:00',
            'retention_until': '2023-01-31T12:00:00',
            'is_deleted': False
        }]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        result = voice_data_repository.find_by_user_id('user123', data_type='recording', limit=10)
        
        assert len(result) == 1
        assert result[0].data_id == 'voice123'
        mock_db_manager.execute_query.assert_called_with(
            "SELECT * FROM voice_data WHERE user_id = ? AND data_type = ? AND is_deleted = 0 ORDER BY created_at DESC LIMIT ?",
            ('user123', 'recording', 10),
            fetch=True
        )
    
    def test_mark_as_deleted_success(self, voice_data_repository, mock_db_manager):
        """Test successful marking as deleted."""
        mock_db_manager.execute_query.return_value = None
        
        result = voice_data_repository.mark_as_deleted('voice123')
        
        assert result is True
        mock_db_manager.execute_query.assert_called_with(
            "UPDATE voice_data SET is_deleted = 1 WHERE data_id = ?",
            ('voice123',)
        )


class TestAuditLogRepository:
    """Test AuditLogRepository functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            yield mock_db
    
    @pytest.fixture
    def audit_log_repository(self, mock_db_manager):
        """Create audit log repository for testing."""
        return AuditLogRepository()
    
    def test_save_success(self, audit_log_repository, mock_db_manager):
        """Test successful audit log save."""
        audit_log = AuditLog.create(
            event_type="LOGIN_SUCCESS",
            user_id="user123"
        )
        
        mock_db_manager.transaction.return_value.__enter__.return_value = Mock()
        
        result = audit_log_repository.save(audit_log)
        
        assert result is True
        mock_db_manager.transaction.assert_called_once()
    
    def test_find_by_user_id_success(self, audit_log_repository, mock_db_manager):
        """Test successful find by user ID."""
        mock_data = [{
            'log_id': 'log123',
            'timestamp': '2023-01-01T12:00:00',
            'event_type': 'LOGIN_SUCCESS',
            'user_id': 'user123',
            'session_id': 'session123',
            'details': '{"ip_address": "192.168.1.1"}',
            'severity': 'INFO'
        }]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        result = audit_log_repository.find_by_user_id('user123', limit=10)
        
        assert len(result) == 1
        assert result[0].log_id == 'log123'
        assert result[0].event_type == 'LOGIN_SUCCESS'
    
    def test_find_by_date_range_with_event_type(self, audit_log_repository, mock_db_manager):
        """Test find by date range with event type filter."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        mock_data = [{
            'log_id': 'log123',
            'timestamp': '2023-01-01T12:00:00',
            'event_type': 'LOGIN_SUCCESS',
            'user_id': 'user123',
            'session_id': 'session123',
            'details': '{"ip_address": "192.168.1.1"}',
            'severity': 'INFO'
        }]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        result = audit_log_repository.find_by_date_range(start_date, end_date, event_type='LOGIN_SUCCESS')
        
        assert len(result) == 1
        assert result[0].event_type == 'LOGIN_SUCCESS'
        mock_db_manager.execute_query.assert_called_with(
            "SELECT * FROM audit_logs WHERE timestamp BETWEEN ? AND ? AND event_type = ? ORDER BY timestamp DESC",
            (start_date.isoformat(), end_date.isoformat(), 'LOGIN_SUCCESS'),
            fetch=True
        )


class TestConsentRepository:
    """Test ConsentRepository functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        with patch('database.models.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            yield mock_db
    
    @pytest.fixture
    def consent_repository(self, mock_db_manager):
        """Create consent repository for testing."""
        return ConsentRepository()
    
    def test_save_success(self, consent_repository, mock_db_manager):
        """Test successful consent save."""
        consent = ConsentRecord.create(
            user_id="user123",
            consent_type="VOICE_RECORDING",
            granted=True
        )
        
        mock_db_manager.transaction.return_value.__enter__.return_value = Mock()
        
        result = consent_repository.save(consent)
        
        assert result is True
        mock_db_manager.transaction.assert_called_once()
    
    def test_find_by_user_id_with_type(self, consent_repository, mock_db_manager):
        """Test find consent by user ID with type filter."""
        mock_data = [{
            'consent_id': 'consent123',
            'user_id': 'user123',
            'consent_type': 'VOICE_RECORDING',
            'granted': True,
            'timestamp': '2023-01-01T12:00:00',
            'version': '1.0',
            'details': '{"purpose": "Therapy"}',
            'revoked_at': None
        }]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        result = consent_repository.find_by_user_id('user123', consent_type='VOICE_RECORDING')
        
        assert len(result) == 1
        assert result[0].consent_id == 'consent123'
        assert result[0].consent_type == 'VOICE_RECORDING'
        mock_db_manager.execute_query.assert_called_with(
            "SELECT * FROM consent_records WHERE user_id = ? AND consent_type = ? ORDER BY timestamp DESC",
            ('user123', 'VOICE_RECORDING'),
            fetch=True
        )
    
    def test_has_active_consent_true(self, consent_repository, mock_db_manager):
        """Test checking for active consent when it exists."""
        mock_db_manager.execute_query.return_value = [{'granted': True}]
        
        result = consent_repository.has_active_consent('user123', 'VOICE_RECORDING')
        
        assert result is True
        mock_db_manager.execute_query.assert_called_with(
            "SELECT granted FROM consent_records WHERE user_id = ? AND consent_type = ? AND revoked_at IS NULL ORDER BY timestamp DESC LIMIT 1",
            ('user123', 'VOICE_RECORDING'),
            fetch=True
        )
    
    def test_has_active_consent_false(self, consent_repository, mock_db_manager):
        """Test checking for active consent when it doesn't exist."""
        mock_db_manager.execute_query.return_value = []
        
        result = consent_repository.has_active_consent('user123', 'VOICE_RECORDING')
        
        assert result is False
    
    def test_has_active_consent_denied(self, consent_repository, mock_db_manager):
        """Test checking for active consent when it was denied."""
        mock_db_manager.execute_query.return_value = [{'granted': False}]
        
        result = consent_repository.has_active_consent('user123', 'VOICE_RECORDING')
        
        assert result is False