"""
Comprehensive unit tests for voice/enhanced_security.py module.
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock torch/whisper to avoid import conflicts
sys.modules['torch'] = Mock()
sys.modules['whisper'] = Mock()

try:
    from voice.enhanced_security import (
        SecurityLevel, AccessControlError, SecurityEvent, Session, SecurityConfig,
        SessionManager, MockAuditLogger, EnhancedAccessManager, EnhancedAccessControl,
        VoiceSecurity, get_voice_security_instance, _voice_security_instance
    )
except ImportError as e:
    pytest.skip(f"voice.enhanced_security module not available: {e}", allow_module_level=True)
except Exception as e:
    pytest.skip(f"Error importing voice.enhanced_security: {e}", allow_module_level=True)


class TestSecurityLevel:
    """Test SecurityLevel enum."""
    
    def test_security_level_values(self):
        """Test security level enum values."""
        assert SecurityLevel.GUEST.value == 1
        assert SecurityLevel.PATIENT.value == 2
        assert SecurityLevel.THERAPIST.value == 3
        assert SecurityLevel.ADMIN.value == 4
    
    def test_access_level_alias(self):
        """Test AccessLevel alias exists."""
        from voice.enhanced_security import AccessLevel
        assert AccessLevel == SecurityLevel


class TestSecurityEvent:
    """Test SecurityEvent dataclass."""
    
    def test_security_event_creation(self):
        """Test security event creation."""
        event = SecurityEvent(
            timestamp="2023-01-01T00:00:00",
            event_type="test_event",
            user_id="user123",
            resource="test_resource",
            action="test_action",
            result="success"
        )
        
        assert event.timestamp == "2023-01-01T00:00:00"
        assert event.event_type == "test_event"
        assert event.user_id == "user123"
        assert event.resource == "test_resource"
        assert event.action == "test_action"
        assert event.result == "success"
        assert event.details == {}
        assert event.ip_address == "127.0.0.1"
    
    def test_security_event_with_details(self):
        """Test security event with details."""
        details = {"key": "value", "session_id": "sess123"}
        event = SecurityEvent(
            timestamp="2023-01-01T00:00:00",
            event_type="test_event",
            user_id="user123",
            resource="test_resource",
            action="test_action",
            result="success",
            details=details,
            ip_address="192.168.1.1"
        )
        
        assert event.details == details
        assert event.ip_address == "192.168.1.1"


class TestSession:
    """Test Session dataclass."""
    
    def test_session_creation(self):
        """Test session creation."""
        created_at = datetime.now()
        last_accessed = datetime.now()
        metadata = {"test": "data"}
        
        session = Session(
            session_id="sess123",
            user_id="user123",
            access_level=SecurityLevel.PATIENT,
            created_at=created_at,
            last_accessed=last_accessed,
            metadata=metadata,
            active=True
        )
        
        assert session.session_id == "sess123"
        assert session.user_id == "user123"
        assert session.access_level == SecurityLevel.PATIENT
        assert session.created_at == created_at
        assert session.last_accessed == last_accessed
        assert session.metadata == metadata
        assert session.active is True


class TestSecurityConfig:
    """Test SecurityConfig class."""
    
    def test_security_config_defaults(self):
        """Test security config with defaults."""
        config = SecurityConfig()
        
        assert config.encryption_enabled is True
        assert config.consent_required is True
        assert config.privacy_mode is False
        assert config.hipaa_compliance_enabled is True
        assert config.data_retention_days == 30
        assert config.audit_logging_enabled is True
        assert config.session_timeout_minutes == 30
        assert config.max_login_attempts == 3
    
    def test_security_config_custom(self):
        """Test security config with custom values."""
        config = SecurityConfig(
            encryption_enabled=False,
            consent_required=False,
            privacy_mode=True,
            hipaa_compliance_enabled=False,
            data_retention_days=60,
            audit_logging_enabled=False,
            session_timeout_minutes=60,
            max_login_attempts=5
        )
        
        assert config.encryption_enabled is False
        assert config.consent_required is False
        assert config.privacy_mode is True
        assert config.hipaa_compliance_enabled is False
        assert config.data_retention_days == 60
        assert config.audit_logging_enabled is False
        assert config.session_timeout_minutes == 60
        assert config.max_login_attempts == 5


class TestSessionManager:
    """Test SessionManager class."""
    
    def test_session_manager_initialization(self):
        """Test session manager initialization."""
        config = {"max_sessions_per_user": 3, "session_timeout_minutes": 60}
        manager = SessionManager(config)
        
        assert manager.max_sessions_per_user == 3
        assert manager.session_timeout_minutes == 60
        assert manager.sessions == {}
        assert manager.user_sessions == {}
    
    def test_session_manager_default_config(self):
        """Test session manager with default config."""
        manager = SessionManager()
        
        assert manager.max_sessions_per_user == 5
        assert manager.session_timeout_minutes == 30
    
    def test_create_session_success(self):
        """Test successful session creation."""
        manager = SessionManager()
        session_id = manager.create_session("user123", SecurityLevel.PATIENT)
        
        assert session_id is not None
        assert len(session_id) == 16  # SHA256 hash truncated to 16 chars
        
        # Check session was stored
        session = manager.get_session(session_id)
        assert session is not None
        assert session.user_id == "user123"
        assert session.access_level == SecurityLevel.PATIENT
        assert session.active is True
        
        # Check user sessions tracking
        assert "user123" in manager.user_sessions
        assert session_id in manager.user_sessions["user123"]
    
    def test_create_session_with_metadata(self):
        """Test session creation with metadata."""
        manager = SessionManager()
        metadata = {"ip_address": "192.168.1.1", "user_agent": "test"}
        session_id = manager.create_session("user123", SecurityLevel.ADMIN, metadata)
        
        session = manager.get_session(session_id)
        assert session.metadata == metadata
    
    def test_create_session_max_sessions_exceeded(self):
        """Test session creation when max sessions exceeded."""
        manager = SessionManager({"max_sessions_per_user": 2})
        
        # Create maximum sessions
        sess1 = manager.create_session("user123", SecurityLevel.PATIENT)
        sess2 = manager.create_session("user123", SecurityLevel.PATIENT)
        
        # Try to create one more - should raise exception
        with pytest.raises(AccessControlError, match="Maximum sessions.*exceeded"):
            manager.create_session("user123", SecurityLevel.PATIENT)
    
    def test_validate_session_success(self):
        """Test successful session validation."""
        manager = SessionManager()
        session_id = manager.create_session("user123", SecurityLevel.PATIENT)
        
        # Should be valid immediately
        assert manager.validate_session(session_id) is True
        
        # Check last_accessed was updated
        session = manager.get_session(session_id)
        assert session.last_accessed > session.created_at
    
    def test_validate_session_not_found(self):
        """Test validation of non-existent session."""
        manager = SessionManager()
        assert manager.validate_session("nonexistent") is False
    
    def test_validate_session_inactive(self):
        """Test validation of inactive session."""
        manager = SessionManager()
        session_id = manager.create_session("user123", SecurityLevel.PATIENT)
        
        # Invalidate session
        manager.invalidate_session(session_id)
        
        # Should no longer be valid
        assert manager.validate_session(session_id) is False
    
    def test_validate_session_expired(self):
        """Test validation of expired session."""
        manager = SessionManager({"session_timeout_minutes": 1})
        session_id = manager.create_session("user123", SecurityLevel.PATIENT)
        
        # Get the session and manually set it to expired
        session = manager.get_session(session_id)
        session.last_accessed = datetime.now() - timedelta(minutes=2)
        
        # Should be invalid due to timeout
        assert manager.validate_session(session_id) is False
    
    def test_invalidate_session(self):
        """Test session invalidation."""
        manager = SessionManager()
        session_id = manager.create_session("user123", SecurityLevel.PATIENT)
        
        # Invalidate session
        manager.invalidate_session(session_id)
        
        # Session should be inactive
        session = manager.get_session(session_id)
        assert session is not None
        assert session.active is False
        
        # Should be removed from user sessions
        assert session_id not in manager.user_sessions["user123"]
    
    def test_invalidate_nonexistent_session(self):
        """Test invalidating non-existent session."""
        manager = SessionManager()
        # Should not raise exception
        manager.invalidate_session("nonexistent")
    
    def test_active_sessions_property(self):
        """Test getting active sessions."""
        manager = SessionManager()
        
        # Create sessions
        sess1 = manager.create_session("user1", SecurityLevel.PATIENT)
        sess2 = manager.create_session("user2", SecurityLevel.THERAPIST)
        sess3 = manager.create_session("user3", SecurityLevel.ADMIN)
        
        # Invalidate one session
        manager.invalidate_session(sess2)
        
        # Get active sessions
        active_sessions = manager.active_sessions
        assert len(active_sessions) == 2
        
        session_ids = [s.session_id for s in active_sessions]
        assert sess1 in session_ids
        assert sess3 in session_ids
        assert sess2 not in session_ids
    
    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        manager = SessionManager({"session_timeout_minutes": 1})
        
        # Create sessions
        sess1 = manager.create_session("user1", SecurityLevel.PATIENT)
        sess2 = manager.create_session("user2", SecurityLevel.THERAPIST)
        
        # Manually set sessions to expired
        session1 = manager.get_session(sess1)
        session2 = manager.get_session(sess2)
        session1.last_accessed = datetime.now() - timedelta(minutes=2)
        session2.last_accessed = datetime.now() - timedelta(minutes=2)
        
        # Cleanup expired sessions
        manager.cleanup_expired_sessions()
        
        # Both sessions should be invalidated
        assert manager.validate_session(sess1) is False
        assert manager.validate_session(sess2) is False
    
    def test_thread_safety(self):
        """Test thread safety of session operations."""
        manager = SessionManager()
        session_ids = []
        errors = []
        
        def create_sessions(user_id, count):
            try:
                for i in range(count):
                    session_id = manager.create_session(f"{user_id}_{i}", SecurityLevel.PATIENT)
                    session_ids.append(session_id)
            except Exception as e:
                errors.append(e)
        
        # Create sessions from multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_sessions, args=(f"user_{i}", 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Check all sessions were created
        assert len(session_ids) == 50
        
        # Check all sessions are valid
        for session_id in session_ids:
            assert manager.validate_session(session_id) is True


class TestMockAuditLogger:
    """Test MockAuditLogger class."""
    
    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        logger = MockAuditLogger()
        
        assert logger.events == []
        assert logger.session_logs_cache == {}
    
    def test_log_event(self):
        """Test event logging."""
        logger = MockAuditLogger()
        
        logger.log_event(
            event_type="test_event",
            user_id="user123",
            action="test_action",
            resource="test_resource",
            result="success",
            details={"key": "value"}
        )
        
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == "test_event"
        assert event.user_id == "user123"
        assert event.action == "test_action"
        assert event.resource == "test_resource"
        assert event.result == "success"
        assert event.details == {"key": "value"}
    
    def test_log_event_with_session_cache(self):
        """Test event logging with session cache."""
        logger = MockAuditLogger()
        
        details = {"session_id": "sess123"}
        logger.log_event(
            event_type="test_event",
            user_id="user123",
            action="test_action",
            resource="test_resource",
            result="success",
            details=details
        )
        
        # Check event was added to session cache
        assert "sess123" in logger.session_logs_cache
        assert len(logger.session_logs_cache["sess123"]) == 1
        assert logger.session_logs_cache["sess123"][0] == logger.events[0]
    
    def test_get_events_by_type(self):
        """Test getting events by type."""
        logger = MockAuditLogger()
        
        # Log different types of events
        logger.log_event("type1", "user1", "action1", "resource1", "success")
        logger.log_event("type2", "user2", "action2", "resource2", "success")
        logger.log_event("type1", "user3", "action3", "resource3", "success")
        
        # Get events by type
        type1_events = logger.get_events_by_type("type1")
        assert len(type1_events) == 2
        assert all(e.event_type == "type1" for e in type1_events)
        
        type2_events = logger.get_events_by_type("type2")
        assert len(type2_events) == 1
        assert type2_events[0].event_type == "type2"
        
        type3_events = logger.get_events_by_type("type3")
        assert len(type3_events) == 0
    
    def test_get_events_by_user(self):
        """Test getting events by user."""
        logger = MockAuditLogger()
        
        # Log events for different users
        logger.log_event("type1", "user1", "action1", "resource1", "success")
        logger.log_event("type2", "user2", "action2", "resource2", "success")
        logger.log_event("type3", "user1", "action3", "resource3", "success")
        
        # Get events by user
        user1_events = logger.get_events_by_user("user1")
        assert len(user1_events) == 2
        assert all(e.user_id == "user1" for e in user1_events)
        
        user2_events = logger.get_events_by_user("user2")
        assert len(user2_events) == 1
        assert user2_events[0].user_id == "user2"
        
        user3_events = logger.get_events_by_user("user3")
        assert len(user3_events) == 0
    
    def test_clear_events(self):
        """Test clearing all events."""
        logger = MockAuditLogger()
        
        # Log some events
        logger.log_event("type1", "user1", "action1", "resource1", "success")
        logger.log_event("type2", "user2", "action2", "resource2", "success")
        
        # Clear events
        logger.clear_events()
        
        # Check everything is cleared
        assert len(logger.events) == 0
        assert len(logger.session_logs_cache) == 0
    
    def test_thread_safety(self):
        """Test thread safety of audit logging."""
        logger = MockAuditLogger()
        errors = []
        
        def log_events(user_id, count):
            try:
                for i in range(count):
                    logger.log_event(
                        event_type=f"event_{i}",
                        user_id=user_id,
                        action=f"action_{i}",
                        resource=f"resource_{i}",
                        result="success"
                    )
            except Exception as e:
                errors.append(e)
        
        # Log events from multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_events, args=(f"user_{i}", 20))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Check all events were logged
        assert len(logger.events) == 100


class TestEnhancedAccessManager:
    """Test EnhancedAccessManager class."""
    
    def test_access_manager_initialization(self):
        """Test access manager initialization."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        assert manager.security == security_mock
        assert manager.access_records == {}
        assert manager.role_assignments == {}
        assert manager.logger is not None
    
    def test_assign_role(self):
        """Test role assignment."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        manager.assign_role("user123", SecurityLevel.ADMIN)
        
        assert manager.role_assignments["user123"] == SecurityLevel.ADMIN
        
        # Check security event was logged
        security_mock._log_security_event.assert_called_once_with(
            event_type="role_assignment",
            user_id="user123",
            action="assign_role",
            resource="user_management",
            result="success",
            details={'role': SecurityLevel.ADMIN.value}
        )
    
    def test_get_user_role_assigned(self):
        """Test getting user role when explicitly assigned."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        manager.assign_role("user123", SecurityLevel.THERAPIST)
        role = manager.get_user_role("user123")
        
        assert role == SecurityLevel.THERAPIST
    
    def test_get_user_role_from_pattern(self):
        """Test getting user role from user_id pattern."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        # Test various patterns - the implementation checks for role.value as prefix
        assert manager.get_user_role("2_patient_123") == SecurityLevel.PATIENT
        assert manager.get_user_role("3_therapist_456") == SecurityLevel.THERAPIST
        assert manager.get_user_role("4_admin_789") == SecurityLevel.ADMIN
        assert manager.get_user_role("1_guest_123") == SecurityLevel.GUEST
    
    def test_get_user_role_invalid(self):
        """Test getting user role with invalid input."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        # Test None and empty string
        assert manager.get_user_role(None) == SecurityLevel.GUEST
        assert manager.get_user_role("") == SecurityLevel.GUEST
        assert manager.get_user_role("invalid_user") == SecurityLevel.GUEST
    
    def test_grant_access(self):
        """Test granting access to a resource."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        manager.grant_access("user123", "resource1", "read")
        
        assert "user123" in manager.access_records
        assert "resource1" in manager.access_records["user123"]
        assert "read" in manager.access_records["user123"]["resource1"]
        
        # Check security event was logged
        security_mock._log_security_event.assert_called_once_with(
            event_type="access_granted",
            user_id="user123",
            action="grant_access",
            resource="resource1",
            result="success",
            details={'permission': 'read'}
        )
    
    def test_grant_multiple_permissions(self):
        """Test granting multiple permissions to a resource."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        manager.grant_access("user123", "resource1", "read")
        manager.grant_access("user123", "resource1", "write")
        manager.grant_access("user123", "resource2", "execute")
        
        assert len(manager.access_records["user123"]["resource1"]) == 2
        assert "read" in manager.access_records["user123"]["resource1"]
        assert "write" in manager.access_records["user123"]["resource1"]
        assert len(manager.access_records["user123"]["resource2"]) == 1
        assert "execute" in manager.access_records["user123"]["resource2"]
    
    def test_has_access_explicit_grant(self):
        """Test access check with explicit grant."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        manager.grant_access("user123", "resource1", "read")
        
        assert manager.has_access("user123", "resource1", "read") is True
        assert manager.has_access("user123", "resource1", "write") is False
        assert manager.has_access("user123", "resource2", "read") is False
    
    def test_has_access_role_based(self):
        """Test access check with role-based permissions."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        # Test patient role
        assert manager.has_access("patient_123", "own_voice_data", "read") is True
        assert manager.has_access("patient_123", "own_voice_data", "update_own") is True
        assert manager.has_access("patient_123", "admin_panel", "read") is False
        
        # Test therapist role
        assert manager.has_access("therapist_456", "therapy_sessions", "read") is True
        assert manager.has_access("therapist_456", "therapy_sessions", "create") is True
        assert manager.has_access("therapist_456", "admin_panel", "read") is False
        
        # Test admin role
        assert manager.has_access("admin_789", "admin_panel", "full_access") is True
        assert manager.has_access("admin_789", "all_patient_data", "delete") is True
        assert manager.has_access("admin_789", "any_resource", "any_permission") is True
    
    def test_has_access_invalid_user(self):
        """Test access check with invalid user."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        assert manager.has_access(None, "resource1", "read") is False
        assert manager.has_access("", "resource1", "read") is False
        assert manager.has_access(123, "resource1", "read") is False
    
    def test_check_resource_ownership(self):
        """Test resource ownership checking."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        # Patient accessing own data
        assert manager._check_resource_ownership("patient_123", "own_voice_data", "read") is True
        assert manager._check_resource_ownership("patient_123", "own_voice_data", "write") is True
        
        # Therapist accessing assigned patient data
        assert manager._check_resource_ownership("therapist_456", "assigned_patient_data", "read") is True
        assert manager._check_resource_ownership("therapist_456", "assigned_patient_data", "write") is True
        
        # Admin accessing admin panel
        assert manager._check_resource_ownership("admin_789", "admin_panel", "read") is True
        assert manager._check_resource_ownership("admin_789", "admin_panel", "full_access") is True
        
        # Invalid combinations
        assert manager._check_resource_ownership("patient_123", "admin_panel", "read") is False
        assert manager._check_resource_ownership("guest_123", "own_voice_data", "read") is False
    
    def test_revoke_access(self):
        """Test revoking access to a resource."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        # Grant access first
        manager.grant_access("user123", "resource1", "read")
        manager.grant_access("user123", "resource1", "write")
        
        # Revoke one permission
        manager.revoke_access("user123", "resource1", "read")
        
        assert "read" not in manager.access_records["user123"]["resource1"]
        assert "write" in manager.access_records["user123"]["resource1"]
        
        # Check security event was logged
        security_mock._log_security_event.assert_called_with(
            event_type="access_revoked",
            user_id="user123",
            action="revoke_access",
            resource="resource1",
            result="success",
            details={'permission': 'read'}
        )
    
    def test_revoke_nonexistent_access(self):
        """Test revoking non-existent access."""
        security_mock = Mock()
        manager = EnhancedAccessManager(security_mock)
        
        # Should not raise exception
        manager.revoke_access("user123", "resource1", "read")
        
        # Should not log event since access didn't exist
        security_mock._log_security_event.assert_not_called()


class TestEnhancedAccessControl:
    """Test EnhancedAccessControl class."""
    
    def test_access_control_initialization(self):
        """Test access control initialization."""
        config = {
            "max_sessions_per_user": 3,
            "session_timeout_minutes": 60,
            "require_mfa_for_admin": False,
            "audit_all_access": False
        }
        access_control = EnhancedAccessControl(config)
        
        assert access_control.max_sessions_per_user == 3
        assert access_control.session_timeout_minutes == 60
        assert access_control.require_mfa_for_admin is False
        assert access_control.audit_all_access is False
        assert access_control.session_manager is not None
        assert access_control.audit_logger is not None
        assert access_control.access_manager is not None
        assert access_control.access_attempts == {}
    
    def test_access_control_default_config(self):
        """Test access control with default config."""
        access_control = EnhancedAccessControl()
        
        assert access_control.max_sessions_per_user == 5
        assert access_control.session_timeout_minutes == 30
        assert access_control.require_mfa_for_admin is True
        assert access_control.audit_all_access is True
    
    def test_check_access_valid_session(self):
        """Test access check with valid session."""
        access_control = EnhancedAccessControl()
        
        # Create session
        session_id = access_control.session_manager.create_session(
            "user123", SecurityLevel.PATIENT
        )
        
        # Check access for patient operation
        result = access_control.check_access(session_id, "read_own")
        assert result is True
    
    def test_check_access_invalid_session(self):
        """Test access check with invalid session."""
        access_control = EnhancedAccessControl()
        
        # Check access with invalid session
        result = access_control.check_access("invalid_session", "read_own")
        assert result is False
    
    def test_check_access_insufficient_permissions(self):
        """Test access check with insufficient permissions."""
        access_control = EnhancedAccessControl()
        
        # Create guest session
        session_id = access_control.session_manager.create_session(
            "guest123", SecurityLevel.GUEST
        )
        
        # Try admin operation
        result = access_control.check_access(session_id, "admin_operation")
        assert result is False
    
    def test_check_operation_access(self):
        """Test operation access checking by level."""
        access_control = EnhancedAccessControl()
        
        # Test guest permissions
        assert access_control._check_operation_access(SecurityLevel.GUEST, "read_public") is True
        assert access_control._check_operation_access(SecurityLevel.GUEST, "read_own") is False
        
        # Test patient permissions
        assert access_control._check_operation_access(SecurityLevel.PATIENT, "read_own") is True
        assert access_control._check_operation_access(SecurityLevel.PATIENT, "admin_operation") is False
        
        # Test therapist permissions
        assert access_control._check_operation_access(SecurityLevel.THERAPIST, "read_patient") is True
        assert access_control._check_operation_access(SecurityLevel.THERAPIST, "delete_all") is False
        
        # Test admin permissions
        assert access_control._check_operation_access(SecurityLevel.ADMIN, "admin_operation") is True
        assert access_control._check_operation_access(SecurityLevel.ADMIN, "delete_all") is True
    
    def test_create_session_success(self):
        """Test successful session creation."""
        access_control = EnhancedAccessControl()
        
        session_id = access_control.create_session("user123")
        
        assert session_id is not None
        assert access_control.session_manager.validate_session(session_id) is True
    
    def test_create_session_failure(self):
        """Test session creation failure."""
        access_control = EnhancedAccessControl({"max_sessions_per_user": 1})
        
        # Create first session
        access_control.create_session("user123")
        
        # Try to create second session - should raise exception
        with pytest.raises(AccessControlError):
            access_control.create_session("user123")
    
    def test_invalidate_session(self):
        """Test session invalidation."""
        access_control = EnhancedAccessControl()
        
        # Create session
        session_id = access_control.create_session("user123")
        
        # Invalidate session
        access_control.invalidate_session(session_id, "user123")
        
        # Session should be invalid
        assert access_control.session_manager.validate_session(session_id) is False
    
    def test_record_failed_attempt(self):
        """Test recording failed access attempts."""
        access_control = EnhancedAccessControl()
        
        # Record failed attempts
        access_control._record_failed_attempt("user123", "resource1", "read")
        access_control._record_failed_attempt("user123", "resource1", "write")
        access_control._record_failed_attempt("user123", "resource2", "read")
        
        assert "user123" in access_control.access_attempts
        assert len(access_control.access_attempts["user123"]) == 3
        
        # Check attempt details
        attempts = access_control.access_attempts["user123"]
        assert attempts[0]["resource"] == "resource1"
        assert attempts[0]["permission"] == "read"
        assert attempts[0]["reason"] == "access_denied"
    
    def test_record_failed_attempt_limit(self):
        """Test failed attempt limit enforcement."""
        access_control = EnhancedAccessControl()
        
        # Record more than the limit (10)
        for i in range(15):
            access_control._record_failed_attempt("user123", "resource1", "read")
        
        # Should only keep last 10 attempts
        assert len(access_control.access_attempts["user123"]) == 10
    
    def test_get_security_events(self):
        """Test getting security events."""
        access_control = EnhancedAccessControl()
        
        # Log some events
        access_control.audit_logger.log_event(
            "type1", "user1", "action1", "resource1", "success"
        )
        access_control.audit_logger.log_event(
            "type2", "user2", "action2", "resource2", "success"
        )
        
        # Get all events
        all_events = access_control.get_security_events()
        assert len(all_events) == 2
        
        # Get events by type
        type1_events = access_control.get_security_events("type1")
        assert len(type1_events) == 1
        assert type1_events[0].event_type == "type1"


class TestVoiceSecurity:
    """Test VoiceSecurity class."""
    
    def test_voice_security_initialization(self):
        """Test voice security initialization."""
        config = SecurityConfig(
            encryption_enabled=False,
            data_retention_days=60,
            session_timeout_minutes=45,
            max_login_attempts=5
        )
        voice_security = VoiceSecurity(config)
        
        assert voice_security.config == config
        assert voice_security.data_retention_days == 60
        assert voice_security.session_timeout_minutes == 45
        assert voice_security.max_login_attempts == 5
        assert voice_security.audit_logger is not None
        assert voice_security.access_manager is not None
    
    def test_voice_security_default_config(self):
        """Test voice security with default config."""
        voice_security = VoiceSecurity()
        
        assert voice_security.data_retention_days == 30
        assert voice_security.session_timeout_minutes == 30
        assert voice_security.max_login_attempts == 3
    
    def test_initialize_test_roles(self):
        """Test initialization of test roles."""
        voice_security = VoiceSecurity()
        
        # Check test roles were assigned
        assert voice_security.access_manager.get_user_role("patient_123") == SecurityLevel.PATIENT
        assert voice_security.access_manager.get_user_role("therapist_456") == SecurityLevel.THERAPIST
        assert voice_security.access_manager.get_user_role("admin_789") == SecurityLevel.ADMIN
        assert voice_security.access_manager.get_user_role("guest_123") == SecurityLevel.GUEST
    
    def test_log_security_event(self):
        """Test security event logging."""
        # Clear global instance to ensure clean test
        global _voice_security_instance
        _voice_security_instance = None
        
        config = SecurityConfig(audit_logging_enabled=True)
        voice_security = VoiceSecurity(config)
        
        # Clear existing events from initialization
        voice_security.clear_audit_logs()
        
        voice_security._log_security_event(
            event_type="test_event",
            user_id="user123",
            action="test_action",
            resource="test_resource",
            result="success",
            details={"key": "value"}
        )
        
        # Check event was logged
        events = voice_security.get_security_events()
        assert len(events) == 1
        assert events[0].event_type == "test_event"
        assert events[0].user_id == "user123"
    
    def test_log_security_event_disabled(self):
        """Test security event logging when disabled."""
        config = SecurityConfig(audit_logging_enabled=False)
        voice_security = VoiceSecurity(config)
        
        voice_security._log_security_event(
            event_type="test_event",
            user_id="user123",
            action="test_action",
            resource="test_resource",
            result="success"
        )
        
        # Check no event was logged
        events = voice_security.get_security_events()
        assert len(events) == 0
    
    def test_get_security_events(self):
        """Test getting security events."""
        # Clear global instance to ensure clean test
        global _voice_security_instance
        _voice_security_instance = None
        
        voice_security = VoiceSecurity()
        
        # Clear existing events from initialization
        voice_security.clear_audit_logs()
        
        # Log some events
        voice_security._log_security_event("type1", "user1", "action1", "resource1", "success")
        voice_security._log_security_event("type2", "user2", "action2", "resource2", "success")
        voice_security._log_security_event("type1", "user3", "action3", "resource3", "success")
        
        # Get all events
        all_events = voice_security.get_security_events()
        assert len(all_events) == 3
        
        # Get events by type
        type1_events = voice_security.get_security_events("type1")
        assert len(type1_events) == 2
        assert all(e.event_type == "type1" for e in type1_events)
    
    def test_clear_audit_logs(self):
        """Test clearing audit logs."""
        voice_security = VoiceSecurity()
        
        # Log some events
        voice_security._log_security_event("type1", "user1", "action1", "resource1", "success")
        voice_security._log_security_event("type2", "user2", "action2", "resource2", "success")
        
        # Clear logs
        voice_security.clear_audit_logs()
        
        # Check logs are cleared
        events = voice_security.get_security_events()
        assert len(events) == 0


class TestModuleFunctions:
    """Test module-level functions."""
    
    def test_get_voice_security_instance(self):
        """Test getting voice security instance."""
        # First call should create new instance
        instance1 = get_voice_security_instance()
        assert instance1 is not None
        assert isinstance(instance1, VoiceSecurity)
        
        # Second call should return same instance
        instance2 = get_voice_security_instance()
        assert instance1 is instance2
    
    def test_get_voice_security_instance_with_config(self):
        """Test getting voice security instance with custom config."""
        # Clear global instance to ensure clean test
        global _voice_security_instance
        _voice_security_instance = None
        
        config = SecurityConfig(data_retention_days=60)
        
        # First call with config should create new instance
        instance1 = get_voice_security_instance(config)
        assert instance1.data_retention_days == 60
        
        # Second call without config should return same instance
        instance2 = get_voice_security_instance()
        assert instance1 is instance2
        assert instance2.data_retention_days == 60


if __name__ == "__main__":
    pytest.main([__file__])