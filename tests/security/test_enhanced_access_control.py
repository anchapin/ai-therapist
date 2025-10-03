"""
Enhanced Access Control Tests

This module contains tests for enhanced access control mechanisms
in the AI Therapist voice services.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to ensure proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import security modules
try:
    from voice.enhanced_security import (
        EnhancedAccessControl,
        AccessLevel,
        AccessControlError,
        SessionManager,
        AuditLogger
    )
    ENHANCED_SECURITY_AVAILABLE = True
except ImportError:
    ENHANCED_SECURITY_AVAILABLE = False
    EnhancedAccessControl = None
    AccessLevel = None
    AccessControlError = None
    SessionManager = None
    AuditLogger = None


@pytest.mark.skipif(not ENHANCED_SECURITY_AVAILABLE, reason="Enhanced security module not available")
class TestEnhancedAccessControl:
    """Test enhanced access control functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.access_control = EnhancedAccessControl({
            'max_sessions_per_user': 5,
            'session_timeout_minutes': 30,
            'require_mfa_for_admin': True,
            'audit_all_access': True
        })
        self.session_manager = SessionManager()
        self.audit_logger = AuditLogger()

    def test_access_level_hierarchy(self):
        """Test access level hierarchy enforcement."""
        # Test that higher levels have more privileges
        assert AccessLevel.ADMIN.value > AccessLevel.THERAPIST.value
        assert AccessLevel.THERAPIST.value > AccessLevel.PATIENT.value
        assert AccessLevel.PATIENT.value > AccessLevel.GUEST.value

    def test_session_creation_and_validation(self):
        """Test session creation and validation."""
        user_id = "test_user"
        access_level = AccessLevel.PATIENT

        # Create session
        session_id = self.session_manager.create_session(
            user_id=user_id,
            access_level=access_level,
            metadata={"ip": "127.0.0.1"}
        )

        assert session_id is not None
        assert self.session_manager.validate_session(session_id) is True

        # Check session details
        session = self.session_manager.get_session(session_id)
        assert session.user_id == user_id
        assert session.access_level == access_level

    def test_session_expiration(self):
        """Test session expiration handling."""
        # Create session with short timeout for testing
        short_timeout_manager = SessionManager({
            'session_timeout_minutes': 0.01  # 0.6 seconds
        })

        user_id = "test_user"
        session_id = short_timeout_manager.create_session(
            user_id=user_id,
            access_level=AccessLevel.PATIENT
        )

        # Session should be valid initially
        assert short_timeout_manager.validate_session(session_id) is True

        # Wait for session to expire
        time.sleep(1)

        # Session should now be invalid
        assert short_timeout_manager.validate_session(session_id) is False

    def test_concurrent_session_limits(self):
        """Test concurrent session limits per user."""
        user_id = "test_user"
        max_sessions = 3

        # Create access control with session limit
        limited_ac = EnhancedAccessControl({
            'max_sessions_per_user': max_sessions
        })

        # Create sessions up to the limit
        session_ids = []
        for i in range(max_sessions):
            session_id = limited_ac.create_user_session(
                user_id=user_id,
                access_level=AccessLevel.PATIENT
            )
            session_ids.append(session_id)
            assert session_id is not None

        # Try to create one more session (should fail)
        extra_session = limited_ac.create_user_session(
            user_id=user_id,
            access_level=AccessLevel.PATIENT
        )
        assert extra_session is None

    def test_access_control_enforcement(self):
        """Test access control enforcement for different operations."""
        # Create sessions for different access levels
        admin_session = self.access_control.create_user_session(
            user_id="admin_user",
            access_level=AccessLevel.ADMIN
        )

        therapist_session = self.access_control.create_user_session(
            user_id="therapist_user",
            access_level=AccessLevel.THERAPIST
        )

        patient_session = self.access_control.create_user_session(
            user_id="patient_user",
            access_level=AccessLevel.PATIENT
        )

        # Test admin operations (should succeed)
        assert self.access_control.check_access(
            admin_session, "admin_operation"
        ) is True

        # Test therapist operations (should succeed for therapist, fail for patient)
        assert self.access_control.check_access(
            therapist_session, "therapy_operation"
        ) is True

        assert self.access_control.check_access(
            patient_session, "therapy_operation"
        ) is False

        # Test patient operations (should succeed for all)
        assert self.access_control.check_access(
            patient_session, "patient_operation"
        ) is True

    def test_audit_logging(self):
        """Test audit logging for access control events."""
        # Enable audit logging
        self.access_control.enable_audit_logging(self.audit_logger)

        # Perform some access-controlled operations
        session_id = self.access_control.create_user_session(
            user_id="test_user",
            access_level=AccessLevel.PATIENT
        )

        # Check access (should be logged)
        self.access_control.check_access(
            session_id, "patient_operation"
        )

        # Get audit events
        events = self.audit_logger.get_events()
        assert len(events) > 0

        # Verify event structure
        access_events = [e for e in events if e['event_type'] == 'access_check']
        assert len(access_events) > 0

        event = access_events[0]
        assert 'user_id' in event
        assert 'operation' in event
        assert 'result' in event
        assert 'timestamp' in event

    def test_concurrent_access_control(self):
        """Test access control under concurrent load."""
        num_threads = 10
        operations_per_thread = 20
        results = []

        def access_control_worker(worker_id):
            # Create session for this worker
            session_id = self.access_control.create_user_session(
                user_id=f"worker_{worker_id}",
                access_level=AccessLevel.PATIENT
            )

            # Perform operations
            worker_results = []
            for i in range(operations_per_thread):
                operation = f"operation_{i % 5}"
                result = self.access_control.check_access(session_id, operation)
                worker_results.append(result)

            results.append(worker_results)

        # Start concurrent workers
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=access_control_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30.0)

        # Verify results
        assert len(results) == num_threads
        for worker_results in results:
            assert len(worker_results) == operations_per_thread

    def test_access_control_error_handling(self):
        """Test error handling in access control."""
        # Test with invalid session
        assert self.access_control.check_access(
            "invalid_session", "operation"
        ) is False

        # Test with None session
        assert self.access_control.check_access(
            None, "operation"
        ) is False

        # Test with None operation
        session_id = self.access_control.create_user_session(
            user_id="test_user",
            access_level=AccessLevel.PATIENT
        )
        assert self.access_control.check_access(
            session_id, None
        ) is False

    def test_session_cleanup(self):
        """Test session cleanup and resource management."""
        # Create multiple sessions
        session_ids = []
        for i in range(10):
            session_id = self.session_manager.create_session(
                user_id=f"user_{i}",
                access_level=AccessLevel.PATIENT
            )
            session_ids.append(session_id)

        # Verify sessions exist
        assert len(self.session_manager.active_sessions) == 10

        # Clean up expired sessions
        self.session_manager.cleanup_expired_sessions()

        # Sessions should still exist (not expired yet)
        assert len(self.session_manager.active_sessions) == 10

        # Manually expire sessions
        for session_id in session_ids:
            session = self.session_manager.get_session(session_id)
            session.created_time = time.time() - 3600  # 1 hour ago

        # Clean up expired sessions
        self.session_manager.cleanup_expired_sessions()

        # Sessions should now be cleaned up
        assert len(self.session_manager.active_sessions) == 0

    def test_access_control_configuration(self):
        """Test access control configuration."""
        # Test with custom configuration
        custom_config = {
            'max_sessions_per_user': 10,
            'session_timeout_minutes': 60,
            'require_mfa_for_admin': False,
            'audit_all_access': False
        }

        custom_ac = EnhancedAccessControl(custom_config)

        # Verify configuration was applied
        assert custom_ac.max_sessions_per_user == 10
        assert custom_ac.session_timeout_minutes == 60
        assert custom_ac.require_mfa_for_admin is False
        assert custom_ac.audit_all_access is False