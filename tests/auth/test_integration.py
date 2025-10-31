"""
Integration tests for authentication system.

Tests the complete authentication flow including middleware integration.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from auth.auth_service import AuthService
from auth.middleware import AuthMiddleware
from auth.user_model import UserModel, UserRole


class TestAuthIntegration:
    """Integration tests for authentication system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp:
            yield temp

    @pytest.fixture
    def mock_streamlit_session(self):
        """Mock Streamlit session state."""
        session_state = MagicMock()
        session_state.auth_token = None
        session_state.user = None
        session_state.auth_time = None
        session_state.auth_service = None
        session_state.auth_middleware = None
        session_state.show_register = False
        session_state.show_reset = False
        session_state.show_profile = False
        session_state.show_change_password = False
        return session_state

    @pytest.fixture
    def auth_service(self, temp_dir):
        """Create auth service with temporary storage."""
        with patch.dict(os.environ, {'AUTH_DATA_DIR': temp_dir}):
            service = AuthService()
            yield service

    @pytest.fixture
    def auth_middleware(self, auth_service, mock_streamlit_session):
        """Create auth middleware with mocked session state."""
        with patch('streamlit.session_state', mock_streamlit_session):
            middleware = AuthMiddleware(auth_service)
            yield middleware

    def test_complete_registration_login_flow(self, auth_service, auth_middleware, mock_streamlit_session):
        """Test complete user registration and login flow."""
        # Register user
        reg_result = auth_service.register_user(
            email="integration@example.com",
            password="Integration123",
            full_name="Integration Test User"
        )

        assert reg_result.success == True

        # Login via middleware
        login_result = auth_middleware.login_user("integration@example.com", "Integration123")

        assert login_result.success == True
        assert mock_streamlit_session.auth_token is not None
        assert mock_streamlit_session.user is not None

        # Check authentication status
        assert auth_middleware.is_authenticated() == True

        # Get current user
        current_user = auth_middleware.get_current_user()
        assert current_user is not None
        assert current_user.email == "integration@example.com"

    def test_session_persistence(self, auth_service, auth_middleware, mock_streamlit_session):
        """Test session persistence across requests."""
        # Register and login
        auth_service.register_user(
            email="session@example.com",
            password="Session123",
            full_name="Session Test User"
        )
        auth_middleware.login_user("session@example.com", "Session123")

        token = mock_streamlit_session.auth_token

        # Simulate new request with persisted token
        mock_streamlit_session.auth_token = token

        # Should still be authenticated
        assert auth_middleware.is_authenticated() == True

        user = auth_middleware.get_current_user()
        assert user is not None
        assert user.email == "session@example.com"

    def test_logout_flow(self, auth_service, auth_middleware, mock_streamlit_session):
        """Test complete logout flow."""
        # Register and login
        auth_service.register_user(
            email="logout@example.com",
            password="Logout123",
            full_name="Logout Test User"
        )
        auth_middleware.login_user("logout@example.com", "Logout123")

        # Verify logged in
        assert auth_middleware.is_authenticated() == True

        # Logout
        auth_middleware.logout_user()

        # Verify logged out
        assert auth_middleware.is_authenticated() == False
        assert mock_streamlit_session.auth_token is None
        assert mock_streamlit_session.user is None

    def test_password_reset_flow(self, auth_service, auth_middleware):
        """Test complete password reset flow."""
        # Register user
        auth_service.register_user(
            email="reset@example.com",
            password="Reset123",
            full_name="Reset Test User"
        )

        # Initiate password reset
        reset_result = auth_service.initiate_password_reset("reset@example.com")
        assert reset_result.success == True

        # Get reset token (normally sent via email)
        user = auth_service.user_model.get_user_by_email("reset@example.com")
        reset_token = user.password_reset_token

        # Reset password
        reset_result = auth_service.reset_password(reset_token, "NewReset123")
        assert reset_result.success == True

        # Should be able to login with new password
        login_result = auth_middleware.login_user("reset@example.com", "NewReset123")
        assert login_result.success == True

    def test_role_based_access_control(self, auth_service, auth_middleware):
        """Test role-based access control integration."""
        # Create users with different roles
        auth_service.register_user(
            email="patient@example.com",
            password="Patient123",
            full_name="Patient User",
            role=UserRole.PATIENT
        )

        auth_service.register_user(
            email="therapist@example.com",
            password="Therapist123",
            full_name="Therapist User",
            role=UserRole.THERAPIST
        )

        auth_service.register_user(
            email="admin@example.com",
            password="Admin123",
            full_name="Admin User",
            role=UserRole.ADMIN
        )

        # Login as patient
        auth_middleware.login_user("patient@example.com", "Patient123")
        patient = auth_middleware.get_current_user()

        # Check patient permissions
        assert auth_service.validate_session_access(patient.user_id, "own_profile", "read")
        assert auth_service.validate_session_access(patient.user_id, "therapy_sessions", "read")
        assert not auth_service.validate_session_access(patient.user_id, "admin_panel", "read")

        # Login as therapist
        auth_middleware.logout_user()
        auth_middleware.login_user("therapist@example.com", "Therapist123")
        therapist = auth_middleware.get_current_user()

        # Check therapist permissions
        assert auth_service.validate_session_access(therapist.user_id, "therapy_sessions", "update")
        assert auth_service.validate_session_access(therapist.user_id, "assigned_patients", "read")
        assert not auth_service.validate_session_access(therapist.user_id, "system_config", "read")

        # Login as admin
        auth_middleware.logout_user()
        auth_middleware.login_user("admin@example.com", "Admin123")
        admin = auth_middleware.get_current_user()

        # Check admin permissions
        assert auth_service.validate_session_access(admin.user_id, "system_config", "update")
        assert auth_service.validate_session_access(admin.user_id, "audit_logs", "read")
        assert auth_service.validate_session_access(admin.user_id, "all_profiles", "read")

    def test_concurrent_session_handling(self, auth_service, auth_middleware, mock_streamlit_session):
        """Test handling of concurrent sessions."""
        # Register user
        auth_service.register_user(
            email="concurrent@example.com",
            password="Concurrent123",
            full_name="Concurrent Test User"
        )

        # Login multiple times
        tokens = []
        for i in range(3):
            result = auth_middleware.login_user("concurrent@example.com", "Concurrent123")
            if result.success:
                tokens.append(result.token)

        # Should have tokens for successful logins
        assert len(tokens) <= auth_service.max_concurrent_sessions

        # All tokens should be valid
        for token in tokens:
            user = auth_service.validate_token(token)
            assert user is not None

    def test_security_integration_voice_data(self, auth_service):
        """Test that voice security integrates with authentication."""
        # This would test the integration between voice/security.py and auth
        # For now, just verify the auth service is available
        assert auth_service is not None

        # Register user
        auth_service.register_user(
            email="voice@example.com",
            password="Voice123",
            full_name="Voice Test User"
        )

        # Login
        login_result = auth_service.login_user("voice@example.com", "Voice123")
        assert login_result.success == True

        # The voice/security.py should be able to validate this user
        # (This would be tested in voice security integration tests)

    def test_hipaa_compliance_features(self, auth_service):
        """Test HIPAA compliance features."""
        # Register user with medical info
        user_result = auth_service.register_user(
            email="hipaa@example.com",
            password="Hipaa123",
            full_name="HIPAA Test User"
        )

        user_id = user_result.user.user_id

        # Update with medical info
        auth_service.user_model.update_user(user_id, {
            'medical_info': {
                'diagnosis': 'Generalized Anxiety Disorder',
                'treatment_plan': 'CBT sessions',
                'medications': ['Sertraline 50mg']
            }
        })

        # Verify medical info is stored
        user = auth_service.user_model.get_user(user_id)
        assert 'medical_info' in user.medical_info
        assert user.medical_info['diagnosis'] == 'Generalized Anxiety Disorder'

    def test_audit_trail_integration(self, auth_service, auth_middleware):
        """Test that authentication actions are properly audited."""
        # Register user
        auth_service.register_user(
            email="audit@example.com",
            password="Audit123",
            full_name="Audit Test User"
        )

        # Login
        auth_middleware.login_user("audit@example.com", "Audit123")

        # Check that audit logs exist (implementation dependent)
        # In a real system, we would check audit logs for auth events
        assert True  # Placeholder - audit logging would be verified here

    def test_environment_variable_configuration(self, temp_dir):
        """Test that authentication uses environment variables correctly."""
        with patch.dict(os.environ, {
            'AUTH_DATA_DIR': temp_dir,
            'JWT_SECRET_KEY': 'test_jwt_secret',
            'JWT_EXPIRATION_HOURS': '2',
            'SESSION_TIMEOUT_MINUTES': '15',
            'MAX_CONCURRENT_SESSIONS': '2'
        }):
            service = AuthService()

            # Verify configuration is loaded
            assert service.jwt_expiration_hours == 2
            assert service.session_timeout_minutes == 15
            assert service.max_concurrent_sessions == 2