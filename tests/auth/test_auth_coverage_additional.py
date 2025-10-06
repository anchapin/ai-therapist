"""
Additional tests to improve auth module coverage to 90%+.
Focuses on covering the missing lines identified in coverage report.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import jwt

from auth.auth_service import AuthService, AuthResult, AuthSession
from auth.middleware import AuthMiddleware
from auth.user_model import UserModel, UserProfile, UserRole, UserStatus


class TestAuthServiceAdditionalCoverage:
    """Additional tests for AuthService to improve coverage."""

    @pytest.fixture
    def mock_user_model(self):
        """Create mock user model."""
        mock_model = Mock(spec=UserModel)
        
        # Mock user for testing
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "test_user_123"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.PATIENT
        mock_user.status = UserStatus.ACTIVE
        mock_user.created_at = datetime.now()
        mock_user.updated_at = datetime.now()
        mock_user.last_login = None
        mock_user.login_attempts = 0
        mock_user.account_locked_until = None
        mock_user.password_reset_token = None
        mock_user.password_reset_expires = None
        mock_user.preferences = {}
        mock_user.medical_info = {}
        mock_user.is_locked.return_value = False
        mock_user.can_access_resource.return_value = True
        
        mock_model.create_user.return_value = mock_user
        mock_model.authenticate_user.return_value = mock_user
        mock_model.get_user.return_value = mock_user
        mock_model.get_user_by_email.return_value = mock_user
        
        return mock_model

    @pytest.fixture
    def auth_service_with_mocks(self, mock_user_model):
        """Create auth service with mocked dependencies."""
        # Set JWT secret for testing
        os.environ["JWT_SECRET_KEY"] = "test-secret-key"
        os.environ["JWT_EXPIRATION_HOURS"] = "24"
        os.environ["SESSION_TIMEOUT_MINUTES"] = "30"
        os.environ["MAX_CONCURRENT_SESSIONS"] = "5"
        
        # Mock the session repository to avoid database issues
        with patch('auth.auth_service.SessionRepository') as mock_repo_class:
            mock_repo = Mock()
            mock_repo.save.return_value = True
            mock_repo.find_by_id.return_value = None
            mock_repo.find_by_user_id.return_value = []
            mock_repo_class.return_value = mock_repo
            
            service = AuthService(user_model=mock_user_model)
            service.session_repo = mock_repo
            
            yield service

    def test_register_user_non_patient_role_forced_to_patient(self, auth_service_with_mocks, mock_user_model):
        """Test that non-patient roles are forced to patient role."""
        result = auth_service_with_mocks.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User",
            role=UserRole.ADMIN  # Try to register as admin
        )
        
        assert result.success is True
        # Should be called with PATIENT role, not ADMIN
        mock_user_model.create_user.assert_called_once_with(
            "test@example.com", "TestPass123", "Test User", UserRole.PATIENT
        )

    def test_register_user_exception_handling(self, auth_service_with_mocks, mock_user_model):
        """Test exception handling during user registration."""
        mock_user_model.create_user.side_effect = Exception("Database error")
        
        result = auth_service_with_mocks.register_user(
            email="test@example.com",
            password="TestPass123",
            full_name="Test User"
        )
        
        assert result.success is False
        assert result.error_message == "Registration failed"

    def test_login_user_session_creation_failure(self, auth_service_with_mocks, mock_user_model):
        """Test login when session creation fails."""
        mock_user_model.authenticate_user.return_value = Mock(
            status=UserStatus.ACTIVE,
            is_locked=Mock(return_value=False)
        )
        
        with patch.object(auth_service_with_mocks, '_create_session', return_value=None):
            result = auth_service_with_mocks.login_user(
                email="test@example.com",
                password="TestPass123"
            )
        
        assert result.success is False
        assert result.error_message == "Failed to create session"

    def test_login_user_exception_handling(self, auth_service_with_mocks, mock_user_model):
        """Test exception handling during login."""
        mock_user_model.authenticate_user.side_effect = Exception("Auth error")
        
        result = auth_service_with_mocks.login_user(
            email="test@example.com",
            password="TestPass123"
        )
        
        assert result.success is False
        assert result.error_message == "Login failed"

    def test_validate_token_missing_user_id(self, auth_service_with_mocks):
        """Test token validation with missing user_id."""
        # Create token without user_id
        token = jwt.encode(
            {
                'email': 'test@example.com',
                'role': 'patient',
                'session_id': 'session_123',
                'iat': int(datetime.now().timestamp()),
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp()),
                'iss': 'ai-therapist'
            },
            auth_service_with_mocks.jwt_secret,
            algorithm=auth_service_with_mocks.jwt_algorithm
        )
        
        user = auth_service_with_mocks.validate_token(token)
        assert user is None

    def test_validate_token_inactive_user(self, auth_service_with_mocks, mock_user_model):
        """Test token validation with inactive user."""
        # Create valid token
        token = jwt.encode(
            {
                'user_id': 'test_user_123',
                'email': 'test@example.com',
                'role': 'patient',
                'session_id': 'session_123',
                'iat': int(datetime.now().timestamp()),
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp()),
                'iss': 'ai-therapist'
            },
            auth_service_with_mocks.jwt_secret,
            algorithm=auth_service_with_mocks.jwt_algorithm
        )
        
        # Mock inactive user
        mock_user = Mock(spec=UserProfile)
        mock_user.status = UserStatus.INACTIVE
        mock_user_model.get_user.return_value = mock_user
        
        with patch.object(auth_service_with_mocks, '_is_session_valid', return_value=True):
            user = auth_service_with_mocks.validate_token(token)
        
        assert user is None

    def test_validate_token_exception_handling(self, auth_service_with_mocks):
        """Test exception handling during token validation."""
        with patch.object(auth_service_with_mocks, '_is_session_valid', side_effect=Exception("Validation error")):
            user = auth_service_with_mocks.validate_token("invalid_token")
        
        assert user is None

    def test_logout_user_exception_handling(self, auth_service_with_mocks):
        """Test exception handling during logout."""
        with patch('jwt.decode', side_effect=Exception("Decode error")):
            result = auth_service_with_mocks.logout_user("invalid_token")
        
        assert result is False

    def test_refresh_token_invalid_session(self, auth_service_with_mocks, mock_user_model):
        """Test token refresh with invalid session."""
        # Create valid token
        token = jwt.encode(
            {
                'user_id': 'test_user_123',
                'session_id': 'session_123',
                'iat': int(datetime.now().timestamp()),
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp()),
                'iss': 'ai-therapist'
            },
            auth_service_with_mocks.jwt_secret,
            algorithm=auth_service_with_mocks.jwt_algorithm
        )
        
        # Mock invalid session
        mock_db_session = Mock()
        mock_db_session.is_active = False
        auth_service_with_mocks.session_repo.find_by_id.return_value = mock_db_session
        
        new_token = auth_service_with_mocks.refresh_token(token)
        assert new_token is None

    def test_refresh_token_exception_handling(self, auth_service_with_mocks):
        """Test exception handling during token refresh."""
        with patch('jwt.decode', side_effect=Exception("Decode error")):
            new_token = auth_service_with_mocks.refresh_token("invalid_token")
        
        assert new_token is None

    def test_initiate_password_reset_exception_handling(self, auth_service_with_mocks, mock_user_model):
        """Test exception handling during password reset initiation."""
        mock_user_model.initiate_password_reset.side_effect = Exception("Reset error")
        
        result = auth_service_with_mocks.initiate_password_reset("test@example.com")
        
        assert result.success is False
        assert result.error_message == "Password reset failed"

    def test_reset_password_exception_handling(self, auth_service_with_mocks, mock_user_model):
        """Test exception handling during password reset."""
        mock_user_model.reset_password.side_effect = Exception("Reset error")
        
        result = auth_service_with_mocks.reset_password("token_123", "NewPassword123")
        
        assert result.success is False
        assert result.error_message == "Password reset failed"

    def test_change_password_exception_handling(self, auth_service_with_mocks, mock_user_model):
        """Test exception handling during password change."""
        mock_user_model.change_password.side_effect = Exception("Change error")
        
        result = auth_service_with_mocks.change_password("user_123", "OldPass123", "NewPass123")
        
        assert result.success is False
        assert result.error_message == "Password change failed"

    def test_create_session_exception_handling(self, auth_service_with_mocks):
        """Test exception handling during session creation."""
        with patch.object(auth_service_with_mocks.session_repo, 'find_by_user_id', side_effect=Exception("Session error")):
            session = auth_service_with_mocks._create_session("user_123")
        
        assert session is None

    def test_invalidate_session_no_session(self, auth_service_with_mocks):
        """Test invalidating non-existent session."""
        auth_service_with_mocks.session_repo.find_by_id.return_value = None
        
        # Should not raise an exception
        auth_service_with_mocks._invalidate_session("nonexistent_session", "user_123")

    def test_is_session_valid_no_session(self, auth_service_with_mocks):
        """Test session validation with non-existent session."""
        auth_service_with_mocks.session_repo.find_by_id.return_value = None
        
        result = auth_service_with_mocks._is_session_valid("nonexistent_session", "user_123")
        assert result is False

    def test_is_session_valid_wrong_user(self, auth_service_with_mocks):
        """Test session validation with wrong user."""
        mock_session = Mock()
        mock_session.user_id = "different_user"
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        
        auth_service_with_mocks.session_repo.find_by_id.return_value = mock_session
        
        result = auth_service_with_mocks._is_session_valid("session_123", "user_123")
        assert result is False

    def test_is_session_valid_inactive_session(self, auth_service_with_mocks):
        """Test session validation with inactive session."""
        mock_session = Mock()
        mock_session.user_id = "user_123"
        mock_session.is_active = False
        
        auth_service_with_mocks.session_repo.find_by_id.return_value = mock_session
        
        result = auth_service_with_mocks._is_session_valid("session_123", "user_123")
        assert result is False

    def test_is_session_valid_expired_session(self, auth_service_with_mocks):
        """Test session validation with expired session."""
        mock_session = Mock()
        mock_session.user_id = "user_123"
        mock_session.is_active = True
        mock_session.is_expired.return_value = True
        
        auth_service_with_mocks.session_repo.find_by_id.return_value = mock_session
        
        result = auth_service_with_mocks._is_session_valid("session_123", "user_123")
        assert result is False

    def test_generate_session_id(self, auth_service_with_mocks):
        """Test session ID generation."""
        session_id1 = auth_service_with_mocks._generate_session_id()
        session_id2 = auth_service_with_mocks._generate_session_id()
        
        assert session_id1 is not None
        assert session_id2 is not None
        assert session_id1 != session_id2
        assert session_id1.startswith("session_")
        assert session_id2.startswith("session_")

    def test_background_cleanup_exception_handling(self, auth_service_with_mocks):
        """Test exception handling in background cleanup."""
        with patch.object(auth_service_with_mocks, '_cleanup_expired_sessions', side_effect=Exception("Cleanup error")):
            # Should not raise an exception
            auth_service_with_mocks._background_cleanup()
            # Sleep is called, then exception is caught

    def test_cleanup_expired_sessions(self, auth_service_with_mocks):
        """Test cleanup of expired sessions."""
        with patch('auth.auth_service.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_db.health_check.return_value = {
                'table_counts': {'sessions': 5, 'users': 10}
            }
            mock_db.cleanup_expired_data.return_value = 3
            mock_get_db.return_value = mock_db
            
            auth_service_with_mocks._cleanup_expired_sessions()
            
            mock_db.cleanup_expired_data.assert_called_once()

    def test_get_auth_statistics(self, auth_service_with_mocks):
        """Test getting authentication statistics."""
        with patch('auth.auth_service.get_database_manager') as mock_get_db:
            mock_db = Mock()
            mock_db.health_check.return_value = {
                'table_counts': {'sessions': 5, 'users': 10}
            }
            mock_get_db.return_value = mock_db
            
            stats = auth_service_with_mocks.get_auth_statistics()
            
            assert stats['total_users'] == 10
            assert stats['active_sessions'] == 5
            assert stats['total_sessions_created'] == 5

    def test_filter_user_for_response_none_user(self, auth_service_with_mocks):
        """Test filtering user response with None user."""
        result = auth_service_with_mocks._filter_user_for_response(None)
        assert result is None


class TestMiddlewareAdditionalCoverage:
    """Additional tests for AuthMiddleware to improve coverage."""

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock auth service."""
        mock_service = Mock(spec=AuthService)
        
        # Mock user for testing
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "test_user_123"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.PATIENT
        mock_user.status = UserStatus.ACTIVE
        mock_user.last_login = datetime.now()
        
        mock_service.validate_token.return_value = mock_user
        mock_service.login_user.return_value = AuthResult(
            success=True,
            user=mock_user,
            token="jwt_token_123",
            session=Mock(created_at=datetime.now())
        )
        mock_service.logout_user.return_value = True
        mock_service.register_user.return_value = AuthResult(success=True, user=mock_user)
        mock_service.initiate_password_reset.return_value = AuthResult(success=True)
        mock_service.change_password.return_value = AuthResult(success=True)
        
        return mock_service

    @pytest.fixture
    def auth_middleware(self, mock_auth_service):
        """Create auth middleware with mocked dependencies."""
        return AuthMiddleware(mock_auth_service)

    def test_login_user_no_session_in_result(self, auth_middleware, mock_auth_service):
        """Test login when result has no session."""
        mock_auth_service.login_user.return_value = AuthResult(
            success=True,
            user=Mock(),
            token="jwt_token_123",
            session=None  # No session
        )
        
        with patch('auth.middleware.st') as mock_st:
            mock_st.session_state = {}
            
            result = auth_middleware.login_user("test@example.com", "password")
            
            assert result.success is True
            assert mock_st.session_state['auth_token'] == "jwt_token_123"
            assert 'auth_time' not in mock_st.session_state

    def test_show_register_form_cancel(self, auth_middleware):
        """Test register form cancellation."""
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {}
            
            # Mock form and button
            mock_form = Mock()
            mock_form.__enter__ = Mock(return_value=None)
            mock_form.__exit__ = Mock(return_value=None)
            mock_st.form.return_value = mock_form
            
            # Mock columns
            mock_col1 = Mock()
            mock_col2 = Mock()
            mock_st.columns.return_value = [mock_col1, mock_col2]
            
            # Mock form submit button to return True for cancel
            mock_st.form_submit_button = Mock(side_effect=lambda label, **kwargs: label == "Cancel")
            
            auth_middleware.show_register_form()
            
            assert mock_st.session_state.get('show_register') is False
            mock_st.rerun.assert_called()

    def test_show_password_reset_form_cancel(self, auth_middleware):
        """Test password reset form cancellation."""
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {}
            
            # Mock form and button
            mock_form = Mock()
            mock_form.__enter__ = Mock(return_value=None)
            mock_form.__exit__ = Mock(return_value=None)
            mock_st.form.return_value = mock_form
            
            # Mock columns
            mock_col1 = Mock()
            mock_col2 = Mock()
            mock_st.columns.return_value = [mock_col1, mock_col2]
            
            # Mock form submit button to return True for cancel
            mock_st.form_submit_button = Mock(side_effect=lambda label, **kwargs: label == "Cancel")
            
            auth_middleware.show_password_reset_form()
            
            assert mock_st.session_state.get('show_reset') is False
            mock_st.rerun.assert_called()

    def test_show_user_menu_logout(self, auth_middleware):
        """Test user menu logout."""
        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = "Test User"
        
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {'auth_token': 'valid_token'}
            
            # Mock sidebar
            mock_sidebar = Mock()
            mock_sidebar.__enter__ = Mock(return_value=None)
            mock_sidebar.__exit__ = Mock(return_value=None)
            mock_st.sidebar = mock_sidebar
            
            # Mock button to return True for logout
            mock_st.button = Mock(side_effect=lambda label: label == "🚪 Logout")
            
            auth_middleware.auth_service.validate_token.return_value = mock_user
            auth_middleware.show_user_menu()
            
            auth_middleware.auth_service.logout_user.assert_called_once_with('valid_token')

    def test_show_profile_settings_cancel(self, auth_middleware):
        """Test profile settings cancellation."""
        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = "Test User"
        
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {'auth_token': 'valid_token'}
            
            # Mock form and button
            mock_form = Mock()
            mock_form.__enter__ = Mock(return_value=None)
            mock_form.__exit__ = Mock(return_value=None)
            mock_st.form.return_value = mock_form
            
            # Mock form submit button to return True for cancel
            mock_st.form_submit_button = Mock(side_effect=lambda label, **kwargs: label == "Cancel")
            
            auth_middleware.auth_service.validate_token.return_value = mock_user
            auth_middleware.show_profile_settings()
            
            assert mock_st.session_state.get('show_profile') is False
            mock_st.rerun.assert_called()

    def test_show_change_password_form_cancel(self, auth_middleware):
        """Test change password form cancellation."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "test_user_123"
        
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {'auth_token': 'valid_token'}
            
            # Mock form and button
            mock_form = Mock()
            mock_form.__enter__ = Mock(return_value=None)
            mock_form.__exit__ = Mock(return_value=None)
            mock_st.form.return_value = mock_form
            
            # Mock form submit button to return True for cancel
            mock_st.form_submit_button = Mock(side_effect=lambda label, **kwargs: label == "Cancel")
            
            auth_middleware.auth_service.validate_token.return_value = mock_user
            auth_middleware.show_change_password_form()
            
            assert mock_st.session_state.get('show_change_password') is False
            mock_st.rerun.assert_called()

    def test_show_change_password_form_validation_errors(self, auth_middleware):
        """Test change password form validation errors."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "test_user_123"
        
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {'auth_token': 'valid_token'}
            
            # Mock form and button
            mock_form = Mock()
            mock_form.__enter__ = Mock(return_value=None)
            mock_form.__exit__ = Mock(return_value=None)
            mock_st.form.return_value = mock_form
            
            # Mock form submit button to return True for change password
            mock_st.form_submit_button = Mock(side_effect=lambda label, **kwargs: label == "Change Password")
            
            # Mock session state with empty fields
            mock_st.session_state.update({
                'old_password': '',
                'new_password': '',
                'confirm_new': ''
            })
            
            auth_middleware.auth_service.validate_token.return_value = mock_user
            auth_middleware.show_change_password_form()
            
            # Should show error for empty fields
            mock_st.error.assert_called_with("Please fill in all fields.")

    def test_show_change_password_form_failure(self, auth_middleware):
        """Test change password form with authentication failure."""
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "test_user_123"
        
        with patch('auth.middleware.st') as mock_st:
            # Mock session state
            mock_st.session_state = {'auth_token': 'valid_token'}
            
            # Mock form and button
            mock_form = Mock()
            mock_form.__enter__ = Mock(return_value=None)
            mock_form.__exit__ = Mock(return_value=None)
            mock_st.form.return_value = mock_form
            
            # Mock form submit button to return True for change password
            mock_st.form_submit_button = Mock(side_effect=lambda label, **kwargs: label == "Change Password")
            
            # Mock session state with valid fields
            mock_st.session_state.update({
                'old_password': 'OldPass123',
                'new_password': 'NewPass123',
                'confirm_new': 'NewPass123'
            })
            
            # Mock auth service to return failure
            auth_middleware.auth_service.change_password.return_value = AuthResult(
                success=False,
                error_message="Authentication failed"
            )
            
            auth_middleware.auth_service.validate_token.return_value = mock_user
            auth_middleware.show_change_password_form()
            
            # Should show error message
            mock_st.error.assert_called_with("Password change failed: Authentication failed")


class TestUserModelAdditionalCoverage:
    """Additional tests for UserModel to improve coverage."""

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository."""
        mock_repo = Mock()
        mock_repo.save.return_value = True
        mock_repo.find_by_id.return_value = None
        mock_repo.find_by_email.return_value = None
        mock_repo.find_all.return_value = []
        return mock_repo

    @pytest.fixture
    def user_model_with_mocks(self, mock_user_repo):
        """Create user model with mocked dependencies."""
        with patch('auth.user_model.UserRepository', return_value=mock_user_repo):
            model = UserModel()
            model.user_repo = mock_user_repo
            return model

    def test_create_user_save_failure(self, user_model_with_mocks, mock_user_repo):
        """Test user creation when save fails."""
        mock_user_repo.save.return_value = False
        
        with patch('auth.user_model.User.create') as mock_create:
            mock_create.return_value = Mock()
            
            with pytest.raises(ValueError, match="Failed to create user account"):
                user_model_with_mocks.create_user(
                    email="test@example.com",
                    password="TestPass123",
                    full_name="Test User"
                )

    def test_update_user_save_failure(self, user_model_with_mocks, mock_user_repo):
        """Test user update when save fails."""
        mock_db_user = Mock()
        mock_db_user.updated_at = None
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = False
        
        updates = {'full_name': 'New Name'}
        result = user_model_with_mocks.update_user("user_123", updates)
        
        assert result is False

    def test_change_password_save_failure(self, user_model_with_mocks, mock_user_repo):
        """Test password change when save fails."""
        mock_db_user = Mock()
        mock_db_user.password_hash = bcrypt.hashpw("OldPass123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        mock_db_user.updated_at = None
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = False
        
        result = user_model_with_mocks.change_password("user_123", "OldPass123", "NewPass123")
        
        assert result is False

    def test_initiate_password_reset_save_failure(self, user_model_with_mocks, mock_user_repo):
        """Test password reset initiation when save fails."""
        mock_db_user = Mock()
        mock_db_user.password_reset_token = None
        mock_db_user.password_reset_expires = None
        mock_db_user.updated_at = None
        
        mock_user_repo.find_by_email.return_value = mock_db_user
        mock_user_repo.save.return_value = False
        
        token = user_model_with_mocks.initiate_password_reset("test@example.com")
        
        assert token is None

    def test_deactivate_user_save_failure(self, user_model_with_mocks, mock_user_repo):
        """Test user deactivation when save fails."""
        mock_db_user = Mock()
        mock_db_user.status = UserStatus.ACTIVE
        mock_db_user.updated_at = None
        
        mock_user_repo.find_by_id.return_value = mock_db_user
        mock_user_repo.save.return_value = False
        
        result = user_model_with_mocks.deactivate_user("user_123")
        
        assert result is False

    def test_migrate_legacy_data_no_files(self, user_model_with_mocks):
        """Test legacy data migration when no files exist."""
        # Should not raise any exceptions
        user_model_with_mocks._migrate_legacy_data()

    def test_migrate_legacy_data_exception_handling(self, user_model_with_mocks):
        """Test exception handling during legacy data migration."""
        with patch('os.path.exists', side_effect=Exception("File error")):
            # Should not raise any exceptions
            user_model_with_mocks._migrate_legacy_data()

    def test_migrate_legacy_data_database_not_empty(self, user_model_with_mocks, mock_user_repo):
        """Test legacy data migration when database is not empty."""
        mock_user_repo.find_all.return_value = [Mock()]  # Database has data
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=FileNotFoundError("No file")):
                # Should not raise any exceptions
                user_model_with_mocks._migrate_legacy_data()