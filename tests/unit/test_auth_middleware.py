"""
Comprehensive unit tests for auth/middleware.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Mock streamlit to avoid import issues
with patch.dict('sys.modules', {'streamlit': Mock()}):
    from auth.middleware import AuthMiddleware
    from auth.auth_service import AuthService, AuthResult
    from auth.user_model import UserProfile, UserRole


class TestAuthMiddleware:
    """Test AuthMiddleware functionality."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Create a mock auth service."""
        auth_service = Mock(spec=AuthService)
        return auth_service
    
    @pytest.fixture
    def mock_streamlit(self):
        """Create a mock streamlit module."""
        st_mock = Mock()
        st_mock.session_state = {}
        return st_mock
    
    @pytest.fixture
    def auth_middleware(self, mock_auth_service):
        """Create auth middleware with mocked dependencies."""
        with patch('auth.middleware.st') as mock_st:
            middleware = AuthMiddleware(mock_auth_service)
            middleware.st = mock_st
            return middleware
    
    def test_auth_middleware_initialization(self, mock_auth_service):
        """Test auth middleware initialization."""
        with patch('auth.middleware.st') as mock_st:
            middleware = AuthMiddleware(mock_auth_service)
            
            assert middleware.auth_service == mock_auth_service
            assert hasattr(middleware, 'st')
    
    def test_login_required_authenticated(self, auth_middleware, mock_streamlit):
        """Test login_required decorator when user is authenticated."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        # Mock the is_authenticated method
        with patch.object(auth_middleware, 'is_authenticated', return_value=True):
            @auth_middleware.login_required
            def protected_function():
                return "protected_content"
            
            result = protected_function()
            
            assert result == "protected_content"
    
    def test_login_required_not_authenticated(self, auth_middleware, mock_streamlit):
        """Test login_required decorator when user is not authenticated."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        # Mock the is_authenticated and show_login_form methods
        with patch.object(auth_middleware, 'is_authenticated', return_value=False):
            with patch.object(auth_middleware, 'show_login_form') as mock_show_login:
                @auth_middleware.login_required
                def protected_function():
                    return "protected_content"
                
                result = protected_function()
                
                assert result is None
                mock_show_login.assert_called_once()
    
    def test_role_required_success(self, auth_middleware, mock_streamlit):
        """Test role_required decorator when user has required role."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.THERAPIST
        
        # Mock the is_authenticated and get_current_user methods
        with patch.object(auth_middleware, 'is_authenticated', return_value=True):
            with patch.object(auth_middleware, 'get_current_user', return_value=mock_user):
                @auth_middleware.role_required([UserRole.THERAPIST, UserRole.ADMIN])
                def protected_function():
                    return "protected_content"
                
                result = protected_function()
                
                assert result == "protected_content"
    
    def test_role_required_not_authenticated(self, auth_middleware, mock_streamlit):
        """Test role_required decorator when user is not authenticated."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        # Mock the is_authenticated and show_login_form methods
        with patch.object(auth_middleware, 'is_authenticated', return_value=False):
            with patch.object(auth_middleware, 'show_login_form') as mock_show_login:
                @auth_middleware.role_required([UserRole.ADMIN])
                def protected_function():
                    return "protected_content"
                
                result = protected_function()
                
                assert result is None
                mock_show_login.assert_called_once()
    
    def test_role_required_insufficient_permissions(self, auth_middleware, mock_streamlit):
        """Test role_required decorator when user lacks required role."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        mock_user.role = UserRole.PATIENT
        
        # Mock the is_authenticated, get_current_user, and st.error methods
        with patch.object(auth_middleware, 'is_authenticated', return_value=True):
            with patch.object(auth_middleware, 'get_current_user', return_value=mock_user):
                with patch.object(auth_middleware.st, 'error') as mock_error:
                    @auth_middleware.role_required([UserRole.ADMIN])
                    def protected_function():
                        return "protected_content"
                    
                    result = protected_function()
                    
                    assert result is None
                    mock_error.assert_called_once_with("Access denied. Insufficient permissions.")
    
    def test_is_authenticated_true(self, auth_middleware, mock_streamlit):
        """Test is_authenticated when user is authenticated."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.is_authenticated()
        
        assert result is True
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_is_authenticated_false_no_token(self, auth_middleware, mock_streamlit):
        """Test is_authenticated when no token is present."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.is_authenticated()
        
        assert result is False
        auth_middleware.auth_service.validate_token.assert_not_called()
    
    def test_is_authenticated_false_invalid_token(self, auth_middleware, mock_streamlit):
        """Test is_authenticated when token is invalid."""
        # Mock streamlit session state with invalid token
        mock_streamlit.session_state = {'auth_token': 'invalid_token'}
        
        auth_middleware.auth_service.validate_token.return_value = None
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.is_authenticated()
        
        assert result is False
        auth_middleware.auth_service.validate_token.assert_called_once_with('invalid_token')
    
    def test_get_current_user_success(self, auth_middleware, mock_streamlit):
        """Test get_current_user when user is authenticated."""
        # Mock streamlit session state with valid token
        mock_streamlit.session_state = {'auth_token': 'valid_token'}
        
        mock_user = Mock(spec=UserProfile)
        auth_middleware.auth_service.validate_token.return_value = mock_user
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.get_current_user()
        
        assert result == mock_user
        auth_middleware.auth_service.validate_token.assert_called_once_with('valid_token')
    
    def test_get_current_user_no_token(self, auth_middleware, mock_streamlit):
        """Test get_current_user when no token is present."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.get_current_user()
        
        assert result is None
        auth_middleware.auth_service.validate_token.assert_not_called()
    
    def test_login_user_success(self, auth_middleware, mock_streamlit):
        """Test successful user login."""
        # Mock streamlit session state
        mock_streamlit.session_state = {}
        
        mock_user = Mock(spec=UserProfile)
        mock_session = Mock()
        mock_session.created_at = datetime.now()
        
        auth_result = AuthResult(
            success=True,
            user=mock_user,
            token="jwt_token_123",
            session=mock_session
        )
        
        auth_middleware.auth_service.login_user.return_value = auth_result
        
        # Mock the helper methods
        with patch.object(auth_middleware, '_get_client_ip', return_value="192.168.1.1"):
            with patch.object(auth_middleware, '_get_user_agent', return_value="Mozilla/5.0"):
                # Override the st reference
                auth_middleware.st.session_state = mock_streamlit.session_state
                
                result = auth_middleware.login_user("test@example.com", "SecurePass123")
        
        assert result == auth_result
        assert mock_streamlit.session_state['auth_token'] == "jwt_token_123"
        assert mock_streamlit.session_state['user'] == mock_user
        assert mock_streamlit.session_state['auth_time'] == mock_session.created_at
        
        auth_middleware.auth_service.login_user.assert_called_once_with(
            email="test@example.com",
            password="SecurePass123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
    
    def test_login_user_failure(self, auth_middleware, mock_streamlit):
        """Test failed user login."""
        # Mock streamlit session state
        mock_streamlit.session_state = {}
        
        auth_result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        auth_middleware.auth_service.login_user.return_value = auth_result
        
        # Mock the helper methods
        with patch.object(auth_middleware, '_get_client_ip', return_value="192.168.1.1"):
            with patch.object(auth_middleware, '_get_user_agent', return_value="Mozilla/5.0"):
                # Override the st reference
                auth_middleware.st.session_state = mock_streamlit.session_state
                
                result = auth_middleware.login_user("test@example.com", "wrongpassword")
        
        assert result == auth_result
        assert 'auth_token' not in mock_streamlit.session_state
        assert 'user' not in mock_streamlit.session_state
        assert 'auth_time' not in mock_streamlit.session_state
    
    def test_logout_user_success(self, auth_middleware, mock_streamlit):
        """Test successful user logout."""
        # Mock streamlit session state with token
        mock_streamlit.session_state = {
            'auth_token': 'jwt_token_123',
            'user': Mock(spec=UserProfile),
            'auth_time': datetime.now()
        }
        
        auth_middleware.auth_service.logout_user.return_value = True
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.logout_user()
        
        assert result is True
        assert 'auth_token' not in mock_streamlit.session_state
        assert 'user' not in mock_streamlit.session_state
        assert 'auth_time' not in mock_streamlit.session_state
        
        auth_middleware.auth_service.logout_user.assert_called_once_with('jwt_token_123')
    
    def test_logout_user_no_token(self, auth_middleware, mock_streamlit):
        """Test logout when no token is present."""
        # Mock streamlit session state without token
        mock_streamlit.session_state = {}
        
        # Override the st reference
        auth_middleware.st.session_state = mock_streamlit.session_state
        
        result = auth_middleware.logout_user()
        
        assert result is True  # Should still succeed
        auth_middleware.auth_service.logout_user.assert_not_called()
    
    def test_show_login_form(self, auth_middleware):
        """Test showing login form."""
        # Mock streamlit components
        auth_middleware.st.title = Mock()
        auth_middleware.st.markdown = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        auth_middleware.st.button = Mock()
        auth_middleware.st.caption = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons
        mock_col1.form_submit_button.return_value = False
        mock_col2.form_submit_button.return_value = False
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["", ""]
        
        # Mock button
        auth_middleware.st.button.return_value = False
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        auth_middleware.show_login_form()
        
        # Verify basic form elements are called
        auth_middleware.st.title.assert_called_once_with("🔐 Login Required")
        auth_middleware.st.markdown.assert_called_once_with("Please log in to access the AI Therapist.")
        auth_middleware.st.form.assert_called_once_with("login_form")
        auth_middleware.st.button.assert_called_once_with("Forgot Password?")
    
    def test_show_login_form_with_login(self, auth_middleware):
        """Test login form with successful login."""
        # Mock streamlit components
        auth_middleware.st.title = Mock()
        auth_middleware.st.markdown = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        auth_middleware.st.button = Mock()
        auth_middleware.st.caption = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons - login button clicked
        mock_col1.form_submit_button.return_value = True
        mock_col2.form_submit_button.return_value = False
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["test@example.com", "SecurePass123"]
        
        # Mock button
        auth_middleware.st.button.return_value = False
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock successful login
        mock_user = Mock(spec=UserProfile)
        mock_session = Mock()
        mock_session.created_at = datetime.now()
        
        auth_result = AuthResult(
            success=True,
            user=mock_user,
            token="jwt_token_123",
            session=mock_session
        )
        
        auth_middleware.login_user = Mock(return_value=auth_result)
        
        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = Mock(return_value=None)
        mock_spinner.__exit__ = Mock(return_value=None)
        auth_middleware.st.spinner.return_value = mock_spinner
        
        auth_middleware.show_login_form()
        
        # Verify login was attempted
        auth_middleware.login_user.assert_called_once_with("test@example.com", "SecurePass123")
        auth_middleware.st.success.assert_called_once_with("Login successful!")
        auth_middleware.st.rerun.assert_called_once()
    
    def test_show_login_form_with_login_error(self, auth_middleware):
        """Test login form with login error."""
        # Mock streamlit components
        auth_middleware.st.title = Mock()
        auth_middleware.st.markdown = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        auth_middleware.st.button = Mock()
        auth_middleware.st.caption = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons - login button clicked
        mock_col1.form_submit_button.return_value = True
        mock_col2.form_submit_button.return_value = False
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["test@example.com", "wrongpassword"]
        
        # Mock button
        auth_middleware.st.button.return_value = False
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock failed login
        auth_result = AuthResult(
            success=False,
            error_message="Invalid credentials"
        )
        
        auth_middleware.login_user = Mock(return_value=auth_result)
        
        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = Mock(return_value=None)
        mock_spinner.__exit__ = Mock(return_value=None)
        auth_middleware.st.spinner.return_value = mock_spinner
        
        auth_middleware.show_login_form()
        
        # Verify login was attempted
        auth_middleware.login_user.assert_called_once_with("test@example.com", "wrongpassword")
        auth_middleware.st.error.assert_called_once_with("Login failed: Invalid credentials")
    
    def test_show_register_form(self, auth_middleware):
        """Test showing register form."""
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        auth_middleware.st.caption = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons
        mock_col1.form_submit_button.return_value = False
        mock_col2.form_submit_button.return_value = False
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["", "", "", ""]
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        auth_middleware.show_register_form()
        
        # Verify basic form elements are called
        auth_middleware.st.subheader.assert_called_once_with("📝 Register New Account")
        auth_middleware.st.form.assert_called_once_with("register_form")
        auth_middleware.st.caption.assert_called_once_with("Password must be at least 8 characters with uppercase, lowercase, and numbers.")
    
    def test_show_register_form_with_registration(self, auth_middleware):
        """Test register form with successful registration."""
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        auth_middleware.st.caption = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons - register button clicked
        mock_col1.form_submit_button.return_value = True
        mock_col2.form_submit_button.return_value = False
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["Test User", "test@example.com", "SecurePass123", "SecurePass123"]
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock successful registration
        mock_user = Mock(spec=UserProfile)
        auth_result = AuthResult(
            success=True,
            user=mock_user
        )
        
        auth_middleware.auth_service.register_user = Mock(return_value=auth_result)
        
        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = Mock(return_value=None)
        mock_spinner.__exit__ = Mock(return_value=None)
        auth_middleware.st.spinner.return_value = mock_spinner
        
        auth_middleware.show_register_form()
        
        # Verify registration was attempted
        auth_middleware.auth_service.register_user.assert_called_once_with(
            email="test@example.com",
            password="SecurePass123",
            full_name="Test User"
        )
        auth_middleware.st.success.assert_called_once_with("Account created successfully! Please log in.")
        assert auth_middleware.st.session_state['show_register'] is False
        auth_middleware.st.rerun.assert_called_once()
    
    def test_show_password_reset_form(self, auth_middleware):
        """Test showing password reset form."""
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons
        mock_col1.form_submit_button.return_value = False
        mock_col2.form_submit_button.return_value = False
        
        # Mock text input
        auth_middleware.st.text_input.return_value = ""
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        auth_middleware.show_password_reset_form()
        
        # Verify basic form elements are called
        auth_middleware.st.subheader.assert_called_once_with("🔑 Reset Password")
        auth_middleware.st.form.assert_called_once_with("reset_form")
    
    def test_show_password_reset_form_with_reset(self, auth_middleware):
        """Test password reset form with successful reset initiation."""
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.columns = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.spinner = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        auth_middleware.st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock form submit buttons - reset button clicked
        mock_col1.form_submit_button.return_value = True
        mock_col2.form_submit_button.return_value = False
        
        # Mock text input
        auth_middleware.st.text_input.return_value = "test@example.com"
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock successful password reset initiation
        auth_result = AuthResult(success=True)
        auth_middleware.auth_service.initiate_password_reset = Mock(return_value=auth_result)
        
        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = Mock(return_value=None)
        mock_spinner.__exit__ = Mock(return_value=None)
        auth_middleware.st.spinner.return_value = mock_spinner
        
        auth_middleware.show_password_reset_form()
        
        # Verify password reset was attempted
        auth_middleware.auth_service.initiate_password_reset.assert_called_once_with("test@example.com")
        auth_middleware.st.success.assert_called_once_with("Password reset link sent to your email.")
        assert auth_middleware.st.session_state['show_reset'] is False
        auth_middleware.st.rerun.assert_called_once()
    
    def test_show_user_menu(self, auth_middleware):
        """Test showing user menu."""
        # Mock user
        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.PATIENT
        mock_user.last_login = datetime.now()
        
        # Mock streamlit components
        auth_middleware.st.sidebar = Mock()
        auth_middleware.st.markdown = Mock()
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.caption = Mock()
        auth_middleware.st.button = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock the get_current_user method
        with patch.object(auth_middleware, 'get_current_user', return_value=mock_user):
            auth_middleware.show_user_menu()
        
        # Verify menu elements are called
        auth_middleware.st.sidebar.markdown.assert_called_once_with("---")
        auth_middleware.st.sidebar.subheader.assert_called_once_with("👤 Test User")
        auth_middleware.st.sidebar.caption.assert_called()
        auth_middleware.st.sidebar.button.assert_called()
    
    def test_show_user_menu_logout(self, auth_middleware):
        """Test user menu logout functionality."""
        # Mock user
        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.PATIENT
        mock_user.last_login = datetime.now()
        
        # Mock streamlit components
        auth_middleware.st.sidebar = Mock()
        auth_middleware.st.markdown = Mock()
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.caption = Mock()
        auth_middleware.st.button = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock the get_current_user and logout_user methods
        with patch.object(auth_middleware, 'get_current_user', return_value=mock_user):
            with patch.object(auth_middleware, 'logout_user') as mock_logout:
                # Mock logout button clicked
                auth_middleware.st.sidebar.button.side_effect = [False, False, True]  # Profile, Change Password, Logout
                
                auth_middleware.show_user_menu()
        
        # Verify logout was called
        mock_logout.assert_called_once()
        auth_middleware.st.success.assert_called_once_with("Logged out successfully!")
        auth_middleware.st.rerun.assert_called_once()
    
    def test_show_profile_settings(self, auth_middleware):
        """Test showing profile settings."""
        # Mock user
        mock_user = Mock(spec=UserProfile)
        mock_user.full_name = "Test User"
        mock_user.preferences = {"theme": "light"}
        
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.selectbox = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.info = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock form submit buttons
        auth_middleware.st.form_submit_button.side_effect = [False, True]  # Save, Cancel
        
        # Mock text input and selectbox
        auth_middleware.st.text_input.return_value = "Test User"
        auth_middleware.st.selectbox.return_value = "light"
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock the get_current_user method
        with patch.object(auth_middleware, 'get_current_user', return_value=mock_user):
            auth_middleware.show_profile_settings()
        
        # Verify form elements are called
        auth_middleware.st.subheader.assert_called_once_with("👤 Profile Settings")
        auth_middleware.st.form.assert_called_once_with("profile_form")
        auth_middleware.st.text_input.assert_called_once()
        auth_middleware.st.subheader.assert_called_with("Preferences")
        auth_middleware.st.selectbox.assert_called_once()
        
        # Verify cancel was processed
        assert auth_middleware.st.session_state['show_profile'] is False
        auth_middleware.st.rerun.assert_called_once()
    
    def test_show_change_password_form(self, auth_middleware):
        """Test showing change password form."""
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock form submit buttons
        auth_middleware.st.form_submit_button.side_effect = [False, True]  # Change, Cancel
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["", "", ""]
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        auth_middleware.show_change_password_form()
        
        # Verify form elements are called
        auth_middleware.st.subheader.assert_called_once_with("🔑 Change Password")
        auth_middleware.st.form.assert_called_once_with("change_password_form")
        auth_middleware.st.text_input.assert_called()
        
        # Verify cancel was processed
        assert auth_middleware.st.session_state['show_change_password'] is False
        auth_middleware.st.rerun.assert_called_once()
    
    def test_show_change_password_form_with_change(self, auth_middleware):
        """Test change password form with successful password change."""
        # Mock user
        mock_user = Mock(spec=UserProfile)
        mock_user.user_id = "user_123"
        
        # Mock streamlit components
        auth_middleware.st.subheader = Mock()
        auth_middleware.st.form = Mock()
        auth_middleware.st.text_input = Mock()
        auth_middleware.st.form_submit_button = Mock()
        auth_middleware.st.error = Mock()
        auth_middleware.st.success = Mock()
        auth_middleware.st.rerun = Mock()
        
        # Mock form context manager
        mock_form = MagicMock()
        mock_form.__enter__ = Mock(return_value=None)
        mock_form.__exit__ = Mock(return_value=None)
        auth_middleware.st.form.return_value = mock_form
        
        # Mock form submit buttons - change button clicked
        auth_middleware.st.form_submit_button.side_effect = [True, False]  # Change, Cancel
        
        # Mock text inputs
        auth_middleware.st.text_input.side_effect = ["OldPass123", "NewPass123", "NewPass123"]
        
        # Mock session state
        auth_middleware.st.session_state = {}
        
        # Mock successful password change
        auth_result = AuthResult(success=True)
        
        # Mock the get_current_user and auth_service methods
        with patch.object(auth_middleware, 'get_current_user', return_value=mock_user):
            auth_middleware.auth_service.change_password = Mock(return_value=auth_result)
            
            auth_middleware.show_change_password_form()
        
        # Verify password change was attempted
        auth_middleware.auth_service.change_password.assert_called_once_with(
            "user_123", "OldPass123", "NewPass123"
        )
        auth_middleware.st.success.assert_called_once_with("Password changed successfully!")
        assert auth_middleware.st.session_state['show_change_password'] is False
        auth_middleware.st.rerun.assert_called_once()
    
    def test_get_client_ip(self, auth_middleware):
        """Test getting client IP address."""
        result = auth_middleware._get_client_ip()
        
        assert result == "streamlit_client"
    
    def test_get_user_agent(self, auth_middleware):
        """Test getting user agent."""
        result = auth_middleware._get_user_agent()
        
        assert result == "streamlit_browser"