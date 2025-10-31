"""
Comprehensive middleware tests to improve coverage
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from datetime import datetime, timedelta

# Import the middleware
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from auth.middleware import AuthMiddleware


class TestAuthMiddlewareCoverage:
    """Comprehensive tests for AuthMiddleware to improve coverage"""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock auth service"""
        with patch('auth.middleware.AuthService') as mock:
            yield mock
    
    @pytest.fixture
    def mock_user_model(self):
        """Mock user model"""
        with patch('auth.user_model.UserModel') as mock:
            yield mock
    
    @pytest.fixture
    def mock_st_session_state(self):
        """Mock streamlit session state"""
        with patch.object(st, 'session_state', create=True) as mock:
            mock.authenticated = False
            mock.user = None
            mock.token = None
            yield mock
    
    @pytest.fixture
    def mock_st(self):
        """Mock streamlit functions"""
        with patch('auth.middleware.st') as mock:
            # Mock all the streamlit functions used in middleware
            mock.error = Mock()
            mock.success = Mock()
            mock.info = Mock()
            mock.warning = Mock()
            mock.text_input = Mock(return_value="")
            mock.text_area = Mock(return_value="")
            mock.password_input = Mock(return_value="")
            mock.checkbox = Mock(return_value=False)
            mock.selectbox = Mock(return_value="")
            mock.button = Mock(return_value=False)
            mock.form = Mock()
            mock.form_submit_button = Mock(return_value=False)
            mock.columns = Mock(return_value=[Mock(), Mock()])
            mock.sidebar = Mock()
            mock.sidebar.title = Mock()
            mock.sidebar.header = Mock()
            mock.sidebar.text_input = Mock(return_value="")
            mock.sidebar.password_input = Mock(return_value="")
            mock.sidebar.button = Mock(return_value=False)
            mock.sidebar.checkbox = Mock(return_value=False)
            mock.sidebar.selectbox = Mock(return_value="")
            mock.sidebar.markdown = Mock()
            mock.sidebar.divider = Mock()
            mock.experimental_rerun = Mock()
            mock.rerun = Mock()
            mock.get_query_params = Mock(return_value={})
            mock.query_params = Mock()
            mock.query_params.get = Mock(return_value=None)
            yield mock
    
    def test_auth_middleware_init(self, mock_auth_service, mock_user_model):
        """Test AuthMiddleware initialization"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        assert middleware.auth_service == mock_auth_service.return_value
    
    def test_login_required_decorator(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test login_required decorator"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock the decorator function
        @middleware.login_required
        def protected_function():
            return "protected"
        
        # Test when not authenticated
        mock_st_session_state.authenticated = False
        mock_auth_service.return_value.validate_token.return_value = None
        
        with patch.object(middleware, 'show_login_form') as mock_login:
            result = protected_function()
            mock_login.assert_called_once()
        
        # Test when authenticated
        mock_st_session_state.authenticated = True
        mock_auth_service.return_value.validate_token.return_value = {"user_id": "test_user"}
        
        result = protected_function()
        assert result == "protected"
    
    def test_role_required_decorator(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test role_required decorator"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock the decorator function
        @middleware.role_required(["admin"])
        def admin_function():
            return "admin_content"
        
        # Test when not authenticated
        mock_st_session_state.authenticated = False
        mock_auth_service.return_value.validate_token.return_value = None
        
        with patch.object(middleware, 'show_login_form') as mock_login:
            result = admin_function()
            mock_login.assert_called_once()
        
        # Test when authenticated but wrong role
        mock_st_session_state.authenticated = True
        mock_st_session_state.user = {"role": "user"}
        mock_auth_service.return_value.validate_token.return_value = {"user_id": "test_user"}
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"):
            result = admin_function()
            mock_st.error.assert_called()
        
        # Test when authenticated with correct role
        mock_st_session_state.user = {"role": "admin"}
        
        result = admin_function()
        assert result == "admin_content"
    
    def test_is_authenticated(self, mock_st_session_state, mock_auth_service):
        """Test is_authenticated method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Test when authenticated
        mock_st_session_state.authenticated = True
        mock_st_session_state.token = "valid_token"
        mock_auth_service.return_value.validate_token.return_value = {"user_id": "test_user"}
        
        result = middleware.is_authenticated()
        assert result == {"user_id": "test_user"}
        
        # Test when not authenticated
        mock_st_session_state.authenticated = False
        mock_auth_service.return_value.validate_token.return_value = None
        
        result = middleware.is_authenticated()
        assert result is None
    
    def test_get_current_user(self, mock_st_session_state, mock_auth_service):
        """Test get_current_user method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Test when user exists
        mock_user = {"id": "test_id", "email": "test@example.com"}
        mock_st_session_state.user = mock_user
        
        result = middleware.get_current_user()
        assert result == mock_user
        
        # Test when no user
        mock_st_session_state.user = None
        
        result = middleware.get_current_user()
        assert result is None
    
    def test_login_user_success(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test successful login"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock successful login
        login_result = {
            "success": True,
            "user": {"id": "test_id", "email": "test@example.com"},
            "token": "test_token",
            "session_id": "test_session"
        }
        mock_auth_service.return_value.login_user.return_value = login_result
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"), \
             patch.object(middleware, '_get_user_agent', return_value="test_agent"):
            result = middleware.login_user("test@example.com", "password")
        
        assert result["success"] is True
        assert mock_st_session_state.authenticated is True
        assert mock_st_session_state.user == login_result["user"]
        assert mock_st_session_state.token == login_result["token"]
        mock_st.success.assert_called()
    
    def test_login_user_failure(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test failed login"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock failed login
        login_result = {
            "success": False,
            "error": "Invalid credentials"
        }
        mock_auth_service.return_value.login_user.return_value = login_result
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"), \
             patch.object(middleware, '_get_user_agent', return_value="test_agent"):
            result = middleware.login_user("test@example.com", "wrong_password")
        
        assert result["success"] is False
        assert mock_st_session_state.authenticated is False
        mock_st.error.assert_called()
    
    def test_logout_user(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test logout user"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Set up authenticated state
        mock_st_session_state.authenticated = True
        mock_st_session_state.user = {"id": "test_id"}
        mock_st_session_state.token = "test_token"
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"):
            middleware.logout_user()
        
        assert mock_st_session_state.authenticated is False
        assert mock_st_session_state.user is None
        assert mock_st_session_state.token is None
        mock_auth_service.return_value.logout_user.assert_called()
        mock_st.success.assert_called()
        mock_st.experimental_rerun.assert_called()
    
    def test_show_login_form(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test show_login_form method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock form inputs
        mock_st.text_input.side_effect = ["test@example.com", ""]
        mock_st.password_input.return_value = "password"
        mock_st.form_submit_button.return_value = True
        
        with patch.object(middleware, 'login_user', return_value={"success": True}) as mock_login:
            middleware.show_login_form()
        
        mock_st.form.assert_called()
        mock_login.assert_called_with("test@example.com", "password")
    
    def test_show_register_form(self, mock_st, mock_st_session_state, mock_auth_service, mock_user_model):
        """Test show_register_form method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock form inputs
        mock_st.text_input.side_effect = ["test@example.com", "Test User", ""]
        mock_st.password_input.return_value = "password"
        mock_st.form_submit_button.return_value = True
        
        with patch.object(middleware, 'login_user', return_value={"success": True}) as mock_login:
            middleware.show_register_form()
        
        mock_st.form.assert_called()
        mock_user_model.return_value.create_user.assert_called()
        mock_login.assert_called()
    
    def test_show_password_reset_form(self, mock_st, mock_auth_service, mock_user_model):
        """Test show_password_reset_form method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Mock form inputs
        mock_st.text_input.return_value = "test@example.com"
        mock_st.form_submit_button.return_value = True
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"):
            middleware.show_password_reset_form()
        
        mock_st.form.assert_called()
        mock_user_model.return_value.initiate_password_reset.assert_called()
    
    def test_show_user_menu(self, mock_st, mock_st_session_state, mock_auth_service):
        """Test show_user_menu method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Set up user
        mock_user = {"email": "test@example.com", "role": "user"}
        mock_st_session_state.user = mock_user
        
        # Mock sidebar buttons
        mock_st.sidebar.button.side_effect = [False, False, True]  # logout button
        
        with patch.object(middleware, 'logout_user') as mock_logout:
            middleware.show_user_menu()
        
        mock_st.sidebar.header.assert_called()
        mock_logout.assert_called()
    
    def test_show_profile_settings(self, mock_st, mock_st_session_state, mock_auth_service, mock_user_model):
        """Test show_profile_settings method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Set up user
        mock_user = {"id": "test_id", "email": "test@example.com"}
        mock_st_session_state.user = mock_user
        
        # Mock form inputs
        mock_st.text_input.return_value = "Test User"
        mock_st.selectbox.return_value = "user"
        mock_st.form_submit_button.return_value = True
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"):
            middleware.show_profile_settings()
        
        mock_st.form.assert_called()
        mock_user_model.return_value.update_user.assert_called()
    
    def test_show_change_password_form(self, mock_st, mock_st_session_state, mock_auth_service, mock_user_model):
        """Test show_change_password_form method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Set up user
        mock_user = {"id": "test_id"}
        mock_st_session_state.user = mock_user
        
        # Mock form inputs
        mock_st.password_input.side_effect = ["old_password", "new_password", "new_password"]
        mock_st.form_submit_button.return_value = True
        
        with patch.object(middleware, '_get_client_ip', return_value="127.0.0.1"):
            middleware.show_change_password_form()
        
        mock_st.form.assert_called()
        mock_user_model.return_value.change_password.assert_called()
    
    def test_get_client_ip(self, mock_st, mock_auth_service):
        """Test _get_client_ip method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Test with query params
        mock_st.get_query_params.return_value = {"client_ip": "192.168.1.1"}
        result = middleware._get_client_ip()
        assert result == "192.168.1.1"
        
        # Test without query params
        mock_st.get_query_params.return_value = {}
        result = middleware._get_client_ip()
        assert result == "127.0.0.1"
    
    def test_get_user_agent(self, mock_st, mock_auth_service):
        """Test _get_user_agent method"""
        middleware = AuthMiddleware(mock_auth_service.return_value)
        
        # Test with query params
        mock_st.query_params.get.return_value = "Mozilla/5.0"
        result = middleware._get_user_agent()
        assert result == "Mozilla/5.0"
        
        # Test without query params
        mock_st.query_params.get.return_value = None
        result = middleware._get_user_agent()
        assert result == "Unknown"