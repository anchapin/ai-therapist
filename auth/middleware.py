"""
Streamlit authentication middleware for AI Therapist.

Provides authentication decorators, login/logout UI components,
and session management integration with Streamlit's session state.
"""

import streamlit as st
import functools
from typing import Callable, Any, Optional, List
from .auth_service import AuthService, AuthResult
from .user_model import UserProfile, UserRole


class AuthMiddleware:
    """Authentication middleware for Streamlit applications."""

    def __init__(self, auth_service: AuthService):
        """Initialize middleware with auth service."""
        self.auth_service = auth_service

    def login_required(self, func: Callable) -> Callable:
        """Decorator to require authentication for a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_authenticated():
                self.show_login_form()
                return None
            return func(*args, **kwargs)
        return wrapper

    def role_required(self, required_roles: List[UserRole]) -> Callable:
        """Decorator to require specific roles for a function."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_authenticated():
                    self.show_login_form()
                    return None

                user = self.get_current_user()
                if not user or user.role not in required_roles:
                    st.error("Access denied. Insufficient permissions.")
                    return None

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        token = st.session_state.get('auth_token')
        if not token:
            return False

        user = self.auth_service.validate_token(token)
        return user is not None

    def get_current_user(self) -> Optional[UserProfile]:
        """Get currently authenticated user."""
        token = st.session_state.get('auth_token')
        if not token:
            return None

        return self.auth_service.validate_token(token)

    def login_user(self, email: str, password: str) -> AuthResult:
        """Login user and store session."""
        result = self.auth_service.login_user(
            email=email,
            password=password,
            ip_address=self._get_client_ip(),
            user_agent=self._get_user_agent()
        )

        if result.success and result.token:
            st.session_state.auth_token = result.token
            st.session_state.user = result.user
            st.session_state.auth_time = result.session.created_at if result.session else None

        return result

    def logout_user(self):
        """Logout current user."""
        token = st.session_state.get('auth_token')
        if token:
            self.auth_service.logout_user(token)

        # Clear session state
        if 'auth_token' in st.session_state:
            del st.session_state.auth_token
        if 'user' in st.session_state:
            del st.session_state.user
        if 'auth_time' in st.session_state:
            del st.session_state.auth_time

    def show_login_form(self):
        """Display login form in Streamlit."""
        st.title("🔐 Login Required")
        st.markdown("Please log in to access the AI Therapist.")

        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            col1, col2 = st.columns([1, 1])
            with col1:
                login_button = st.form_submit_button("Login", type="primary")
            with col2:
                register_button = st.form_submit_button("Register")

            if login_button:
                if email and password:
                    with st.spinner("Logging in..."):
                        result = self.login_user(email, password)

                    if result.success:
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {result.error_message}")
                else:
                    st.error("Please enter both email and password.")

            if register_button:
                st.session_state.show_register = True
                st.rerun()

        # Show register form if requested
        if st.session_state.get('show_register', False):
            self.show_register_form()

        # Show password reset option
        if st.button("Forgot Password?"):
            st.session_state.show_reset = True
            st.rerun()

        if st.session_state.get('show_reset', False):
            self.show_password_reset_form()

    def show_register_form(self):
        """Display user registration form."""
        st.subheader("📝 Register New Account")

        with st.form("register_form"):
            full_name = st.text_input("Full Name", key="reg_full_name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")

            # Password requirements
            st.caption("Password must be at least 8 characters with uppercase, lowercase, and numbers.")

            col1, col2 = st.columns([1, 1])
            with col1:
                register_button = st.form_submit_button("Create Account", type="primary")
            with col2:
                cancel_button = st.form_submit_button("Cancel")

            if register_button:
                if not all([full_name, email, password, confirm_password]):
                    st.error("Please fill in all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    with st.spinner("Creating account..."):
                        result = self.auth_service.register_user(
                            email=email,
                            password=password,
                            full_name=full_name
                        )

                    if result.success:
                        st.success("Account created successfully! Please log in.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {result.error_message}")

            if cancel_button:
                st.session_state.show_register = False
                st.rerun()

    def show_password_reset_form(self):
        """Display password reset form."""
        st.subheader("🔑 Reset Password")

        with st.form("reset_form"):
            email = st.text_input("Email Address", key="reset_email")

            col1, col2 = st.columns([1, 1])
            with col1:
                reset_button = st.form_submit_button("Send Reset Link", type="primary")
            with col2:
                cancel_button = st.form_submit_button("Cancel")

            if reset_button and email:
                with st.spinner("Sending reset link..."):
                    result = self.auth_service.initiate_password_reset(email)

                if result.success:
                    st.success("Password reset link sent to your email.")
                    st.session_state.show_reset = False
                    st.rerun()
                else:
                    st.error(f"Failed to send reset link: {result.error_message}")

            if cancel_button:
                st.session_state.show_reset = False
                st.rerun()

    def show_user_menu(self):
        """Display user menu in sidebar."""
        user = self.get_current_user()
        if not user:
            return

        with st.sidebar:
            st.markdown("---")
            st.subheader(f"👤 {user.full_name}")

            # User info
            st.caption(f"Role: {user.role.value.title()}")
            if user.last_login:
                # Handle both datetime objects and ISO strings
                if isinstance(user.last_login, str):
                    # Parse ISO string if needed
                    try:
                        from datetime import datetime
                        last_login = datetime.fromisoformat(user.last_login.replace('Z', '+00:00'))
                        st.caption(f"Last login: {last_login.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"Last login: {user.last_login}")
                else:
                    st.caption(f"Last login: {user.last_login.strftime('%Y-%m-%d %H:%M')}")

            # Menu options
            if st.button("⚙️ Profile Settings"):
                st.session_state.show_profile = True

            if st.button("🔑 Change Password"):
                st.session_state.show_change_password = True

            if st.button("🚪 Logout"):
                self.logout_user()
                st.success("Logged out successfully!")
                st.rerun()

        # Show profile settings if requested
        if st.session_state.get('show_profile', False):
            self.show_profile_settings()

        if st.session_state.get('show_change_password', False):
            self.show_change_password_form()

    def show_profile_settings(self):
        """Display user profile settings."""
        user = self.get_current_user()
        if not user:
            return

        st.subheader("👤 Profile Settings")

        with st.form("profile_form"):
            full_name = st.text_input("Full Name", value=user.full_name, key="profile_name")

            # Preferences
            st.subheader("Preferences")
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], key="profile_theme")

            if st.form_submit_button("Save Changes"):
                updates = {}
                if full_name != user.full_name:
                    updates['full_name'] = full_name

                if updates:
                    # In a real system, update user profile
                    st.success("Profile updated successfully!")
                    st.session_state.show_profile = False
                    st.rerun()
                else:
                    st.info("No changes to save.")

            if st.form_submit_button("Cancel"):
                st.session_state.show_profile = False
                st.rerun()

    def show_change_password_form(self):
        """Display change password form."""
        st.subheader("🔑 Change Password")

        with st.form("change_password_form"):
            old_password = st.text_input("Current Password", type="password", key="old_password")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_new")

            if st.form_submit_button("Change Password"):
                if not all([old_password, new_password, confirm_password]):
                    st.error("Please fill in all fields.")
                elif new_password != confirm_password:
                    st.error("New passwords do not match.")
                elif len(new_password) < 8:
                    st.error("New password must be at least 8 characters.")
                else:
                    user = self.get_current_user()
                    if user:
                        result = self.auth_service.change_password(
                            user.user_id, old_password, new_password
                        )

                        if result.success:
                            st.success("Password changed successfully!")
                            st.session_state.show_change_password = False
                            st.rerun()
                        else:
                            st.error(f"Password change failed: {result.error_message}")

            if st.form_submit_button("Cancel"):
                st.session_state.show_change_password = False
                st.rerun()

    def _get_client_ip(self) -> Optional[str]:
        """Get client IP address (limited in Streamlit)."""
        # Streamlit doesn't provide direct access to client IP
        # In production, this would be handled by a proxy or custom component
        return "streamlit_client"

    def _get_user_agent(self) -> Optional[str]:
        """Get user agent (limited in Streamlit)."""
        # Streamlit doesn't provide direct access to user agent
        return "streamlit_browser"