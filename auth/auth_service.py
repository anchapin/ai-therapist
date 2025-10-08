"""
Authentication service for AI Therapist.

Provides JWT token management, session handling, user registration/login,
and password reset functionality with HIPAA compliance.
"""

import os
import jwt
import json
import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

from .user_model import UserModel, UserProfile, UserRole, UserStatus

# Database imports - use robust import that works in both test and runtime environments
try:
    # Try relative import first (for normal package structure)
    from ..database.models import SessionRepository
except ImportError:
    try:
        # Try absolute import (for when auth is treated as top-level package)
        from database.models import SessionRepository
    except ImportError:
        # Create mock repository for testing when database is not available
        class MockRepository:
            def __init__(self):
                self.sessions = {}

            def save(self, session):
                self.sessions[session.session_id] = session
                return True

            def find_by_id(self, session_id):
                return self.sessions.get(session_id)

            def find_by_user_id(self, user_id, active_only=True):
                sessions = []
                for session in self.sessions.values():
                    if session.user_id == user_id and (not active_only or session.is_active):
                        sessions.append(session)
                return sessions

        SessionRepository = MockRepository


@dataclass
class AuthSession:
    """User authentication session."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = data['created_at'].isoformat()
        data['expires_at'] = data['expires_at'].isoformat()
        return data


@dataclass
class AuthResult:
    """Result of authentication attempt."""
    success: bool
    user: Optional[UserProfile] = None
    token: Optional[str] = None
    session: Optional[AuthSession] = None
    error_message: Optional[str] = None


class AuthService:
    """Comprehensive authentication service with JWT and session management."""

    def __init__(self, user_model: Optional[UserModel] = None):
        """Initialize authentication service."""
        # JWT configuration from environment
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", "ai-therapist-jwt-secret-change-in-production")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

        # Session configuration
        self.session_timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
        self.max_concurrent_sessions = int(os.getenv("MAX_CONCURRENT_SESSIONS", "5"))

        # User model
        self.user_model = user_model or UserModel()

        # Database session repository
        self.session_repo = SessionRepository()

        # Background cleanup
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()

    def register_user(self, email: str, password: str, full_name: str,
                     role: UserRole = UserRole.PATIENT) -> AuthResult:
        """Register a new user account."""
        try:
            # Validate role permissions (only admin can create other roles)
            if role != UserRole.PATIENT:
                # In a real system, check current user's role
                # For now, allow patient role only
                role = UserRole.PATIENT

            # Create user
            user = self.user_model.create_user(email, password, full_name, role)

            print(f"User registered successfully: {email}")
            # Don't filter user data for response - use the full UserProfile object
            return AuthResult(success=True, user=user)

        except ValueError as e:
            return AuthResult(success=False, error_message=str(e))
        except Exception as e:
            print(f"Registration error: {e}")
            return AuthResult(success=False, error_message="Registration failed")

    def login_user(self, email: str, password: str, ip_address: str = None,
                  user_agent: str = None) -> AuthResult:
        """Authenticate user and create session."""
        try:
            # Authenticate user
            user = self.user_model.authenticate_user(email, password)
            if not user:
                return AuthResult(success=False, error_message="Invalid credentials")

            # Check account status
            if user.status.value != UserStatus.ACTIVE.value:
                return AuthResult(success=False, error_message="Account is not active")

            # Check if account is locked
            if user.is_locked():
                return AuthResult(success=False, error_message="Account is temporarily locked")

            # Create session
            session = self._create_session(user.user_id, ip_address, user_agent)
            if not session:
                return AuthResult(success=False, error_message="Failed to create session")

            # Generate JWT token
            token = self._generate_jwt_token(user, session)

            print(f"User logged in successfully: {email}")
            # Don't filter user data for token generation - use the full UserProfile object
            return AuthResult(success=True, user=user, token=token, session=session)

        except Exception as e:
            print(f"Login error: {e}")
            return AuthResult(success=False, error_message="Login failed")

    def validate_token(self, token: str) -> Optional[UserProfile]:
        """Validate JWT token and return user if valid."""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Check expiration
            exp = payload.get('exp')
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                return None

            # Get user
            user_id = payload.get('user_id')
            if not user_id:
                return None

            user = self.user_model.get_user(user_id)
            if not user or user.status.value != UserStatus.ACTIVE.value:
                return None

            # Check session
            session_id = payload.get('session_id')
            if session_id and not self._is_session_valid(session_id, user_id):
                return None

            return user

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            print(f"Token validation error: {e}")
            return None

    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating session."""
        try:
            # Decode token to get session info
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm],
                               options={"verify_exp": False})

            session_id = payload.get('session_id')
            user_id = payload.get('user_id')

            if session_id:
                self._invalidate_session(session_id, user_id)

            print(f"User logged out: {user_id}")
            return True

        except Exception as e:
            print(f"Logout error: {e}")
            return False

    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh JWT token if valid."""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            user_id = payload.get('user_id')
            session_id = payload.get('session_id')

            # Validate user and session
            user = self.user_model.get_user(user_id)
            if not user or not self._is_session_valid(session_id, user_id):
                return None

            # Get session from database
            db_session = self.session_repo.find_by_id(session_id)
            if not db_session or not db_session.is_active:
                return None

            # Convert to AuthSession for token generation
            session = AuthSession(
                session_id=db_session.session_id,
                user_id=db_session.user_id,
                created_at=db_session.created_at,
                expires_at=db_session.expires_at,
                ip_address=db_session.ip_address,
                user_agent=db_session.user_agent,
                is_active=db_session.is_active
            )

            new_token = self._generate_jwt_token(user, session)
            print(f"Token refreshed for user: {user_id}")
            return new_token

        except Exception as e:
            print(f"Token refresh error: {e}")
            return None

    def initiate_password_reset(self, email: str) -> AuthResult:
        """Initiate password reset process."""
        try:
            reset_token = self.user_model.initiate_password_reset(email)
            if reset_token:
                # In a real system, send email here
                print(f"Password reset initiated for: {email}")
                return AuthResult(success=True)
            else:
                return AuthResult(success=False, error_message="User not found")

        except Exception as e:
            print(f"Password reset initiation error: {e}")
            return AuthResult(success=False, error_message="Password reset failed")

    def reset_password(self, reset_token: str, new_password: str) -> AuthResult:
        """Reset password using reset token."""
        try:
            if self.user_model.reset_password(reset_token, new_password):
                print("Password reset completed successfully")
                return AuthResult(success=True)
            else:
                return AuthResult(success=False, error_message="Invalid or expired reset token")

        except Exception as e:
            print(f"Password reset error: {e}")
            return AuthResult(success=False, error_message="Password reset failed")

    def change_password(self, user_id: str, old_password: str, new_password: str) -> AuthResult:
        """Change user password."""
        try:
            if self.user_model.change_password(user_id, old_password, new_password):
                print(f"Password changed successfully for user: {user_id}")
                return AuthResult(success=True)
            else:
                return AuthResult(success=False, error_message="Password change failed")

        except Exception as e:
            print(f"Password change error: {e}")
            return AuthResult(success=False, error_message="Password change failed")

    def get_user_sessions(self, user_id: str) -> List[AuthSession]:
        """Get all active sessions for a user."""
        db_sessions = self.session_repo.find_by_user_id(user_id, active_only=True)
        # Convert database sessions to AuthSession objects
        auth_sessions = []
        for db_session in db_sessions:
            auth_session = AuthSession(
                session_id=db_session.session_id,
                user_id=db_session.user_id,
                created_at=db_session.created_at,
                expires_at=db_session.expires_at,
                ip_address=db_session.ip_address,
                user_agent=db_session.user_agent,
                is_active=db_session.is_active
            )
            auth_sessions.append(auth_session)
        return auth_sessions

    def invalidate_user_sessions(self, user_id: str, keep_current: str = None) -> int:
        """Invalidate all sessions for a user except optionally one."""
        db_sessions = self.session_repo.find_by_user_id(user_id, active_only=True)
        invalidated = 0

        for db_session in db_sessions:
            if db_session.session_id != keep_current:
                db_session.is_active = False
                if self.session_repo.save(db_session):
                    invalidated += 1

        print(f"Invalidated {invalidated} sessions for user: {user_id}")
        return invalidated

    def validate_session_access(self, user_id: str, resource: str, permission: str) -> bool:
        """Validate if user has permission for a resource."""
        user = self.user_model.get_user(user_id)
        if not user:
            return False

        return user.can_access_resource(resource, permission)

    def _create_session(self, user_id: str, ip_address: str = None,
                        user_agent: str = None) -> Optional[AuthSession]:
        """Create a new authentication session."""
        try:
            # Check concurrent session limit
            user_sessions = self.session_repo.find_by_user_id(user_id, active_only=True)
            active_sessions = [s for s in user_sessions if s.is_active]

            if len(active_sessions) >= self.max_concurrent_sessions:
                # Remove oldest session
                oldest_session = min(active_sessions, key=lambda s: s.created_at)
                self._invalidate_session(oldest_session.session_id, user_id)

            # Create new database session
            try:
                from ..database.models import Session
            except ImportError:
                try:
                    from database.models import Session
                except ImportError:
                    # Create mock Session for testing
                    from dataclasses import dataclass
                    from datetime import datetime, timedelta

                    @dataclass
                    class Session:
                        session_id: str = ""
                        user_id: str = ""
                        created_at: datetime = None
                        expires_at: datetime = None
                        ip_address: str = None
                        user_agent: str = None
                        is_active: bool = True

                        @classmethod
                        def create(cls, user_id, session_timeout_minutes=30, ip_address=None, user_agent=None):
                            now = datetime.now()
                            import secrets
                            return cls(
                                session_id=f"session_{secrets.token_hex(8)}",
                                user_id=user_id,
                                created_at=now,
                                expires_at=now + timedelta(minutes=session_timeout_minutes),
                                ip_address=ip_address,
                                user_agent=user_agent,
                                is_active=True
                            )

                        def is_expired(self):
                            return datetime.now() > self.expires_at
            db_session = Session.create(
                user_id=user_id,
                session_timeout_minutes=self.session_timeout_minutes,
                ip_address=ip_address,
                user_agent=user_agent
            )

            # Save to database
            if not self.session_repo.save(db_session):
                return None

            # Convert to AuthSession for backward compatibility
            session = AuthSession(
                session_id=db_session.session_id,
                user_id=db_session.user_id,
                created_at=db_session.created_at,
                expires_at=db_session.expires_at,
                ip_address=db_session.ip_address,
                user_agent=db_session.user_agent,
                is_active=db_session.is_active
            )

            return session

        except Exception as e:
            print(f"Session creation error: {e}")
            return None

    def _invalidate_session(self, session_id: str, user_id: str):
        """Invalidate a session."""
        db_session = self.session_repo.find_by_id(session_id)
        if db_session:
            db_session.is_active = False
            self.session_repo.save(db_session)

    def _is_session_valid(self, session_id: str, user_id: str) -> bool:
        """Check if session is valid."""
        db_session = self.session_repo.find_by_id(session_id)
        if not db_session or not db_session.is_active or db_session.user_id != user_id:
            return False
        return not db_session.is_expired()

    def _generate_jwt_token(self, user: UserProfile, session: AuthSession) -> str:
        """Generate JWT token for user session."""
        now = datetime.now()
        expires_at = now + timedelta(hours=self.jwt_expiration_hours)

        payload = {
            'user_id': user.user_id,
            'email': user.email,
            'role': user.role.value,
            'session_id': session.session_id,
            'iat': int(now.timestamp()),
            'exp': int(expires_at.timestamp()),
            'iss': 'ai-therapist'
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{secrets.token_hex(16)}"

    def _background_cleanup(self):
        """Background thread for session cleanup."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._cleanup_expired_sessions()
            except Exception as e:
                print(f"Session cleanup error: {e}")

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        # Use database cleanup method
        from ..database.db_manager import get_database_manager
        db = get_database_manager()
        cleaned = db.cleanup_expired_data()

        if cleaned > 0:
            print(f"Cleaned up {cleaned} expired sessions and data")

    def get_auth_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        from ..database.db_manager import get_database_manager
        db = get_database_manager()
        health = db.health_check()

        return {
            'total_users': health['table_counts'].get('users', 0),
            'active_sessions': health['table_counts'].get('sessions', 0),
            'total_sessions_created': health['table_counts'].get('sessions', 0)
        }

    def _filter_user_for_response(self, user: UserProfile, requesting_user_role: str = None) -> UserProfile:
        """
        Filter user data for API responses based on requesting user's role.

        Args:
            user: The user profile to filter
            requesting_user_role: Role of the user making the request

        Returns:
            Filtered user profile
        """
        if not user:
            return user

        # Create a filtered version using the to_dict method with PII protection
        filtered_data = user.to_dict(user_role=requesting_user_role, include_sensitive=False)

        # Convert back to UserProfile-like object for compatibility
        # (Note: This creates a dict-like object, not a full UserProfile)
        return filtered_data