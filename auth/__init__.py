"""
Authentication module for AI Therapist.

Provides user authentication, session management, and JWT token handling
with HIPAA compliance and security best practices.
"""

from .auth_service import AuthService
from .user_model import UserModel, UserRole, UserStatus

__all__ = ['AuthService', 'UserModel', 'UserRole', 'UserStatus']