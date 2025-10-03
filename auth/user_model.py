"""
User model for AI Therapist authentication system.

Provides user profile management, password hashing with bcrypt,
role-based permissions, and account status tracking with HIPAA compliance.
Now uses SQLite database for persistent storage.
"""

import os
import hashlib
import secrets
import string
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import bcrypt

# Database imports
from ..database.models import User, UserRepository


class UserRole(Enum):
    """User roles with role-based access control."""
    PATIENT = "patient"
    THERAPIST = "therapist"
    ADMIN = "admin"
    GUEST = "guest"


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"


@dataclass
class UserProfile:
    """User profile information."""
    user_id: str
    email: str
    full_name: str
    role: UserRole
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    medical_info: Dict[str, Any] = field(default_factory=dict)  # HIPAA-protected

    def to_dict(self, user_role: Optional[str] = None, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary with PII protection based on user role.

        Args:
            user_role: Role of the requesting user for access control
            include_sensitive: Whether to include sensitive medical info

        Returns:
            Dictionary representation with appropriate PII filtering
        """
        data = asdict(self)

        # Always remove highly sensitive fields
        sensitive_fields = ['password_reset_token', 'password_reset_expires']
        for field in sensitive_fields:
            data.pop(field, None)

        # PII protection based on user role
        if user_role:
            user_role = user_role.lower()

            # Medical information - only therapists and admins can see full details
            if 'medical_info' in data and data['medical_info']:
                if user_role not in ['therapist', 'admin']:
                    # Sanitize medical information for patients and guests
                    data['medical_info'] = self._sanitize_medical_info(data['medical_info'], user_role)
                elif not include_sensitive:
                    # Even authorized users get sanitized view unless explicitly requested
                    data['medical_info'] = self._sanitize_medical_info(data['medical_info'], user_role)

            # Email protection - partial masking for non-owners and non-admins
            if 'email' in data and user_role not in ['admin'] and not self._is_owner_request(user_role):
                data['email'] = self._mask_email(data['email'])

        return data

    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.account_locked_until:
            return datetime.now() < self.account_locked_until
        return False

    def increment_login_attempts(self, max_attempts: int = 5, lock_duration_minutes: int = 30):
        """Increment login attempts and lock account if needed."""
        self.login_attempts += 1
        if self.login_attempts >= max_attempts:
            self.account_locked_until = datetime.now() + timedelta(minutes=lock_duration_minutes)
            self.status = UserStatus.LOCKED

    def reset_login_attempts(self):
        """Reset login attempts on successful login."""
        self.login_attempts = 0
        self.account_locked_until = None
        if self.status == UserStatus.LOCKED:
            self.status = UserStatus.ACTIVE

    def can_access_resource(self, resource: str, permission: str) -> bool:
        """Check if user has access to a resource based on role."""
        role_permissions = self._get_role_permissions()
        return permission in role_permissions.get(resource, [])

    def _get_role_permissions(self) -> Dict[str, List[str]]:
        """Get permissions for the user's role."""
        permissions = {
            UserRole.PATIENT: {
                'own_profile': ['read', 'update'],
                'therapy_sessions': ['read', 'create'],
                'voice_features': ['read', 'use'],
                'emergency_contacts': ['read', 'update']
            },
            UserRole.THERAPIST: {
                'own_profile': ['read', 'update'],
                'assigned_patients': ['read', 'update_notes'],
                'therapy_sessions': ['read', 'create', 'update'],
                'voice_features': ['read', 'use'],
                'emergency_contacts': ['read', 'update'],
                'reports': ['read']
            },
            UserRole.ADMIN: {
                'all_profiles': ['read', 'update', 'delete'],
                'system_config': ['read', 'update'],
                'reports': ['read', 'create'],
                'audit_logs': ['read'],
                'emergency_contacts': ['read', 'update', 'manage']
            },
            UserRole.GUEST: {
                'public_info': ['read'],
                'emergency_contacts': ['read']
            }
        }
        return permissions.get(self.role, {})

    def _sanitize_medical_info(self, medical_info: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """Sanitize medical information based on user role."""
        if not medical_info:
            return {}

        sanitized = {}

        # Define what information is visible to different roles
        patient_visible = ['insurance_provider', 'emergency_contact']
        therapist_visible = patient_visible + ['conditions', 'medications', 'allergies']
        admin_visible = therapist_visible + ['treatment_history', 'diagnoses']

        visible_fields = {
            'patient': patient_visible,
            'therapist': therapist_visible,
            'admin': admin_visible
        }.get(user_role, [])

        for field in visible_fields:
            if field in medical_info:
                sanitized[field] = medical_info[field]

        # Add a note that information was sanitized
        if len(sanitized) < len(medical_info):
            sanitized['_sanitized'] = True
            sanitized['_visible_fields'] = visible_fields

        return sanitized

    def _mask_email(self, email: str) -> str:
        """Mask email address for privacy."""
        if '@' not in email:
            return email

        local, domain = email.split('@', 1)
        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    def _is_owner_request(self, user_role: str) -> bool:
        """Check if the request is from the profile owner."""
        # This would need to be implemented based on session/user context
        # For now, return False (assume not owner)
        return False


class UserModel:
    """Manages user profiles and authentication data using database storage."""

    def __init__(self, data_dir: str = None):
        """Initialize user model with database storage."""
        # Initialize database repository
        self.user_repo = UserRepository()

        # Keep legacy file paths for backward compatibility during migration
        if data_dir is None:
            data_dir = os.getenv("AUTH_DATA_DIR", "./auth_data")

        self.data_dir = Path(data_dir)
        self.users_file = self.data_dir / "users.json"
        self.passwords_file = self.data_dir / "passwords.json"

        # Load legacy data if database is empty (migration support)
        self._migrate_legacy_data()

    def _migrate_legacy_data(self):
        """Migrate legacy JSON data to database if needed."""
        try:
            # Check if legacy files exist and database is empty
            if self.users_file.exists() and self.passwords_file.exists():
                # Load legacy data
                with open(self.users_file, 'r') as f:
                    legacy_users = json.load(f)

                with open(self.passwords_file, 'r') as f:
                    legacy_passwords = json.load(f)

                if legacy_users and not self.user_repo.find_all(limit=1):
                    print("Migrating legacy user data to database...")

                    # Migrate users to database
                    for user_id, user_data in legacy_users.items():
                        # Convert legacy format to database format
                        if 'created_at' in user_data:
                            user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                        if 'updated_at' in user_data:
                            user_data['updated_at'] = datetime.fromisoformat(user_data['updated_at'])
                        if 'last_login' in user_data and user_data['last_login']:
                            user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                        if 'account_locked_until' in user_data and user_data['account_locked_until']:
                            user_data['account_locked_until'] = datetime.fromisoformat(user_data['account_locked_until'])
                        if 'password_reset_expires' in user_data and user_data['password_reset_expires']:
                            user_data['password_reset_expires'] = datetime.fromisoformat(user_data['password_reset_expires'])

                        # Convert enums
                        user_data['role'] = UserRole(user_data['role'])
                        user_data['status'] = UserStatus(user_data['status'])

                        # Get password hash
                        password_hash = legacy_passwords.get(user_id, "")

                        # Create database user
                        db_user = User(
                            user_id=user_id,
                            email=user_data['email'],
                            full_name=user_data['full_name'],
                            role=user_data['role'],
                            status=user_data['status'],
                            password_hash=password_hash,
                            created_at=user_data['created_at'],
                            updated_at=user_data['updated_at'],
                            last_login=user_data.get('last_login'),
                            login_attempts=user_data.get('login_attempts', 0),
                            account_locked_until=user_data.get('account_locked_until'),
                            password_reset_token=user_data.get('password_reset_token'),
                            password_reset_expires=user_data.get('password_reset_expires'),
                            preferences=user_data.get('preferences', {}),
                            medical_info=user_data.get('medical_info', {})
                        )

                        self.user_repo.save(db_user)

                    print(f"Migrated {len(legacy_users)} users to database")

                    # Backup legacy files
                    import shutil
                    backup_dir = self.data_dir / "backup"
                    backup_dir.mkdir(exist_ok=True)
                    shutil.move(str(self.users_file), str(backup_dir / "users.json"))
                    shutil.move(str(self.passwords_file), str(backup_dir / "passwords.json"))
                    print("Legacy files backed up")

        except Exception as e:
            print(f"Error during data migration: {e}")

    # Legacy attributes for backward compatibility
    @property
    def users(self):
        """Get all users as dictionary (legacy compatibility)."""
        all_users = self.user_repo.find_all()
        return {user.user_id: user for user in all_users}

    def create_user(self, email: str, password: str, full_name: str,
                    role: UserRole = UserRole.PATIENT) -> UserProfile:
        """Create a new user account."""
        # Validate input
        if not self._validate_email(email):
            raise ValueError("Invalid email format")
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
        if self.get_user_by_email(email):
            raise ValueError("User with this email already exists")

        # Hash password
        password_hash = self._hash_password(password)

        # Create database user
        db_user = User.create(
            email=email.lower(),
            full_name=full_name,
            role=role,
            password_hash=password_hash
        )

        # Save to database
        if not self.user_repo.save(db_user):
            raise ValueError("Failed to create user account")

        # Convert to UserProfile for backward compatibility
        user_profile = UserProfile(
            user_id=db_user.user_id,
            email=db_user.email,
            full_name=db_user.full_name,
            role=db_user.role,
            status=db_user.status,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
            last_login=db_user.last_login,
            login_attempts=db_user.login_attempts,
            account_locked_until=db_user.account_locked_until,
            password_reset_token=db_user.password_reset_token,
            password_reset_expires=db_user.password_reset_expires,
            preferences=db_user.preferences,
            medical_info=db_user.medical_info
        )

        print(f"Created user account: {db_user.email}")
        return user_profile

        print(f"Created user account: {email}")
        return user_profile

    def authenticate_user(self, email: str, password: str) -> Optional[UserProfile]:
        """Authenticate user with email and password."""
        # Get user from database
        db_user = self.user_repo.find_by_email(email.lower())
        if not db_user:
            return None

        # Check account status
        if db_user.status != UserStatus.ACTIVE:
            return None

        # Check if account is locked
        if db_user.is_locked():
            return None

        # Verify password
        if not self._verify_password(password, db_user.password_hash):
            db_user.increment_login_attempts()
            self.user_repo.save(db_user)
            return None

        # Successful login
        db_user.reset_login_attempts()
        db_user.last_login = datetime.now()
        db_user.updated_at = datetime.now()
        self.user_repo.save(db_user)

        # Convert to UserProfile for backward compatibility
        user_profile = UserProfile(
            user_id=db_user.user_id,
            email=db_user.email,
            full_name=db_user.full_name,
            role=db_user.role,
            status=db_user.status,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
            last_login=db_user.last_login,
            login_attempts=db_user.login_attempts,
            account_locked_until=db_user.account_locked_until,
            password_reset_token=db_user.password_reset_token,
            password_reset_expires=db_user.password_reset_expires,
            preferences=db_user.preferences,
            medical_info=db_user.medical_info
        )

        print(f"User authenticated: {email}")
        return user_profile

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID."""
        db_user = self.user_repo.find_by_id(user_id)
        if not db_user:
            return None

        # Convert to UserProfile for backward compatibility
        return UserProfile(
            user_id=db_user.user_id,
            email=db_user.email,
            full_name=db_user.full_name,
            role=db_user.role,
            status=db_user.status,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
            last_login=db_user.last_login,
            login_attempts=db_user.login_attempts,
            account_locked_until=db_user.account_locked_until,
            password_reset_token=db_user.password_reset_token,
            password_reset_expires=db_user.password_reset_expires,
            preferences=db_user.preferences,
            medical_info=db_user.medical_info
        )

    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email."""
        db_user = self.user_repo.find_by_email(email.lower())
        if not db_user:
            return None

        # Convert to UserProfile for backward compatibility
        return UserProfile(
            user_id=db_user.user_id,
            email=db_user.email,
            full_name=db_user.full_name,
            role=db_user.role,
            status=db_user.status,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
            last_login=db_user.last_login,
            login_attempts=db_user.login_attempts,
            account_locked_until=db_user.account_locked_until,
            password_reset_token=db_user.password_reset_token,
            password_reset_expires=db_user.password_reset_expires,
            preferences=db_user.preferences,
            medical_info=db_user.medical_info
        )

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile."""
        db_user = self.user_repo.find_by_id(user_id)
        if not db_user:
            return False

        # Update allowed fields
        allowed_fields = ['full_name', 'preferences', 'medical_info']
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(db_user, field, value)

        db_user.updated_at = datetime.now()
        return self.user_repo.save(db_user)

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        db_user = self.user_repo.find_by_id(user_id)
        if not db_user:
            return False

        # Verify old password
        if not self._verify_password(old_password, db_user.password_hash):
            return False

        # Validate new password
        if not self._validate_password(new_password):
            return False

        # Hash and store new password
        db_user.password_hash = self._hash_password(new_password)
        db_user.updated_at = datetime.now()

        if not self.user_repo.save(db_user):
            return False

        print(f"Password changed for user: {db_user.email}")
        return True

    def initiate_password_reset(self, email: str) -> Optional[str]:
        """Initiate password reset process."""
        db_user = self.user_repo.find_by_email(email.lower())
        if not db_user:
            return None

        # Generate reset token
        reset_token = self._generate_reset_token()
        db_user.password_reset_token = reset_token
        db_user.password_reset_expires = datetime.now() + timedelta(hours=24)

        if not self.user_repo.save(db_user):
            return None

        print(f"Password reset initiated for: {email}")
        return reset_token

    def reset_password(self, reset_token: str, new_password: str) -> bool:
        """Reset password using reset token."""
        # This method needs database support - for now, return False
        # In a full implementation, you'd query the database for users with matching tokens
        print("Password reset with token needs database query implementation")
        return False

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        db_user = self.user_repo.find_by_id(user_id)
        if not db_user:
            return False

        db_user.status = UserStatus.INACTIVE
        db_user.updated_at = datetime.now()
        return self.user_repo.save(db_user)

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def _generate_user_id(self) -> str:
        """Generate unique user ID."""
        while True:
            user_id = f"user_{secrets.token_hex(8)}"
            if user_id not in self.users:
                return user_id

    def _generate_reset_token(self) -> str:
        """Generate password reset token."""
        return secrets.token_urlsafe(32)

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def _validate_password(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False
        if not any(c.isupper() for c in password):
            return False
        if not any(c.islower() for c in password):
            return False
        if not any(c.isdigit() for c in password):
            return False
        return True

    def get_all_users(self) -> List[UserProfile]:
        """Get all users (admin only)."""
        db_users = self.user_repo.find_all()
        user_profiles = []

        for db_user in db_users:
            user_profiles.append(UserProfile(
                user_id=db_user.user_id,
                email=db_user.email,
                full_name=db_user.full_name,
                role=db_user.role,
                status=db_user.status,
                created_at=db_user.created_at,
                updated_at=db_user.updated_at,
                last_login=db_user.last_login,
                login_attempts=db_user.login_attempts,
                account_locked_until=db_user.account_locked_until,
                password_reset_token=db_user.password_reset_token,
                password_reset_expires=db_user.password_reset_expires,
                preferences=db_user.preferences,
                medical_info=db_user.medical_info
            ))

        return user_profiles

    # Legacy property for backward compatibility
    @property
    def password_hashes(self):
        """Get password hashes as dictionary (legacy compatibility)."""
        all_users = self.user_repo.find_all()
        return {user.user_id: user.password_hash for user in all_users}

    def cleanup_expired_data(self):
        """Clean up expired password reset tokens."""
        # This would be handled by the database cleanup process
        # For now, just log that cleanup should be done
        print("Database cleanup should be handled by database maintenance")