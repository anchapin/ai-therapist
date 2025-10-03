"""
Database Models for AI Therapist.

Provides SQLAlchemy-style model classes for database operations,
with proper data types, relationships, and HIPAA compliance features.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import logging

from .db_manager import get_database_manager, DatabaseError
from ..auth.user_model import UserRole, UserStatus


@dataclass
class BaseModel:
    """Base model with common database operations."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings for JSON storage
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create model instance from dictionary."""
        # Convert ISO strings back to datetime objects
        for key, value in data.items():
            if key.endswith('_at') or key.endswith('_until') or key.endswith('_expires'):
                if isinstance(value, str) and value:
                    try:
                        data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        pass  # Keep as string if parsing fails
        return cls(**data)


@dataclass
class User(BaseModel):
    """User model for database storage."""

    user_id: str
    email: str
    full_name: str
    role: UserRole
    status: UserStatus
    password_hash: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    medical_info: Dict[str, Any] = field(default_factory=dict)  # HIPAA protected

    @classmethod
    def create(cls, email: str, full_name: str, role: UserRole, password_hash: str) -> 'User':
        """Create a new user."""
        now = datetime.now()
        user_id = f"user_{now.strftime('%Y%m%d_%H%M%S')}_{hash(email) % 10000:04d}"

        return cls(
            user_id=user_id,
            email=email.lower(),
            full_name=full_name,
            role=role,
            status=UserStatus.ACTIVE,
            password_hash=password_hash,
            created_at=now,
            updated_at=now
        )

    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.account_locked_until:
            return datetime.now() < self.account_locked_until
        return False

    def increment_login_attempts(self, max_attempts: int = 5, lock_duration_minutes: int = 30):
        """Increment login attempts and lock if needed."""
        self.login_attempts += 1
        if self.login_attempts >= max_attempts:
            self.account_locked_until = datetime.now() + timedelta(minutes=lock_duration_minutes)
            self.status = UserStatus.LOCKED

    def reset_login_attempts(self):
        """Reset login attempts."""
        self.login_attempts = 0
        self.account_locked_until = None
        if self.status == UserStatus.LOCKED:
            self.status = UserStatus.ACTIVE


@dataclass
class Session(BaseModel):
    """Session model for database storage."""

    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

    @classmethod
    def create(cls, user_id: str, session_timeout_minutes: int = 30,
               ip_address: str = None, user_agent: str = None) -> 'Session':
        """Create a new session."""
        now = datetime.now()
        session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}_{hash(user_id + str(now)) % 1000000:06d}"

        expires_at = now + timedelta(minutes=session_timeout_minutes)

        return cls(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at

    def extend(self, minutes: int = 30):
        """Extend session expiration."""
        self.expires_at = datetime.now() + timedelta(minutes=minutes)


@dataclass
class VoiceData(BaseModel):
    """Voice data model for encrypted voice recordings and metadata."""

    data_id: str
    user_id: str
    session_id: Optional[str]
    data_type: str  # 'recording', 'transcription', 'analysis'
    encrypted_data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    retention_until: Optional[datetime] = None
    is_deleted: bool = False

    @classmethod
    def create(cls, user_id: str, data_type: str, encrypted_data: bytes,
               session_id: str = None, metadata: Dict[str, Any] = None,
               retention_days: int = 30) -> 'VoiceData':
        """Create new voice data entry."""
        now = datetime.now()
        data_id = f"voice_{now.strftime('%Y%m%d_%H%M%S')}_{hash(user_id + data_type + str(now)) % 1000000:06d}"

        retention_until = None
        if retention_days > 0:
            retention_until = now + timedelta(days=retention_days)

        return cls(
            data_id=data_id,
            user_id=user_id,
            session_id=session_id,
            data_type=data_type,
            encrypted_data=encrypted_data,
            metadata=metadata or {},
            created_at=now,
            retention_until=retention_until
        )

    def is_expired(self) -> bool:
        """Check if data has expired based on retention policy."""
        if self.retention_until:
            return datetime.now() > self.retention_until
        return False


@dataclass
class AuditLog(BaseModel):
    """Audit log entry for HIPAA compliance tracking."""

    log_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "INFO"

    @classmethod
    def create(cls, event_type: str, user_id: str = None, session_id: str = None,
               details: Dict[str, Any] = None, severity: str = "INFO") -> 'AuditLog':
        """Create new audit log entry."""
        now = datetime.now()
        log_id = f"log_{now.strftime('%Y%m%d_%H%M%S')}_{hash(str(details) + str(now)) % 1000000:06d}"

        return cls(
            log_id=log_id,
            timestamp=now,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            severity=severity
        )


@dataclass
class ConsentRecord(BaseModel):
    """Consent record for patient consent management."""

    consent_id: str
    user_id: str
    consent_type: str
    granted: bool
    timestamp: datetime
    version: str
    details: Dict[str, Any] = field(default_factory=dict)
    revoked_at: Optional[datetime] = None

    @classmethod
    def create(cls, user_id: str, consent_type: str, granted: bool,
               version: str = "1.0", details: Dict[str, Any] = None) -> 'ConsentRecord':
        """Create new consent record."""
        now = datetime.now()
        consent_id = f"consent_{now.strftime('%Y%m%d_%H%M%S')}_{hash(user_id + consent_type + version) % 1000000:06d}"

        return cls(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=now,
            version=version,
            details=details or {}
        )

    def revoke(self):
        """Revoke this consent."""
        self.revoked_at = datetime.now()
        self.granted = False

    def is_active(self) -> bool:
        """Check if consent is currently active."""
        return self.granted and self.revoked_at is None


class UserRepository:
    """Repository for User model operations."""

    def __init__(self):
        self.db = get_database_manager()
        self.logger = logging.getLogger(__name__)

    def save(self, user: User) -> bool:
        """Save user to database."""
        try:
            with self.db.transaction() as conn:
                # Convert model to database format
                data = user.to_dict()
                data['role'] = user.role.value
                data['status'] = user.status.value
                data['preferences'] = json.dumps(user.preferences)
                data['medical_info'] = json.dumps(user.medical_info)

                # Insert or replace user
                conn.execute('''
                    INSERT OR REPLACE INTO users
                    (user_id, email, full_name, role, status, created_at, updated_at,
                     last_login, login_attempts, account_locked_until,
                     password_reset_token, password_reset_expires,
                     preferences, medical_info, password_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['user_id'], data['email'], data['full_name'], data['role'], data['status'],
                    data['created_at'], data['updated_at'], data['last_login'],
                    data['login_attempts'], data['account_locked_until'],
                    data['password_reset_token'], data['password_reset_expires'],
                    data['preferences'], data['medical_info'], data['password_hash']
                ))

            return True
        except Exception as e:
            self.logger.error(f"Failed to save user {user.user_id}: {e}")
            return False

    def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID."""
        try:
            result = self.db.execute_query(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
                fetch=True
            )

            if result:
                data = result[0]
                # Convert database format to model
                data['role'] = UserRole(data['role'])
                data['status'] = UserStatus(data['status'])
                data['preferences'] = json.loads(data['preferences'] or '{}')
                data['medical_info'] = json.loads(data['medical_info'] or '{}')

                return User.from_dict(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to find user {user_id}: {e}")
            return None

    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        try:
            result = self.db.execute_query(
                "SELECT * FROM users WHERE email = ?",
                (email.lower(),),
                fetch=True
            )

            if result:
                data = result[0]
                # Convert database format to model
                data['role'] = UserRole(data['role'])
                data['status'] = UserStatus(data['status'])
                data['preferences'] = json.loads(data['preferences'] or '{}')
                data['medical_info'] = json.loads(data['medical_info'] or '{}')

                return User.from_dict(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to find user by email {email}: {e}")
            return None

    def find_all(self, status: UserStatus = None, limit: int = 100) -> List[User]:
        """Find all users with optional status filter."""
        try:
            if status:
                result = self.db.execute_query(
                    "SELECT * FROM users WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                    fetch=True
                )
            else:
                result = self.db.execute_query(
                    "SELECT * FROM users ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                    fetch=True
                )

            users = []
            for data in result:
                # Convert database format to model
                data['role'] = UserRole(data['role'])
                data['status'] = UserStatus(data['status'])
                data['preferences'] = json.loads(data['preferences'] or '{}')
                data['medical_info'] = json.loads(data['medical_info'] or '{}')
                users.append(User.from_dict(data))

            return users
        except Exception as e:
            self.logger.error(f"Failed to find users: {e}")
            return []

    def update(self, user: User) -> bool:
        """Update user in database."""
        user.updated_at = datetime.now()
        return self.save(user)

    def delete(self, user_id: str) -> bool:
        """Delete user from database."""
        try:
            self.db.execute_query(
                "DELETE FROM users WHERE user_id = ?",
                (user_id,)
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {e}")
            return False


class SessionRepository:
    """Repository for Session model operations."""

    def __init__(self):
        self.db = get_database_manager()
        self.logger = logging.getLogger(__name__)

    def save(self, session: Session) -> bool:
        """Save session to database."""
        try:
            with self.db.transaction() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO sessions
                    (session_id, user_id, created_at, expires_at, ip_address, user_agent, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id, session.user_id, session.created_at.isoformat(),
                    session.expires_at.isoformat(), session.ip_address, session.user_agent,
                    session.is_active
                ))
            return True
        except Exception as e:
            self.logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    def find_by_id(self, session_id: str) -> Optional[Session]:
        """Find session by ID."""
        try:
            result = self.db.execute_query(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
                fetch=True
            )

            if result:
                return Session.from_dict(result[0])
            return None
        except Exception as e:
            self.logger.error(f"Failed to find session {session_id}: {e}")
            return None

    def find_by_user_id(self, user_id: str, active_only: bool = True) -> List[Session]:
        """Find sessions by user ID."""
        try:
            if active_only:
                result = self.db.execute_query(
                    "SELECT * FROM sessions WHERE user_id = ? AND is_active = 1 ORDER BY created_at DESC",
                    (user_id,),
                    fetch=True
                )
            else:
                result = self.db.execute_query(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                    fetch=True
                )

            return [Session.from_dict(data) for data in result]
        except Exception as e:
            self.logger.error(f"Failed to find sessions for user {user_id}: {e}")
            return []

    def update(self, session: Session) -> bool:
        """Update session in database."""
        return self.save(session)

    def delete_expired(self) -> int:
        """Delete expired sessions."""
        try:
            result = self.db.execute_query(
                "DELETE FROM sessions WHERE expires_at < ? OR is_active = 0",
                (datetime.now().isoformat(),)
            )
            return result.rowcount if hasattr(result, 'rowcount') else 0
        except Exception as e:
            self.logger.error(f"Failed to delete expired sessions: {e}")
            return 0


class VoiceDataRepository:
    """Repository for VoiceData model operations."""

    def __init__(self):
        self.db = get_database_manager()
        self.logger = logging.getLogger(__name__)

    def save(self, voice_data: VoiceData) -> bool:
        """Save voice data to database."""
        try:
            with self.db.transaction() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO voice_data
                    (data_id, user_id, session_id, data_type, encrypted_data, metadata,
                     created_at, retention_until, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    voice_data.data_id, voice_data.user_id, voice_data.session_id,
                    voice_data.data_type, voice_data.encrypted_data,
                    json.dumps(voice_data.metadata), voice_data.created_at.isoformat(),
                    voice_data.retention_until.isoformat() if voice_data.retention_until else None,
                    voice_data.is_deleted
                ))
            return True
        except Exception as e:
            self.logger.error(f"Failed to save voice data {voice_data.data_id}: {e}")
            return False

    def find_by_id(self, data_id: str) -> Optional[VoiceData]:
        """Find voice data by ID."""
        try:
            result = self.db.execute_query(
                "SELECT * FROM voice_data WHERE data_id = ? AND is_deleted = 0",
                (data_id,),
                fetch=True
            )

            if result:
                data = result[0]
                data['metadata'] = json.loads(data['metadata'] or '{}')
                return VoiceData.from_dict(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to find voice data {data_id}: {e}")
            return None

    def find_by_user_id(self, user_id: str, data_type: str = None, limit: int = 50) -> List[VoiceData]:
        """Find voice data by user ID."""
        try:
            if data_type:
                result = self.db.execute_query(
                    "SELECT * FROM voice_data WHERE user_id = ? AND data_type = ? AND is_deleted = 0 ORDER BY created_at DESC LIMIT ?",
                    (user_id, data_type, limit),
                    fetch=True
                )
            else:
                result = self.db.execute_query(
                    "SELECT * FROM voice_data WHERE user_id = ? AND is_deleted = 0 ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                    fetch=True
                )

            voice_data_list = []
            for data in result:
                data['metadata'] = json.loads(data['metadata'] or '{}')
                voice_data_list.append(VoiceData.from_dict(data))

            return voice_data_list
        except Exception as e:
            self.logger.error(f"Failed to find voice data for user {user_id}: {e}")
            return []

    def mark_as_deleted(self, data_id: str) -> bool:
        """Soft delete voice data."""
        try:
            self.db.execute_query(
                "UPDATE voice_data SET is_deleted = 1 WHERE data_id = ?",
                (data_id,)
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to mark voice data {data_id} as deleted: {e}")
            return False


class AuditLogRepository:
    """Repository for AuditLog model operations."""

    def __init__(self):
        self.db = get_database_manager()
        self.logger = logging.getLogger(__name__)

    def save(self, audit_log: AuditLog) -> bool:
        """Save audit log to database."""
        try:
            with self.db.transaction() as conn:
                conn.execute('''
                    INSERT INTO audit_logs
                    (log_id, timestamp, event_type, user_id, session_id, details, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_log.log_id, audit_log.timestamp.isoformat(), audit_log.event_type,
                    audit_log.user_id, audit_log.session_id, json.dumps(audit_log.details),
                    audit_log.severity
                ))
            return True
        except Exception as e:
            self.logger.error(f"Failed to save audit log {audit_log.log_id}: {e}")
            return False

    def find_by_user_id(self, user_id: str, limit: int = 100) -> List[AuditLog]:
        """Find audit logs by user ID."""
        try:
            result = self.db.execute_query(
                "SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit),
                fetch=True
            )

            audit_logs = []
            for data in result:
                data['details'] = json.loads(data['details'] or '{}')
                audit_logs.append(AuditLog.from_dict(data))

            return audit_logs
        except Exception as e:
            self.logger.error(f"Failed to find audit logs for user {user_id}: {e}")
            return []

    def find_by_date_range(self, start_date: datetime, end_date: datetime,
                          event_type: str = None) -> List[AuditLog]:
        """Find audit logs by date range."""
        try:
            if event_type:
                result = self.db.execute_query(
                    "SELECT * FROM audit_logs WHERE timestamp BETWEEN ? AND ? AND event_type = ? ORDER BY timestamp DESC",
                    (start_date.isoformat(), end_date.isoformat(), event_type),
                    fetch=True
                )
            else:
                result = self.db.execute_query(
                    "SELECT * FROM audit_logs WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC",
                    (start_date.isoformat(), end_date.isoformat()),
                    fetch=True
                )

            audit_logs = []
            for data in result:
                data['details'] = json.loads(data['details'] or '{}')
                audit_logs.append(AuditLog.from_dict(data))

            return audit_logs
        except Exception as e:
            self.logger.error(f"Failed to find audit logs by date range: {e}")
            return []


class ConsentRepository:
    """Repository for ConsentRecord model operations."""

    def __init__(self):
        self.db = get_database_manager()
        self.logger = logging.getLogger(__name__)

    def save(self, consent: ConsentRecord) -> bool:
        """Save consent record to database."""
        try:
            with self.db.transaction() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO consent_records
                    (consent_id, user_id, consent_type, granted, timestamp, version, details, revoked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    consent.consent_id, consent.user_id, consent.consent_type, consent.granted,
                    consent.timestamp.isoformat(), consent.version, json.dumps(consent.details),
                    consent.revoked_at.isoformat() if consent.revoked_at else None
                ))
            return True
        except Exception as e:
            self.logger.error(f"Failed to save consent {consent.consent_id}: {e}")
            return False

    def find_by_user_id(self, user_id: str, consent_type: str = None) -> List[ConsentRecord]:
        """Find consent records by user ID."""
        try:
            if consent_type:
                result = self.db.execute_query(
                    "SELECT * FROM consent_records WHERE user_id = ? AND consent_type = ? ORDER BY timestamp DESC",
                    (user_id, consent_type),
                    fetch=True
                )
            else:
                result = self.db.execute_query(
                    "SELECT * FROM consent_records WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id,),
                    fetch=True
                )

            consents = []
            for data in result:
                data['details'] = json.loads(data['details'] or '{}')
                consents.append(ConsentRecord.from_dict(data))

            return consents
        except Exception as e:
            self.logger.error(f"Failed to find consents for user {user_id}: {e}")
            return []

    def has_active_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has active consent for a type."""
        try:
            result = self.db.execute_query(
                "SELECT granted FROM consent_records WHERE user_id = ? AND consent_type = ? AND revoked_at IS NULL ORDER BY timestamp DESC LIMIT 1",
                (user_id, consent_type),
                fetch=True
            )

            if result:
                return bool(result[0]['granted'])
            return False
        except Exception as e:
            self.logger.error(f"Failed to check consent for user {user_id}: {e}")
            return False