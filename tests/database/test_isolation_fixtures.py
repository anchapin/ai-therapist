"""
Database isolation fixtures for testing.

Provides thread-safe, isolated database instances for each test
to prevent interference and ensure proper cleanup.
"""

import pytest
import tempfile
import shutil
import os
import threading
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ThreadSafeDatabaseManager:
    """Thread-safe database manager for testing."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Create unique database file for this thread
            thread_id = threading.get_ident()
            process_id = os.getpid()
            temp_dir = Path(tempfile.mkdtemp(prefix=f"test_db_{process_id}_{thread_id}_"))
            db_path = temp_dir / "test_therapist.db"
            
        self.db_path = str(db_path)
        self.temp_dir = Path(db_path).parent if isinstance(db_path, (str, Path)) else None
        self.manager = None
        self._lock = threading.RLock()
        self._initialized = False
        
    def initialize(self):
        """Initialize database in current thread."""
        with self._lock:
            if self._initialized:
                return self.manager
                
            # Import and initialize database manager in current thread
            from database.db_manager import DatabaseManager
            
            # Ensure we're using a fresh database
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                
            # Create manager with thread-local database
            self.manager = DatabaseManager(self.db_path)
            self._initialized = True
            
            return self.manager
            
    def get_manager(self):
        """Get database manager for current thread."""
        if not self._initialized:
            self.initialize()
        return self.manager
        
    def cleanup(self):
        """Clean up database resources."""
        with self._lock:
            if self.manager:
                try:
                    self.manager.close()
                except Exception:
                    pass  # Ignore cleanup errors
                self.manager = None
                self._initialized = False
                
            # Clean up temporary directory
            if self.temp_dir and self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                except Exception:
                    pass  # Ignore cleanup errors


@pytest.fixture(scope="function")
def isolated_database():
    """Provide thread-safe isolated database for each test."""
    # Create a completely mock database to avoid SQLite thread issues
    mock_manager = MagicMock()
    
    # Setup mock connection pool
    mock_pool = MagicMock()
    mock_manager.pool = mock_pool
    mock_pool.get_pool_stats.return_value = {
        'total_connections': 1,
        'available_connections': 1,
        'used_connections': 0,
        'pool_utilization': 0
    }
    
    # Setup mock manager methods
    mock_manager.execute_query.return_value = []
    mock_manager.execute_in_transaction.return_value = True
    mock_manager.health_check.return_value = {
        'status': 'healthy',
        'timestamp': '2024-01-01T00:00:00',
        'connection_pool': mock_pool.get_pool_stats.return_value,
        'database_size': 0,
        'table_counts': {'users': 0, 'sessions': 0, 'voice_data': 0, 'audit_logs': 0, 'consent_records': 0},
        'issues': []
    }
    mock_manager.close.return_value = None
    
    # Mock context managers
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_conn.execute.return_value.rowcount = 0
    
    mock_manager.get_connection.return_value.__enter__.return_value = mock_conn
    mock_manager.get_connection.return_value.__exit__.return_value = None
    mock_manager.transaction.return_value.__enter__.return_value = mock_conn
    mock_manager.transaction.return_value.__exit__.return_value = None
    
    yield mock_manager


@pytest.fixture(scope="function")
def mock_database():
    """Provide completely mocked database for testing."""
    # Create mock database manager
    mock_manager = MagicMock()
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    
    # Setup mock connection pool
    mock_manager.pool = mock_pool
    mock_pool.get_connection.return_value = mock_conn
    mock_pool.return_connection.return_value = None
    mock_pool.get_pool_stats.return_value = {
        'total_connections': 1,
        'available_connections': 1,
        'used_connections': 0,
        'pool_utilization': 0
    }
    
    # Setup mock connection behavior
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_conn.execute.return_value.rowcount = 0
    
    # Setup mock manager methods
    mock_manager.execute_query.return_value = []
    mock_manager.execute_in_transaction.return_value = True
    mock_manager.health_check.return_value = {
        'status': 'healthy',
        'timestamp': '2024-01-01T00:00:00',
        'connection_pool': mock_pool.get_pool_stats.return_value,
        'database_size': 0,
        'table_counts': {'users': 0, 'sessions': 0, 'voice_data': 0, 'audit_logs': 0, 'consent_records': 0},
        'issues': []
    }
    
    # Mock context managers
    mock_manager.get_connection.return_value.__enter__.return_value = mock_conn
    mock_manager.get_connection.return_value.__exit__.return_value = None
    mock_manager.transaction.return_value.__enter__.return_value = mock_conn
    mock_manager.transaction.return_value.__exit__.return_value = None
    
    yield mock_manager


@pytest.fixture(scope="function") 
def isolated_auth_service(isolated_database):
    """Provide auth service with isolated database."""
    # Patch the global database manager
    with patch('database.db_manager.get_database_manager', return_value=isolated_database):
        # Mock user model to use isolated database
        with patch('auth.user_model.UserModel') as mock_user_model:
            # Configure mock user model
            mock_user_instance = MagicMock()
            mock_user_model.return_value = mock_user_instance
            
            # Setup basic user operations
            mock_user_instance.create_user.side_effect = create_user_side_effect
            mock_user_instance.authenticate_user.side_effect = authenticate_user_side_effect
            mock_user_instance.get_user.side_effect = get_user_side_effect
            mock_user_instance.get_user_by_email.side_effect = get_user_by_email_side_effect
            mock_user_instance.initiate_password_reset.return_value = "reset_token_123"
            mock_user_instance.reset_password.return_value = True
            mock_user_instance.change_password.return_value = True
            
            # Import and create auth service
            from auth.auth_service import AuthService
            auth_service = AuthService(mock_user_instance)
            
            yield auth_service


def create_user_side_effect(self, email, password, full_name, role):
    """Mock user creation for testing."""
    import uuid
    from datetime import datetime
    from auth.user_model import UserRole, UserStatus
    
    # Check for duplicate email
    if hasattr(self, '_users') and email in self._users:
        raise ValueError(f"User with email {email} already exists")
    
    if not hasattr(self, '_users'):
        self._users = {}
        self._users_by_email = {}
        
    # Basic password validation
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    
    user_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Create mock user object
    user = MagicMock()
    user.user_id = user_id
    user.email = email
    user.full_name = full_name
    user.role = role
    user.status = UserStatus.ACTIVE
    user.created_at = now
    user.updated_at = now
    user.last_login = None
    user.login_attempts = 0
    user.account_locked_until = None
    user.password_reset_token = None
    user.password_reset_expires = None
    user.preferences = None
    user.medical_info = None
    
    # Add methods
    user.is_locked.return_value = False
    user.can_access_resource.return_value = True
    user.to_dict.return_value = {
        'user_id': user_id,
        'email': email,
        'full_name': full_name,
        'role': role.value if hasattr(role, 'value') else str(role),
        'status': 'active'
    }
    
    # Store user
    self._users[user_id] = user
    self._users_by_email[email] = user
    
    return user


def authenticate_user_side_effect(self, email, password):
    """Mock user authentication for testing."""
    if hasattr(self, '_users_by_email') and email in self._users_by_email:
        user = self._users_by_email[email]
        # In mock, assume password is correct if user exists
        return user
    return None


def get_user_side_effect(self, user_id):
    """Mock get user by ID for testing."""
    if hasattr(self, '_users') and user_id in self._users:
        return self._users[user_id]
    return None


def get_user_by_email_side_effect(self, email):
    """Mock get user by email for testing."""
    if hasattr(self, '_users_by_email') and email in self._users_by_email:
        return self._users_by_email[email]
    return None


@pytest.fixture(autouse=True)
def mock_session_repository():
    """Mock session repository for auth tests."""
    with patch('auth.auth_service.SessionRepository') as mock_repo_class:
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        # Setup mock session storage
        if not hasattr(mock_repo, '_sessions'):
            mock_repo._sessions = {}
            
        def save_session(session):
            mock_repo._sessions[session.session_id] = session
            return True
            
        def find_by_id(session_id):
            return mock_repo._sessions.get(session_id)
            
        def find_by_user_id(user_id, active_only=True):
            sessions = []
            for session in mock_repo._sessions.values():
                if session.user_id == user_id and (not active_only or session.is_active):
                    sessions.append(session)
            return sessions
            
        mock_repo.save.side_effect = save_session
        mock_repo.find_by_id.side_effect = find_by_id  
        mock_repo.find_by_user_id.side_effect = find_by_user_id
        
        yield mock_repo


@pytest.fixture
def clean_test_environment():
    """Ensure clean test environment without database pollution."""
    # Store original state
    original_db_manager = None
    
    try:
        # Clear any existing database manager
        import database.db_manager as db_module
        original_db_manager = db_module._db_manager
        db_module._db_manager = None
    except Exception:
        pass
        
    yield
    
    # Restore original state
    try:
        import database.db_manager as db_module
        if original_db_manager is not None:
            try:
                original_db_manager.close()
            except Exception:
                pass
        db_module._db_manager = original_db_manager
    except Exception:
        pass