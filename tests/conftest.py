"""
Pytest configuration and fixtures for voice feature testing.
Comprehensive environment isolation and cleanup for reliable testing.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
import json
import os
import sys
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Remove problematic asyncio configuration


class TestEnvironmentManager:
    """Manages isolated test environments to prevent interference."""

    def __init__(self):
        # Create unique temporary directory using process ID and thread ID
        process_id = os.getpid()
        thread_id = threading.get_ident()
        timestamp = int(time.time() * 1000000)  # microseconds
        unique_id = f"{process_id}_{thread_id}_{timestamp}"
        self.temp_dir: Path = Path(tempfile.mkdtemp(prefix=f"ai_therapist_test_{unique_id}_"))
        self.original_env: Dict[str, str] = {}
        self.test_env_file: Path = self.temp_dir / ".env.test"
        self.voice_profiles_dir: Path = self.temp_dir / "voice_profiles"
        self.credentials_dir: Path = self.temp_dir / "credentials"
        self.logs_dir: Path = self.temp_dir / "test_logs"
        self.vectorstore_dir: Path = self.temp_dir / "test_vectorstore"
        self.process_id = process_id
        self.thread_id = thread_id

    def setup_test_environment(self):
        """Set up isolated test environment."""
        # Create necessary directories
        self.voice_profiles_dir.mkdir(exist_ok=True)
        self.credentials_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.vectorstore_dir.mkdir(exist_ok=True)

        # Store original environment variables
        test_env_vars = [
            'OPENAI_API_KEY', 'ELEVENLABS_API_KEY', 'ELEVENLABS_VOICE_ID',
            'GOOGLE_CLOUD_PROJECT_ID', 'GOOGLE_CLOUD_CREDENTIALS_PATH',
            'VOICE_PROFILE_PATH', 'VOICE_LOG_LEVEL', 'VOICE_LOGGING_ENABLED',
            'VOICE_ENCRYPTION_ENABLED', 'VOICE_DATA_RETENTION_HOURS',
            'OLLAMA_HOST', 'OLLAMA_MODEL', 'OLLAMA_EMBEDDING_MODEL',
            'KNOWLEDGE_PATH', 'VECTORSTORE_PATH'
        ]

        for var in test_env_vars:
            self.original_env[var] = os.getenv(var, '')

        # Create test environment file with MOCK configuration
        test_env_content = {
            'OPENAI_API_KEY': 'mock_openai_key_for_testing',
            'ELEVENLABS_API_KEY': 'mock_elevenlabs_key_for_testing',
            'ELEVENLABS_VOICE_ID': 'mock_voice_id_for_testing',
            'GOOGLE_CLOUD_PROJECT_ID': 'mock-project-id-for-testing',
            'GOOGLE_CLOUD_CREDENTIALS_PATH': str(self.credentials_dir / 'google-cloud-credentials.json'),
            'VOICE_PROFILE_PATH': str(self.voice_profiles_dir),
            'VOICE_LOG_LEVEL': 'DEBUG',
            'VOICE_LOGGING_ENABLED': 'true',
            'VOICE_ENCRYPTION_ENABLED': 'true',
            'VOICE_DATA_RETENTION_HOURS': '1',
            'OLLAMA_HOST': 'http://localhost:11434',
            'OLLAMA_MODEL': 'llama3.2:latest',
            'OLLAMA_EMBEDDING_MODEL': 'nomic-embed-text:latest',
            'KNOWLEDGE_PATH': str(self.temp_dir / 'test_knowledge'),
            'VECTORSTORE_PATH': str(self.vectorstore_dir),
            # Force mock mode for all voice services
            'VOICE_MOCK_MODE': 'true',
            'VOICE_FORCE_MOCK_SERVICES': 'true',
            'STT_PROVIDER': 'mock',
            'TTS_PROVIDER': 'mock',
            'MOCK_AUDIO_INPUT': 'true',
            'DISABLE_REAL_API_CALLS': 'true'
        }

        with open(self.test_env_file, 'w') as f:
            for key, value in test_env_content.items():
                f.write(f"{key}={value}\n")

        # Create test Google Cloud credentials
        google_creds = {
            "type": "service_account",
            "project_id": "test-project-id-for-testing",
            "private_key_id": "test-key-id-for-testing",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest-private-key-for-testing-purposes-only\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project-id-for-testing.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project-id-for-testing.iam.gserviceaccount.com"
        }

        with open(self.credentials_dir / 'google-cloud-credentials.json', 'w') as f:
            json.dump(google_creds, f, indent=2)

    def apply_test_environment(self):
        """Apply test environment variables."""
        # Load test environment file
        if os.path.exists(self.test_env_file):
            with open(self.test_env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    def restore_environment(self):
        """Restore original environment variables."""
        for key, value in self.original_env.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def cleanup(self):
        """Clean up test environment."""
        # Restore original environment
        self.restore_environment()

        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_env_manager():
    """Session-scoped test environment manager."""
    manager = TestEnvironmentManager()
    manager.setup_test_environment()
    manager.apply_test_environment()

    yield manager

    manager.cleanup()


@pytest.fixture(scope="function")
def isolated_test_env(test_env_manager):
    """Function-scoped isolated test environment."""
    # Apply test environment for each test
    test_env_manager.apply_test_environment()
    yield test_env_manager
    # Environment cleanup is handled by session-scoped manager


@pytest.fixture
def temp_voice_profiles_dir(isolated_test_env):
    """Temporary voice profiles directory for testing."""
    return isolated_test_env.voice_profiles_dir


@pytest.fixture
def temp_credentials_dir(isolated_test_env):
    """Temporary credentials directory for testing."""
    return isolated_test_env.credentials_dir


@pytest.fixture
def temp_logs_dir(isolated_test_env):
    """Temporary logs directory for testing."""
    return isolated_test_env.logs_dir


@pytest.fixture
def temp_vectorstore_dir(isolated_test_env):
    """Temporary vectorstore directory for testing."""
    return isolated_test_env.vectorstore_dir


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API for testing."""
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock audio speech response
        mock_speech_response = MagicMock()
        mock_speech_response.content = b"test audio content"
        mock_client.audio.speech.create.return_value = mock_speech_response

        # Mock whisper transcription response
        mock_transcript_response = MagicMock()
        mock_transcript_response.text = "Test transcription"
        mock_client.audio.transcriptions.create.return_value = mock_transcript_response

        yield mock_client


@pytest.fixture
def mock_elevenlabs_api():
    """Mock ElevenLabs API for testing."""
    with patch('elevenlabs.api') as mock_elevenlabs:
        mock_tts_response = MagicMock()
        mock_tts_response.audio = b"test elevenlabs audio"
        mock_elevenlabs.generate.return_value = mock_tts_response

        yield mock_elevenlabs


@pytest.fixture
def mock_google_speech_api():
    """Mock Google Speech API for testing."""
    with patch('google.cloud.speech') as mock_speech:
        mock_client = MagicMock()
        mock_speech.SpeechClient.return_value = mock_client

        # Mock speech recognition response
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_result.transcript = "Test Google speech recognition"
        mock_result.confidence = 0.95
        mock_response.results = [mock_result]
        mock_client.recognize.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_voice_config():
    """Mock VoiceConfig for testing."""
    config = MagicMock()
    config.voice_enabled = True
    config.voice_input_enabled = True
    config.voice_output_enabled = True
    config.voice_commands_enabled = True
    config.security_enabled = True
    config.session_timeout_minutes = 30
    config.audio_sample_rate = 16000
    config.audio_channels = 1
    config.audio_chunk_size = 1024
    
    # Add comprehensive mock configurations to prevent hanging
    config.voice_profiles = {}
    config.voice_command_timeout = 30000
    config.voice_command_wake_word = "therapist"
    config.voice_command_min_confidence = 0.7
    config.voice_command_max_duration = 10000
    config.default_voice_profile = "default"
    config.stt_provider = "mock"
    config.tts_provider = "mock"
    config.stt_language = "en-US"
    config.stt_timeout = 10.0
    config.stt_max_retries = 2
    config.tts_voice = "alloy"
    config.tts_model = "tts-1"
    config.max_concurrent_sessions = 5
    config.max_concurrent_requests = 3
    config.request_timeout_seconds = 10.0
    config.cache_enabled = True
    config.cache_size_mb = 50
    config.encryption_enabled = True
    config.data_retention_days = 30
    
    # Mock nested configurations
    config.audio = MagicMock()
    config.audio.max_buffer_size = 300
    config.audio.max_memory_mb = 50
    config.audio.sample_rate = 16000
    config.audio.channels = 1
    config.audio.chunk_size = 1024
    config.audio.recording_timeout = 5.0
    config.audio.playback_enabled = True
    
    config.performance = MagicMock()
    config.performance.buffer_size = 2048
    config.performance.processing_timeout = 15000
    config.performance.max_concurrent_requests = 3
    config.performance.cache_enabled = True
    config.performance.response_timeout = 5000
    
    config.security = MagicMock()
    config.security.audit_logging_enabled = True
    config.security.session_timeout_minutes = 30
    config.security.max_login_attempts = 3
    config.security.data_retention_days = 30
    config.security.encryption_key_rotation_days = 90
    
    config.get_missing_api_keys.return_value = []
    config.validate_configuration.return_value = []
    return config


@pytest.fixture(scope="session", autouse=True)
def mock_audio_hardware():
    """Mock audio hardware dependencies to prevent CI failures."""
    # Mock pyaudio to prevent hardware dependency issues
    with patch.dict('sys.modules', {
        'pyaudio': MagicMock(),
        'sounddevice': MagicMock(),
        'soundfile': MagicMock(),
        'librosa': MagicMock(),
        'webrtcvad': MagicMock(),
        'silero_vad': MagicMock()
    }):
        # Mock pyaudio.PyAudio
        mock_pyaudio = MagicMock()
        mock_pyaudio.get_device_count.return_value = 2
        mock_pyaudio.get_device_info_by_index.return_value = {
            'name': 'Mock Audio Device',
            'maxInputChannels': 1,
            'maxOutputChannels': 2
        }
        
        # Add to sys.modules
        sys.modules['pyaudio'].PyAudio = MagicMock(return_value=mock_pyaudio)
        sys.modules['sounddevice'].default.samplerate = 16000
        sys.modules['sounddevice'].default.channels = 1
        
        yield


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    # Generate 1 second of 16kHz audio
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767
    audio_data = audio_data.astype(np.int16)

    return audio_data.tobytes()


@pytest.fixture
def mock_audio_data():
    """Mock AudioData object for testing."""
    try:
        from voice.audio_processor import AudioData
    except ImportError:
        # Fallback for testing
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mocks'))
        from mock_audio_processor import AudioData

    # Generate sample audio data
    sample_rate = 16000
    duration = 1.0
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.1  # Lower amplitude for float32
    audio_data = audio_data.astype(np.float32)

    return AudioData(
        data=audio_data,
        sample_rate=sample_rate,
        channels=1,
        format="float32",
        duration=duration
    )


@pytest.fixture
def test_voice_profile_data():
    """Sample voice profile data for testing."""
    return {
        "name": "test_therapist",
        "description": "Test therapist voice profile",
        "voice_id": "test_voice_id",
        "language": "en-US",
        "gender": "neutral",
        "age": "adult",
        "pitch": 1.0,
        "speed": 1.0,
        "volume": 0.8,
        "emotion": "calm",
        "style": "conversational"
    }


@pytest.fixture(autouse=True)
def setup_test_logging(isolated_test_env):
    """Set up logging for individual tests with unique file paths."""
    # Create unique log file name using process ID, thread ID, and timestamp
    process_id = os.getpid()
    thread_id = threading.get_ident()
    timestamp = int(time.time() * 1000000)  # microseconds
    unique_id = f"{process_id}_{thread_id}_{timestamp}"
    
    log_file = isolated_test_env.logs_dir / f"test_output_{unique_id}.log"

    # Configure test-specific logging
    test_logger = logging.getLogger(f'test_session_{unique_id}')
    test_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in test_logger.handlers[:]:
        handler.close()
        test_logger.removeHandler(handler)

    # Add file handler for test logs with proper error handling
    try:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        test_logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        # Fallback to console logging if file handler fails
        print(f"Warning: Could not create log file {log_file}: {e}")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        test_logger.addHandler(console_handler)
        log_file = None  # Mark that we're using console fallback

    # Store log file path for test access
    test_logger.log_file = log_file
    test_logger.unique_id = unique_id

    yield test_logger

    # Cleanup: close handlers and remove log file after test
    for handler in test_logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass  # Ignore errors during cleanup
        test_logger.removeHandler(handler)

    # Remove log file if it exists and was created
    if log_file and log_file.exists():
        try:
            log_file.unlink(missing_ok=True)
        except (OSError, IOError):
            pass  # Ignore errors during cleanup


@pytest.fixture
def assert_no_environment_pollution():
    """Fixture to ensure tests don't pollute the main environment."""
    initial_env = set(os.environ.keys())

    yield

    # Check for new environment variables after test
    final_env = set(os.environ.keys())
    new_vars = final_env - initial_env

    # Allow certain test-specific variables
    allowed_new_vars = {
        'PYTEST_CURRENT_TEST', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD',
        'PYTEST_PLUGINS', 'PYTEST_VERSION'
    }

    unexpected_vars = new_vars - allowed_new_vars
    if unexpected_vars:
        pytest.fail(f"Test polluted environment with new variables: {unexpected_vars}")


# Database fixtures for consistent testing across all test types

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


@pytest.fixture
def auth_service(isolated_database):
    """Provide auth service with isolated database."""
    # Patch the global database manager
    with patch('database.db_manager.get_database_manager', return_value=isolated_database):
        # Mock user model to use isolated database
        with patch('auth.user_model.UserModel') as mock_user_model:
            # Configure mock user model
            mock_user_instance = MagicMock()
            mock_user_instance._users = {}  # Initialize user storage
            mock_user_instance._users_by_email = {}  # Initialize email storage
            mock_user_model.return_value = mock_user_instance
            
            # Setup basic user operations with proper binding
            mock_user_instance.create_user.side_effect = lambda email=None, password=None, full_name=None, role=None: create_user_side_effect(mock_user_instance, email, password, full_name, role)
            mock_user_instance.authenticate_user.side_effect = lambda email=None, password=None: authenticate_user_side_effect(mock_user_instance, email, password)
            mock_user_instance.get_user.side_effect = lambda user_id=None: get_user_side_effect(mock_user_instance, user_id)
            mock_user_instance.get_user_by_email.side_effect = lambda email=None: get_user_by_email_side_effect(mock_user_instance, email)
            mock_user_instance.initiate_password_reset.return_value = "reset_token_123"
            mock_user_instance.reset_password.return_value = True
            mock_user_instance.change_password.return_value = True
            
            # Import and create auth service
            from auth.auth_service import AuthService
            auth_service = AuthService(mock_user_instance)
            
            yield auth_service


# Helper functions for auth service fixture
def create_user_side_effect(self, email=None, password=None, full_name=None, role=None):
    """Mock user creation for testing."""
    if email is None or password is None or full_name is None or role is None:
        return None
    
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


def authenticate_user_side_effect(self, email=None, password=None):
    """Mock user authentication for testing."""
    if email is None or password is None:
        return None
        
    if hasattr(self, '_users_by_email') and email in self._users_by_email:
        user = self._users_by_email[email]
        # In mock, assume password is correct if user exists
        user.last_login = datetime.now()
        return user
    return None


def get_user_side_effect(self, user_id=None):
    """Mock get user by ID for testing."""
    if user_id is None:
        return None
        
    if hasattr(self, '_users') and user_id in self._users:
        return self._users[user_id]
    return None


def get_user_by_email_side_effect(self, email=None):
    """Mock get user by email for testing."""
    if email is None:
        return None
        
    if hasattr(self, '_users_by_email') and email in self._users_by_email:
        return self._users_by_email[email]
    return None


# Additional utility fixtures for common test scenarios

@pytest.fixture
def minimal_voice_config():
    """Minimal VoiceConfig for basic testing."""
    from voice.config import VoiceConfig
    config = VoiceConfig()
    config.voice_enabled = True
    config.voice_input_enabled = True
    config.voice_output_enabled = True
    config.voice_commands_enabled = False  # Disable for minimal config
    return config


@pytest.fixture
def secure_voice_config():
    """VoiceConfig with security features enabled."""
    from voice.config import VoiceConfig
    config = VoiceConfig()
    config.security.encryption_enabled = True
    config.security.audit_logging_enabled = True
    config.security.consent_required = True
    config.security.hipaa_compliance_enabled = True
    config.security.gdpr_compliance_enabled = True
    return config


@pytest.fixture
def performance_voice_config():
    """VoiceConfig optimized for performance testing."""
    from voice.config import VoiceConfig
    config = VoiceConfig()
    config.performance.cache_enabled = True
    config.performance.cache_size = 50
    config.performance.streaming_enabled = True
    config.performance.parallel_processing = True
    config.performance.max_concurrent_requests = 3
    return config


@pytest.fixture(autouse=True)
def force_mock_configuration(isolated_test_env):
    """Force all tests to use mock configurations instead of real APIs."""
    # Apply mock environment variables
    os.environ.update({
        'VOICE_MOCK_MODE': 'true',
        'VOICE_FORCE_MOCK_SERVICES': 'true',
        'STT_PROVIDER': 'mock',
        'TTS_PROVIDER': 'mock',
        'MOCK_AUDIO_INPUT': 'true',
        'DISABLE_REAL_API_CALLS': 'true',
        'OPENAI_API_KEY': 'mock_openai_key_for_testing',
        'ELEVENLABS_API_KEY': 'mock_elevenlabs_key_for_testing',
        'ELEVENLABS_VOICE_ID': 'mock_voice_id_for_testing',
        'GOOGLE_CLOUD_PROJECT_ID': 'mock-project-id-for-testing'
    })
    
    # Mock sounddevice and other audio dependencies
    sys.modules['sounddevice'] = MagicMock()
    sys.modules['sounddevice'].default = MagicMock()
    sys.modules['sounddevice'].rec = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
    sys.modules['sounddevice'].play = MagicMock()
    sys.modules['sounddevice'].query_devices = MagicMock(return_value=[])
    sys.modules['sounddevice'].default.device = (0, 0)
    
    # Mock pyaudio
    pyaudio_mock = MagicMock()
    pyaudio_mock.PyAudio.return_value = MagicMock()
    pyaudio_mock.PyAudio.return_value.get_device_info_by_index.return_value = {
        'name': 'Mock Device',
        'maxInputChannels': 1,
        'maxOutputChannels': 2
    }
    sys.modules['pyaudio'] = pyaudio_mock
    
    yield
    
    # Cleanup is handled by the isolated_test_env fixture


@pytest.fixture
def mock_voice_config_from_file():
    """Load mock configuration from voice/mock_config.py."""
    from voice.mock_config import create_mock_voice_config
    return create_mock_voice_config()
