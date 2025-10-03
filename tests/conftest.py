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

# Configure asyncio for testing - proper initialization
def pytest_configure(config):
    """Configure pytest for async testing."""
    pytest_asyncio.plugin.pytest_configure(config)


class TestEnvironmentManager:
    """Manages isolated test environments to prevent interference."""

    def __init__(self):
        self.temp_dir: Path = Path(tempfile.mkdtemp(prefix="ai_therapist_test_"))
        self.original_env: Dict[str, str] = {}
        self.test_env_file: Path = self.temp_dir / ".env.test"
        self.voice_profiles_dir: Path = self.temp_dir / "voice_profiles"
        self.credentials_dir: Path = self.temp_dir / "credentials"
        self.logs_dir: Path = self.temp_dir / "test_logs"
        self.vectorstore_dir: Path = self.temp_dir / "test_vectorstore"

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

        # Create test environment file
        test_env_content = {
            'OPENAI_API_KEY': 'sk-test-openai-api-key-for-testing-purposes-only-not-functional',
            'ELEVENLABS_API_KEY': 'sk-test-elevenlabs-api-key-for-testing-purposes-only-not-functional',
            'ELEVENLABS_VOICE_ID': 'test_voice_id_for_testing',
            'GOOGLE_CLOUD_PROJECT_ID': 'test-project-id-for-testing',
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
            'VECTORSTORE_PATH': str(self.vectorstore_dir)
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
    config.get_missing_api_keys.return_value = []
    config.validate_configuration.return_value = []
    return config


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
    from voice.audio_processor import AudioData

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
    """Set up logging for individual tests."""
    log_file = isolated_test_env.logs_dir / "test_output.log"

    # Configure test-specific logging
    test_logger = logging.getLogger('test_session')
    test_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in test_logger.handlers[:]:
        test_logger.removeHandler(handler)

    # Add file handler for test logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    test_logger.addHandler(file_handler)

    # Store log file path for test access
    test_logger.log_file = log_file

    yield test_logger

    # Cleanup: remove test log file after test
    if log_file.exists():
        log_file.unlink(missing_ok=True)


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
