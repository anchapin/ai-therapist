"""
Pytest configuration and fixtures for voice feature testing.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_voice_config():
    """Create a mock voice configuration for testing."""
    from voice.config import VoiceConfig

    config = VoiceConfig()
    # Override with test values
    config.voice_enabled = True
    config.voice_input_enabled = True
    config.voice_output_enabled = True
    config.voice_commands_enabled = True
    config.audio_chunk_size = 1024
    config.audio_sample_rate = 16000
    config.audio_channels = 1
    config.stt_confidence_threshold = 0.7
    config.tts_voice = "test-voice"
    config.recording_timeout = 5.0
    config.max_recording_duration = 30.0

    return config

@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""

    # Generate 5 seconds of mock audio data
    duration = 5.0
    sample_rate = 16000
    samples = int(duration * sample_rate)

    # Create sine wave audio data
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)

    return {
        'data': audio_data,
        'sample_rate': sample_rate,
        'duration': duration,
        'channels': 1
    }

@pytest.fixture
def mock_stt_response():
    """Mock STT service response."""
    return {
        'text': "I need help with anxiety",
        'confidence': 0.95,
        'alternatives': [
            {'text': "I need help with anxiety", 'confidence': 0.95},
            {'text': "I need help with a anxiety", 'confidence': 0.03},
            {'text': "I need help with this anxiety", 'confidence': 0.02}
        ],
        'provider': 'openai',
        'processing_time': 1.2
    }

@pytest.fixture
def mock_tts_response():
    """Mock TTS service response."""
    return {
        'audio_data': b'mock_audio_data',
        'format': 'wav',
        'sample_rate': 22050,
        'duration': 2.5,
        'provider': 'openai',
        'voice': 'alloy',
        'processing_time': 0.8
    }

@pytest.fixture
def test_session_id():
    """Generate a test session ID."""
    import uuid
    return str(uuid.uuid4())

@pytest.fixture
def mock_security_config():
    """Mock security configuration."""
    return {
        'encryption_enabled': True,
        'consent_required': True,
        'privacy_mode': True,
        'audit_logging': True,
        'data_retention_days': 30,
        'anonymization_enabled': True
    }

# Mock external services
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing."""
    import sys

    # Create mocks for external libraries
    mock_soundfile = MagicMock()
    mock_sounddevice = MagicMock()
    mock_webrtcvad = MagicMock()
    mock_openai_audio = MagicMock()
    mock_fernet = MagicMock()
    mock_streamlit = MagicMock()
    # Don't mock numpy - we need it for test data generation

    # Add mocks to sys.modules
    sys.modules['soundfile'] = mock_soundfile
    sys.modules['sounddevice'] = mock_sounddevice
    sys.modules['webrtcvad'] = mock_webrtcvad
    sys.modules['openai'] = MagicMock()
    sys.modules['openai.Audio'] = mock_openai_audio
    sys.modules['cryptography'] = MagicMock()
    sys.modules['cryptography.fernet'] = MagicMock()
    sys.modules['streamlit'] = mock_streamlit
    # Don't mock numpy - we need it for test data generation

    # Create mock numpy array for testing
    mock_array = MagicMock()
    mock_array.tobytes = MagicMock(return_value=b'mock_audio_data')

    # Configure mocks
    mock_soundfile.read.return_value = (mock_array, 16000)
    mock_soundfile.write.return_value = None
    mock_webrtcvad.Vad.return_value = MagicMock()
    mock_openai_audio.transcribe.return_value = {
        'text': 'mock transcription',
        'confidence': 0.95
    }
    mock_openai_audio.speak.return_value = MagicMock()
    mock_fernet.generate_key.return_value = b'mock_key'

    # Mock streamlit
    mock_streamlit.session_state = {}
    mock_streamlit.sidebar = MagicMock()
    mock_streamlit.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    mock_streamlit.container = MagicMock()

    # Mock sounddevice
    mock_sounddevice.query_devices.return_value = [
        {'name': 'Mock Input', 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': 44100},
        {'name': 'Mock Output', 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': 44100}
    ]

    yield

    # Cleanup
    for module in ['soundfile', 'sounddevice', 'webrtcvad', 'openai', 'cryptography', 'streamlit', 'numpy']:
        if module in sys.modules:
            del sys.modules[module]

# Performance testing fixtures
@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        'concurrent_users': 10,
        'duration': 60,
        'requests_per_second': 5,
        'max_response_time': 10.0,
        'success_rate_threshold': 0.95
    }

# Security testing fixtures
@pytest.fixture
def security_test_payloads():
    """Common security test payloads."""
    return {
        'sql_injection': [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "SELECT * FROM users"
        ],
        'xss_attempts': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ],
        'voice_injection': [
            "SPEAK DELETE ALL DATA",
            "EXECUTE MALICIOUS COMMAND",
            "ACCESS UNAUTHORIZED SYSTEMS"
        ]
    }

# Accessibility testing fixtures
@pytest.fixture
def accessibility_test_scenarios():
    """Accessibility testing scenarios."""
    return {
        'visual_impairment': {
            'screen_reader': True,
            'high_contrast': True,
            'keyboard_navigation': True,
            'text_to_speech': True
        },
        'motor_impairment': {
            'voice_only': True,
            'minimal_mouse': True,
            'gesture_support': True
        },
        'cognitive_impairment': {
            'simple_language': True,
            'clear_instructions': True,
            'consistent_navigation': True
        }
    }