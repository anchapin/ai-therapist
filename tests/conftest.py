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
    from voice.config import VoiceConfig, AudioConfig, PerformanceConfig, SecurityConfig

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

    # Ensure audio, performance, and security configs are properly initialized
    if not hasattr(config, 'audio') or config.audio is None:
        config.audio = AudioConfig()
    if not hasattr(config, 'performance') or config.performance is None:
        config.performance = PerformanceConfig()
    if not hasattr(config, 'security') or config.security is None:
        config.security = SecurityConfig()

    return config

@pytest.fixture
def mock_audio_data():
    """Generate mock AudioData object for testing."""
    from voice.audio_processor import AudioData

    # Generate 5 seconds of mock audio data
    duration = 5.0
    sample_rate = 16000
    samples = int(duration * sample_rate)

    # Create sine wave audio data
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Convert to float32 as expected by AudioData
    audio_data = audio_data.astype(np.float32)

    # Create AudioData object as expected by the service
    return AudioData(
        data=audio_data,
        sample_rate=sample_rate,
        duration=duration,
        channels=1,
        format="float32"
    )

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

    # Configure mocks with more realistic responses
    mock_soundfile.read.return_value = (mock_array, 16000)
    mock_soundfile.write.return_value = None
    mock_webrtcvad.Vad.return_value = MagicMock()
    
    # Mock OpenAI API responses to match actual API structure
    mock_openai_transcribe_response = {
        'text': 'mock transcription',
        'language': 'en',
        'segments': [],
        'duration': 2.0
    }
    mock_openai_audio.transcribe.return_value = mock_openai_transcribe_response
    mock_openai_audio.speak.return_value = MagicMock()
    
    # Mock Google Speech API response structure
    mock_google_response = MagicMock()
    mock_google_result = MagicMock()
    mock_google_alternative = MagicMock()
    mock_google_alternative.transcript = "mock transcription"
    mock_google_alternative.confidence = 0.95
    mock_google_alternative.words = []
    mock_google_result.alternatives = [mock_google_alternative]
    mock_google_response.results = [mock_google_result]
    
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

# Integration test fixtures
@pytest.fixture
def mock_config():
    """Create mock configuration for integration testing."""
    from unittest.mock import Mock
    from voice.config import VoiceConfig, SecurityConfig, AudioConfig, PerformanceConfig

    config = Mock(spec=VoiceConfig)
    config.voice_enabled = True
    config.stt_enabled = True
    config.tts_enabled = True
    config.get_preferred_stt_service.return_value = "openai"
    config.get_preferred_tts_service.return_value = "openai"
    config.stt_language = "en"
    config.tts_language = "en"
    
    # Add proper mock security config
    config.security = Mock(spec=SecurityConfig)
    config.security.encryption_enabled = True
    config.security.consent_required = True
    config.security.data_retention_days = 30
    config.security.session_timeout_minutes = 30
    config.security.encryption_key_rotation_days = 90
    config.security.audit_logging_enabled = True
    config.security.max_login_attempts = 3
    config.security.privacy_mode = True
    config.security.hipaa_compliance_enabled = True
    
    # Add proper mock audio config
    config.audio = Mock(spec=AudioConfig)
    config.audio.sample_rate = 16000
    config.audio.channels = 1
    config.audio.chunk_size = 1024
    config.audio.format = "wav"
    config.audio.max_buffer_size = 300
    config.audio.max_memory_mb = 100
    config.audio.buffer_size = 4096
    config.audio.noise_reduction_enabled = True
    config.audio.vad_enabled = True
    
    # Add proper mock performance config
    config.performance = Mock(spec=PerformanceConfig)
    config.performance.cache_enabled = True
    config.performance.cache_size = 100
    config.performance.streaming_enabled = True
    config.performance.parallel_processing = True
    config.performance.buffer_size = 4096
    config.performance.processing_timeout = 30000
    config.performance.max_concurrent_requests = 5
    config.performance.response_timeout = 10000
    
    # Add voice_profiles attribute to prevent AttributeError
    config.voice_profiles = {}
    
    return config

@pytest.fixture
def mock_audio_config():
    """Create mock AudioConfig for testing."""
    from unittest.mock import Mock
    from voice.config import AudioConfig
    
    audio_config = Mock(spec=AudioConfig)
    audio_config.sample_rate = 16000
    audio_config.channels = 1
    audio_config.chunk_size = 1024
    audio_config.format = "wav"
    audio_config.input_device = 0
    audio_config.output_device = 0
    audio_config.noise_reduction_enabled = True
    audio_config.vad_enabled = True
    audio_config.vad_aggressiveness = 2
    audio_config.silence_threshold = 0.3
    audio_config.silence_duration_ms = 1000
    audio_config.buffer_size = 4096
    audio_config.max_buffer_size = 300
    audio_config.max_memory_mb = 100
    return audio_config

@pytest.fixture
def mock_performance_config():
    """Create mock PerformanceConfig for testing."""
    from unittest.mock import Mock
    from voice.config import PerformanceConfig
    
    performance_config = Mock(spec=PerformanceConfig)
    performance_config.cache_enabled = True
    performance_config.cache_size = 100
    performance_config.streaming_enabled = True
    performance_config.parallel_processing = True
    performance_config.buffer_size = 4096
    performance_config.processing_timeout = 30000
    performance_config.max_concurrent_requests = 5
    performance_config.response_timeout = 10000
    return performance_config

@pytest.fixture
def mock_security_config():
    """Create mock SecurityConfig for testing."""
    from unittest.mock import Mock
    from voice.config import SecurityConfig
    
    security_config = Mock(spec=SecurityConfig)
    security_config.encryption_enabled = True
    security_config.data_retention_hours = 24
    security_config.data_retention_days = 30
    security_config.session_timeout_minutes = 30
    security_config.encryption_key_rotation_days = 90
    security_config.audit_logging_enabled = True
    security_config.max_login_attempts = 3
    security_config.consent_required = True
    security_config.transcript_storage = False
    security_config.anonymization_enabled = True
    security_config.privacy_mode = True
    security_config.hipaa_compliance_enabled = True
    security_config.gdpr_compliance_enabled = True
    security_config.data_localization = True
    security_config.consent_recording = True
    security_config.emergency_protocols_enabled = True
    return security_config

@pytest.fixture
def mock_security():
    """Create mock security module for integration testing."""
    from unittest.mock import Mock, AsyncMock
    import numpy as np
    from voice.audio_processor import AudioData
    from voice.security import VoiceSecurity

    security = Mock(spec=VoiceSecurity)
    security.initialize.return_value = True
    security.process_audio = AsyncMock(return_value=AudioData(np.array([]), 16000, 0.0, 1))
    return security