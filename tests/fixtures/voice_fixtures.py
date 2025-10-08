"""
Voice testing fixtures for consistent voice feature testing.

Provides fixtures for voice configuration, audio processing, and service mocking
to ensure reliable and isolated voice testing.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_voice_config():
    """Comprehensive mock VoiceConfig for testing."""
    config = MagicMock()
    
    # Basic voice settings
    config.voice_enabled = True
    config.voice_input_enabled = True
    config.voice_output_enabled = True
    config.voice_commands_enabled = True
    config.security_enabled = True
    
    # Audio settings
    config.audio_sample_rate = 16000
    config.audio_channels = 1
    config.audio_chunk_size = 1024
    config.audio_max_buffer_size = 300
    config.audio_max_memory_mb = 50
    config.audio_recording_timeout = 5.0
    config.audio_playback_enabled = True
    
    # Voice command settings
    config.voice_command_timeout = 30000
    config.voice_command_wake_word = "therapist"
    config.voice_command_min_confidence = 0.7
    config.voice_command_max_duration = 10000
    
    # STT/TTS settings
    config.default_voice_profile = "default"
    config.stt_provider = "mock"
    config.tts_provider = "mock"
    config.stt_language = "en-US"
    config.stt_timeout = 10.0
    config.stt_max_retries = 2
    config.tts_voice = "alloy"
    config.tts_model = "tts-1"
    
    # Performance settings
    config.max_concurrent_sessions = 5
    config.max_concurrent_requests = 3
    config.request_timeout_seconds = 10.0
    config.cache_enabled = True
    config.cache_size_mb = 50
    config.response_timeout = 5000
    
    # Security settings
    config.encryption_enabled = True
    config.data_retention_days = 30
    config.consent_required = True
    config.hipaa_compliance_enabled = True
    config.audit_logging_enabled = True
    
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
    
    # Mock validation methods
    config.get_missing_api_keys.return_value = []
    config.validate_configuration.return_value = []
    
    return config


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
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mocks'))
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
def mock_stt_service():
    """Mock STT service for testing."""
    with patch('voice.stt_service.STTService') as mock_stt_class:
        mock_stt = MagicMock()
        mock_stt_class.return_value = mock_stt
        
        # Mock transcription
        mock_stt.transcribe_audio.return_value = {
            'text': 'Hello, this is a test transcription.',
            'confidence': 0.95,
            'provider': 'mock'
        }
        
        # Mock real-time transcription
        mock_stt.start_real_time_transcription.return_value = True
        mock_stt.stop_real_time_transcription.return_value = True
        mock_stt.is_transcribing.return_value = False
        
        # Mock health check
        mock_stt.health_check.return_value = {
            'status': 'healthy',
            'provider': 'mock',
            'supported_languages': ['en-US', 'es-ES'],
            'model_info': {'name': 'mock-model', 'version': '1.0'}
        }
        
        yield mock_stt


@pytest.fixture
def mock_tts_service():
    """Mock TTS service for testing."""
    with patch('voice.tts_service.TTSService') as mock_tts_class:
        mock_tts = MagicMock()
        mock_tts_class.return_value = mock_tts
        
        # Mock synthesis
        mock_tts.synthesize_speech.return_value = b"mock_audio_data"
        mock_tts.synthesize_speech_stream.return_value = iter([b"chunk1", b"chunk2"])
        
        # Mock real-time synthesis
        mock_tts.start_real_time_synthesis.return_value = True
        mock_tts.stop_real_time_synthesis.return_value = True
        mock_tts.is_synthesizing.return_value = False
        
        # Mock voice management
        mock_tts.get_available_voices.return_value = [
            {'id': 'alloy', 'name': 'Alloy', 'language': 'en-US'},
            {'id': 'echo', 'name': 'Echo', 'language': 'en-US'}
        ]
        
        # Mock health check
        mock_tts.health_check.return_value = {
            'status': 'healthy',
            'provider': 'mock',
            'available_voices': 2,
            'model_info': {'name': 'mock-tts-model', 'version': '1.0'}
        }
        
        yield mock_tts


@pytest.fixture
def mock_audio_processor():
    """Mock audio processor for testing."""
    with patch('voice.audio_processor.AudioProcessor') as mock_processor_class:
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # Mock audio recording
        mock_processor.start_recording.return_value = True
        mock_processor.stop_recording.return_value = True
        mock_processor.is_recording.return_value = False
        
        # Mock audio playback
        mock_processor.play_audio.return_value = True
        mock_processor.stop_playback.return_value = True
        mock_processor.is_playing.return_value = False
        
        # Mock audio processing
        mock_processor.process_audio.return_value = mock_audio_data()
        mock_processor.apply_noise_reduction.return_value = mock_audio_data()
        
        # Mock device management
        mock_processor.get_input_devices.return_value = [
            {'id': 0, 'name': 'Default Input Device', 'channels': 1}
        ]
        mock_processor.get_output_devices.return_value = [
            {'id': 0, 'name': 'Default Output Device', 'channels': 2}
        ]
        
        # Mock health check
        mock_processor.health_check.return_value = {
            'status': 'healthy',
            'input_devices': 1,
            'output_devices': 1,
            'sample_rate': 16000,
            'buffer_size': 1024
        }
        
        yield mock_processor


@pytest.fixture
def mock_voice_service(mock_stt_service, mock_tts_service, mock_audio_processor):
    """Mock complete voice service for testing."""
    with patch('voice.voice_service.VoiceService') as mock_voice_class:
        mock_voice = MagicMock()
        mock_voice_class.return_value = mock_voice
        
        # Mock service initialization
        mock_voice.initialize.return_value = True
        mock_voice.is_initialized.return_value = True
        
        # Mock voice session management
        mock_voice.start_voice_session.return_value = {
            'session_id': 'test_session_123',
            'status': 'active',
            'created_at': '2024-01-01T00:00:00'
        }
        mock_voice.end_voice_session.return_value = True
        
        # Mock voice interaction
        mock_voice.process_voice_input.return_value = {
            'text': 'Hello therapist',
            'confidence': 0.95,
            'processing_time': 1.2
        }
        mock_voice.generate_voice_output.return_value = {
            'audio_data': b"synthesized_audio",
            'duration': 2.5,
            'provider': 'mock'
        }
        
        # Mock command processing
        mock_voice.process_voice_command.return_value = {
            'is_command': False,
            'text': 'Hello therapist'
        }
        
        # Mock health check
        mock_voice.health_check.return_value = {
            'status': 'healthy',
            'services': {
                'stt': 'healthy',
                'tts': 'healthy',
                'audio_processor': 'healthy'
            },
            'active_sessions': 0,
            'uptime': 3600
        }
        
        yield mock_voice


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


@pytest.fixture
def mock_voice_profiles():
    """Mock voice profiles for testing."""
    return {
        "default": {
            "voice_id": "alloy",
            "provider": "openai",
            "language": "en-US",
            "gender": "neutral",
            "pitch": 1.0,
            "speed": 1.0,
            "volume": 0.8
        },
        "therapist": {
            "voice_id": "nova",
            "provider": "elevenlabs",
            "language": "en-US",
            "gender": "female",
            "pitch": 0.9,
            "speed": 0.85,
            "volume": 0.7
        },
        "calm": {
            "voice_id": "echo",
            "provider": "openai",
            "language": "en-US",
            "gender": "male",
            "pitch": 0.8,
            "speed": 0.75,
            "volume": 0.6
        }
    }


@pytest.fixture
def voice_test_environment(mock_voice_config, mock_stt_service, mock_tts_service, 
                          mock_audio_processor, mock_voice_service):
    """Complete voice test environment with all services mocked."""
    return {
        'config': mock_voice_config,
        'stt_service': mock_stt_service,
        'tts_service': mock_tts_service,
        'audio_processor': mock_audio_processor,
        'voice_service': mock_voice_service
    }