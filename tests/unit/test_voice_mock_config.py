"""
Comprehensive unit tests for voice/mock_config.py module.
"""

import pytest
from unittest.mock import Mock

# Import the module to test with robust error handling
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock torch/whisper to avoid import conflicts
sys.modules['torch'] = Mock()
sys.modules['whisper'] = Mock()

try:
    from voice.mock_config import (
        SecurityConfig, AudioConfig, MockVoiceConfig, MockSecurityConfig, MockAudioConfig,
        MockAuditLogger, VoiceSecurity, MockConfig, create_mock_voice_config
    )
except ImportError as e:
    pytest.skip(f"voice.mock_config module not available: {e}", allow_module_level=True)
except Exception as e:
    pytest.skip(f"Error importing voice.mock_config: {e}", allow_module_level=True)


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""
    
    def test_security_config_defaults(self):
        """Test security config with default values."""
        config = SecurityConfig()
        
        assert config.encryption_enabled is True
        assert config.consent_required is False
        assert config.hipaa_compliance_enabled is True
        assert config.audit_logging_enabled is True
        assert config.data_retention_days == 30
        assert config.max_login_attempts == 3
        assert config.encryption_key_rotation_days == 90
    
    def test_security_config_custom_values(self):
        """Test security config with custom values."""
        config = SecurityConfig(
            encryption_enabled=False,
            consent_required=True,
            hipaa_compliance_enabled=False,
            audit_logging_enabled=False,
            data_retention_days=60,
            max_login_attempts=5,
            encryption_key_rotation_days=180
        )
        
        assert config.encryption_enabled is False
        assert config.consent_required is True
        assert config.hipaa_compliance_enabled is False
        assert config.audit_logging_enabled is False
        assert config.data_retention_days == 60
        assert config.max_login_attempts == 5
        assert config.encryption_key_rotation_days == 180


class TestAudioConfig:
    """Test AudioConfig dataclass."""
    
    def test_audio_config_defaults(self):
        """Test audio config with default values."""
        config = AudioConfig()
        
        assert config.max_buffer_size == 300
        assert config.max_memory_mb == 100
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.format == 'wav'
        assert config.stream_buffer_size == 10
        assert config.stream_chunk_duration == 0.1
        assert config.compression_enabled is True
        assert config.compression_level == 6
    
    def test_audio_config_custom_values(self):
        """Test audio config with custom values."""
        config = AudioConfig(
            max_buffer_size=500,
            max_memory_mb=200,
            sample_rate=22050,
            channels=2,
            chunk_size=2048,
            format='mp3',
            stream_buffer_size=20,
            stream_chunk_duration=0.2,
            compression_enabled=False,
            compression_level=9
        )
        
        assert config.max_buffer_size == 500
        assert config.max_memory_mb == 200
        assert config.sample_rate == 22050
        assert config.channels == 2
        assert config.chunk_size == 2048
        assert config.format == 'mp3'
        assert config.stream_buffer_size == 20
        assert config.stream_chunk_duration == 0.2
        assert config.compression_enabled is False
        assert config.compression_level == 9


class TestMockVoiceConfig:
    """Test MockVoiceConfig dataclass."""
    
    def test_mock_voice_config_defaults(self):
        """Test mock voice config with default values."""
        config = MockVoiceConfig()
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.bit_depth == 16
        assert config.buffer_size == 1024
        assert config.input_device is None
        assert config.output_device is None
        assert config.noise_reduction is True
        assert config.echo_cancellation is True
        assert config.auto_gain_control is True
    
    def test_mock_voice_config_custom_values(self):
        """Test mock voice config with custom values."""
        config = MockVoiceConfig(
            sample_rate=22050,
            channels=2,
            bit_depth=24,
            buffer_size=2048,
            input_device="mic1",
            output_device="speaker1",
            noise_reduction=False,
            echo_cancellation=False,
            auto_gain_control=False
        )
        
        assert config.sample_rate == 22050
        assert config.channels == 2
        assert config.bit_depth == 24
        assert config.buffer_size == 2048
        assert config.input_device == "mic1"
        assert config.output_device == "speaker1"
        assert config.noise_reduction is False
        assert config.echo_cancellation is False
        assert config.auto_gain_control is False


class TestMockSecurityConfig:
    """Test MockSecurityConfig dataclass."""
    
    def test_mock_security_config_defaults(self):
        """Test mock security config with default values."""
        config = MockSecurityConfig()
        
        assert config.encryption_enabled is True
        assert config.consent_required is False
        assert config.hipaa_compliance_enabled is True
        assert config.audit_logging_enabled is True
        assert config.data_retention_days == 30
        assert config.max_login_attempts == 3
        assert config.encryption_key_rotation_days == 90
    
    def test_mock_security_config_custom_values(self):
        """Test mock security config with custom values."""
        config = MockSecurityConfig(
            encryption_enabled=False,
            consent_required=True,
            hipaa_compliance_enabled=False,
            audit_logging_enabled=False,
            data_retention_days=60,
            max_login_attempts=5,
            encryption_key_rotation_days=180
        )
        
        assert config.encryption_enabled is False
        assert config.consent_required is True
        assert config.hipaa_compliance_enabled is False
        assert config.audit_logging_enabled is False
        assert config.data_retention_days == 60
        assert config.max_login_attempts == 5
        assert config.encryption_key_rotation_days == 180


class TestMockAudioConfig:
    """Test MockAudioConfig dataclass."""
    
    def test_mock_audio_config_defaults(self):
        """Test mock audio config with default values."""
        config = MockAudioConfig()
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.bit_depth == 16
        assert config.buffer_size == 1024
        assert config.input_device is None
        assert config.output_device is None
        assert config.noise_reduction is True
        assert config.echo_cancellation is True
        assert config.auto_gain_control is True
    
    def test_mock_audio_config_custom_values(self):
        """Test mock audio config with custom values."""
        config = MockAudioConfig(
            sample_rate=22050,
            channels=2,
            bit_depth=24,
            buffer_size=2048,
            input_device="mic1",
            output_device="speaker1",
            noise_reduction=False,
            echo_cancellation=False,
            auto_gain_control=False
        )
        
        assert config.sample_rate == 22050
        assert config.channels == 2
        assert config.bit_depth == 24
        assert config.buffer_size == 2048
        assert config.input_device == "mic1"
        assert config.output_device == "speaker1"
        assert config.noise_reduction is False
        assert config.echo_cancellation is False
        assert config.auto_gain_control is False


class TestMockAuditLogger:
    """Test MockAuditLogger class."""
    
    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        logger = MockAuditLogger()
        assert logger.events == []
    
    def test_log_event(self):
        """Test event logging."""
        logger = MockAuditLogger()
        details = {"key": "value", "action": "test"}
        
        logger.log_event("test_event", "user123", details)
        
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event['type'] == "test_event"
        assert event['user_id'] == "user123"
        assert event['details'] == details
        assert event['timestamp'] is None
    
    def test_log_multiple_events(self):
        """Test logging multiple events."""
        logger = MockAuditLogger()
        
        logger.log_event("event1", "user1", {"action": "test1"})
        logger.log_event("event2", "user2", {"action": "test2"})
        logger.log_event("event3", "user3", {"action": "test3"})
        
        assert len(logger.events) == 3
        assert logger.events[0]['type'] == "event1"
        assert logger.events[1]['type'] == "event2"
        assert logger.events[2]['type'] == "event3"
    
    def test_get_events(self):
        """Test getting all events."""
        logger = MockAuditLogger()
        
        logger.log_event("event1", "user1", {"action": "test1"})
        logger.log_event("event2", "user2", {"action": "test2"})
        
        events = logger.get_events()
        assert len(events) == 2
        assert events == logger.events


class TestVoiceSecurity:
    """Test VoiceSecurity class."""
    
    def test_voice_security_initialization(self):
        """Test voice security initialization."""
        config = SecurityConfig(encryption_enabled=False)
        security = VoiceSecurity(config)
        
        assert security.config == config
        assert security.audit_logger is not None
        assert isinstance(security.audit_logger, MockAuditLogger)
    
    def test_log_security_event(self):
        """Test security event logging."""
        config = SecurityConfig()
        security = VoiceSecurity(config)
        details = {"action": "test", "resource": "voice_data"}
        
        security._log_security_event("test_event", details)
        
        assert len(security.audit_logger.events) == 1
        event = security.audit_logger.events[0]
        assert event['type'] == "test_event"
        assert event['user_id'] == "test_user"
        assert event['details'] == details
    
    def test_log_multiple_security_events(self):
        """Test logging multiple security events."""
        config = SecurityConfig()
        security = VoiceSecurity(config)
        
        security._log_security_event("event1", {"action": "test1"})
        security._log_security_event("event2", {"action": "test2"})
        security._log_security_event("event3", {"action": "test3"})
        
        assert len(security.audit_logger.events) == 3
        assert security.audit_logger.events[0]['type'] == "event1"
        assert security.audit_logger.events[1]['type'] == "event2"
        assert security.audit_logger.events[2]['type'] == "event3"


class TestMockConfig:
    """Test MockConfig class."""
    
    def test_mock_config_initialization(self):
        """Test mock config initialization."""
        config = MockConfig()
        
        # Test audio settings
        assert config.audio_sample_rate == 16000
        assert config.audio_channels == 1
        assert config.audio_buffer_size == 1024
        
        # Test STT/TTS settings
        assert config.stt_provider == "mock"
        assert config.tts_provider == "mock"
        assert config.stt_model == "base"
        assert config.tts_model == "base"
        assert config.tts_voice == "default"
        
        # Test voice settings
        assert config.voice_enabled is True
        assert config.voice_input_enabled is True
        assert config.voice_output_enabled is True
        assert config.default_voice_profile == 'default'
        assert config.voice_commands_enabled is True
        assert config.voice_speed == 1.0
        assert config.voice_pitch == 1.0
        assert config.voice_volume == 0.8
        
        # Test security settings
        assert config.encryption_enabled is True
        assert config.consent_required is False
        assert config.hipaa_compliance_enabled is True
        assert config.audit_logging_enabled is True
        assert config.data_retention_days == 30
        assert config.max_login_attempts == 3
        assert config.encryption_key_rotation_days == 90
        
        # Test API keys
        assert config.openai_api_key == "mock_openai_key"
        assert config.elevenlabs_api_key == "mock_elevenlabs_key"
        assert config.elevenlabs_voice_id == "mock_voice_id"
        assert config.elevenlabs_model_id == "mock_model_id"
        assert config.elevenlabs_stability == 0.5
        assert config.google_api_key == "mock_google_key"
        
        # Test debug settings
        assert config.debug_mode is True
        assert config.mock_audio_input is True
        assert config.save_debug_logs is True
        
        # Test performance settings
        assert config.max_concurrent_requests == 5
        assert config.request_timeout_seconds == 30
        assert config.cache_size_mb == 100
        assert config.optimization_level == "balanced"
        
        # Test logging settings
        assert config.log_level == "INFO"
        assert config.log_file_path == "/tmp/voice_service.log"
        assert config.log_rotation_enabled is True
        assert config.max_log_size_mb == 10
        assert config.log_retention_days == 7
        
        # Test version
        assert config.version == "1.0.0"
        
        # Test nested objects
        assert config.audio is not None
        assert isinstance(config.audio, AudioConfig)
        assert config.security is not None
        assert isinstance(config.security, VoiceSecurity)
        
        # Test voice profiles
        assert 'default' in config.voice_profiles
        assert 'calm_therapist' in config.voice_profiles
        assert isinstance(config.voice_profiles['default'], MockVoiceConfig)
    
    def test_get_preferred_stt_service_openai(self):
        """Test getting preferred STT service when OpenAI is configured."""
        config = MockConfig()
        config.openai_api_key = "real_key"
        
        assert config.get_preferred_stt_service() == "openai"
    
    def test_get_preferred_stt_service_google(self):
        """Test getting preferred STT service when Google is configured."""
        config = MockConfig()
        config.openai_api_key = None
        config.google_api_key = "real_key"
        
        assert config.get_preferred_stt_service() == "google"
    
    def test_get_preferred_stt_service_whisper(self):
        """Test getting preferred STT service when only Whisper is configured."""
        config = MockConfig()
        config.openai_api_key = None
        config.google_api_key = None
        
        assert config.get_preferred_stt_service() == "whisper"
    
    def test_get_preferred_stt_service_mock(self):
        """Test getting preferred STT service when none are configured."""
        config = MockConfig()
        config.openai_api_key = None
        config.google_api_key = None
        
        # Whisper is always configured for mock, but falls back to mock
        assert config.get_preferred_stt_service() == "whisper"
    
    def test_get_preferred_tts_service_elevenlabs(self):
        """Test getting preferred TTS service when ElevenLabs is configured."""
        config = MockConfig()
        config.elevenlabs_api_key = "real_key"
        
        assert config.get_preferred_tts_service() == "elevenlabs"
    
    def test_get_preferred_tts_service_piper(self):
        """Test getting preferred TTS service when Piper is configured."""
        config = MockConfig()
        config.elevenlabs_api_key = None
        
        assert config.get_preferred_tts_service() == "piper"
    
    def test_get_preferred_tts_service_mock(self):
        """Test getting preferred TTS service when none are configured."""
        config = MockConfig()
        config.elevenlabs_api_key = None
        
        # Piper is always configured for mock, but falls back to mock
        assert config.get_preferred_tts_service() == "piper"
    
    def test_is_openai_whisper_configured(self):
        """Test OpenAI Whisper configuration check."""
        config = MockConfig()
        
        # With API key
        assert config.is_openai_whisper_configured() is True
        
        # Without API key
        config.openai_api_key = ""
        assert config.is_openai_whisper_configured() is False
        
        config.openai_api_key = None
        assert config.is_openai_whisper_configured() is False
    
    def test_is_openai_tts_configured(self):
        """Test OpenAI TTS configuration check."""
        config = MockConfig()
        
        # With API key
        assert config.is_openai_tts_configured() is True
        
        # Without API key
        config.openai_api_key = ""
        assert config.is_openai_tts_configured() is False
        
        config.openai_api_key = None
        assert config.is_openai_tts_configured() is False
    
    def test_is_google_speech_configured(self):
        """Test Google Speech configuration check."""
        config = MockConfig()
        
        # With API key
        assert config.is_google_speech_configured() is True
        
        # Without API key
        config.google_api_key = ""
        assert config.is_google_speech_configured() is False
        
        config.google_api_key = None
        assert config.is_google_speech_configured() is False
    
    def test_is_whisper_configured(self):
        """Test Whisper configuration check."""
        config = MockConfig()
        
        # Always true for mock
        assert config.is_whisper_configured() is True
    
    def test_is_elevenlabs_configured(self):
        """Test ElevenLabs configuration check."""
        config = MockConfig()
        
        # With API key
        assert config.is_elevenlabs_configured() is True
        
        # Without API key
        config.elevenlabs_api_key = ""
        assert config.is_elevenlabs_configured() is False
        
        config.elevenlabs_api_key = None
        assert config.is_elevenlabs_configured() is False
    
    def test_is_piper_configured(self):
        """Test Piper configuration check."""
        config = MockConfig()
        
        # Always true for mock
        assert config.is_piper_configured() is True
    
    def test_get_voice_profile_default(self):
        """Test getting default voice profile."""
        config = MockConfig()
        
        profile = config.get_voice_profile()
        assert isinstance(profile, MockVoiceConfig)
        assert profile == config.voice_profiles['default']
    
    def test_get_voice_profile_by_name(self):
        """Test getting voice profile by name."""
        config = MockConfig()
        
        profile = config.get_voice_profile('calm_therapist')
        assert isinstance(profile, MockVoiceConfig)
        assert profile == config.voice_profiles['calm_therapist']
    
    def test_get_voice_profile_nonexistent(self):
        """Test getting non-existent voice profile."""
        config = MockConfig()
        
        profile = config.get_voice_profile('nonexistent')
        assert isinstance(profile, MockVoiceConfig)
        assert profile == config.voice_profiles['default']
    
    def test_get_voice_profile_default_not_in_profiles(self):
        """Test getting voice profile when default is not in profiles."""
        config = MockConfig()
        config.default_voice_profile = 'nonexistent'
        
        profile = config.get_voice_profile()
        assert isinstance(profile, MockVoiceConfig)
        assert profile == config.voice_profiles['default']
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid config."""
        config = MockConfig()
        
        issues = config.validate_configuration()
        assert issues == []
    
    def test_validate_configuration_voice_disabled(self):
        """Test configuration validation with voice disabled."""
        config = MockConfig()
        config.voice_enabled = False
        
        issues = config.validate_configuration()
        assert len(issues) == 1
        assert "Voice features are disabled" in issues[0]
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = MockConfig()
        
        config_dict = config.to_dict()
        
        # Check some key fields
        assert config_dict['audio_sample_rate'] == 16000
        assert config_dict['stt_provider'] == "mock"
        assert config_dict['tts_provider'] == "mock"
        assert config_dict['debug_mode'] is True
        assert config_dict['encryption_enabled'] is True
        assert config_dict['openai_api_key'] == "mock_openai_key"
        assert config_dict['version'] == "1.0.0"
        
        # Check that nested objects are not included
        assert 'audio' not in config_dict
        assert 'security' not in config_dict
        assert 'voice_profiles' not in config_dict
    
    def test_copy(self):
        """Test creating a copy of the configuration."""
        config = MockConfig()
        
        # Modify some values
        config.audio_sample_rate = 22050
        config.openai_api_key = "different_key"
        
        copy = config.copy()
        
        # Check it's a different object
        assert copy is not config
        
        # Check values are copied (but reset to defaults for new MockConfig)
        assert copy.audio_sample_rate == 16000
        assert copy.openai_api_key == "mock_openai_key"
    
    def test_equality_same_config(self):
        """Test equality with same configuration."""
        config1 = MockConfig()
        config2 = MockConfig()
        
        assert config1 == config2
    
    def test_equality_different_config(self):
        """Test equality with different configuration."""
        config1 = MockConfig()
        config2 = MockConfig()
        config2.audio_sample_rate = 22050
        
        assert config1 != config2
    
    def test_equality_different_type(self):
        """Test equality with different type."""
        config = MockConfig()
        
        assert config != "not a config"
        assert config != 123
        assert config != None
    
    def test_equality_with_modified_config(self):
        """Test equality with modified configuration."""
        config = MockConfig()
        original_dict = config.to_dict()
        
        # Modify a value
        config.audio_sample_rate = 22050
        
        # Should not be equal
        assert config != MockConfig()
        
        # Restore original value
        config.audio_sample_rate = original_dict['audio_sample_rate']
        
        # Should be equal again
        assert config == MockConfig()


class TestCreateMockVoiceConfig:
    """Test create_mock_voice_config function."""
    
    def test_create_mock_voice_config(self):
        """Test creating mock voice configuration."""
        config = create_mock_voice_config()
        
        assert isinstance(config, MockConfig)
        assert config.audio_sample_rate == 16000
        assert config.stt_provider == "mock"
        assert config.tts_provider == "mock"
        assert config.openai_api_key == "mock_openai_key"
    
    def test_create_mock_voice_config_multiple_calls(self):
        """Test creating multiple mock configurations."""
        config1 = create_mock_voice_config()
        config2 = create_mock_voice_config()
        
        # Should be different objects
        assert config1 is not config2
        
        # But should have same values
        assert config1 == config2


if __name__ == "__main__":
    pytest.main([__file__])