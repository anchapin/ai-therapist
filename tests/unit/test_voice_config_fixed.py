"""
Unit tests for voice configuration module.

This module tests the VoiceConfig, VoiceProfile, AudioConfig, SecurityConfig, 
and PerformanceConfig classes to ensure proper configuration management.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

# Import the actual voice config classes
from voice.config import (
    VoiceProfile, AudioConfig, SecurityConfig, PerformanceConfig, VoiceConfig
)


class TestVoiceProfile:
    """Test VoiceProfile class functionality."""
    
    def test_voice_profile_creation(self):
        """Test creating a voice profile with required parameters."""
        profile = VoiceProfile(
            name="test_profile",
            description="Test profile",
            voice_id="test_voice_id"
        )
        
        assert profile.name == "test_profile"
        assert profile.description == "Test profile"
        assert profile.voice_id == "test_voice_id"
        assert profile.language == "en-US"  # Default value
        assert profile.gender == "neutral"  # Default value
        assert profile.age == "adult"  # Default value
        assert profile.pitch == 1.0  # Default value
        assert profile.speed == 1.0  # Default value
        assert profile.volume == 1.0  # Default value
        assert profile.emotion == "calm"  # Default value
        assert profile.accent == "neutral"  # Default value
        assert profile.style == "conversational"  # Default value
        assert profile.elevenlabs_settings == {}  # Default value
        assert profile.piper_settings == {}  # Default value
    
    def test_voice_profile_with_all_parameters(self):
        """Test creating a voice profile with all parameters."""
        elevenlabs_settings = {"stability": 0.5, "similarity_boost": 0.8}
        piper_settings = {"noise_scale": 0.667, "length_scale": 1.0}
        
        profile = VoiceProfile(
            name="full_profile",
            description="Full profile with all parameters",
            voice_id="full_voice_id",
            language="en-GB",
            gender="male",
            age="senior",
            pitch=0.9,
            speed=1.1,
            volume=0.8,
            emotion="happy",
            accent="british",
            style="formal",
            elevenlabs_settings=elevenlabs_settings,
            piper_settings=piper_settings
        )
        
        assert profile.name == "full_profile"
        assert profile.description == "Full profile with all parameters"
        assert profile.voice_id == "full_voice_id"
        assert profile.language == "en-GB"
        assert profile.gender == "male"
        assert profile.age == "senior"
        assert profile.pitch == 0.9
        assert profile.speed == 1.1
        assert profile.volume == 0.8
        assert profile.emotion == "happy"
        assert profile.accent == "british"
        assert profile.style == "formal"
        assert profile.elevenlabs_settings == elevenlabs_settings
        assert profile.piper_settings == piper_settings
    
    def test_voice_profile_from_dict(self):
        """Test creating a voice profile from a dictionary."""
        data = {
            "name": "dict_profile",
            "description": "Profile from dict",
            "voice_id": "dict_voice_id",
            "language": "fr-FR",
            "gender": "female",
            "pitch": 0.95
        }
        
        profile = VoiceProfile.from_dict(data)
        
        assert profile.name == "dict_profile"
        assert profile.description == "Profile from dict"
        assert profile.voice_id == "dict_voice_id"
        assert profile.language == "fr-FR"
        assert profile.gender == "female"
        assert profile.pitch == 0.95
        # Default values for missing fields
        assert profile.age == "adult"
        assert profile.speed == 1.0
    
    def test_voice_profile_to_dict(self):
        """Test converting a voice profile to a dictionary."""
        profile = VoiceProfile(
            name="to_dict_profile",
            description="Profile to dict",
            voice_id="to_dict_voice_id",
            language="es-ES",
            gender="neutral",
            pitch=1.05
        )
        
        result = profile.to_dict()
        
        assert result["name"] == "to_dict_profile"
        assert result["description"] == "Profile to dict"
        assert result["voice_id"] == "to_dict_voice_id"
        assert result["language"] == "es-ES"
        assert result["gender"] == "neutral"
        assert result["pitch"] == 1.05
        # Check that all expected fields are present
        expected_fields = [
            "name", "description", "voice_id", "language", "gender", "age",
            "pitch", "speed", "volume", "emotion", "accent", "style",
            "elevenlabs_settings", "piper_settings"
        ]
        for field in expected_fields:
            assert field in result


class TestAudioConfig:
    """Test AudioConfig class functionality."""
    
    def test_audio_config_defaults(self):
        """Test creating an audio config with default values."""
        config = AudioConfig()
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.format == "wav"
        assert config.input_device == 0
        assert config.output_device == 0
        assert config.noise_reduction_enabled is True
        assert config.vad_enabled is True
        assert config.vad_aggressiveness == 2
        assert config.silence_threshold == 0.3
        assert config.silence_duration_ms == 1000
        assert config.buffer_size == 4096
        assert config.max_buffer_size == 300
        assert config.max_memory_mb == 100
    
    def test_audio_config_custom_values(self):
        """Test creating an audio config with custom values."""
        config = AudioConfig(
            sample_rate=22050,
            channels=2,
            chunk_size=2048,
            format="mp3",
            noise_reduction_enabled=False,
            vad_enabled=False,
            vad_aggressiveness=3,
            silence_threshold=0.5,
            silence_duration_ms=2000,
            buffer_size=8192,
            max_buffer_size=500,
            max_memory_mb=200
        )
        
        assert config.sample_rate == 22050
        assert config.channels == 2
        assert config.chunk_size == 2048
        assert config.format == "mp3"
        assert config.noise_reduction_enabled is False
        assert config.vad_enabled is False
        assert config.vad_aggressiveness == 3
        assert config.silence_threshold == 0.5
        assert config.silence_duration_ms == 2000
        assert config.buffer_size == 8192
        assert config.max_buffer_size == 500
        assert config.max_memory_mb == 200


class TestSecurityConfig:
    """Test SecurityConfig class functionality."""
    
    def test_security_config_defaults(self):
        """Test creating a security config with default values."""
        config = SecurityConfig()
        
        assert config.encryption_enabled is True
        assert config.data_retention_hours == 24
        assert config.data_retention_days == 30
        assert config.session_timeout_minutes == 30
        assert config.encryption_key_rotation_days == 90
        assert config.audit_logging_enabled is True
        assert config.max_login_attempts == 3
        assert config.consent_required is True
        assert config.transcript_storage is False
        assert config.anonymization_enabled is True
        assert config.privacy_mode is True
        assert config.hipaa_compliance_enabled is True
        assert config.gdpr_compliance_enabled is True
        assert config.data_localization is True
        assert config.consent_recording is True
        assert config.emergency_protocols_enabled is True
    
    def test_security_config_custom_values(self):
        """Test creating a security config with custom values."""
        config = SecurityConfig(
            encryption_enabled=False,
            data_retention_hours=48,
            data_retention_days=60,
            session_timeout_minutes=60,
            encryption_key_rotation_days=180,
            audit_logging_enabled=False,
            max_login_attempts=5,
            consent_required=False,
            transcript_storage=True,
            anonymization_enabled=False,
            privacy_mode=False,
            hipaa_compliance_enabled=False,
            gdpr_compliance_enabled=False,
            data_localization=False,
            consent_recording=False,
            emergency_protocols_enabled=False
        )
        
        assert config.encryption_enabled is False
        assert config.data_retention_hours == 48
        assert config.data_retention_days == 60
        assert config.session_timeout_minutes == 60
        assert config.encryption_key_rotation_days == 180
        assert config.audit_logging_enabled is False
        assert config.max_login_attempts == 5
        assert config.consent_required is False
        assert config.transcript_storage is True
        assert config.anonymization_enabled is False
        assert config.privacy_mode is False
        assert config.hipaa_compliance_enabled is False
        assert config.gdpr_compliance_enabled is False
        assert config.data_localization is False
        assert config.consent_recording is False
        assert config.emergency_protocols_enabled is False


class TestPerformanceConfig:
    """Test PerformanceConfig class functionality."""
    
    def test_performance_config_defaults(self):
        """Test creating a performance config with default values."""
        config = PerformanceConfig()
        
        assert config.cache_enabled is True
        assert config.cache_size == 100
        assert config.streaming_enabled is True
        assert config.parallel_processing is True
        assert config.buffer_size == 4096
        assert config.processing_timeout == 30000
        assert config.max_concurrent_requests == 5
        assert config.response_timeout == 10000
    
    def test_performance_config_custom_values(self):
        """Test creating a performance config with custom values."""
        config = PerformanceConfig(
            cache_enabled=False,
            cache_size=200,
            streaming_enabled=False,
            parallel_processing=False,
            buffer_size=8192,
            processing_timeout=60000,
            max_concurrent_requests=10,
            response_timeout=20000
        )
        
        assert config.cache_enabled is False
        assert config.cache_size == 200
        assert config.streaming_enabled is False
        assert config.parallel_processing is False
        assert config.buffer_size == 8192
        assert config.processing_timeout == 60000
        assert config.max_concurrent_requests == 10
        assert config.response_timeout == 20000


class TestVoiceConfig:
    """Test VoiceConfig class functionality."""
    
    def test_voice_config_defaults(self):
        """Test creating a voice config with default values."""
        config = VoiceConfig()
        
        # Test feature toggles
        assert config.voice_enabled is True
        assert config.voice_input_enabled is True
        assert config.voice_output_enabled is True
        assert config.voice_commands_enabled is True
        assert config.security_enabled is True
        
        # Test session and timeout configuration
        assert config.session_timeout_minutes == 30
        assert config.session_timeout == 1800.0
        assert config.recording_timeout == 10.0
        assert config.max_concurrent_sessions == 100
        assert config.audio_processing_timeout == 10.0
        
        # Test nested configs
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.performance, PerformanceConfig)
        
        # Test voice profiles
        assert isinstance(config.voice_profiles, dict)
        assert len(config.voice_profiles) > 0  # Should have default profiles
        
        # Test service configurations
        assert config.elevenlabs_model == "eleven_multilingual_v2"
        assert config.elevenlabs_voice_speed == 1.0
        assert config.elevenlabs_voice_stability == 0.5
        assert config.elevenlabs_voice_similarity_boost == 0.8
        
        assert config.google_speech_language_code == "en-US"
        assert config.google_speech_model == "latest_long"
        assert config.google_speech_enable_automatic_punctuation is True
        assert config.google_speech_enable_word_time_offsets is True
        assert config.google_speech_max_alternatives == 1
        
        assert config.openai_whisper_model == "whisper-1"
        assert config.openai_whisper_language == "en"
        assert config.openai_whisper_temperature == 0.0
        
        assert config.whisper_model == "base"
        assert config.whisper_language == "en"
        assert config.whisper_temperature == 0.0
        assert config.whisper_beam_size == 5
        assert config.whisper_best_of == 5
        
        # Test voice profile configuration
        assert config.voice_profile_path == "./voice_profiles"
        assert config.default_voice_profile == "calm_therapist"
        assert config.voice_customization_enabled is True
        assert config.voice_pitch_adjustment == 1.0
        assert config.voice_speed_adjustment == 1.0
        assert config.voice_volume_adjustment == 1.0
        
        # Test voice command configuration
        assert config.voice_command_wake_word == "therapist"
        assert config.voice_command_timeout == 5000
        assert config.voice_command_max_duration == 10000
        assert config.voice_command_min_confidence == 0.7
        assert config.voice_command_debug_mode is False
        
        # Test logging configuration
        assert config.voice_logging_enabled is True
        assert config.voice_log_level == "INFO"
        assert config.voice_metrics_enabled is True
        assert config.voice_error_reporting is True
        assert config.voice_performance_monitoring is True
    
    def test_voice_config_mock_mode(self):
        """Test voice config in mock mode."""
        with patch.dict(os.environ, {
            "VOICE_MOCK_MODE": "true",
            "VOICE_FORCE_MOCK_SERVICES": "true"
        }, clear=True):
            config = VoiceConfig()
            
            assert config.get_preferred_stt_service() == "mock"
            assert config.get_preferred_tts_service() == "mock"
            assert config.openai_api_key == "mock_openai_key_for_testing"
            assert config.elevenlabs_api_key == "mock_elevenlabs_key_for_testing"
            assert config.elevenlabs_voice_id == "mock_voice_id_for_testing"
            assert config.google_cloud_project_id == "mock-project-id-for-testing"
    
    def test_voice_config_environment_variables(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "VOICE_ENABLED": "false",
            "VOICE_INPUT_ENABLED": "false",
            "VOICE_OUTPUT_ENABLED": "false",
            "VOICE_COMMANDS_ENABLED": "false",
            "VOICE_SECURITY_ENABLED": "false",
            "VOICE_MOCK_MODE": "false",
            "VOICE_FORCE_MOCK_SERVICES": "false",
            "ELEVENLABS_API_KEY": "test_elevenlabs_key",
            "ELEVENLABS_VOICE_ID": "test_voice_id",
            "ELEVENLABS_MODEL": "test_model",
            "ELEVENLABS_VOICE_SPEED": "1.5",
            "ELEVENLABS_VOICE_STABILITY": "0.7",
            "ELEVENLABS_VOICE_SIMILARITY_BOOST": "0.9",
            "GOOGLE_CLOUD_CREDENTIALS_PATH": "/path/to/credentials.json",
            "GOOGLE_CLOUD_PROJECT_ID": "test-project",
            "GOOGLE_SPEECH_LANGUAGE_CODE": "es-ES",
            "GOOGLE_SPEECH_MODEL": "test_model",
            "GOOGLE_SPEECH_ENABLE_AUTOMATIC_PUNCTUATION": "false",
            "GOOGLE_SPEECH_ENABLE_WORD_TIME_OFFSETS": "false",
            "GOOGLE_SPEECH_MAX_ALTERNATIVES": "3",
            "OPENAI_WHISPER_MODEL": "whisper-test",
            "OPENAI_WHISPER_LANGUAGE": "es",
            "OPENAI_WHISPER_TEMPERATURE": "0.5",
            "WHISPER_MODEL": "test-base",
            "WHISPER_LANGUAGE": "fr",
            "WHISPER_TEMPERATURE": "0.3",
            "WHISPER_BEAM_SIZE": "3",
            "WHISPER_BEST_OF": "3",
            "PIPER_TTS_MODEL_PATH": "/path/to/piper/model",
            "PIPER_TTS_SPEAKER_ID": "1",
            "PIPER_TTS_NOISE_SCALE": "0.5",
            "PIPER_TTS_LENGTH_SCALE": "1.2",
            "PIPER_TTS_NOISE_W": "0.7",
            "VOICE_PROFILE_PATH": "/test/profiles",
            "DEFAULT_VOICE_PROFILE": "test_profile",
            "VOICE_CUSTOMIZATION_ENABLED": "false",
            "VOICE_PITCH_ADJUSTMENT": "1.2",
            "VOICE_SPEED_ADJUSTMENT": "1.3",
            "VOICE_VOLUME_ADJUSTMENT": "0.8",
            "VOICE_COMMAND_WAKE_WORD": "assistant",
            "VOICE_COMMAND_TIMEOUT": "3000",
            "VOICE_COMMAND_MAX_DURATION": "8000",
            "VOICE_COMMAND_MIN_CONFIDENCE": "0.8",
            "VOICE_COMMAND_DEBUG_MODE": "true",
            "VOICE_LOGGING_ENABLED": "false",
            "VOICE_LOG_LEVEL": "DEBUG",
            "VOICE_METRICS_ENABLED": "false",
            "VOICE_ERROR_REPORTING": "false",
            "VOICE_PERFORMANCE_MONITORING": "false",
            "VOICE_SESSION_TIMEOUT_MINUTES": "60",
            "VOICE_SESSION_TIMEOUT": "3600.0",
            "VOICE_RECORDING_TIMEOUT": "15.0",
            "VOICE_MAX_CONCURRENT_SESSIONS": "50",
            "VOICE_AUDIO_PROCESSING_TIMEOUT": "15.0",
            "VOICE_ENCRYPTION_ENABLED": "false",
            "VOICE_DATA_RETENTION_HOURS": "48",
            "VOICE_DATA_RETENTION_DAYS": "60",
            "VOICE_SESSION_TIMEOUT_MINUTES": "60",
            "VOICE_ENCRYPTION_KEY_ROTATION_DAYS": "180",
            "VOICE_AUDIT_LOGGING_ENABLED": "false",
            "VOICE_MAX_LOGIN_ATTEMPTS": "5",
            "VOICE_CONSENT_REQUIRED": "false",
            "VOICE_TRANSCRIPT_STORAGE": "true",
            "VOICE_ANONYMIZATION_ENABLED": "false",
            "VOICE_PRIVACY_MODE": "false",
            "VOICE_HIPAA_COMPLIANCE_ENABLED": "false",
            "VOICE_GDPR_COMPLIANCE_ENABLED": "false",
            "VOICE_DATA_LOCALIZATION": "false",
            "VOICE_CONSENT_RECORDING": "false",
            "VOICE_EMERGENCY_PROTOCOLS_ENABLED": "false",
            "VOICE_AUDIO_SAMPLE_RATE": "22050",
            "VOICE_AUDIO_CHANNELS": "2",
            "VOICE_AUDIO_CHUNK_SIZE": "2048",
            "VOICE_AUDIO_MAX_BUFFER_SIZE": "600",
            "VOICE_AUDIO_MAX_MEMORY_MB": "200",
            "VOICE_CACHE_ENABLED": "false",
            "VOICE_CACHE_SIZE": "200",
            "VOICE_STREAMING_ENABLED": "false",
            "VOICE_PARALLEL_PROCESSING": "false",
            "VOICE_BUFFER_SIZE": "8192",
            "VOICE_PROCESSING_TIMEOUT": "60000"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = VoiceConfig()
            
            # Test feature toggles
            assert config.voice_enabled is False
            assert config.voice_input_enabled is False
            assert config.voice_output_enabled is False
            assert config.voice_commands_enabled is False
            assert config.security_enabled is False
            
            # Test service configurations
            assert config.elevenlabs_api_key == "test_elevenlabs_key"
            assert config.elevenlabs_voice_id == "test_voice_id"
            assert config.elevenlabs_model == "test_model"
            assert config.elevenlabs_voice_speed == 1.5
            assert config.elevenlabs_voice_stability == 0.7
            assert config.elevenlabs_voice_similarity_boost == 0.9
            
            assert config.google_cloud_credentials_path == "/path/to/credentials.json"
            assert config.google_cloud_project_id == "test-project"
            assert config.google_speech_language_code == "es-ES"
            assert config.google_speech_model == "test_model"
            assert config.google_speech_enable_automatic_punctuation is False
            assert config.google_speech_enable_word_time_offsets is False
            assert config.google_speech_max_alternatives == 3
            
            assert config.openai_whisper_model == "whisper-test"
            assert config.openai_whisper_language == "es"
            assert config.openai_whisper_temperature == 0.5
            
            assert config.whisper_model == "test-base"
            assert config.whisper_language == "fr"
            assert config.whisper_temperature == 0.3
            assert config.whisper_beam_size == 3
            assert config.whisper_best_of == 3
            
            assert config.piper_tts_model_path == "/path/to/piper/model"
            assert config.piper_tts_speaker_id == 1
            assert config.piper_tts_noise_scale == 0.5
            assert config.piper_tts_length_scale == 1.2
            assert config.piper_tts_noise_w == 0.7
            
            # Test voice profile configuration
            assert config.voice_profile_path == "/test/profiles"
            assert config.default_voice_profile == "test_profile"
            assert config.voice_customization_enabled is False
            assert config.voice_pitch_adjustment == 1.2
            assert config.voice_speed_adjustment == 1.3
            assert config.voice_volume_adjustment == 0.8
            
            # Test voice command configuration
            assert config.voice_command_wake_word == "assistant"
            assert config.voice_command_timeout == 3000
            assert config.voice_command_max_duration == 8000
            assert config.voice_command_min_confidence == 0.8
            assert config.voice_command_debug_mode is True
            
            # Test logging configuration
            assert config.voice_logging_enabled is False
            assert config.voice_log_level == "DEBUG"
            assert config.voice_metrics_enabled is False
            assert config.voice_error_reporting is False
            assert config.voice_performance_monitoring is False
            
            # Test session and timeout configuration
            assert config.session_timeout_minutes == 60
            assert config.session_timeout == 3600.0
            assert config.recording_timeout == 15.0
            assert config.max_concurrent_sessions == 50
            assert config.audio_processing_timeout == 15.0
            
            # Test security configuration
            assert config.security.encryption_enabled is False
            assert config.security.data_retention_hours == 48
            assert config.security.data_retention_days == 60
            assert config.security.session_timeout_minutes == 60
            assert config.security.encryption_key_rotation_days == 180
            assert config.security.audit_logging_enabled is False
            assert config.security.max_login_attempts == 5
            assert config.security.consent_required is False
            assert config.security.transcript_storage is True
            assert config.security.anonymization_enabled is False
            assert config.security.privacy_mode is False
            assert config.security.hipaa_compliance_enabled is False
            assert config.security.gdpr_compliance_enabled is False
            assert config.security.data_localization is False
            assert config.security.consent_recording is False
            assert config.security.emergency_protocols_enabled is False
            
            # Test audio configuration
            assert config.audio.sample_rate == 22050
            assert config.audio.channels == 2
            assert config.audio.chunk_size == 2048
            assert config.audio.max_buffer_size == 600
            assert config.audio.max_memory_mb == 200
            
            # Test performance configuration
            assert config.performance.cache_enabled is False
            assert config.performance.cache_size == 200
            assert config.performance.streaming_enabled is False
            assert config.performance.parallel_processing is False
            assert config.performance.buffer_size == 8192
            assert config.performance.processing_timeout == 60000
    
    def test_get_voice_profile(self):
        """Test getting a voice profile by name."""
        config = VoiceConfig()
        
        # Test getting default profile
        profile = config.get_voice_profile()
        assert isinstance(profile, VoiceProfile)
        
        # Test getting specific profile
        if "calm_therapist" in config.voice_profiles:
            profile = config.get_voice_profile("calm_therapist")
            assert profile.name == "calm_therapist"
        
        # Test getting non-existent profile
        profile = config.get_voice_profile("non_existent")
        assert isinstance(profile, VoiceProfile)  # Should return default or first available
    
    def test_get_voice_profile_settings(self):
        """Test getting detailed settings for a voice profile."""
        config = VoiceConfig()
        
        # Test getting settings for existing profile
        if "calm_therapist" in config.voice_profiles:
            settings = config.get_voice_profile_settings("calm_therapist")
            assert isinstance(settings, dict)
            assert settings["name"] == "calm_therapist"
            assert "description" in settings
            assert "voice_id" in settings
            assert "language" in settings
            assert "gender" in settings
            assert "pitch" in settings
            assert "speed" in settings
            assert "volume" in settings
            assert "emotion" in settings
            assert "style" in settings
            assert "elevenlabs_settings" in settings
            assert "piper_settings" in settings
        
        # Test getting settings for non-existent profile
        with pytest.raises(ValueError, match="Voice profile 'non_existent' not found"):
            config.get_voice_profile_settings("non_existent")
    
    def test_save_voice_profile(self):
        """Test saving a voice profile to file."""
        config = VoiceConfig()
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config.voice_profile_path = temp_dir
            
            # Create a new profile
            profile = VoiceProfile(
                name="test_save_profile",
                description="Test save profile",
                voice_id="test_save_voice_id",
                language="en-GB",
                gender="male",
                pitch=0.9
            )
            
            # Save the profile
            config.save_voice_profile(profile)
            
            # Check that the profile was saved
            assert "test_save_profile" in config.voice_profiles
            assert config.voice_profiles["test_save_profile"] == profile
            
            # Check that the file was created
            profile_file = Path(temp_dir) / "test_save_profile.json"
            assert profile_file.exists()
            
            # Check the file content
            with open(profile_file, 'r') as f:
                data = json.load(f)
                assert data["name"] == "test_save_profile"
                assert data["description"] == "Test save profile"
                assert data["voice_id"] == "test_save_voice_id"
                assert data["language"] == "en-GB"
                assert data["gender"] == "male"
                assert data["pitch"] == 0.9
    
    def test_is_elevenlabs_configured(self):
        """Test checking if ElevenLabs is configured."""
        config = VoiceConfig()
        
        # Test with no API key or voice ID
        config.elevenlabs_api_key = None
        config.elevenlabs_voice_id = None
        assert config.is_elevenlabs_configured() is False
        
        # Test with API key but no voice ID
        config.elevenlabs_api_key = "test_api_key"
        config.elevenlabs_voice_id = None
        assert config.is_elevenlabs_configured() is False
        
        # Test with voice ID but no API key
        config.elevenlabs_api_key = None
        config.elevenlabs_voice_id = "test_voice_id"
        assert config.is_elevenlabs_configured() is False
        
        # Test with both API key and voice ID
        config.elevenlabs_api_key = "test_api_key"
        config.elevenlabs_voice_id = "test_voice_id"
        assert config.is_elevenlabs_configured() is True
    
    def test_is_openai_whisper_configured(self):
        """Test checking if OpenAI Whisper is configured."""
        config = VoiceConfig()
        
        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            assert config.is_openai_whisper_configured() is False
        
        # Test with API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"}):
            assert config.is_openai_whisper_configured() is True
    
    def test_is_openai_tts_configured(self):
        """Test checking if OpenAI TTS is configured."""
        config = VoiceConfig()
        
        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            assert config.is_openai_tts_configured() is False
        
        # Test with API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"}):
            assert config.is_openai_tts_configured() is True
    
    def test_is_google_speech_configured(self):
        """Test checking if Google Speech is configured."""
        config = VoiceConfig()
        
        # Test with no credentials or project ID
        config.google_cloud_credentials_path = None
        config.google_cloud_project_id = None
        assert config.is_google_speech_configured() is False
        
        # Test with credentials but no project ID
        config.google_cloud_credentials_path = "/path/to/credentials.json"
        config.google_cloud_project_id = None
        assert config.is_google_speech_configured() is False
        
        # Test with project ID but no credentials
        config.google_cloud_credentials_path = None
        config.google_cloud_project_id = "test-project"
        assert config.is_google_speech_configured() is False
        
        # Test with both credentials and project ID
        config.google_cloud_credentials_path = "/path/to/credentials.json"
        config.google_cloud_project_id = "test-project"
        assert config.is_google_speech_configured() is True
    
    def test_is_whisper_configured(self):
        """Test checking if Whisper is configured."""
        config = VoiceConfig()
        
        # Whisper works offline by default
        assert config.is_whisper_configured() is True
    
    def test_is_piper_configured(self):
        """Test checking if Piper TTS is configured."""
        config = VoiceConfig()
        
        # Test with no model path
        config.piper_tts_model_path = None
        assert config.is_piper_configured() is False
        
        # Test with model path
        config.piper_tts_model_path = "/path/to/piper/model"
        assert config.is_piper_configured() is True
    
    def test_get_preferred_stt_service(self):
        """Test getting preferred STT service."""
        config = VoiceConfig()
        
        # Test with mock mode
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "true"}, clear=True):
            assert config.get_preferred_stt_service() == "mock"
        
        # Test with force mock services
        with patch.dict(os.environ, {"VOICE_FORCE_MOCK_SERVICES": "true"}, clear=True):
            assert config.get_preferred_stt_service() == "mock"
        
        # Test with STT provider environment variable
        with patch.dict(os.environ, {"STT_PROVIDER": "mock"}, clear=True):
            assert config.get_preferred_stt_service() == "mock"
        
        # Test with OpenAI Whisper configured
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key", "VOICE_MOCK_MODE": "false"}, clear=True):
            assert config.get_preferred_stt_service() == "openai"
        
        # Test with Google Speech configured
        config.google_cloud_credentials_path = "/path/to/credentials.json"
        config.google_cloud_project_id = "test-project"
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "false"}, clear=True):
            assert config.get_preferred_stt_service() == "google"
        
        # Test with Whisper configured (default)
        config.google_cloud_credentials_path = None
        config.google_cloud_project_id = None
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "false"}, clear=True):
            assert config.get_preferred_stt_service() == "whisper"
    
    def test_get_preferred_tts_service(self):
        """Test getting preferred TTS service."""
        config = VoiceConfig()
        
        # Test with mock mode
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "true"}, clear=True):
            assert config.get_preferred_tts_service() == "mock"
        
        # Test with force mock services
        with patch.dict(os.environ, {"VOICE_FORCE_MOCK_SERVICES": "true"}, clear=True):
            assert config.get_preferred_tts_service() == "mock"
        
        # Test with TTS provider environment variable
        with patch.dict(os.environ, {"TTS_PROVIDER": "mock"}, clear=True):
            assert config.get_preferred_tts_service() == "mock"
        
        # Test with ElevenLabs configured
        config.elevenlabs_api_key = "test_api_key"
        config.elevenlabs_voice_id = "test_voice_id"
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "false"}, clear=True):
            assert config.get_preferred_tts_service() == "elevenlabs"
        
        # Test with Piper configured
        config.elevenlabs_api_key = None
        config.elevenlabs_voice_id = None
        config.piper_tts_model_path = "/path/to/piper/model"
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "false"}, clear=True):
            assert config.get_preferred_tts_service() == "piper"
        
        # Test with no service configured (default to mock)
        config.elevenlabs_api_key = None
        config.elevenlabs_voice_id = None
        config.piper_tts_model_path = None
        with patch.dict(os.environ, {"VOICE_MOCK_MODE": "false"}, clear=True):
            assert config.get_preferred_tts_service() == "mock"
    
    def test_validate_configuration(self):
        """Test validating configuration."""
        config = VoiceConfig()
        
        # Test with valid configuration
        issues = config.validate_configuration()
        assert isinstance(issues, list)
        
        # Test with voice disabled
        config.voice_enabled = False
        issues = config.validate_configuration()
        assert any("Voice features are disabled" in issue for issue in issues)
        
        # Test with voice input enabled but no STT service (except Whisper which is always available)
        config.voice_enabled = True
        config.voice_input_enabled = True
        config.elevenlabs_api_key = None
        config.elevenlabs_voice_id = None
        config.google_cloud_credentials_path = None
        config.google_cloud_project_id = None
        # Mock is_whisper_configured to return False for this test
        with patch.object(config, 'is_whisper_configured', return_value=False):
            with patch.dict(os.environ, {"VOICE_MOCK_MODE": "false"}, clear=True):
                issues = config.validate_configuration()
                assert any("No STT service configured" in issue for issue in issues)
        
        # Test with voice output enabled but no TTS service
        config.voice_input_enabled = False
        config.voice_output_enabled = True
        config.elevenlabs_api_key = None
        config.elevenlabs_voice_id = None
        config.piper_tts_model_path = None
        issues = config.validate_configuration()
        assert any("No TTS service configured" in issue for issue in issues)
        
        # Test with voice commands enabled but voice input disabled
        config.voice_output_enabled = False
        config.voice_commands_enabled = True
        issues = config.validate_configuration()
        assert any("Voice commands require voice input to be enabled" in issue for issue in issues)
        
        # Test with encryption enabled but no credentials directory
        config.voice_commands_enabled = False
        config.security.encryption_enabled = True
        with patch("os.path.exists", return_value=False):
            issues = config.validate_configuration()
            assert any("Encryption enabled but no credentials directory found" in issue for issue in issues)
        
        # Test with Google Cloud credentials file not found
        config.security.encryption_enabled = False
        config.google_cloud_credentials_path = "/nonexistent/path/credentials.json"
        config.google_cloud_project_id = "test-project"
        issues = config.validate_configuration()
        assert any("Google Cloud credentials file not found" in issue for issue in issues)
        
        # Test with Piper TTS model not found
        config.google_cloud_credentials_path = None
        config.google_cloud_project_id = None
        config.piper_tts_model_path = "/nonexistent/path/model"
        issues = config.validate_configuration()
        assert any("Piper TTS model not found" in issue for issue in issues)
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = VoiceConfig()
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert "voice_enabled" in result
        assert "voice_input_enabled" in result
        assert "voice_output_enabled" in result
        assert "voice_commands_enabled" in result
        assert "elevenlabs_configured" in result
        assert "openai_whisper_configured" in result
        assert "google_speech_configured" in result
        assert "whisper_configured" in result
        assert "piper_configured" in result
        assert "preferred_stt_service" in result
        assert "preferred_tts_service" in result
        assert "voice_profiles_count" in result
        assert "default_voice_profile" in result
        assert "security_encryption" in result
        assert "performance_cache" in result
        assert "performance_streaming" in result
        
        # Check values
        assert result["voice_enabled"] is True
        assert result["voice_input_enabled"] is True
        assert result["voice_output_enabled"] is True
        assert result["voice_commands_enabled"] is True
        assert isinstance(result["voice_profiles_count"], int)
        assert result["default_voice_profile"] == "calm_therapist"
        assert result["security_encryption"] is True
        assert result["performance_cache"] is True
        assert result["performance_streaming"] is True
    
    def test_from_env(self):
        """Test creating config from environment variables."""
        config = VoiceConfig.from_env()
        
        assert isinstance(config, VoiceConfig)
        assert config.voice_enabled is True  # Default value
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "voice_enabled": False,
            "voice_input_enabled": False,
            "voice_output_enabled": False,
            "voice_commands_enabled": False,
            "elevenlabs_api_key": "test_api_key",
            "elevenlabs_voice_id": "test_voice_id"
        }
        
        config = VoiceConfig.from_dict(data)
        
        assert config.voice_enabled is False
        assert config.voice_input_enabled is False
        assert config.voice_output_enabled is False
        assert config.voice_commands_enabled is False
        assert config.elevenlabs_api_key == "test_api_key"
        assert config.elevenlabs_voice_id == "test_voice_id"
    
    def test_to_json(self):
        """Test serializing configuration to JSON string."""
        config = VoiceConfig()
        
        result = config.to_json()
        
        assert isinstance(result, str)
        
        # Try to parse the JSON
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "voice_enabled" in data
    
    def test_from_json(self):
        """Test deserializing configuration from JSON string."""
        data = {
            "voice_enabled": False,
            "voice_input_enabled": False,
            "voice_output_enabled": False,
            "voice_commands_enabled": False,
            "elevenlabs_api_key": "test_api_key",
            "elevenlabs_voice_id": "test_voice_id"
        }
        
        json_str = json.dumps(data)
        config = VoiceConfig.from_json(json_str)
        
        assert isinstance(config, VoiceConfig)
        assert config.voice_enabled is False
        assert config.voice_input_enabled is False
        assert config.voice_output_enabled is False
        assert config.voice_commands_enabled is False
        assert config.elevenlabs_api_key == "test_api_key"
        assert config.elevenlabs_voice_id == "test_voice_id"
    
    def test_from_json_invalid(self):
        """Test deserializing configuration from invalid JSON string."""
        invalid_json = "{invalid json"
        
        config = VoiceConfig.from_json(invalid_json)
        
        # Should return a default config
        assert isinstance(config, VoiceConfig)
        assert config.voice_enabled is True  # Default value
    
    def test_security_property_accessors(self):
        """Test security property accessors for backwards compatibility."""
        config = VoiceConfig()
        
        # Test getting and setting encryption_key_rotation_days
        assert config.encryption_key_rotation_days == config.security.encryption_key_rotation_days
        config.encryption_key_rotation_days = 180
        assert config.encryption_key_rotation_days == 180
        assert config.security.encryption_key_rotation_days == 180
        
        # Test getting and setting data_retention_days
        assert config.data_retention_days == config.security.data_retention_days
        config.data_retention_days = 60
        assert config.data_retention_days == 60
        assert config.security.data_retention_days == 60
        
        # Test getting and setting audit_logging_enabled
        assert config.audit_logging_enabled == config.security.audit_logging_enabled
        config.audit_logging_enabled = False
        assert config.audit_logging_enabled is False
        assert config.security.audit_logging_enabled is False
        
        # Test getting and setting max_login_attempts
        assert config.max_login_attempts == config.security.max_login_attempts
        config.max_login_attempts = 5
        assert config.max_login_attempts == 5
        assert config.security.max_login_attempts == 5
        
        # Test getting and setting encryption_enabled
        assert config.encryption_enabled == config.security.encryption_enabled
        config.encryption_enabled = False
        assert config.encryption_enabled is False
        assert config.security.encryption_enabled is False
        
        # Test getting and setting consent_required
        assert config.consent_required == config.security.consent_required
        config.consent_required = False
        assert config.consent_required is False
        assert config.security.consent_required is False
        
        # Test getting and setting hipaa_compliance_enabled
        assert config.hipaa_compliance_enabled == config.security.hipaa_compliance_enabled
        config.hipaa_compliance_enabled = False
        assert config.hipaa_compliance_enabled is False
        assert config.security.hipaa_compliance_enabled is False
    
    def test_load_voice_profiles_from_directory(self):
        """Test loading voice profiles from directory."""
        config = VoiceConfig()
        
        # Create a temporary directory with profile files
        with tempfile.TemporaryDirectory() as temp_dir:
            config.voice_profile_path = temp_dir
            
            # Create profile files
            profile1_data = {
                "name": "test_profile_1",
                "description": "Test profile 1",
                "voice_id": "test_voice_id_1",
                "language": "en-US",
                "gender": "female"
            }
            
            profile2_data = {
                "name": "test_profile_2",
                "description": "Test profile 2",
                "voice_id": "test_voice_id_2",
                "language": "en-GB",
                "gender": "male"
            }
            
            with open(Path(temp_dir) / "test_profile_1.json", 'w') as f:
                json.dump(profile1_data, f)
            
            with open(Path(temp_dir) / "test_profile_2.json", 'w') as f:
                json.dump(profile2_data, f)
            
            # Create an invalid profile file
            with open(Path(temp_dir) / "invalid_profile.json", 'w') as f:
                f.write("{invalid json")
            
            # Load profiles
            config._load_voice_profiles()
            
            # Check that valid profiles were loaded
            assert "test_profile_1" in config.voice_profiles
            assert "test_profile_2" in config.voice_profiles
            assert config.voice_profiles["test_profile_1"].name == "test_profile_1"
            assert config.voice_profiles["test_profile_2"].name == "test_profile_2"
            
            # Check that invalid profile was skipped
            assert "invalid_profile" not in config.voice_profiles
    
    def test_create_default_profiles(self):
        """Test creating default voice profiles."""
        config = VoiceConfig()
        
        # Clear existing profiles
        config.voice_profiles = {}
        
        # Create default profiles
        config._create_default_profiles()
        
        # Check that default profiles were created
        assert "calm_therapist" in config.voice_profiles
        assert "empathetic" in config.voice_profiles
        assert "professional" in config.voice_profiles
        
        # Check profile attributes
        calm_profile = config.voice_profiles["calm_therapist"]
        assert calm_profile.name == "calm_therapist"
        assert calm_profile.description == "Calm and soothing voice for therapy sessions"
        assert calm_profile.voice_id == "default"
        assert calm_profile.language == "en-US"
        assert calm_profile.gender == "female"
        assert calm_profile.age == "adult"
        assert calm_profile.pitch == 0.9
        assert calm_profile.speed == 0.9
        assert calm_profile.volume == 0.8
        assert calm_profile.emotion == "calm"
        assert calm_profile.style == "conversational"
        assert "stability" in calm_profile.elevenlabs_settings
        assert "similarity_boost" in calm_profile.elevenlabs_settings
        assert "style" in calm_profile.elevenlabs_settings
        
        empathetic_profile = config.voice_profiles["empathetic"]
        assert empathetic_profile.name == "empathetic"
        assert empathetic_profile.description == "Empathetic and understanding voice"
        assert empathetic_profile.emotion == "empathetic"
        assert empathetic_profile.style == "caring"
        
        professional_profile = config.voice_profiles["professional"]
        assert professional_profile.name == "professional"
        assert professional_profile.description == "Professional and authoritative voice"
        assert professional_profile.gender == "male"
        assert professional_profile.emotion == "neutral"
        assert professional_profile.style == "professional"