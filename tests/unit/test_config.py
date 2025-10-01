"""
Unit tests for Voice Configuration functionality.

Tests comprehensive configuration management including:
- Voice configuration initialization
- Environment variable handling
- Configuration validation
- Default values management
- Configuration updates and persistence
- Voice profile management
- Audio settings configuration
- Security settings
- Provider configuration
- Feature toggles
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from dataclasses import asdict
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import test utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.test_utils import (
    setup_voice_module_mocks,
    clear_module_cache
)

# Set up mocks
setup_voice_module_mocks(project_root)

# Import config module
try:
    from voice.config import (
        VoiceConfig, VoiceProfile, AudioConfig, STTConfig, TTSConfig,
        SecurityConfig, load_config, save_config, validate_config,
        get_default_config, merge_configs, ConfigError
    )
except ImportError:
    # If config module fails to import, create minimal mocks
    VoiceConfig = MagicMock
    VoiceProfile = MagicMock
    AudioConfig = MagicMock
    STTConfig = MagicMock
    TTSConfig = MagicMock
    SecurityConfig = MagicMock
    load_config = MagicMock
    save_config = MagicMock
    validate_config = MagicMock
    get_default_config = MagicMock
    merge_configs = MagicMock
    ConfigError = Exception


class TestVoiceConfig:
    """Test Voice Configuration functionality."""

    @pytest.fixture
    def default_config(self):
        """Create default voice configuration."""
        return VoiceConfig()

    @pytest.fixture
    def custom_config_data(self):
        """Create custom configuration data."""
        return {
            "voice_enabled": True,
            "voice_input_enabled": True,
            "voice_output_enabled": True,
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "stt_provider": "openai",
            "stt_model": "whisper-1",
            "tts_provider": "openai",
            "tts_model": "tts-1",
            "tts_voice": "alloy",
            "encryption_enabled": True,
            "consent_required": True,
            "hipaa_compliance_enabled": True,
            "data_retention_days": 30,
            "session_timeout_minutes": 30
        }

    @pytest.fixture
    def mock_env_vars(self):
        """Create mock environment variables."""
        return {
            "VOICE_ENABLED": "true",
            "VOICE_INPUT_ENABLED": "true",
            "VOICE_OUTPUT_ENABLED": "true",
            "OPENAI_API_KEY": "test_openai_key",
            "ELEVENLABS_API_KEY": "test_elevenlabs_key",
            "VOICE_SAMPLE_RATE": "16000",
            "VOICE_CHANNELS": "1",
            "STT_PROVIDER": "openai",
            "TTS_PROVIDER": "openai",
            "ENCRYPTION_ENABLED": "true",
            "HIPAA_COMPLIANCE_ENABLED": "true"
        }

    def test_default_config_initialization(self, default_config):
        """Test default configuration initialization."""
        config = default_config

        # Test default values
        assert config.voice_enabled == True
        assert config.voice_input_enabled == True
        assert config.voice_output_enabled == True
        assert config.audio_sample_rate == 16000
        assert config.audio_channels == 1
        assert config.stt_provider == "openai"
        assert config.tts_provider == "openai"
        assert config.encryption_enabled == True
        assert config.consent_required == True
        assert config.hipaa_compliance_enabled == True

    def test_config_from_dict(self, custom_config_data):
        """Test configuration creation from dictionary."""
        config = VoiceConfig.from_dict(custom_config_data)

        assert config.voice_enabled == custom_config_data["voice_enabled"]
        assert config.audio_sample_rate == custom_config_data["audio_sample_rate"]
        assert config.stt_provider == custom_config_data["stt_provider"]
        assert config.tts_provider == custom_config_data["tts_provider"]
        assert config.encryption_enabled == custom_config_data["encryption_enabled"]

    def test_config_to_dict(self, default_config):
        """Test configuration conversion to dictionary."""
        config_dict = default_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "voice_enabled" in config_dict
        assert "audio_sample_rate" in config_dict
        assert "stt_provider" in config_dict
        assert "tts_provider" in config_dict
        assert "encryption_enabled" in config_dict

    def test_config_from_environment_variables(self, mock_env_vars):
        """Test configuration loading from environment variables."""
        with patch.dict(os.environ, mock_env_vars):
            config = VoiceConfig.from_env()

            assert config.voice_enabled == True
            assert config.voice_input_enabled == True
            assert config.voice_output_enabled == True
            assert config.audio_sample_rate == 16000
            assert config.audio_channels == 1
            assert config.stt_provider == "openai"
            assert config.tts_provider == "openai"
            assert config.encryption_enabled == True
            assert config.hipaa_compliance_enabled == True

    def test_config_validation(self, default_config):
        """Test configuration validation."""
        # Test valid configuration
        is_valid = validate_config(default_config)
        assert is_valid == True

        # Test invalid audio sample rate
        invalid_config = VoiceConfig()
        invalid_config.audio_sample_rate = 0  # Invalid
        is_valid = validate_config(invalid_config)
        assert is_valid == False

        # Test invalid channels
        invalid_config = VoiceConfig()
        invalid_config.audio_channels = 0  # Invalid
        is_valid = validate_config(invalid_config)
        assert is_valid == False

        # Test invalid provider
        invalid_config = VoiceConfig()
        invalid_config.stt_provider = "invalid_provider"
        is_valid = validate_config(invalid_config)
        assert is_valid == False

    def test_config_merge(self, default_config, custom_config_data):
        """Test configuration merging."""
        custom_config = VoiceConfig.from_dict(custom_config_data)

        # Merge configurations
        merged_config = merge_configs(default_config, custom_config)

        # Verify custom values override defaults
        assert merged_config.audio_sample_rate == custom_config_data["audio_sample_rate"]
        assert merged_config.stt_provider == custom_config_data["stt_provider"]
        assert merged_config.data_retention_days == custom_config_data["data_retention_days"]

        # Verify default values are preserved where not overridden
        assert hasattr(merged_config, 'voice_enabled')  # Should still have default attributes

    def test_config_save_and_load(self, default_config, custom_config_data):
        """Test configuration save and load functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save configuration
            custom_config = VoiceConfig.from_dict(custom_config_data)
            save_success = save_config(custom_config, temp_path)
            assert save_success == True

            # Load configuration
            loaded_config = load_config(temp_path)
            assert loaded_config is not None
            assert loaded_config.audio_sample_rate == custom_config_data["audio_sample_rate"]
            assert loaded_config.stt_provider == custom_config_data["stt_provider"]

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_voice_profile_creation(self):
        """Test voice profile creation and management."""
        profile_data = {
            "name": "calm_therapist",
            "gender": "female",
            "age_range": "adult",
            "language": "en",
            "accent": "neutral",
            "provider": "openai",
            "voice_id": "alloy",
            "settings": {
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 0.9,
                "emotion": "calm"
            }
        }

        profile = VoiceProfile.from_dict(profile_data)

        assert profile.name == "calm_therapist"
        assert profile.gender == "female"
        assert profile.age_range == "adult"
        assert profile.language == "en"
        assert profile.provider == "openai"
        assert profile.voice_id == "alloy"
        assert profile.settings["speed"] == 1.0
        assert profile.settings["emotion"] == "calm"

    def test_audio_config_settings(self):
        """Test audio configuration settings."""
        audio_config = AudioConfig()

        # Test default audio settings
        assert audio_config.sample_rate == 16000
        assert audio_config.channels == 1
        assert audio_config.chunk_size == 1024
        assert audio_config.format == "wav"

        # Test audio configuration validation
        audio_config.sample_rate = 44100
        audio_config.channels = 2
        audio_config.chunk_size = 2048

        assert audio_config.sample_rate == 44100
        assert audio_config.channels == 2
        assert audio_config.chunk_size == 2048

    def test_stt_config_settings(self):
        """Test speech-to-text configuration settings."""
        stt_config = STTConfig()

        # Test default STT settings
        assert stt_config.provider == "openai"
        assert stt_config.model == "whisper-1"
        assert stt_config.language == "en"
        assert stt_config.confidence_threshold == 0.7

        # Test STT configuration updates
        stt_config.provider = "google"
        stt_config.model = "speech_v1"
        stt_config.confidence_threshold = 0.8

        assert stt_config.provider == "google"
        assert stt_config.model == "speech_v1"
        assert stt_config.confidence_threshold == 0.8

    def test_tts_config_settings(self):
        """Test text-to-speech configuration settings."""
        tts_config = TTSConfig()

        # Test default TTS settings
        assert tts_config.provider == "openai"
        assert tts_config.model == "tts-1"
        assert tts_config.voice == "alloy"
        assert tts_config.speed == 1.0
        assert tts_config.volume == 1.0

        # Test TTS configuration updates
        tts_config.provider = "elevenlabs"
        tts_config.voice = "rachel"
        tts_config.speed = 1.2
        tts_config.volume = 0.8

        assert tts_config.provider == "elevenlabs"
        assert tts_config.voice == "rachel"
        assert tts_config.speed == 1.2
        assert tts_config.volume == 0.8

    def test_security_config_settings(self):
        """Test security configuration settings."""
        security_config = SecurityConfig()

        # Test default security settings
        assert security_config.encryption_enabled == True
        assert security_config.consent_required == True
        assert security_config.hipaa_compliance_enabled == True
        assert security_config.data_retention_days == 30
        assert security_config.session_timeout_minutes == 30

        # Test security configuration updates
        security_config.data_retention_days = 90
        security_config.session_timeout_minutes = 60
        security_config.encryption_key_rotation_days = 180

        assert security_config.data_retention_days == 90
        assert security_config.session_timeout_minutes == 60
        assert security_config.encryption_key_rotation_days == 180

    def test_config_error_handling(self):
        """Test configuration error handling."""
        # Test invalid configuration file
        with patch('builtins.open', side_effect=FileNotFoundError("Config file not found")):
            with pytest.raises(FileNotFoundError):
                load_config("nonexistent_config.json")

        # Test invalid JSON in configuration
        with patch('builtins.open', mock_open(read_data="invalid json content")):
            with pytest.raises(json.JSONDecodeError):
                load_config("invalid_config.json")

        # Test missing required fields
        incomplete_config_data = {
            "voice_enabled": True
            # Missing required fields
        }

        with pytest.raises((KeyError, ValueError, ConfigError)):
            VoiceConfig.from_dict(incomplete_config_data)

    def test_environment_variable_override(self, default_config):
        """Test environment variable override functionality."""
        env_overrides = {
            "VOICE_SAMPLE_RATE": "22050",
            "VOICE_CHANNELS": "2",
            "STT_PROVIDER": "google",
            "TTS_VOICE": "echo",
            "DATA_RETENTION_DAYS": "60"
        }

        with patch.dict(os.environ, env_overrides):
            config = VoiceConfig.from_env()

            assert config.audio_sample_rate == 22050
            assert config.audio_channels == 2
            assert config.stt_provider == "google"
            assert config.tts_voice == "echo"
            assert config.data_retention_days == 60

    def test_config_type_validation(self):
        """Test configuration type validation."""
        # Test with invalid types
        invalid_configs = [
            {"audio_sample_rate": "not_a_number"},  # Should be int
            {"voice_enabled": "not_a_boolean"},     # Should be bool
            {"stt_provider": 123},                  # Should be str
            {"data_retention_days": -1}              # Should be positive
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError, ConfigError)):
                VoiceConfig.from_dict(invalid_config)

    def test_config_defaults_getter(self):
        """Test default configuration getter."""
        default_config = get_default_config()

        assert default_config is not None
        assert isinstance(default_config, VoiceConfig)
        assert default_config.voice_enabled == True
        assert default_config.audio_sample_rate == 16000
        assert default_config.stt_provider == "openai"
        assert default_config.tts_provider == "openai"

    def test_config_copy_and_equality(self, default_config, custom_config_data):
        """Test configuration copying and equality comparison."""
        # Create config from custom data
        original_config = VoiceConfig.from_dict(custom_config_data)

        # Test copying
        copied_config = original_config.copy()
        assert copied_config is not original_config  # Different objects
        assert copied_config.audio_sample_rate == original_config.audio_sample_rate
        assert copied_config.stt_provider == original_config.stt_provider

        # Test equality
        assert copied_config == original_config

        # Test inequality
        modified_config = original_config.copy()
        modified_config.audio_sample_rate = 22050
        assert modified_config != original_config

    def test_config_serialization(self, custom_config_data):
        """Test configuration serialization to JSON."""
        config = VoiceConfig.from_dict(custom_config_data)

        # Serialize to JSON
        json_str = config.to_json()
        assert json_str is not None
        assert isinstance(json_str, str)

        # Deserialize from JSON
        deserialized_config = VoiceConfig.from_json(json_str)
        assert deserialized_config.audio_sample_rate == config.audio_sample_rate
        assert deserialized_config.stt_provider == config.stt_provider

    def test_config_feature_flags(self):
        """Test configuration feature flags."""
        config = VoiceConfig()

        # Test default feature flags
        assert hasattr(config, 'voice_enabled')
        assert hasattr(config, 'voice_input_enabled')
        assert hasattr(config, 'voice_output_enabled')
        assert hasattr(config, 'voice_commands_enabled')
        assert hasattr(config, 'encryption_enabled')
        assert hasattr(config, 'hipaa_compliance_enabled')

        # Test feature flag updates
        config.voice_commands_enabled = False
        config.encryption_enabled = False

        assert config.voice_commands_enabled == False
        assert config.encryption_enabled == False

    def test_config_provider_specific_settings(self):
        """Test provider-specific configuration settings."""
        config = VoiceConfig()

        # Test OpenAI specific settings
        config.openai_model = "gpt-4"
        config.openai_temperature = 0.7
        config.openai_max_tokens = 1000

        assert config.openai_model == "gpt-4"
        assert config.openai_temperature == 0.7
        assert config.openai_max_tokens == 1000

        # Test ElevenLabs specific settings
        config.elevenlabs_voice_id = "rachel"
        config.elevenlabs_model_id = "eleven_multilingual_v2"
        config.elevenlabs_stability = 0.75

        assert config.elevenlabs_voice_id == "rachel"
        assert config.elevenlabs_model_id == "eleven_multilingual_v2"
        assert config.elevenlabs_stability == 0.75

    def test_config_audio_quality_settings(self):
        """Test audio quality configuration settings."""
        config = VoiceConfig()

        # Test audio quality settings
        config.audio_quality_preset = "high"
        config.noise_reduction_enabled = True
        config.echo_cancellation_enabled = True
        config.auto_gain_control_enabled = True

        assert config.audio_quality_preset == "high"
        assert config.noise_reduction_enabled == True
        assert config.echo_cancellation_enabled == True
        assert config.auto_gain_control_enabled == True

    def test_config_performance_settings(self):
        """Test performance-related configuration settings."""
        config = VoiceConfig()

        # Test performance settings
        config.max_concurrent_requests = 10
        config.request_timeout_seconds = 30
        config.cache_enabled = True
        config.cache_size_mb = 100
        config.optimization_level = "balanced"

        assert config.max_concurrent_requests == 10
        assert config.request_timeout_seconds == 30
        assert config.cache_enabled == True
        assert config.cache_size_mb == 100
        assert config.optimization_level == "balanced"

    def test_config_logging_settings(self):
        """Test logging configuration settings."""
        config = VoiceConfig()

        # Test logging settings
        config.log_level = "INFO"
        config.log_file_path = "/tmp/voice_app.log"
        config.log_rotation_enabled = True
        config.max_log_size_mb = 10
        config.log_retention_days = 7

        assert config.log_level == "INFO"
        assert config.log_file_path == "/tmp/voice_app.log"
        assert config.log_rotation_enabled == True
        assert config.max_log_size_mb == 10
        assert config.log_retention_days == 7

    def test_config_development_settings(self):
        """Test development and debugging configuration settings."""
        config = VoiceConfig()

        # Test development settings
        config.debug_mode = False
        config.development_mode = False
        config.api_endpoint_override = None
        config.mock_audio_input = False
        config.save_debug_logs = False

        assert config.debug_mode == False
        assert config.development_mode == False
        assert config.api_endpoint_override is None
        assert config.mock_audio_input == False
        assert config.save_debug_logs == False

    def test_config_compatibility_check(self):
        """Test configuration compatibility checking."""
        current_config = VoiceConfig()
        current_config.version = "2.0.0"

        # Test version compatibility
        compatible_versions = ["2.0.0", "2.1.0", "2.0.1"]
        for version in compatible_versions:
            is_compatible = current_config.is_compatible_with_version(version)
            assert is_compatible == True

        incompatible_versions = ["1.0.0", "3.0.0"]
        for version in incompatible_versions:
            is_compatible = current_config.is_compatible_with_version(version)
            assert is_compatible == False

    def test_config_migration(self):
        """Test configuration migration between versions."""
        old_config_data = {
            "voice_enabled": True,
            "sample_rate": 16000,  # Old field name
            "stt_provider": "openai"
            # Missing new fields
        }

        # Migrate old config to new format
        migrated_config = VoiceConfig.migrate_from_version(old_config_data, "1.0.0")

        assert migrated_config is not None
        assert migrated_config.voice_enabled == True
        assert migrated_config.audio_sample_rate == 16000  # Migrated from old field
        assert migrated_config.stt_provider == "openai"
        # New fields should have default values
        assert hasattr(migrated_config, 'voice_input_enabled')

    def test_config_template_generation(self):
        """Test configuration template generation."""
        # Generate configuration template
        template = VoiceConfig.generate_template()

        assert isinstance(template, dict)
        assert "voice_enabled" in template
        assert "audio_sample_rate" in template
        assert "stt_provider" in template
        assert "tts_provider" in template

        # Generate template with comments
        template_with_comments = VoiceConfig.generate_template_with_comments()
        assert isinstance(template_with_comments, str)
        assert "voice_enabled" in template_with_comments

    def test_config_environment_specific_loading(self):
        """Test loading configuration for specific environments."""
        environments = ["development", "staging", "production"]

        for env in environments:
            with patch.dict(os.environ, {"APP_ENVIRONMENT": env}):
                config = VoiceConfig.load_for_environment(env)

                assert config is not None
                # Environment-specific settings should be applied
                if env == "production":
                    assert config.debug_mode == False
                    assert config.log_level in ["INFO", "WARNING", "ERROR"]
                elif env == "development":
                    assert config.debug_mode == True

    def test_config_sensitivity_validation(self):
        """Test validation of sensitive configuration values."""
        config = VoiceConfig()

        # Test sensitive data handling
        sensitive_fields = [
            'openai_api_key',
            'elevenlabs_api_key',
            'encryption_key',
            'database_password'
        ]

        for field in sensitive_fields:
            # Set sensitive value
            setattr(config, field, "sensitive_value_123")

            # Verify value is masked in string representation
            config_str = str(config)
            assert "sensitive_value_123" not in config_str
            assert field in config_str

    def test_config_update_validation(self, default_config):
        """Test validation during configuration updates."""
        config = default_config

        # Test valid update
        valid_updates = {
            "audio_sample_rate": 22050,
            "voice_enabled": False,
            "data_retention_days": 90
        }

        update_success = config.update_config(valid_updates)
        assert update_success == True
        assert config.audio_sample_rate == 22050
        assert config.voice_enabled == False
        assert config.data_retention_days == 90

        # Test invalid update
        invalid_updates = {
            "audio_sample_rate": -1,  # Invalid
            "voice_enabled": "not_boolean"  # Invalid type
        }

        update_success = config.update_config(invalid_updates)
        assert update_success == False

    def test_config_backup_and_restore(self, custom_config_data):
        """Test configuration backup and restore functionality."""
        original_config = VoiceConfig.from_dict(custom_config_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "config_backup.json"

            # Create backup
            backup_success = original_config.create_backup(backup_path)
            assert backup_success == True
            assert backup_path.exists()

            # Modify original config
            original_config.audio_sample_rate = 48000

            # Restore from backup
            restore_success = original_config.restore_from_backup(backup_path)
            assert restore_success == True
            assert original_config.audio_sample_rate == custom_config_data["audio_sample_rate"]

    def test_config_diff_generation(self):
        """Test configuration difference generation."""
        config1 = VoiceConfig()
        config1.audio_sample_rate = 16000
        config1.voice_enabled = True

        config2 = VoiceConfig()
        config2.audio_sample_rate = 22050
        config2.voice_enabled = False

        # Generate diff
        diff = VoiceConfig.generate_diff(config1, config2)

        assert isinstance(diff, dict)
        assert "audio_sample_rate" in diff
        assert "voice_enabled" in diff
        assert diff["audio_sample_rate"]["old"] == 16000
        assert diff["audio_sample_rate"]["new"] == 22050
        assert diff["voice_enabled"]["old"] == True
        assert diff["voice_enabled"]["new"] == False

    @pytest.mark.parametrize("config_scenario", [
        "minimal_config",
        "maximal_config",
        "invalid_audio_config",
        "missing_api_keys",
        "conflicting_settings"
    ])
    def test_config_scenarios(self, config_scenario):
        """Test various configuration scenarios."""
        if config_scenario == "minimal_config":
            # Test minimal required configuration
            minimal_data = {
                "voice_enabled": True,
                "audio_sample_rate": 16000
            }

            config = VoiceConfig.from_dict(minimal_data)
            assert config.voice_enabled == True
            assert config.audio_sample_rate == 16000
            # Other fields should have defaults

        elif config_scenario == "maximal_config":
            # Test configuration with all possible settings
            maximal_data = {
                "voice_enabled": True,
                "voice_input_enabled": True,
                "voice_output_enabled": True,
                "voice_commands_enabled": True,
                "audio_sample_rate": 48000,
                "audio_channels": 2,
                "stt_provider": "openai",
                "stt_model": "whisper-1",
                "tts_provider": "elevenlabs",
                "tts_model": "eleven_multilingual_v2",
                "encryption_enabled": True,
                "hipaa_compliance_enabled": True,
                "debug_mode": True,
                "log_level": "DEBUG"
            }

            config = VoiceConfig.from_dict(maximal_data)
            assert config.audio_sample_rate == 48000
            assert config.audio_channels == 2
            assert config.debug_mode == True

        elif config_scenario == "invalid_audio_config":
            # Test with invalid audio configuration
            invalid_data = {
                "audio_sample_rate": 0,  # Invalid
                "audio_channels": -1     # Invalid
            }

            with pytest.raises((ValueError, ConfigError)):
                VoiceConfig.from_dict(invalid_data)

        elif config_scenario == "missing_api_keys":
            # Test configuration with missing API keys
            config = VoiceConfig()
            config.stt_provider = "openai"
            config.tts_provider = "elevenlabs"

            # Should detect missing API keys
            missing_keys = config.get_missing_api_keys()
            assert "openai_api_key" in missing_keys
            assert "elevenlabs_api_key" in missing_keys

        elif config_scenario == "conflicting_settings":
            # Test configuration with conflicting settings
            conflicting_data = {
                "voice_enabled": False,
                "voice_input_enabled": True,  # Conflicts with voice_enabled
                "encryption_enabled": True,
                "hipaa_compliance_enabled": False  # May conflict
            }

            config = VoiceConfig.from_dict(conflicting_data)
            warnings = config.detect_conflicts()
            assert len(warnings) > 0