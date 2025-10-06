"""
Tests for voice configuration module
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Test the voice configuration
class TestVoiceConfig:
    """Test VoiceConfig class."""
    
    def test_config_initialization_default(self):
        """Test default configuration initialization."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig()
        
        # Verify default values
        assert config.voice_enabled is True
        assert config.stt_provider == "openai"
        assert config.tts_provider == "openai"
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.format == "int16"
    
    @patch.dict(os.environ, {
        'VOICE_ENABLED': 'false',
        'STT_PROVIDER': 'google',
        'TTS_PROVIDER': 'elevenlabs',
        'OPENAI_API_KEY': 'test_openai_key',
        'ELEVENLABS_API_KEY': 'test_elevenlabs_key',
        'ELEVENLABS_VOICE_ID': 'test_voice_id',
        'GOOGLE_CLOUD_CREDENTIALS_PATH': '/test/path'
    })
    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig()
        
        # Verify environment values are loaded
        assert config.voice_enabled is False
        assert config.stt_provider == "google"
        assert config.tts_provider == "elevenlabs"
        assert config.openai_api_key == "test_openai_key"
        assert config.elevenlabs_api_key == "test_elevenlabs_key"
        assert config.elevenlabs_voice_id == "test_voice_id"
        assert config.google_cloud_credentials_path == "/test/path"
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig(
            voice_enabled=False,
            stt_provider="whisper",
            tts_provider="piper",
            sample_rate=48000,
            channels=2,
            chunk_size=2048
        )
        
        # Verify custom values
        assert config.voice_enabled is False
        assert config.stt_provider == "whisper"
        assert config.tts_provider == "piper"
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.chunk_size == 2048
    
    def test_get_preferred_stt_service(self):
        """Test getting preferred STT service."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig(stt_provider="openai")
        assert config.get_preferred_stt_service() == "openai"
        
        config = VoiceConfig(stt_provider="google")
        assert config.get_preferred_stt_service() == "google"
        
        config = VoiceConfig(stt_provider="whisper")
        assert config.get_preferred_stt_service() == "whisper"
    
    def test_get_preferred_tts_service(self):
        """Test getting preferred TTS service."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig(tts_provider="openai")
        assert config.get_preferred_tts_service() == "openai"
        
        config = VoiceConfig(tts_provider="elevenlabs")
        assert config.get_preferred_tts_service() == "elevenlabs"
        
        config = VoiceConfig(tts_provider="piper")
        assert config.get_preferred_tts_service() == "piper"
    
    def test_is_stt_available(self):
        """Test STT availability check."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig()
        
        # With API key
        config.openai_api_key = "test_key"
        assert config.is_stt_available("openai") is True
        
        # Without API key
        config.openai_api_key = None
        assert config.is_stt_available("openai") is False
        
        # Invalid provider
        assert config.is_stt_available("invalid") is False
    
    def test_is_tts_available(self):
        """Test TTS availability check."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig()
        
        # With API key
        config.openai_api_key = "test_key"
        assert config.is_tts_available("openai") is True
        
        # Without API key
        config.openai_api_key = None
        assert config.is_tts_available("openai") is False
        
        # Invalid provider
        assert config.is_tts_available("invalid") is False
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        from voice.config import VoiceConfig
        
        config = VoiceConfig(
            voice_enabled=True,
            stt_provider="openai",
            tts_provider="elevenlabs"
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['voice_enabled'] is True
        assert result['stt_provider'] == "openai"
        assert result['tts_provider'] == "elevenlabs"
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        from voice.config import VoiceConfig
        
        data = {
            'voice_enabled': False,
            'stt_provider': 'google',
            'tts_provider': 'piper',
            'sample_rate': 48000
        }
        
        config = VoiceConfig.from_dict(data)
        
        assert config.voice_enabled is False
        assert config.stt_provider == "google"
        assert config.tts_provider == "piper"
        assert config.sample_rate == 48000
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        from voice.config import VoiceConfig
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Create and save config
            config = VoiceConfig(
                voice_enabled=False,
                stt_provider="whisper",
                tts_provider="piper"
            )
            config.save_to_file(config_path)
            
            # Load config
            loaded_config = VoiceConfig.load_from_file(config_path)
            
            # Verify loaded config
            assert loaded_config.voice_enabled is False
            assert loaded_config.stt_provider == "whisper"
            assert loaded_config.tts_provider == "piper"
        finally:
            os.unlink(config_path)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        from voice.config import VoiceConfig
        
        # Should return default config
        config = VoiceConfig.load_from_file("/nonexistent/path.json")
        assert config.voice_enabled is True  # Default value
    
    def test_validate_config(self):
        """Test configuration validation."""
        from voice.config import VoiceConfig
        
        # Valid config
        config = VoiceConfig()
        assert config.validate() is True
        
        # Invalid sample rate
        config.sample_rate = -1
        assert config.validate() is False
        
        # Invalid channels
        config.sample_rate = 16000
        config.channels = 0
        assert config.validate() is False
        
        # Invalid chunk size
        config.channels = 1
        config.chunk_size = -1
        assert config.validate() is False


class TestVoiceProfile:
    """Test VoiceProfile class."""
    
    def test_profile_initialization(self):
        """Test voice profile initialization."""
        from voice.config import VoiceProfile
        
        profile = VoiceProfile(
            name="Test Profile",
            provider="openai",
            voice_id="test_voice",
            speed=1.0,
            pitch=1.0,
            volume=0.8
        )
        
        assert profile.name == "Test Profile"
        assert profile.provider == "openai"
        assert profile.voice_id == "test_voice"
        assert profile.speed == 1.0
        assert profile.pitch == 1.0
        assert profile.volume == 0.8
    
    def test_profile_defaults(self):
        """Test voice profile default values."""
        from voice.config import VoiceProfile
        
        profile = VoiceProfile(name="Default")
        
        assert profile.name == "Default"
        assert profile.provider == "openai"
        assert profile.voice_id == "alloy"
        assert profile.speed == 1.0
        assert profile.pitch == 1.0
        assert profile.volume == 0.9
    
    def test_profile_to_dict(self):
        """Test converting profile to dictionary."""
        from voice.config import VoiceProfile
        
        profile = VoiceProfile(
            name="Test",
            provider="elevenlabs",
            voice_id="voice_123"
        )
        
        result = profile.to_dict()
        
        assert isinstance(result, dict)
        assert result['name'] == "Test"
        assert result['provider'] == "elevenlabs"
        assert result['voice_id'] == "voice_123"
    
    def test_profile_from_dict(self):
        """Test creating profile from dictionary."""
        from voice.config import VoiceProfile
        
        data = {
            'name': 'Test Profile',
            'provider': 'piper',
            'voice_id': 'piper_voice',
            'speed': 1.2,
            'pitch': 0.9
        }
        
        profile = VoiceProfile.from_dict(data)
        
        assert profile.name == "Test Profile"
        assert profile.provider == "piper"
        assert profile.voice_id == "piper_voice"
        assert profile.speed == 1.2
        assert profile.pitch == 0.9
    
    def test_get_therapeutic_profiles(self):
        """Test getting therapeutic voice profiles."""
        from voice.config import get_therapeutic_profiles
        
        profiles = get_therapeutic_profiles()
        
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        
        # Check profile structure
        for profile in profiles:
            assert isinstance(profile, VoiceProfile)
            assert profile.name
            assert profile.provider
            assert profile.voice_id
    
    def test_get_profile_by_name(self):
        """Test getting profile by name."""
        from voice.config import get_profile_by_name
        
        # Get existing profile
        profile = get_profile_by_name("calm")
        assert profile is not None
        assert profile.name == "calm"
        
        # Get non-existent profile
        profile = get_profile_by_name("nonexistent")
        assert profile is None
    
    def test_create_custom_profile(self):
        """Test creating custom profile."""
        from voice.config import create_custom_profile
        
        profile = create_custom_profile(
            name="Custom",
            provider="openai",
            voice_id="custom_voice",
            speed=0.9,
            volume=0.7
        )
        
        assert profile.name == "Custom"
        assert profile.provider == "openai"
        assert profile.voice_id == "custom_voice"
        assert profile.speed == 0.9
        assert profile.volume == 0.7


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        from voice.config import get_default_config
        
        config = get_default_config()
        
        assert isinstance(config, VoiceConfig)
        assert config.voice_enabled is True
        assert config.stt_provider == "openai"
        assert config.tts_provider == "openai"
    
    def test_merge_configs(self):
        """Test merging configurations."""
        from voice.config import VoiceConfig, merge_configs
        
        base_config = VoiceConfig(
            voice_enabled=True,
            stt_provider="openai",
            sample_rate=16000
        )
        
        override_config = VoiceConfig(
            voice_enabled=False,
            tts_provider="elevenlabs",
            sample_rate=48000
        )
        
        merged = merge_configs(base_config, override_config)
        
        # Should use override values
        assert merged.voice_enabled is False
        assert merged.tts_provider == "elevenlabs"
        assert merged.sample_rate == 48000
        # Should keep base value for non-overridden fields
        assert merged.stt_provider == "openai"
    
    def test_validate_provider(self):
        """Test provider validation."""
        from voice.config import validate_provider
        
        # Valid providers
        assert validate_provider("openai", "stt") is True
        assert validate_provider("openai", "tts") is True
        assert validate_provider("elevenlabs", "tts") is True
        assert validate_provider("google", "stt") is True
        assert validate_provider("whisper", "stt") is True
        assert validate_provider("piper", "tts") is True
        
        # Invalid providers
        assert validate_provider("invalid", "stt") is False
        assert validate_provider("invalid", "tts") is False
        
        # Invalid service type
        assert validate_provider("openai", "invalid") is False
    
    def test_get_provider_requirements(self):
        """Test getting provider requirements."""
        from voice.config import get_provider_requirements
        
        # OpenAI requirements
        reqs = get_provider_requirements("openai")
        assert "api_key" in reqs
        assert reqs["api_key"] is True
        
        # ElevenLabs requirements
        reqs = get_provider_requirements("elevenlabs")
        assert "api_key" in reqs
        assert "voice_id" in reqs
        
        # Google requirements
        reqs = get_provider_requirements("google")
        assert "credentials_path" in reqs
        
        # Invalid provider
        reqs = get_provider_requirements("invalid")
        assert reqs == {}