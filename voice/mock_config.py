"""
Mock configuration for testing voice services.

This module provides mock configurations for testing voice services
without requiring actual API keys or external services.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Mock security configuration for testing."""
    encryption_enabled: bool = True
    consent_required: bool = False
    hipaa_compliance_enabled: bool = True
    audit_logging_enabled: bool = True
    data_retention_days: int = 30
    max_login_attempts: int = 3
    encryption_key_rotation_days: int = 90


@dataclass
class AudioConfig:
    """Mock audio configuration for testing."""
    max_buffer_size: int = 300
    max_memory_mb: int = 100
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = 'wav'
    stream_buffer_size: int = 10
    stream_chunk_duration: float = 0.1
    compression_enabled: bool = True
    compression_level: int = 6


@dataclass
class MockVoiceConfig:
    """Mock voice configuration for testing."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    buffer_size: int = 1024
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    noise_reduction: bool = True
    echo_cancellation: bool = True
    auto_gain_control: bool = True


@dataclass
class MockSecurityConfig:
    """Mock security configuration for testing."""
    encryption_enabled: bool = True
    consent_required: bool = False
    hipaa_compliance_enabled: bool = True
    audit_logging_enabled: bool = True
    data_retention_days: int = 30
    max_login_attempts: int = 3
    encryption_key_rotation_days: int = 90


@dataclass
class MockAudioConfig:
    """Mock audio configuration for testing."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    buffer_size: int = 1024
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    noise_reduction: bool = True
    echo_cancellation: bool = True
    auto_gain_control: bool = True


class MockAuditLogger:
    """Mock audit logger for testing security features."""
    
    def __init__(self):
        self.events = []
    
    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any]) -> None:
        """Log a security event."""
        self.events.append({
            'type': event_type,
            'user_id': user_id,
            'details': details,
            'timestamp': None  # Would be actual timestamp in real implementation
        })
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all logged events."""
        return self.events


class VoiceSecurity:
    """Mock voice security implementation for testing."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_logger = MockAuditLogger()
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a security event."""
        self.audit_logger.log_event(event_type, "test_user", details)


class MockConfig:
    """Mock configuration for testing voice services."""
    
    def __init__(self):
        # Audio settings
        self.audio_sample_rate = 16000
        self.audio_channels = 1
        self.audio_buffer_size = 1024
        
        # STT/TTS settings
        self.stt_provider = "mock"
        self.tts_provider = "mock"
        self.stt_model = "base"
        self.tts_model = "base"
        self.tts_voice = "default"
        
        # Voice settings
        self.voice_enabled = True
        self.voice_input_enabled = True
        self.voice_output_enabled = True
        self.default_voice_profile = 'default'
        self.voice_commands_enabled = True
        self.voice_speed = 1.0
        self.voice_pitch = 1.0
        self.voice_volume = 0.8
        
        # Security settings
        self.encryption_enabled = True
        self.consent_required = False
        self.hipaa_compliance_enabled = True
        self.audit_logging_enabled = True
        self.data_retention_days = 30
        self.max_login_attempts = 3
        self.encryption_key_rotation_days = 90
        
        # API keys (mock)
        self.openai_api_key = "mock_openai_key"
        self.elevenlabs_api_key = "mock_elevenlabs_key"
        self.elevenlabs_voice_id = "mock_voice_id"
        self.elevenlabs_model_id = "mock_model_id"
        self.elevenlabs_stability = 0.5
        self.google_api_key = "mock_google_key"
        
        # Debug settings
        self.debug_mode = True
        self.mock_audio_input = True
        self.save_debug_logs = True
        
        # Performance settings
        self.max_concurrent_requests = 5
        self.request_timeout_seconds = 30
        self.cache_size_mb = 100
        self.optimization_level = "balanced"
        
        # Logging settings
        self.log_level = "INFO"
        import tempfile
        # Create a secure temporary file with proper permissions
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.log', 
            prefix='voice_service_',
            delete=False
        )
        temp_file.close()  # Close the file handle but keep the path
        self.log_file_path = temp_file.name
        self.log_rotation_enabled = True
        self.max_log_size_mb = 10
        self.log_retention_days = 7
        
        # Version
        self.version = "1.0.0"
        
        # Create nested audio config object
        self.audio = AudioConfig()
        
        # Initialize security
        self.security = VoiceSecurity(SecurityConfig())
        
        # Voice profiles
        self.voice_profiles = {
            'default': MockVoiceConfig(),
            'calm_therapist': MockVoiceConfig()
        }
    
    def get_preferred_stt_service(self) -> str:
        """Get the preferred STT service."""
        if self.is_openai_whisper_configured():
            return "openai"
        elif self.is_google_speech_configured():
            return "google"
        elif self.is_whisper_configured():
            return "whisper"
        else:
            return "mock"
    
    def get_preferred_tts_service(self) -> str:
        """Get the preferred TTS service."""
        if self.is_elevenlabs_configured():
            return "elevenlabs"
        elif self.is_piper_configured():
            return "piper"
        else:
            return "mock"
    
    def is_openai_whisper_configured(self) -> bool:
        """Check if OpenAI Whisper is configured."""
        return bool(self.openai_api_key)
    
    def is_openai_tts_configured(self) -> bool:
        """Check if OpenAI TTS is configured."""
        return bool(self.openai_api_key)
    
    def is_google_speech_configured(self) -> bool:
        """Check if Google Speech is configured."""
        return bool(self.google_api_key)
    
    def is_whisper_configured(self) -> bool:
        """Check if Whisper is configured."""
        return True  # Always true for mock
    
    def is_elevenlabs_configured(self) -> bool:
        """Check if ElevenLabs is configured."""
        return bool(self.elevenlabs_api_key)
    
    def is_piper_configured(self) -> bool:
        """Check if Piper TTS is configured."""
        return True  # Always true for mock
    
    def get_voice_profile(self, profile_name: Optional[str] = None) -> MockVoiceConfig:
        """Get voice profile by name or default."""
        if profile_name and profile_name in self.voice_profiles:
            return self.voice_profiles[profile_name]
        elif self.default_voice_profile in self.voice_profiles:
            return self.voice_profiles[self.default_voice_profile]
        else:
            return self.voice_profiles.get('default', MockVoiceConfig())
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        if not self.voice_enabled:
            issues.append("Voice features are disabled")
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'audio_sample_rate': self.audio_sample_rate,
            'audio_channels': self.audio_channels,
            'stt_provider': self.stt_provider,
            'tts_provider': self.tts_provider,
            'debug_mode': self.debug_mode,
            'mock_audio_input': self.mock_audio_input,
            'encryption_enabled': self.encryption_enabled,
            'consent_required': self.consent_required,
            'hipaa_compliance_enabled': self.hipaa_compliance_enabled,
            'audit_logging_enabled': self.audit_logging_enabled,
            'data_retention_days': self.data_retention_days,
            'max_login_attempts': self.max_login_attempts,
            'encryption_key_rotation_days': self.encryption_key_rotation_days,
            'openai_api_key': self.openai_api_key,
            'elevenlabs_api_key': self.elevenlabs_api_key,
            'elevenlabs_voice_id': self.elevenlabs_voice_id,
            'elevenlabs_model_id': self.elevenlabs_model_id,
            'elevenlabs_stability': self.elevenlabs_stability,
            'google_api_key': self.google_api_key,
            'max_concurrent_requests': self.max_concurrent_requests,
            'request_timeout_seconds': self.request_timeout_seconds,
            'cache_size_mb': self.cache_size_mb,
            'optimization_level': self.optimization_level,
            'log_level': self.log_level,
            'log_file_path': self.log_file_path,
            'log_rotation_enabled': self.log_rotation_enabled,
            'max_log_size_mb': self.max_log_size_mb,
            'log_retention_days': self.log_retention_days,
            'version': self.version
        }
    
    def copy(self) -> 'MockConfig':
        """Create a copy of the configuration."""
        return MockConfig()
    
    def __eq__(self, other) -> bool:
        """Check if two configurations are equal."""
        if not isinstance(other, MockConfig):
            return False
        return self.to_dict() == other.to_dict()


def create_mock_voice_config() -> MockConfig:
    """Create a mock voice configuration for testing."""
    return MockConfig()