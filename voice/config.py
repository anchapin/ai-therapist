"""
Voice Configuration Management

This module handles all configuration aspects of the voice features including:
- Environment variable management
- Voice profile definitions
- Audio processing settings
- Security and privacy settings
- Performance optimization settings
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class VoiceProfile:
    """Voice profile configuration for TTS customization."""
    name: str
    description: str
    voice_id: str
    language: str = "en-US"
    gender: str = "neutral"
    age: str = "adult"
    pitch: float = 1.0
    speed: float = 1.0
    volume: float = 1.0
    emotion: str = "calm"
    accent: str = "neutral"
    style: str = "conversational"
    elevenlabs_settings: Dict[str, Any] = field(default_factory=dict)
    piper_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "wav"
    input_device: int = 0
    output_device: int = 0
    noise_reduction_enabled: bool = True
    vad_enabled: bool = True
    vad_aggressiveness: int = 2
    silence_threshold: float = 0.3
    silence_duration_ms: int = 1000
    buffer_size: int = 4096
    max_buffer_size: int = 300  # Maximum number of audio chunks to store (~30 seconds at 10ms chunks)
    max_memory_mb: int = 100   # Maximum memory usage for audio buffer in MB

@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    encryption_enabled: bool = True
    data_retention_hours: int = 24
    data_retention_days: int = 30
    session_timeout_minutes: int = 30
    encryption_key_rotation_days: int = 90
    audit_logging_enabled: bool = True
    max_login_attempts: int = 3
    consent_required: bool = True
    transcript_storage: bool = False
    anonymization_enabled: bool = True
    privacy_mode: bool = True
    hipaa_compliance_enabled: bool = True
    gdpr_compliance_enabled: bool = True
    data_localization: bool = True
    consent_recording: bool = True
    emergency_protocols_enabled: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    cache_enabled: bool = True
    cache_size: int = 100
    streaming_enabled: bool = True
    parallel_processing: bool = True
    buffer_size: int = 4096
    processing_timeout: int = 30000
    max_concurrent_requests: int = 5
    response_timeout: int = 10000

@dataclass
class VoiceConfig:
    """Main voice configuration class."""
    # Feature toggles
    voice_enabled: bool = True
    voice_input_enabled: bool = True
    voice_output_enabled: bool = True
    voice_commands_enabled: bool = True
    security_enabled: bool = True

    # Session and timeout configuration
    session_timeout_minutes: int = 30
    session_timeout: float = 1800.0  # 30 minutes in seconds
    recording_timeout: float = 10.0
    max_concurrent_sessions: int = 100
    audio_processing_timeout: float = 10.0

    # Audio configuration
    audio: AudioConfig = field(default_factory=AudioConfig)

    # Security configuration
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Performance configuration
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Voice profiles dictionary - initialized in __post_init__
    voice_profiles: Dict[str, VoiceProfile] = field(default_factory=dict)

    # Service configurations
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_speed: float = 1.0
    elevenlabs_voice_stability: float = 0.5
    elevenlabs_voice_similarity_boost: float = 0.8

    google_cloud_credentials_path: Optional[str] = None
    google_cloud_project_id: Optional[str] = None
    google_speech_language_code: str = "en-US"
    google_speech_model: str = "latest_long"
    google_speech_enable_automatic_punctuation: bool = True
    google_speech_enable_word_time_offsets: bool = True
    google_speech_max_alternatives: int = 1

    openai_whisper_model: str = "whisper-1"
    openai_whisper_language: str = "en"
    openai_whisper_temperature: float = 0.0

    whisper_model: str = "base"
    whisper_language: str = "en"
    whisper_temperature: float = 0.0
    whisper_beam_size: int = 5
    whisper_best_of: int = 5

    piper_tts_model_path: Optional[str] = None
    piper_tts_speaker_id: int = 0
    piper_tts_noise_scale: float = 0.667
    piper_tts_length_scale: float = 1.0
    piper_tts_noise_w: float = 0.8

    # Voice profile configuration
    voice_profile_path: str = "./voice_profiles"
    default_voice_profile: str = "calm_therapist"
    voice_customization_enabled: bool = True
    voice_pitch_adjustment: float = 1.0
    voice_speed_adjustment: float = 1.0
    voice_volume_adjustment: float = 1.0

    # Voice command configuration
    voice_command_wake_word: str = "therapist"
    voice_command_timeout: int = 5000
    voice_command_max_duration: int = 10000
    voice_command_min_confidence: float = 0.7
    voice_command_debug_mode: bool = False

    # Logging configuration
    voice_logging_enabled: bool = True
    voice_log_level: str = "INFO"
    voice_metrics_enabled: bool = True
    voice_error_reporting: bool = True
    voice_performance_monitoring: bool = True

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._load_from_env()
        self._load_voice_profiles()

    # Backwards compatibility properties for direct access to security attributes
    @property
    def encryption_key_rotation_days(self) -> int:
        """Direct access to security encryption_key_rotation_days."""
        return self.security.encryption_key_rotation_days

    @encryption_key_rotation_days.setter
    def encryption_key_rotation_days(self, value: int):
        """Set security encryption_key_rotation_days."""
        self.security.encryption_key_rotation_days = value

    @property
    def data_retention_days(self) -> int:
        """Direct access to security data_retention_days."""
        return self.security.data_retention_days

    @data_retention_days.setter
    def data_retention_days(self, value: int):
        """Set security data_retention_days."""
        self.security.data_retention_days = value

    @property
    def audit_logging_enabled(self) -> bool:
        """Direct access to security audit_logging_enabled."""
        return self.security.audit_logging_enabled

    @audit_logging_enabled.setter
    def audit_logging_enabled(self, value: bool):
        """Set security audit_logging_enabled."""
        self.security.audit_logging_enabled = value

    @property
    def max_login_attempts(self) -> int:
        """Direct access to security max_login_attempts."""
        return self.security.max_login_attempts

    @max_login_attempts.setter
    def max_login_attempts(self, value: int):
        """Set security max_login_attempts."""
        self.security.max_login_attempts = value

    @property
    def encryption_enabled(self) -> bool:
        """Direct access to security encryption_enabled."""
        return self.security.encryption_enabled

    @encryption_enabled.setter
    def encryption_enabled(self, value: bool):
        """Set security encryption_enabled."""
        self.security.encryption_enabled = value

    @property
    def consent_required(self) -> bool:
        """Direct access to security consent_required."""
        return self.security.consent_required

    @consent_required.setter
    def consent_required(self, value: bool):
        """Set security consent_required."""
        self.security.consent_required = value

    @property
    def hipaa_compliance_enabled(self) -> bool:
        """Direct access to security hipaa_compliance_enabled."""
        return self.security.hipaa_compliance_enabled

    @hipaa_compliance_enabled.setter
    def hipaa_compliance_enabled(self, value: bool):
        """Set security hipaa_compliance_enabled."""
        self.security.hipaa_compliance_enabled = value

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Feature toggles
        self.voice_enabled = os.getenv("VOICE_ENABLED", "true").lower() == "true"
        self.voice_input_enabled = os.getenv("VOICE_INPUT_ENABLED", "true").lower() == "true"
        self.voice_output_enabled = os.getenv("VOICE_OUTPUT_ENABLED", "true").lower() == "true"
        self.voice_commands_enabled = os.getenv("VOICE_COMMANDS_ENABLED", "true").lower() == "true"
        self.security_enabled = os.getenv("VOICE_SECURITY_ENABLED", "true").lower() == "true"

        # ElevenLabs configuration
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.elevenlabs_voice_speed = float(os.getenv("ELEVENLABS_VOICE_SPEED", "1.0"))
        self.elevenlabs_voice_stability = float(os.getenv("ELEVENLABS_VOICE_STABILITY", "0.5"))
        self.elevenlabs_voice_similarity_boost = float(os.getenv("ELEVENLABS_VOICE_SIMILARITY_BOOST", "0.8"))

        # Google Cloud configuration
        self.google_cloud_credentials_path = os.getenv("GOOGLE_CLOUD_CREDENTIALS_PATH")
        self.google_cloud_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.google_speech_language_code = os.getenv("GOOGLE_SPEECH_LANGUAGE_CODE", "en-US")
        self.google_speech_model = os.getenv("GOOGLE_SPEECH_MODEL", "latest_long")
        self.google_speech_enable_automatic_punctuation = os.getenv("GOOGLE_SPEECH_ENABLE_AUTOMATIC_PUNCTUATION", "true").lower() == "true"
        self.google_speech_enable_word_time_offsets = os.getenv("GOOGLE_SPEECH_ENABLE_WORD_TIME_OFFSETS", "true").lower() == "true"
        self.google_speech_max_alternatives = int(os.getenv("GOOGLE_SPEECH_MAX_ALTERNATIVES", "1"))

        # OpenAI Whisper configuration
        self.openai_whisper_model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
        self.openai_whisper_language = os.getenv("OPENAI_WHISPER_LANGUAGE", "en")
        self.openai_whisper_temperature = float(os.getenv("OPENAI_WHISPER_TEMPERATURE", "0.0"))

        # Whisper configuration
        self.whisper_model = os.getenv("WHISPER_MODEL", "base")
        self.whisper_language = os.getenv("WHISPER_LANGUAGE", "en")
        self.whisper_temperature = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
        self.whisper_beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
        self.whisper_best_of = int(os.getenv("WHISPER_BEST_OF", "5"))

        # Piper TTS configuration
        self.piper_tts_model_path = os.getenv("PIPER_TTS_MODEL_PATH")
        self.piper_tts_speaker_id = int(os.getenv("PIPER_TTS_SPEAKER_ID", "0"))
        self.piper_tts_noise_scale = float(os.getenv("PIPER_TTS_NOISE_SCALE", "0.667"))
        self.piper_tts_length_scale = float(os.getenv("PIPER_TTS_LENGTH_SCALE", "1.0"))
        self.piper_tts_noise_w = float(os.getenv("PIPER_TTS_NOISE_W", "0.8"))

        # Voice profile configuration
        self.voice_profile_path = os.getenv("VOICE_PROFILE_PATH", "./voice_profiles")
        self.default_voice_profile = os.getenv("DEFAULT_VOICE_PROFILE", "calm_therapist")
        self.voice_customization_enabled = os.getenv("VOICE_CUSTOMIZATION_ENABLED", "true").lower() == "true"
        self.voice_pitch_adjustment = float(os.getenv("VOICE_PITCH_ADJUSTMENT", "1.0"))
        self.voice_speed_adjustment = float(os.getenv("VOICE_SPEED_ADJUSTMENT", "1.0"))
        self.voice_volume_adjustment = float(os.getenv("VOICE_VOLUME_ADJUSTMENT", "1.0"))

        # Voice command configuration
        self.voice_command_wake_word = os.getenv("VOICE_COMMAND_WAKE_WORD", "therapist")
        self.voice_command_timeout = int(os.getenv("VOICE_COMMAND_TIMEOUT", "5000"))
        self.voice_command_max_duration = int(os.getenv("VOICE_COMMAND_MAX_DURATION", "10000"))
        self.voice_command_min_confidence = float(os.getenv("VOICE_COMMAND_MIN_CONFIDENCE", "0.7"))
        self.voice_command_debug_mode = os.getenv("VOICE_COMMAND_DEBUG_MODE", "false").lower() == "true"

        # Security configuration
        self.security.encryption_enabled = os.getenv("VOICE_ENCRYPTION_ENABLED", "true").lower() == "true"
        self.security.data_retention_hours = int(os.getenv("VOICE_DATA_RETENTION_HOURS", "24"))
        self.security.data_retention_days = int(os.getenv("VOICE_DATA_RETENTION_DAYS", "30"))
        self.security.session_timeout_minutes = int(os.getenv("VOICE_SESSION_TIMEOUT_MINUTES", "30"))
        self.security.encryption_key_rotation_days = int(os.getenv("VOICE_ENCRYPTION_KEY_ROTATION_DAYS", "90"))
        self.security.audit_logging_enabled = os.getenv("VOICE_AUDIT_LOGGING_ENABLED", "true").lower() == "true"
        self.security.max_login_attempts = int(os.getenv("VOICE_MAX_LOGIN_ATTEMPTS", "3"))
        self.security.consent_required = os.getenv("VOICE_CONSENT_REQUIRED", "true").lower() == "true"
        self.security.transcript_storage = os.getenv("VOICE_TRANSCRIPT_STORAGE", "false").lower() == "true"
        self.security.anonymization_enabled = os.getenv("VOICE_ANONYMIZATION_ENABLED", "true").lower() == "true"
        self.security.privacy_mode = os.getenv("VOICE_PRIVACY_MODE", "true").lower() == "true"
        self.security.hipaa_compliance_enabled = os.getenv("VOICE_HIPAA_COMPLIANCE_ENABLED", "true").lower() == "true"
        self.security.gdpr_compliance_enabled = os.getenv("VOICE_GDPR_COMPLIANCE_ENABLED", "true").lower() == "true"
        self.security.data_localization = os.getenv("VOICE_DATA_LOCALIZATION", "true").lower() == "true"
        self.security.consent_recording = os.getenv("VOICE_CONSENT_RECORDING", "true").lower() == "true"
        self.security.emergency_protocols_enabled = os.getenv("VOICE_EMERGENCY_PROTOCOLS_ENABLED", "true").lower() == "true"

        # Audio configuration
        self.audio.sample_rate = int(os.getenv("VOICE_AUDIO_SAMPLE_RATE", str(self.audio.sample_rate)))
        self.audio.channels = int(os.getenv("VOICE_AUDIO_CHANNELS", str(self.audio.channels)))
        self.audio.chunk_size = int(os.getenv("VOICE_AUDIO_CHUNK_SIZE", str(self.audio.chunk_size)))
        self.audio.max_buffer_size = int(os.getenv("VOICE_AUDIO_MAX_BUFFER_SIZE", str(self.audio.max_buffer_size)))
        self.audio.max_memory_mb = int(os.getenv("VOICE_AUDIO_MAX_MEMORY_MB", str(self.audio.max_memory_mb)))

        # Session and timeout configuration
        self.session_timeout_minutes = int(os.getenv("VOICE_SESSION_TIMEOUT_MINUTES", "30"))
        self.session_timeout = float(os.getenv("VOICE_SESSION_TIMEOUT", "1800.0"))
        self.recording_timeout = float(os.getenv("VOICE_RECORDING_TIMEOUT", "10.0"))
        self.max_concurrent_sessions = int(os.getenv("VOICE_MAX_CONCURRENT_SESSIONS", "100"))
        self.audio_processing_timeout = float(os.getenv("VOICE_AUDIO_PROCESSING_TIMEOUT", "10.0"))

        # Performance configuration
        self.performance.cache_enabled = os.getenv("VOICE_CACHE_ENABLED", "true").lower() == "true"
        self.performance.cache_size = int(os.getenv("VOICE_CACHE_SIZE", "100"))
        self.performance.streaming_enabled = os.getenv("VOICE_STREAMING_ENABLED", "true").lower() == "true"
        self.performance.parallel_processing = os.getenv("VOICE_PARALLEL_PROCESSING", "true").lower() == "true"
        self.performance.buffer_size = int(os.getenv("VOICE_BUFFER_SIZE", "4096"))
        self.performance.processing_timeout = int(os.getenv("VOICE_PROCESSING_TIMEOUT", "30000"))

        # Logging configuration
        self.voice_logging_enabled = os.getenv("VOICE_LOGGING_ENABLED", "true").lower() == "true"
        self.voice_log_level = os.getenv("VOICE_LOG_LEVEL", "INFO")
        self.voice_metrics_enabled = os.getenv("VOICE_METRICS_ENABLED", "true").lower() == "true"
        self.voice_error_reporting = os.getenv("VOICE_ERROR_REPORTING", "true").lower() == "true"
        self.voice_performance_monitoring = os.getenv("VOICE_PERFORMANCE_MONITORING", "true").lower() == "true"

    def _load_voice_profiles(self):
        """Load voice profiles from configuration files."""
        self.voice_profiles = {}
        profiles_dir = Path(self.voice_profile_path)

        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        profile_data = json.load(f)
                        profile = VoiceProfile(**profile_data)
                        self.voice_profiles[profile.name] = profile
                except Exception as e:
                    logging.warning(f"Failed to load voice profile {profile_file}: {str(e)}")

        # Create default profiles if none exist
        if not self.voice_profiles:
            self._create_default_profiles()

    def _create_default_profiles(self):
        """Create default voice profiles."""
        default_profiles = [
            VoiceProfile(
                name="calm_therapist",
                description="Calm and soothing voice for therapy sessions",
                voice_id="default",
                language="en-US",
                gender="female",
                age="adult",
                pitch=0.9,
                speed=0.9,
                volume=0.8,
                emotion="calm",
                style="conversational",
                elevenlabs_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.0
                }
            ),
            VoiceProfile(
                name="empathetic",
                description="Empathetic and understanding voice",
                voice_id="default",
                language="en-US",
                gender="female",
                age="adult",
                pitch=1.0,
                speed=0.95,
                volume=0.9,
                emotion="empathetic",
                style="caring",
                elevenlabs_settings={
                    "stability": 0.4,
                    "similarity_boost": 0.9,
                    "style": 0.2
                }
            ),
            VoiceProfile(
                name="professional",
                description="Professional and authoritative voice",
                voice_id="default",
                language="en-US",
                gender="male",
                age="adult",
                pitch=1.1,
                speed=1.0,
                volume=1.0,
                emotion="neutral",
                style="professional",
                elevenlabs_settings={
                    "stability": 0.7,
                    "similarity_boost": 0.6,
                    "style": 0.0
                }
            )
        ]

        for profile in default_profiles:
            self.voice_profiles[profile.name] = profile

    def get_voice_profile(self, profile_name: Optional[str] = None) -> VoiceProfile:
        """Get voice profile by name or default."""
        if profile_name and profile_name in self.voice_profiles:
            return self.voice_profiles[profile_name]
        elif self.default_voice_profile in self.voice_profiles:
            return self.voice_profiles[self.default_voice_profile]
        else:
            # Return first available profile
            return list(self.voice_profiles.values())[0] if self.voice_profiles else VoiceProfile("default", "Default profile", "default")

    def get_voice_profile_settings(self, profile_name: str) -> Dict[str, Any]:
        """Get detailed settings for a voice profile."""
        if profile_name not in self.voice_profiles:
            raise ValueError(f"Voice profile '{profile_name}' not found")

        profile = self.voice_profiles[profile_name]
        return {
            'name': profile.name,
            'description': profile.description,
            'voice_id': profile.voice_id,
            'language': profile.language,
            'gender': profile.gender,
            'pitch': profile.pitch,
            'speed': profile.speed,
            'volume': profile.volume,
            'emotion': profile.emotion,
            'style': profile.style,
            'elevenlabs_settings': profile.elevenlabs_settings,
            'piper_settings': profile.piper_settings
        }

    def save_voice_profile(self, profile: VoiceProfile):
        """Save voice profile to file."""
        profiles_dir = Path(self.voice_profile_path)
        profiles_dir.mkdir(exist_ok=True)

        profile_file = profiles_dir / f"{profile.name}.json"
        profile_data = {
            'name': profile.name,
            'description': profile.description,
            'voice_id': profile.voice_id,
            'language': profile.language,
            'gender': profile.gender,
            'age': profile.age,
            'pitch': profile.pitch,
            'speed': profile.speed,
            'volume': profile.volume,
            'emotion': profile.emotion,
            'accent': profile.accent,
            'style': profile.style,
            'elevenlabs_settings': profile.elevenlabs_settings,
            'piper_settings': profile.piper_settings
        }

        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)

        self.voice_profiles[profile.name] = profile

    def is_elevenlabs_configured(self) -> bool:
        """Check if ElevenLabs is properly configured."""
        return bool(self.elevenlabs_api_key and self.elevenlabs_voice_id)

    def is_openai_whisper_configured(self) -> bool:
        """Check if OpenAI Whisper API is properly configured."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def is_openai_tts_configured(self) -> bool:
        """Check if OpenAI TTS is properly configured."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def is_google_speech_configured(self) -> bool:
        """Check if Google Speech-to-Text is properly configured."""
        return bool(self.google_cloud_credentials_path and self.google_cloud_project_id)

    def is_whisper_configured(self) -> bool:
        """Check if Whisper is properly configured."""
        return True  # Whisper works offline by default

    def is_piper_configured(self) -> bool:
        """Check if Piper TTS is properly configured."""
        return bool(self.piper_tts_model_path)

    def get_preferred_stt_service(self) -> str:
        """Get preferred STT service based on configuration."""
        if self.is_openai_whisper_configured():
            return "openai"
        elif self.is_google_speech_configured():
            return "google"
        elif self.is_whisper_configured():
            return "whisper"
        else:
            return "none"

    def get_preferred_tts_service(self) -> str:
        """Get preferred TTS service based on configuration."""
        if self.is_elevenlabs_configured():
            return "elevenlabs"
        elif self.is_piper_configured():
            return "piper"
        else:
            return "none"

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check basic voice requirements
        if not self.voice_enabled:
            issues.append("Voice features are disabled")

        # Check STT configuration
        if self.voice_input_enabled:
            if not (self.is_openai_whisper_configured() or self.is_google_speech_configured() or self.is_whisper_configured()):
                issues.append("No STT service configured. Configure OpenAI Whisper API, Google Speech, or Local Whisper")

        # Check TTS configuration
        if self.voice_output_enabled:
            if not (self.is_elevenlabs_configured() or self.is_piper_configured()):
                issues.append("No TTS service configured. Configure ElevenLabs or Piper TTS")

        # Check command configuration
        if self.voice_commands_enabled and not self.voice_input_enabled:
            issues.append("Voice commands require voice input to be enabled")

        # Check security configuration
        if self.security.encryption_enabled and not os.path.exists('./credentials'):
            issues.append("Encryption enabled but no credentials directory found")

        # Check paths
        if self.google_cloud_credentials_path and not os.path.exists(self.google_cloud_credentials_path):
            issues.append(f"Google Cloud credentials file not found: {self.google_cloud_credentials_path}")

        if self.piper_tts_model_path and not os.path.exists(self.piper_tts_model_path):
            issues.append(f"Piper TTS model not found: {self.piper_tts_model_path}")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'voice_enabled': self.voice_enabled,
            'voice_input_enabled': self.voice_input_enabled,
            'voice_output_enabled': self.voice_output_enabled,
            'voice_commands_enabled': self.voice_commands_enabled,
            'elevenlabs_configured': self.is_elevenlabs_configured(),
            'openai_whisper_configured': self.is_openai_whisper_configured(),
            'google_speech_configured': self.is_google_speech_configured(),
            'whisper_configured': self.is_whisper_configured(),
            'piper_configured': self.is_piper_configured(),
            'preferred_stt_service': self.get_preferred_stt_service(),
            'preferred_tts_service': self.get_preferred_tts_service(),
            'voice_profiles_count': len(self.voice_profiles),
            'default_voice_profile': self.default_voice_profile,
            'security_encryption': self.security.encryption_enabled,
            'performance_cache': self.performance.cache_enabled,
            'performance_streaming': self.performance.streaming_enabled
        }