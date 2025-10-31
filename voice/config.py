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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Create VoiceProfile from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert VoiceProfile to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'voice_id': self.voice_id,
            'language': self.language,
            'gender': self.gender,
            'age': self.age,
            'pitch': self.pitch,
            'speed': self.speed,
            'volume': self.volume,
            'emotion': self.emotion,
            'accent': self.accent,
            'style': self.style,
            'elevenlabs_settings': self.elevenlabs_settings,
            'piper_settings': self.piper_settings
        }

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
    # Performance configuration environment variables
    VOICE_MEMORY_THRESHOLD_LOW = int(os.getenv("VOICE_MEMORY_THRESHOLD_LOW", "256"))
    VOICE_MEMORY_THRESHOLD_MEDIUM = int(os.getenv("VOICE_MEMORY_THRESHOLD_MEDIUM", "512"))
    VOICE_MEMORY_THRESHOLD_HIGH = int(os.getenv("VOICE_MEMORY_THRESHOLD_HIGH", "768"))
    VOICE_MEMORY_THRESHOLD_CRITICAL = int(os.getenv("VOICE_MEMORY_THRESHOLD_CRITICAL", "1024"))
    VOICE_MONITORING_INTERVAL = float(os.getenv("VOICE_MONITORING_INTERVAL", "30.0"))
    VOICE_GC_THRESHOLD = int(os.getenv("VOICE_GC_THRESHOLD", "500"))
    VOICE_CLEANUP_INTERVAL = float(os.getenv("VOICE_CLEANUP_INTERVAL", "300.0"))
    VOICE_CACHE_SIZE = int(os.getenv("VOICE_CACHE_SIZE", "1000"))
    VOICE_CACHE_MAX_MEMORY_MB = int(os.getenv("VOICE_CACHE_MAX_MEMORY_MB", "256"))
    VOICE_ENABLE_COMPRESSION = os.getenv("VOICE_ENABLE_COMPRESSION", "true").lower() == "true"
    VOICE_SESSION_CLEANUP_INTERVAL = int(os.getenv("VOICE_SESSION_CLEANUP_INTERVAL", "300"))
    VOICE_MAX_SESSION_AGE = int(os.getenv("VOICE_MAX_SESSION_AGE", "3600"))
    VOICE_INACTIVE_SESSION_TIMEOUT = int(os.getenv("VOICE_INACTIVE_SESSION_TIMEOUT", "1800"))
    VOICE_MAX_CONNECTIONS = int(os.getenv("VOICE_MAX_CONNECTIONS", "5"))
    VOICE_MAX_ASYNC_WORKERS = int(os.getenv("VOICE_MAX_ASYNC_WORKERS", "3"))
    VOICE_STREAM_BUFFER_SIZE = int(os.getenv("VOICE_STREAM_BUFFER_SIZE", "10"))
    VOICE_STREAM_CHUNK_DURATION = float(os.getenv("VOICE_STREAM_CHUNK_DURATION", "0.1"))
    VOICE_COMPRESSION_LEVEL = int(os.getenv("VOICE_COMPRESSION_LEVEL", "6"))
    
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
        # Check for mock mode first
        mock_mode = os.getenv("VOICE_MOCK_MODE", "false").lower() == "true"
        force_mock_services = os.getenv("VOICE_FORCE_MOCK_SERVICES", "false").lower() == "true"
        
        # Feature toggles
        self.voice_enabled = os.getenv("VOICE_ENABLED", "true").lower() == "true"
        self.voice_input_enabled = os.getenv("VOICE_INPUT_ENABLED", "true").lower() == "true"
        self.voice_output_enabled = os.getenv("VOICE_OUTPUT_ENABLED", "true").lower() == "true"
        self.voice_commands_enabled = os.getenv("VOICE_COMMANDS_ENABLED", "true").lower() == "true"
        self.security_enabled = os.getenv("VOICE_SECURITY_ENABLED", "true").lower() == "true"
        
        # Force mock providers if in mock mode
        if mock_mode or force_mock_services:
            self._stt_provider = "mock"
            self._tts_provider = "mock"
            # Use environment variables for mock keys to avoid hardcoded secrets
            self.openai_api_key = os.getenv("MOCK_OPENAI_API_KEY", "")
            self.elevenlabs_api_key = os.getenv("MOCK_ELEVENLABS_API_KEY", "")
            self.elevenlabs_voice_id = os.getenv("MOCK_ELEVENLABS_VOICE_ID", "mock-voice-id")
            self.google_cloud_project_id = os.getenv("MOCK_GOOGLE_CLOUD_PROJECT_ID", "mock-project-id")
            return  # Skip loading real API keys in mock mode

        # ElevenLabs configuration
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.elevenlabs_voice_speed = float(os.getenv("ELEVENLABS_VOICE_SPEED", "1.0"))
        self.elevenlabs_voice_stability = float(os.getenv("ELEVENLABS_VOICE_STABILITY", "0.5"))
        self.elevenlabs_voice_similarity_boost = float(os.getenv("ELEVENLABS_VOICE_SIMILARITY_BOOST", "0.8"))

        # Google Cloud configuration
        google_credentials_path = os.getenv("GOOGLE_CLOUD_CREDENTIALS_PATH")
        if google_credentials_path:
            self.google_cloud_credentials_path = google_credentials_path
        else:
            # Set default path but don't fail if it doesn't exist
            self.google_cloud_credentials_path = "./credentials/google-cloud-credentials.json"
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
        try:
            return bool(self.elevenlabs_api_key and self.elevenlabs_voice_id)
        except AttributeError:
            return False

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
        # Check for mock mode first
        mock_mode = os.getenv("VOICE_MOCK_MODE", "false").lower() == "true"
        force_mock_services = os.getenv("VOICE_FORCE_MOCK_SERVICES", "false").lower() == "true"
        stt_provider_env = os.getenv("STT_PROVIDER", "").lower()
        
        if mock_mode or force_mock_services or stt_provider_env == "mock":
            return "mock"
        elif self.is_whisper_configured():  # Prioritize local Whisper
            return "whisper"
        elif self.is_openai_whisper_configured():
            return "openai"
        elif self.is_google_speech_configured():
            return "google"
        else:
            return "mock"  # Default to mock in testing

    def get_preferred_tts_service(self) -> str:
        """Get preferred TTS service based on configuration."""
        # Check for mock mode first
        mock_mode = os.getenv("VOICE_MOCK_MODE", "false").lower() == "true"
        force_mock_services = os.getenv("VOICE_FORCE_MOCK_SERVICES", "false").lower() == "true"
        tts_provider_env = os.getenv("TTS_PROVIDER", "").lower()
        
        if mock_mode or force_mock_services or tts_provider_env == "mock":
            return "mock"
        elif self.is_elevenlabs_configured():
            return "elevenlabs"
        elif self.is_piper_configured():
            return "piper"
        else:
            return "mock"  # Default to mock in testing

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

    # Additional missing methods for test compatibility

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create config from dictionary."""
        config = cls()
        # Set basic attributes
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        try:
            return json.dumps(self.to_dict(), indent=2)
        except Exception as e:
            logging.error(f"Failed to serialize config to JSON: {e}")
            return "{}"

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize configuration from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Failed to deserialize config from JSON: {e}")
            return cls()

    def copy(self):
        """Create a copy of the configuration."""
        return VoiceConfig(
            voice_enabled=self.voice_enabled,
            voice_input_enabled=self.voice_input_enabled,
            voice_output_enabled=self.voice_output_enabled,
            voice_commands_enabled=self.voice_commands_enabled,
            security_enabled=self.security_enabled,
            session_timeout_minutes=self.session_timeout_minutes,
            session_timeout=self.session_timeout,
            recording_timeout=self.recording_timeout,
            max_concurrent_sessions=self.max_concurrent_sessions,
            audio_processing_timeout=self.audio_processing_timeout,
            audio=self.audio,
            security=self.security,
            performance=self.performance,
            voice_profiles=self.voice_profiles.copy(),
            elevenlabs_api_key=self.elevenlabs_api_key,
            elevenlabs_voice_id=self.elevenlabs_voice_id,
            elevenlabs_model=self.elevenlabs_model,
            elevenlabs_voice_speed=self.elevenlabs_voice_speed,
            elevenlabs_voice_stability=self.elevenlabs_voice_stability,
            elevenlabs_voice_similarity_boost=self.elevenlabs_voice_similarity_boost,
            google_cloud_credentials_path=self.google_cloud_credentials_path,
            google_cloud_project_id=self.google_cloud_project_id,
            google_speech_language_code=self.google_speech_language_code,
            google_speech_model=self.google_speech_model,
            google_speech_enable_automatic_punctuation=self.google_speech_enable_automatic_punctuation,
            google_speech_enable_word_time_offsets=self.google_speech_enable_word_time_offsets,
            google_speech_max_alternatives=self.google_speech_max_alternatives,
            openai_whisper_model=self.openai_whisper_model,
            openai_whisper_language=self.openai_whisper_language,
            openai_whisper_temperature=self.openai_whisper_temperature,
            whisper_model=self.whisper_model,
            whisper_language=self.whisper_language,
            whisper_temperature=self.whisper_temperature,
            whisper_beam_size=self.whisper_beam_size,
            whisper_best_of=self.whisper_best_of,
            piper_tts_model_path=self.piper_tts_model_path,
            piper_tts_speaker_id=self.piper_tts_speaker_id,
            piper_tts_noise_scale=self.piper_tts_noise_scale,
            piper_tts_length_scale=self.piper_tts_length_scale,
            piper_tts_noise_w=self.piper_tts_noise_w,
            voice_profile_path=self.voice_profile_path,
            default_voice_profile=self.default_voice_profile,
            voice_customization_enabled=self.voice_customization_enabled,
            voice_pitch_adjustment=self.voice_pitch_adjustment,
            voice_speed_adjustment=self.voice_speed_adjustment,
            voice_volume_adjustment=self.voice_volume_adjustment,
            voice_command_wake_word=self.voice_command_wake_word,
            voice_command_timeout=self.voice_command_timeout,
            voice_command_max_duration=self.voice_command_max_duration,
            voice_command_min_confidence=self.voice_command_min_confidence,
            voice_command_debug_mode=self.voice_command_debug_mode,
            voice_logging_enabled=self.voice_logging_enabled,
            voice_log_level=self.voice_log_level,
            voice_metrics_enabled=self.voice_metrics_enabled,
            voice_error_reporting=self.voice_error_reporting,
            voice_performance_monitoring=self.voice_performance_monitoring
        )

    def __eq__(self, other):
        """Check equality with another VoiceConfig."""
        if not isinstance(other, VoiceConfig):
            return False
        return (
            self.voice_enabled == other.voice_enabled and
            self.voice_input_enabled == other.voice_input_enabled and
            self.voice_output_enabled == other.voice_output_enabled and
            self.voice_commands_enabled == other.voice_commands_enabled and
            self.audio_sample_rate == other.audio_sample_rate
        )

    def merge(self, other: 'VoiceConfig'):
        """Merge another configuration into this one."""
        if not isinstance(other, VoiceConfig):
            raise ValueError("Can only merge with another VoiceConfig instance")

        # Merge basic attributes
        self.voice_enabled = other.voice_enabled if other.voice_enabled is not None else self.voice_enabled
        self.voice_input_enabled = other.voice_input_enabled if other.voice_input_enabled is not None else self.voice_input_enabled
        self.voice_output_enabled = other.voice_output_enabled if other.voice_output_enabled is not None else self.voice_output_enabled
        self.voice_commands_enabled = other.voice_commands_enabled if other.voice_commands_enabled is not None else self.voice_commands_enabled
        self.audio_sample_rate = other.audio_sample_rate if other.audio_sample_rate is not None else self.audio_sample_rate
        self.data_retention_days = other.data_retention_days if other.data_retention_days is not None else self.data_retention_days

        return self

    def save(self, file_path: str):
        """Save configuration to a file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config to {file_path}: {e}")
            raise

    @classmethod
    def load(cls, file_path: str):
        """Load configuration from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Failed to load config from {file_path}: {e}")
            return cls()

    def get_missing_api_keys(self) -> List[str]:
        """Get list of missing or invalid API keys."""
        missing = []
        invalid_indicators = [
            'your_', 'here', 'test_', 'example', 'placeholder',
            'replace', 'change', 'add_', 'insert_', 'fake', 'dummy'
        ]

        # Check OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            missing.append("openai_api_key")
        elif any(indicator in str(openai_key).lower() for indicator in invalid_indicators):
            missing.append("openai_api_key_invalid")

        # Check Google Cloud configuration
        google_creds_path = self.google_cloud_credentials_path
        google_project_id = self.google_cloud_project_id
        if self.stt_provider == "google" or self.google_cloud_project_id:
            if not (google_creds_path and google_project_id):
                missing.append("google_cloud_credentials")
            elif not os.path.exists(google_creds_path):
                missing.append("google_cloud_credentials_file_missing")
            elif any(indicator in str(google_project_id).lower() for indicator in invalid_indicators):
                missing.append("google_cloud_project_id_invalid")

        # Check ElevenLabs configuration
        elevenlabs_key = self.elevenlabs_api_key
        elevenlabs_voice_id = self.elevenlabs_voice_id
        if self.tts_provider == "elevenlabs" or elevenlabs_key:
            if not (elevenlabs_key and elevenlabs_voice_id):
                missing.append("elevenlabs_api_key")
            elif any(indicator in str(elevenlabs_key).lower() for indicator in invalid_indicators):
                missing.append("elevenlabs_api_key_invalid")
            elif any(indicator in str(elevenlabs_voice_id).lower() for indicator in invalid_indicators):
                missing.append("elevenlabs_voice_id_invalid")

        return missing

    def detect_conflicts(self) -> List[str]:
        """Detect configuration conflicts."""
        conflicts = []

        # Check for conflicting feature combinations
        if not self.voice_enabled and (self.voice_input_enabled or self.voice_output_enabled):
            conflicts.append("Voice features are disabled but individual voice features are enabled")

        if self.voice_commands_enabled and not self.voice_input_enabled:
            conflicts.append("Voice commands require voice input to be enabled")

        if self.encryption_enabled and not self.hipaa_compliance_enabled:
            conflicts.append("Encryption is enabled but HIPAA compliance is disabled")

        return conflicts

    def is_compatible_with_version(self, version: str) -> bool:
        """Check if configuration is compatible with a given version."""
        try:
            # Simple version compatibility check
            current_major = 2
            target_major = int(version.split('.')[0])
            return abs(current_major - target_major) <= 1
        except:
            return False

    @classmethod
    def migrate_from_version(cls, config_data: Dict[str, Any], from_version: str) -> 'VoiceConfig':
        """Migrate configuration from an older version."""
        config = cls.from_dict(config_data)

        # Handle version-specific migrations
        if from_version.startswith("1."):
            # Migrate audio_sample_rate from old location
            if "audio_sample_rate" not in config_data and "sample_rate" in config_data:
                config.audio_sample_rate = config_data["sample_rate"]

        return config

    def create_backup(self, backup_path: Path) -> bool:
        """Create a backup of the current configuration."""
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return False

    def restore_from_backup(self, backup_path: Path) -> bool:
        """Restore configuration from a backup."""
        try:
            with open(backup_path, 'r') as f:
                data = json.load(f)

            # Update current config with backup data
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            return True
        except Exception as e:
            logging.error(f"Failed to restore from backup: {e}")
            return False

    @staticmethod
    def generate_diff(config1: 'VoiceConfig', config2: 'VoiceConfig') -> Dict[str, Dict[str, Any]]:
        """Generate diff between two configurations."""
        diff = {}

        # Compare basic attributes
        attrs_to_compare = [
            'voice_enabled', 'voice_input_enabled', 'voice_output_enabled',
            'voice_commands_enabled', 'audio_sample_rate', 'data_retention_days'
        ]

        for attr in attrs_to_compare:
            val1 = getattr(config1, attr, None)
            val2 = getattr(config2, attr, None)
            if val1 != val2:
                diff[attr] = {'old': val1, 'new': val2}

        return diff

    @classmethod
    def generate_template(cls) -> Dict[str, Any]:
        """Generate a configuration template."""
        template = cls().to_dict()
        # Add comments as keys for documentation
        template["_comment_voice_enabled"] = "Enable/disable all voice features"
        template["_comment_audio_sample_rate"] = "Audio sample rate in Hz"
        template["_comment_stt_provider"] = "Speech-to-text provider: openai, google, whisper"
        template["_comment_tts_provider"] = "Text-to-speech provider: openai, elevenlabs, piper"
        return template

    @classmethod
    def generate_template_with_comments(cls) -> str:
        """Generate a configuration template with comments."""
        template = cls.generate_template()
        lines = ["# Voice Configuration Template", "# This file contains all available configuration options", ""]

        for key, value in template.items():
            if key.startswith("_comment_"):
                lines.append(f"# {template[key]}")
            else:
                lines.append(f'# {key}: {json.dumps(value)}')

        return "\n".join(lines)

    @classmethod
    def load_for_environment(cls, environment: str) -> 'VoiceConfig':
        """Load configuration for a specific environment."""
        config = cls()

        # Environment-specific overrides
        if environment == "production":
            config.debug_mode = False
            config.voice_logging_enabled = True
            config.voice_log_level = "WARNING"
        elif environment == "development":
            config.debug_mode = True
            config.voice_logging_enabled = True
            config.voice_log_level = "DEBUG"
        elif environment == "staging":
            config.debug_mode = False
            config.voice_logging_enabled = True
            config.voice_log_level = "INFO"

        return config

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            for key, value in updates.items():
                if hasattr(self, key):
                    # Basic type validation
                    current_value = getattr(self, key)
                    if isinstance(current_value, bool) and not isinstance(value, bool):
                        continue  # Skip invalid type
                    if isinstance(current_value, int) and not isinstance(value, int):
                        continue  # Skip invalid type
                    if isinstance(current_value, str) and not isinstance(value, str):
                        continue  # Skip invalid type
                    if isinstance(current_value, float) and not isinstance(value, (int, float)):
                        continue  # Skip invalid type

                    setattr(self, key, value)
                else:
                    # Try to set on nested objects
                    if key.startswith('audio_') and hasattr(self.audio, key[6:]):
                        setattr(self.audio, key[6:], value)
                    elif key.startswith('security_') and hasattr(self.security, key[9:]):
                        setattr(self.security, key[9:], value)
                    elif key.startswith('performance_') and hasattr(self.performance, key[12:]):
                        setattr(self.performance, key[12:], value)

            return True
        except Exception as e:
            logging.error(f"Failed to update config: {e}")
            return False

    # Additional properties for backward compatibility
    @property
    def audio_sample_rate(self) -> int:
        """Get audio sample rate."""
        return self.audio.sample_rate

    @audio_sample_rate.setter
    def audio_sample_rate(self, value: int):
        """Set audio sample rate."""
        self.audio.sample_rate = value

    @property
    def audio_channels(self) -> int:
        """Get audio channels."""
        return self.audio.channels

    @audio_channels.setter
    def audio_channels(self, value: int):
        """Set audio channels."""
        self.audio.channels = value

    @property
    def stt_provider(self) -> str:
        """Get STT provider."""
        return getattr(self, '_stt_provider', 'openai')

    @stt_provider.setter
    def stt_provider(self, value: str):
        """Set STT provider."""
        self._stt_provider = value

    @property
    def tts_provider(self) -> str:
        """Get TTS provider."""
        return getattr(self, '_tts_provider', 'openai')

    @tts_provider.setter
    def tts_provider(self, value: str):
        """Set TTS provider."""
        self._tts_provider = value

    @property
    def stt_model(self) -> str:
        """Get STT model."""
        return getattr(self, '_stt_model', 'whisper-1')

    @stt_model.setter
    def stt_model(self, value: str):
        """Set STT model."""
        self._stt_model = value

    @property
    def tts_model(self) -> str:
        """Get TTS model."""
        return getattr(self, '_tts_model', 'tts-1')

    @tts_model.setter
    def tts_model(self, value: str):
        """Set TTS model."""
        self._tts_model = value

    @property
    def tts_voice(self) -> str:
        """Get TTS voice."""
        return getattr(self, '_tts_voice', 'alloy')

    @tts_voice.setter
    def tts_voice(self, value: str):
        """Set TTS voice."""
        self._tts_voice = value

    @property
    def debug_mode(self) -> bool:
        """Get debug mode."""
        return getattr(self, '_debug_mode', False)

    @debug_mode.setter
    def debug_mode(self, value: bool):
        """Set debug mode."""
        self._debug_mode = value

    @property
    def version(self) -> str:
        """Get configuration version."""
        return getattr(self, '_version', '2.0.0')

    @version.setter
    def version(self, value: str):
        """Set configuration version."""
        self._version = value

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        # Check if we have a mock key set first
        mock_mode = os.getenv("VOICE_MOCK_MODE", "false").lower() == "true"
        force_mock_services = os.getenv("VOICE_FORCE_MOCK_SERVICES", "false").lower() == "true"
        
        if mock_mode or force_mock_services:
            return getattr(self, '_openai_api_key', os.getenv("MOCK_OPENAI_API_KEY", ""))
        
        return os.getenv("OPENAI_API_KEY")

    @openai_api_key.setter
    def openai_api_key(self, value: Optional[str]):
        """Set OpenAI API key."""
        self._openai_api_key = value
        if value and not (os.getenv("VOICE_MOCK_MODE", "false").lower() == "true" or
                         os.getenv("VOICE_FORCE_MOCK_SERVICES", "false").lower() == "true"):
            os.environ["OPENAI_API_KEY"] = value
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @property
    def elevenlabs_api_key(self) -> Optional[str]:
        """Get ElevenLabs API key."""
        return getattr(self, '_elevenlabs_api_key', None)

    @elevenlabs_api_key.setter
    def elevenlabs_api_key(self, value: Optional[str]):
        """Set ElevenLabs API key."""
        self._elevenlabs_api_key = value

    @property
    def elevenlabs_voice_id(self) -> Optional[str]:
        """Get ElevenLabs voice ID."""
        return getattr(self, '_elevenlabs_voice_id', None)

    @elevenlabs_voice_id.setter
    def elevenlabs_voice_id(self, value: Optional[str]):
        """Set ElevenLabs voice ID."""
        self._elevenlabs_voice_id = value

    @property
    def elevenlabs_model_id(self) -> str:
        """Get ElevenLabs model ID."""
        return getattr(self, '_elevenlabs_model_id', 'eleven_multilingual_v2')

    @elevenlabs_model_id.setter
    def elevenlabs_model_id(self, value: str):
        """Set ElevenLabs model ID."""
        self._elevenlabs_model_id = value

    @property
    def elevenlabs_stability(self) -> float:
        """Get ElevenLabs stability."""
        return getattr(self, '_elevenlabs_stability', 0.5)

    @elevenlabs_stability.setter
    def elevenlabs_stability(self, value: float):
        """Set ElevenLabs stability."""
        self._elevenlabs_stability = value

    @property
    def audio_quality_preset(self) -> str:
        """Get audio quality preset."""
        return getattr(self, '_audio_quality_preset', 'high')

    @audio_quality_preset.setter
    def audio_quality_preset(self, value: str):
        """Set audio quality preset."""
        self._audio_quality_preset = value

    @property
    def noise_reduction_enabled(self) -> bool:
        """Get noise reduction enabled."""
        return getattr(self, '_noise_reduction_enabled', True)

    @noise_reduction_enabled.setter
    def noise_reduction_enabled(self, value: bool):
        """Set noise reduction enabled."""
        self._noise_reduction_enabled = value

    @property
    def echo_cancellation_enabled(self) -> bool:
        """Get echo cancellation enabled."""
        return getattr(self, '_echo_cancellation_enabled', True)

    @echo_cancellation_enabled.setter
    def echo_cancellation_enabled(self, value: bool):
        """Set echo cancellation enabled."""
        self._echo_cancellation_enabled = value

    @property
    def auto_gain_control_enabled(self) -> bool:
        """Get auto gain control enabled."""
        return getattr(self, '_auto_gain_control_enabled', True)

    @auto_gain_control_enabled.setter
    def auto_gain_control_enabled(self, value: bool):
        """Set auto gain control enabled."""
        self._auto_gain_control_enabled = value

    @property
    def max_concurrent_requests(self) -> int:
        """Get max concurrent requests."""
        return getattr(self, '_max_concurrent_requests', 10)

    @max_concurrent_requests.setter
    def max_concurrent_requests(self, value: int):
        """Set max concurrent requests."""
        self._max_concurrent_requests = value

    @property
    def request_timeout_seconds(self) -> int:
        """Get request timeout seconds."""
        return getattr(self, '_request_timeout_seconds', 30)

    @request_timeout_seconds.setter
    def request_timeout_seconds(self, value: int):
        """Set request timeout seconds."""
        self._request_timeout_seconds = value

    @property
    def cache_size_mb(self) -> int:
        """Get cache size in MB."""
        return getattr(self, '_cache_size_mb', 100)

    @cache_size_mb.setter
    def cache_size_mb(self, value: int):
        """Set cache size in MB."""
        self._cache_size_mb = value

    @property
    def optimization_level(self) -> str:
        """Get optimization level."""
        return getattr(self, '_optimization_level', 'balanced')

    @optimization_level.setter
    def optimization_level(self, value: str):
        """Set optimization level."""
        self._optimization_level = value

    @property
    def log_level(self) -> str:
        """Get log level."""
        return getattr(self, '_log_level', 'INFO')

    @log_level.setter
    def log_level(self, value: str):
        """Set log level."""
        self._log_level = value

    @property
    def log_file_path(self) -> str:
        """Get log file path with secure temporary file."""
        import tempfile
        if not hasattr(self, '_log_file_path'):
            # Create a secure temporary file with proper permissions
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.log', 
                prefix='voice_app_',
                delete=False
            )
            temp_file.close()  # Close the file handle but keep the path
            self._log_file_path = temp_file.name
        return self._log_file_path

    @log_file_path.setter
    def log_file_path(self, value: str):
        """Set log file path."""
        self._log_file_path = value

    @property
    def log_rotation_enabled(self) -> bool:
        """Get log rotation enabled."""
        return getattr(self, '_log_rotation_enabled', True)

    @log_rotation_enabled.setter
    def log_rotation_enabled(self, value: bool):
        """Set log rotation enabled."""
        self._log_rotation_enabled = value

    @property
    def max_log_size_mb(self) -> int:
        """Get max log size in MB."""
        return getattr(self, '_max_log_size_mb', 10)

    @max_log_size_mb.setter
    def max_log_size_mb(self, value: int):
        """Set max log size in MB."""
        self._max_log_size_mb = value

    @property
    def log_retention_days(self) -> int:
        """Get log retention days."""
        return getattr(self, '_log_retention_days', 7)

    @log_retention_days.setter
    def log_retention_days(self, value: int):
        """Set log retention days."""
        self._log_retention_days = value

    @property
    def api_endpoint_override(self) -> Optional[str]:
        """Get API endpoint override."""
        return getattr(self, '_api_endpoint_override', None)

    @api_endpoint_override.setter
    def api_endpoint_override(self, value: Optional[str]):
        """Set API endpoint override."""
        self._api_endpoint_override = value

    @property
    def mock_audio_input(self) -> bool:
        """Get mock audio input."""
        return getattr(self, '_mock_audio_input', False)

    @mock_audio_input.setter
    def mock_audio_input(self, value: bool):
        """Set mock audio input."""
        self._mock_audio_input = value

    @property
    def save_debug_logs(self) -> bool:
        """Get save debug logs."""
        return getattr(self, '_save_debug_logs', False)

    @save_debug_logs.setter
    def save_debug_logs(self, value: bool):
        """Set save debug logs."""
        self._save_debug_logs = value

    @property
    def openai_model(self) -> str:
        """Get OpenAI model."""
        return getattr(self, '_openai_model', 'gpt-4')

    @openai_model.setter
    def openai_model(self, value: str):
        """Set OpenAI model."""
        self._openai_model = value

    @property
    def openai_temperature(self) -> float:
        """Get OpenAI temperature."""
        return getattr(self, '_openai_temperature', 0.7)

    @openai_temperature.setter
    def openai_temperature(self, value: float):
        """Set OpenAI temperature."""
        self._openai_temperature = value

    @property
    def openai_max_tokens(self) -> int:
        """Get OpenAI max tokens."""
        return getattr(self, '_openai_max_tokens', 1000)

    @openai_max_tokens.setter
    def openai_max_tokens(self, value: int):
        """Set OpenAI max tokens."""
        self._openai_max_tokens = value