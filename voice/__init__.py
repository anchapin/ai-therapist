"""
Voice Module for AI Therapist

This module provides comprehensive voice interaction capabilities including:
- Speech-to-Text (STT) services
- Text-to-Speech (TTS) services
- Voice command processing
- Audio processing and analysis
- Security and privacy features
- Voice activity detection
- Voice customization and profiles

Architecture:
- Modular design with separate services for different voice features
- Support for multiple STT/TTS providers (Google, ElevenLabs, Whisper, Piper)
- Configurable voice profiles and command processing
- Comprehensive security and privacy controls
- Performance optimization with caching and streaming
"""

__version__ = "1.0.0"
__author__ = "AI Therapist Team"

# Export main classes and functions
from .config import VoiceConfig, VoiceProfile
from .voice_service import VoiceService
from .audio_processor import AudioProcessor
from .stt_service import STTService
from .tts_service import TTSService
from .security import VoiceSecurity
from .commands import VoiceCommandProcessor

# Optional import for UI components (requires streamlit)
try:
    from .voice_ui import VoiceUIComponents
    _UI_AVAILABLE = True
except ImportError as e:
    # Streamlit or other UI dependencies not available
    VoiceUIComponents = None
    _UI_AVAILABLE = False

__all__ = [
    'VoiceConfig',
    'VoiceProfile',
    'VoiceService',
    'AudioProcessor',
    'STTService',
    'TTSService',
    'VoiceSecurity',
    'VoiceCommandProcessor'
]

# Add UI components to exports only if available
if _UI_AVAILABLE:
    __all__.append('VoiceUIComponents')

# Module initialization
def initialize_voice_module():
    """Initialize the voice module with default configuration."""
    try:
        # Load configuration
        config = VoiceConfig()

        # Initialize security
        security = VoiceSecurity(config)

        # Initialize main voice service
        voice_service = VoiceService(config, security)

        return voice_service
    except Exception as e:
        print(f"Error initializing voice module: {str(e)}")
        return None

# Utility functions
def is_voice_enabled():
    """Check if voice features are enabled."""
    from .config import VoiceConfig
    config = VoiceConfig()
    return config.voice_enabled

def get_available_audio_devices():
    """Get list of available audio input/output devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()

        input_devices = []
        output_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append({
                    "index": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"]
                })
            if device["max_output_channels"] > 0:
                output_devices.append({
                    "index": i,
                    "name": device["name"],
                    "channels": device["max_output_channels"],
                    "sample_rate": device["default_samplerate"]
                })
        return input_devices, output_devices
    except Exception as e:
        print(f"Error getting audio devices: {str(e)}")
        return [], []

# Voice feature availability checks
def check_voice_requirements():
    """Check if system meets voice feature requirements."""
    requirements = {
        'audio_library': False,
        'voice_models': False,
        'credentials': False,
        'audio_devices': False
    }

    # Check audio library
    try:
        import sounddevice as sd
        requirements['audio_library'] = True
    except ImportError:
        pass

    # Check voice models
    try:
        from .config import VoiceConfig
        config = VoiceConfig()
        if config.elevenlabs_api_key or config.google_cloud_credentials_path:
            requirements['voice_models'] = True
    except (ImportError, AttributeError, Exception) as e:
        # Log the error for debugging but don't fail the initialization
        # Voice models are optional for basic functionality
        pass

    # Check credentials
    import os
    if os.path.exists('./credentials'):
        requirements['credentials'] = True

    # Check audio devices
    try:
        input_devices, output_devices = get_available_audio_devices()
        if input_devices and output_devices:
            requirements['audio_devices'] = True
    except (ImportError, OSError, Exception) as e:
        # Audio devices are optional for basic functionality
        # May fail in headless environments or without audio hardware
        pass

    return requirements