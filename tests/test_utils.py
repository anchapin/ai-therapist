"""
Test utilities for voice feature testing.

This module provides utilities to safely import modules and avoid
NumPy reload warnings during test execution.
"""

import sys
import importlib.util
import importlib
from pathlib import Path
from unittest.mock import MagicMock
from typing import Dict, Any, Optional

# Import numpy once at module level to prevent reload warnings
import numpy

# Cache for already loaded modules to prevent reloading
_loaded_modules: Dict[str, Any] = {}


def safe_import_module(module_name: str, module_path: str, package: Optional[str] = None) -> Any:
    """
    Safely import a module without reloading if it's already loaded.

    Args:
        module_name: Name of the module to import
        module_path: Path to the module file
        package: Package name for relative imports

    Returns:
        The imported module
    """
    # Check if module is already loaded in our cache
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]

    # Check if module exists in sys.modules (already imported elsewhere)
    if module_name in sys.modules:
        _loaded_modules[module_name] = sys.modules[module_name]
        return sys.modules[module_name]

    # Load the module for the first time
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")

  
    module = importlib.util.module_from_spec(spec)

    # Set package context for relative imports
    if package:
        module.__package__ = package

    # Register module in sys.modules before loading to prevent circular imports
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        _loaded_modules[module_name] = module
        return module
    except Exception as e:
        # Clean up sys.modules if import fails
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise ImportError(f"Failed to load module {module_name}: {e}")


def setup_voice_module_mocks(project_root: str, mock_voice_module: bool = True) -> None:
    """
    Set up common mocks for voice module testing.

    Args:
        project_root: Root directory of the project
        mock_voice_module: Whether to mock the voice module itself (set to False for config tests)
    """
    # List of modules to mock
    mock_modules = [
        'streamlit',
        'openai',
        'whisper',
        'librosa',
        'soundfile',
        'pyaudio',
        'google.cloud',
        'google.cloud.speech',
        'google.oauth2',
        'google.oauth2.service_account',
        'elevenlabs',
        'webrtcvad',
        'cryptography',
        'cryptography.fernet',
        'noisereduce',
        'pydub',
        'silero_vad',
        'ffmpeg',
        'sounddevice',
        'sounddevice.default',
        'sounddevice.rec',
        'sounddevice.play',
        'sounddevice.query_devices',
        'sounddevice.check_input_settings',
        'sounddevice.check_output_settings'
    ]

    # Create basic mocks
    for module_name in mock_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()

    # Mock specific submodules
    sys.modules['google.cloud.speech'] = MagicMock()
    sys.modules['google.oauth2.service_account'] = MagicMock()

    # Mock cryptography submodules
    crypto_mock = MagicMock()
    sys.modules['cryptography'] = crypto_mock
    sys.modules['cryptography.hazmat'] = MagicMock()
    sys.modules['cryptography.hazmat.primitives'] = MagicMock()
    sys.modules['cryptography.hazmat.primitives.hashes'] = MagicMock()
    sys.modules['cryptography.hazmat.primitives.kdf'] = MagicMock()
    sys.modules['cryptography.hazmat.primitives.kdf.pbkdf2'] = MagicMock()
    sys.modules['cryptography.hazmat.primitives.ciphers'] = MagicMock()
    sys.modules['cryptography.hazmat.primitives.ciphers.modes'] = MagicMock()
    sys.modules['cryptography.hazmat.backends'] = MagicMock()
    sys.modules['cryptography.hazmat.backends.default_backend'] = MagicMock()
    sys.modules['cryptography.fernet'] = MagicMock()

    # Mock openai.Audio for tests
    openai_mock = MagicMock()
    openai_mock.Audio.transcribe.return_value = {
        'text': 'mock transcription',
        'confidence': 0.95
    }
    sys.modules['openai'] = openai_mock

    # Mock sounddevice with proper functionality
    sounddevice_mock = MagicMock()
    sounddevice_mock.default = MagicMock()
    sounddevice_mock.default.device = (0, 0)
    sounddevice_mock.rec = MagicMock(return_value=numpy.array([0.1, 0.2, 0.3]))
    sounddevice_mock.play = MagicMock()
    sounddevice_mock.query_devices = MagicMock(return_value=[
        {'name': 'Mock Input', 'max_input_channels': 2, 'max_output_channels': 0},
        {'name': 'Mock Output', 'max_input_channels': 0, 'max_output_channels': 2}
    ])
    sounddevice_mock.check_input_settings = MagicMock(return_value=True)
    sounddevice_mock.check_output_settings = MagicMock(return_value=True)
    sys.modules['sounddevice'] = sounddevice_mock

    # Mock pyaudio with proper functionality
    pyaudio_mock = MagicMock()
    pyaudio_instance = MagicMock()
    pyaudio_instance.get_device_info_by_index.return_value = {
        'name': 'Mock Device',
        'maxInputChannels': 1,
        'maxOutputChannels': 2
    }
    pyaudio_mock.PyAudio.return_value = pyaudio_instance
    sys.modules['pyaudio'] = pyaudio_mock

    # Create a voice module with proper __path__ to support relative imports (only if requested)
    if mock_voice_module:
        voice_module = MagicMock()
        voice_module.__path__ = [str(Path(project_root) / 'voice')]
        sys.modules['voice'] = voice_module


def get_voice_config_module(project_root: str) -> Any:
    """Get or import the voice config module."""
    return safe_import_module(
        "voice.config",
        str(Path(project_root) / "voice" / "config.py"),
        package="voice"
    )


def get_audio_processor_module(project_root: str) -> Any:
    """Get or import the audio processor module."""
    return safe_import_module(
        "voice.audio_processor",
        str(Path(project_root) / "voice" / "audio_processor.py"),
        package="voice"
    )


def get_stt_service_module(project_root: str) -> Any:
    """Get or import the STT service module."""
    return safe_import_module(
        "voice.stt_service",
        str(Path(project_root) / "voice" / "stt_service.py"),
        package="voice"
    )


def get_tts_service_module(project_root: str) -> Any:
    """Get or import the TTS service module."""
    return safe_import_module(
        "voice.tts_service",
        str(Path(project_root) / "voice" / "tts_service.py"),
        package="voice"
    )


def get_security_module(project_root: str) -> Any:
    """Get or import the security module."""
    return safe_import_module(
        "voice.security",
        str(Path(project_root) / "voice" / "security.py"),
        package="voice"
    )


def clear_module_cache() -> None:
    """Clear the module cache (useful for testing)."""
    global _loaded_modules
    _loaded_modules.clear()