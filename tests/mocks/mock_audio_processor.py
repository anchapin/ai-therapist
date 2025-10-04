"""
Mock Audio Processor for testing.

This module provides a mock audio processor that returns proper numpy arrays
instead of Mock objects, fixing type compatibility issues in tests.
"""

import numpy as np
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from unittest.mock import MagicMock

# Import the real AudioData class
try:
    from voice.audio_processor import AudioData, AudioProcessorState
except ImportError:
    # Fallback for testing
    @dataclass
    class AudioData:
        data: np.ndarray
        sample_rate: int
        duration: float
        channels: int = 1
        format: str = "wav"
    
    class AudioProcessorState:
        IDLE = "idle"
        RECORDING = "recording"
        PROCESSING = "processing"
        PLAYING = "playing"
        ERROR = "error"


class MockAudioProcessor:
    """Mock audio processor that returns proper numpy arrays."""
    
    def __init__(self, config=None):
        """Initialize mock audio processor."""
        self.config = config
        self.state = AudioProcessorState.IDLE
        self.is_recording = False
        self.is_playing = False
        
        # Mock device lists
        self.input_devices = ['mock_microphone']
        self.output_devices = ['mock_speaker']
        self.default_input_device = 'mock_microphone'
        self.default_output_device = 'mock_speaker'
        
        # Audio buffer
        self.audio_buffer = []
        
        # Mock audio data
        self.sample_rate = 16000
        self.channels = 1
        
        # Mock methods that would normally interact with hardware
        self.start_recording = MagicMock(return_value=True)
        self.stop_recording = MagicMock(side_effect=self._mock_stop_recording)
        self.play_audio = MagicMock(return_value=True)
        self.stop_playback = MagicMock(return_value=True)
        self.get_audio_devices = MagicMock(return_value=(self.input_devices, self.output_devices))
        self.save_audio = MagicMock(return_value=True)
        self.load_audio = MagicMock(side_effect=self._mock_load_audio)
        
        # Additional methods for compatibility
        self.detect_audio_devices = MagicMock(return_value=(self.input_devices, self.output_devices))
        self.get_available_features = MagicMock(return_value={
            'audio_capture': True,
            'audio_playback': True,
            'noise_reduction': False,
            'vad': False,
            'quality_analysis': False,
            'format_conversion': True
        })
        self.get_state = MagicMock(return_value=self.state)
        self.is_available = MagicMock(return_value=True)
        self.get_status = MagicMock(return_value={
            'state': self.state,
            'is_recording': self.is_recording,
            'is_playing': self.is_playing,
            'input_devices': len(self.input_devices),
            'output_devices': len(self.output_devices),
            'sample_rate': self.sample_rate,
            'channels': self.channels
        })
        self.get_memory_usage = MagicMock(return_value={
            'buffer_size': 0,
            'max_buffer_size': 300,
            'buffer_usage_percent': 0,
            'memory_usage_bytes': 0,
            'memory_limit_bytes': 100 * 1024 * 1024,
            'memory_usage_percent': 0,
            'chunk_size': 1024,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        })
        self.force_cleanup_buffers = MagicMock(return_value=0)
        self.cleanup = MagicMock()
        
    def _mock_stop_recording(self) -> AudioData:
        """Mock stop recording that returns proper AudioData with numpy array."""
        # Create mock audio data (1 second of silence)
        samples = int(self.sample_rate * 1.0)  # 1 second
        audio_data = np.zeros(samples, dtype=np.float32)
        
        # Add some random noise to make it more realistic
        audio_data += np.random.normal(0, 0.01, samples).astype(np.float32)
        
        return AudioData(
            data=audio_data,
            sample_rate=self.sample_rate,
            duration=1.0,
            channels=self.channels,
            format="wav"
        )
    
    def _mock_load_audio(self, filepath: str) -> Optional[AudioData]:
        """Mock load audio that returns proper AudioData with numpy array."""
        # Create mock audio data (1 second of sine wave)
        samples = int(self.sample_rate * 1.0)  # 1 second
        t = np.linspace(0, 1.0, samples)
        frequency = 440  # A4 note
        audio_data = (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        
        return AudioData(
            data=audio_data,
            sample_rate=self.sample_rate,
            duration=1.0,
            channels=self.channels,
            format="wav"
        )
    
    def create_silent_audio(self, duration: float) -> AudioData:
        """Create silent audio data."""
        samples = int(duration * self.sample_rate)
        silent_data = np.zeros(samples, dtype=np.float32)
        
        return AudioData(
            data=silent_data,
            sample_rate=self.sample_rate,
            duration=duration,
            channels=self.channels,
            format="wav"
        )
    
    def get_recording_duration(self) -> float:
        """Get current recording duration."""
        return 0.0  # Mock implementation
    
    def get_audio_level(self) -> float:
        """Get current audio input level."""
        return 0.0  # Mock implementation


def create_mock_audio_processor(config=None) -> MockAudioProcessor:
    """Create a mock audio processor for testing."""
    return MockAudioProcessor(config)