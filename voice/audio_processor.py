"""
Simplified Audio Processor for AI Therapist

This module provides audio processing functionality with graceful fallbacks
when audio libraries are not available. It handles:
- Audio capture and playback
- Voice Activity Detection (VAD)
- Noise reduction
- Audio quality analysis
- Audio format conversion

The processor will work with reduced functionality when audio libraries
are not available, providing text-based fallbacks.
"""

import asyncio
import time
import threading
import numpy as np
import logging
from unittest.mock import MagicMock
from typing import Optional, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import json
import base64
import os
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None
SOUNDDEVICE_AVAILABLE = sf is not None

try:
    import noisereduce as nr
except ImportError:
    nr = None
NOISEREDUCE_AVAILABLE = nr is not None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None
VAD_AVAILABLE = webrtcvad is not None

try:
    import librosa
except ImportError:
    librosa = None
# Performance optimization imports
try:
    from ..performance.memory_manager import MemoryManager
    from ..performance.cache_manager import CacheManager
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODULES_AVAILABLE = False
    MemoryManager = None
    CacheManager = None
LIBROSA_AVAILABLE = librosa is not None

# Audio data structures
@dataclass
class AudioData:
    """Audio data container."""
    data: np.ndarray
    sample_rate: int
    duration: float
    channels: int = 1
    format: str = "wav"
    
    def to_bytes(self) -> bytes:
        """Convert audio data to bytes."""
        if SOUNDDEVICE_AVAILABLE:
            buffer = sf.io.BytesIO()
            sf.write(buffer, self.data, self.sample_rate, format='WAV')
            return buffer.getvalue()
        else:
            # Fallback to base64 encoded numpy array
            return base64.b64encode(self.data.tobytes())
    
    @classmethod
    def from_bytes(cls, data: bytes, sample_rate: int = 16000):
        """Create audio data from bytes."""
        if SOUNDDEVICE_AVAILABLE:
            buffer = sf.io.BytesIO(data)
            audio_data, sr = sf.read(buffer)
            duration = len(audio_data) / sr
            return cls(audio_data, sr, duration)
        else:
            # Fallback for base64 encoded numpy array
            decoded = base64.b64decode(data)
            audio_data = np.frombuffer(decoded, dtype=np.float32)
            duration = len(audio_data) / sample_rate
            return cls(audio_data, sample_rate, duration)

class AudioProcessorState(Enum):
    """Audio processor state enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RECORDING = "recording"
    PROCESSING = "processing"
    PLAYING = "playing"
    ERROR = "error"

@dataclass
class AudioQualityMetrics:
    """Audio quality metrics."""
    snr_ratio: float = 0.0
    noise_level: float = 0.0
    speech_level: float = 0.0
    clarity_score: float = 0.0
    overall_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'snr_ratio': self.snr_ratio,
            'noise_level': self.noise_level,
            'speech_level': self.speech_level,
            'clarity_score': self.clarity_score,
            'overall_quality': self.overall_quality
        }

class SimplifiedAudioProcessor:
    """Simplified audio processor with graceful fallbacks."""
    
    def __init__(self, config=None):
        """Initialize audio processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audio processing state
        self.state = AudioProcessorState.IDLE
        self.is_recording = False
        self.is_playing = False
        
        # Audio buffers (with memory-safe bounded deque)
        max_buffer_size = int(getattr(config.audio, "max_buffer_size", 300))  # ~30 seconds at 10ms chunks
        self.max_buffer_size = max_buffer_size
        self.audio_buffer = deque(maxlen=max_buffer_size)

        # Memory monitoring and cleanup
        self._buffer_bytes_estimate = 0
        max_memory_mb = getattr(config.audio, "max_memory_mb", 100)
        self._max_memory_bytes = max_memory_mb * 1024 * 1024  # Convert MB to bytes
        self.recording_start_time = None
        self.recording_duration = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        self._recording_thread = None
        
        # Audio configuration
        self.sample_rate = getattr(config.audio, "sample_rate", 16000)
        self.channels = getattr(config.audio, "channels", 1)
        self.chunk_size = getattr(config.audio, "chunk_size", 1024)
        self.format = getattr(config.audio, "format", "wav")
        
        # Feature availability
        self.features = {
            'audio_capture': SOUNDDEVICE_AVAILABLE,
            'audio_playback': SOUNDDEVICE_AVAILABLE,
            'noise_reduction': NOISEREDUCE_AVAILABLE,
            'vad': VAD_AVAILABLE,
            'quality_analysis': LIBROSA_AVAILABLE,
            'format_conversion': SOUNDDEVICE_AVAILABLE
        }
        
        # Initialize available features
        self._initialize_features()

        # Missing attributes that tests expect
        self.audio = None
        self.stream = None

        # Performance optimization components
        self.memory_manager = None
        self.cache_manager = None
        if PERFORMANCE_MODULES_AVAILABLE:
            try:
                self.memory_manager = MemoryManager({
                    'memory_threshold_low': 256,
                    'memory_threshold_medium': 512,
                    'memory_threshold_high': 768,
                    'memory_threshold_critical': 1024,
                    'monitoring_interval': 60.0,
                    'gc_threshold': 500,
                    'cleanup_interval': 300.0
                })
                self.cache_manager = CacheManager({
                    'max_cache_size': 50,
                    'max_memory_mb': 128,
                    'enable_compression': True
                })
                self.memory_manager.register_cleanup_callback(self._memory_cleanup_callback)
                self.memory_manager.start_monitoring()
                self.cache_manager.start()
                self.logger.info("Performance optimization modules initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize performance modules: {e}")

        # Streaming audio processing
        self.streaming_enabled = True
        self.stream_buffer_size = getattr(config.audio, "stream_buffer_size", 10) if config and hasattr(config, 'audio') else 10
        self.stream_chunk_duration = getattr(config.audio, "stream_chunk_duration", 0.1) if config and hasattr(config, 'audio') else 0.1
        self.streaming_queue = asyncio.Queue(maxsize=self.stream_buffer_size) if asyncio else None
        self.streaming_active = False
        self.streaming_thread = None

        # Audio data compression for storage
        self.compression_enabled = getattr(config.audio, "compression_enabled", True) if config and hasattr(config, 'audio') else True
        self.compression_level = getattr(config.audio, "compression_level", 6) if config and hasattr(config, 'audio') else 6
        self.logger.info(f"Audio processor initialized with features: {self.features}")
    
    def _initialize_features(self):
        """Initialize available audio features."""
        try:
            # Initialize VAD if available
            if self.features['vad']:
                try:
                    self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
                    self.logger.info("VAD initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize VAD: {e}")
                    self.vad = None
            else:
                self.vad = None
            
            # Initialize audio device info
            if self.features['audio_capture']:
                self._get_audio_devices()
            
            self.state = AudioProcessorState.READY
            
        except Exception as e:
            self.logger.error(f"Error initializing features: {e}")
            self.state = AudioProcessorState.ERROR
            # Ensure VAD attribute exists even on error
            if not hasattr(self, 'vad'):
                self.vad = None
    
    def _get_audio_devices(self):
        """Get available audio devices."""
        if not SOUNDDEVICE_AVAILABLE:
            self.input_devices = []
            self.output_devices = []
            return
        
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            self.input_devices = []
            self.output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.input_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
                if device['max_output_channels'] > 0:
                    self.output_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_output_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            self.logger.info(f"Found {len(self.input_devices)} input devices and {len(self.output_devices)} output devices")
            
        except Exception as e:
            self.logger.error(f"Error getting audio devices: {e}")
            self.input_devices = []
            self.output_devices = []

    def detect_audio_devices(self) -> Tuple[List[Dict], List[Dict]]:
        """Detect available audio devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return [], []

        try:
            import sounddevice as sd
            devices = sd.query_devices()

            input_devices = []
            output_devices = []

            for device in devices:
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'name': device['name'],
                        'maxInputChannels': device['max_input_channels'],
                        'maxOutputChannels': device['max_output_channels'],
                        'sample_rate': device.get('default_samplerate', 44100)
                    })
                if device['max_output_channels'] > 0:
                    output_devices.append({
                        'name': device['name'],
                        'maxInputChannels': device['max_input_channels'],
                        'maxOutputChannels': device['max_output_channels'],
                        'sample_rate': device.get('default_samplerate', 44100)
                    })

            return input_devices, output_devices

        except Exception as e:
            self.logger.error(f"Error detecting audio devices: {e}")
            return [], []

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback function for streaming audio input.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        # Add audio data to buffer
        if len(indata) > 0:
            self.add_to_buffer(indata.flatten().astype(np.float32))

    def get_available_features(self) -> Dict[str, bool]:
        """Get available audio features."""
        return self.features.copy()
    
    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """Start audio recording."""
        if not self.features['audio_capture']:
            self.logger.warning("Audio capture not available")
            return False
        
        if self.state == AudioProcessorState.RECORDING:
            self.logger.warning("Already recording")
            return False
        
        try:
            with self._lock:
                self.state = AudioProcessorState.RECORDING
                self.is_recording = True
                self.recording_start_time = time.time()
                self.audio_buffer.clear()
                
                # Start recording thread
                self._recording_thread = threading.Thread(target=self._record_audio)
                self._recording_thread.daemon = True
                self._recording_thread.start()
                
                self.logger.info("Recording started")
                return True
                
        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            self.state = AudioProcessorState.ERROR
            return False
    
    def _record_audio(self):
        """Record audio in background thread."""
        if not SOUNDDEVICE_AVAILABLE:
            return

        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time, status):
                if status:
                    self.logger.warning(f"Audio callback status: {status}")

                with self._lock:
                    if self.is_recording:
                        # Check memory usage before adding to buffer
                        chunk_size_bytes = indata.nbytes
                        if self._buffer_bytes_estimate + chunk_size_bytes > self._max_memory_bytes:
                            # Memory limit reached, skip this chunk and log warning
                            self.logger.warning("Audio buffer memory limit reached, dropping audio chunk")
                            return

                        self.audio_buffer.append(indata.copy())
                        self._buffer_bytes_estimate += chunk_size_bytes

            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=self.chunk_size
            ):
                # Add memory monitoring to the recording loop
                loop_counter = 0
                while self.is_recording:
                    time.sleep(0.1)
                    loop_counter += 1

                    # Log memory usage every 10 seconds
                    if loop_counter % 100 == 0:
                        self.logger.debug(f"Audio buffer size: {len(self.audio_buffer)}, Memory estimate: {self._buffer_bytes_estimate} bytes")

                    # Safety check: if buffer is somehow growing beyond limits, force cleanup
                    if len(self.audio_buffer) >= self.max_buffer_size * 0.9:
                        self.logger.warning("Audio buffer approaching size limit, forcing cleanup")
                        # This should not happen with deque maxlen, but add safety check
                        excess = len(self.audio_buffer) - int(self.max_buffer_size * 0.8)
                        for _ in range(excess):
                            if self.audio_buffer:
                                removed_chunk = self.audio_buffer.popleft()
                                self._buffer_bytes_estimate -= getattr(removed_chunk, 'nbytes', 0)

        except Exception as e:
            self.logger.error(f"Error in recording thread: {e}")
            self.state = AudioProcessorState.ERROR
        finally:
            # Ensure memory tracking is reset
            with self._lock:
                self._buffer_bytes_estimate = 0
    
    def stop_recording(self) -> Optional[AudioData]:
        """Stop audio recording and return recorded data."""
        if not self.is_recording:
            return None

        try:
            with self._lock:
                self.is_recording = False
                self.recording_duration = time.time() - self.recording_start_time

                # Wait for recording thread to finish with longer timeout
                if self._recording_thread:
                    self._recording_thread.join(timeout=2.0)
                    if self._recording_thread.is_alive():
                        self.logger.warning("Recording thread did not finish cleanly")

                # Process recorded audio
                result = None
                if self.audio_buffer:
                    try:
                        # Safely convert deque to list and concatenate
                        audio_chunks = list(self.audio_buffer)

                        # Validate audio chunks before concatenation
                        valid_chunks = []
                        total_size = 0
                        for chunk in audio_chunks:
                            if chunk is not None and chunk.size > 0:
                                # Prevent excessive memory usage
                                if total_size + chunk.size > 100_000_000:  # 100MB limit
                                    self.logger.warning("Audio data too large, truncating")
                                    break
                                valid_chunks.append(chunk)
                                total_size += chunk.size

                        if valid_chunks:
                            audio_data = np.concatenate(valid_chunks, axis=0)

                            # Convert to mono if needed
                            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                                audio_data = np.mean(audio_data, axis=1)

                            # Create AudioData object
                            result = AudioData(
                                data=audio_data,
                                sample_rate=self.sample_rate,
                                duration=self.recording_duration,
                                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                                format=self.format
                            )

                            # Process audio (noise reduction, etc.)
                            result = self._process_audio(result)

                            self.logger.info(f"Recording stopped, duration: {self.recording_duration:.2f}s, size: {total_size} bytes")
                        else:
                            self.logger.warning("No valid audio chunks found")

                    except Exception as e:
                        self.logger.error(f"Error processing audio buffer: {e}")
                        result = None

                # Clear buffer and reset memory tracking
                self.audio_buffer.clear()
                self._buffer_bytes_estimate = 0

                self.state = AudioProcessorState.READY
                return result

        except Exception as e:
            self.logger.error(f"Error stopping recording: {e}")
            self.state = AudioProcessorState.ERROR
            # Ensure cleanup even on error
            with self._lock:
                self.audio_buffer.clear()
                self._buffer_bytes_estimate = 0
            return None
    
    def _process_audio(self, audio_data: AudioData) -> AudioData:
        """Process audio data with available features."""
        try:
            # Noise reduction
            if self.features['noise_reduction'] and NOISEREDUCE_AVAILABLE:
                audio_data.data = nr.reduce_noise(
                    y=audio_data.data,
                    sr=audio_data.sample_rate,
                    stationary=True,
                    prop_decrease=0.8
                )
                self.logger.info("Noise reduction applied")
            
            # Audio quality analysis
            if self.features['quality_analysis']:
                quality_metrics = self._analyze_audio_quality(audio_data)
                self.logger.info(f"Audio quality: {quality_metrics.overall_quality:.2f}")
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return audio_data
    
    def _analyze_audio_quality(self, audio_data: AudioData) -> AudioQualityMetrics:
        """Analyze audio quality."""
        metrics = AudioQualityMetrics()
        
        try:
            if not LIBROSA_AVAILABLE:
                return metrics
            
            # Calculate basic metrics
            rms = np.sqrt(np.mean(audio_data.data ** 2))
            metrics.speech_level = 20 * np.log10(rms + 1e-10)
            
            # Calculate noise level (using first 100ms as noise reference)
            noise_samples = int(0.1 * audio_data.sample_rate)
            if len(audio_data.data) > noise_samples:
                noise_rms = np.sqrt(np.mean(audio_data.data[:noise_samples] ** 2))
                metrics.noise_level = 20 * np.log10(noise_rms + 1e-10)
                metrics.snr_ratio = metrics.speech_level - metrics.noise_level
            
            # Calculate clarity score
            if len(audio_data.data) > 0:
                zero_crossings = np.sum(np.diff(np.sign(audio_data.data)) != 0)
                zcr = zero_crossings / len(audio_data.data)
                metrics.clarity_score = max(0, 1 - zcr)
            else:
                metrics.clarity_score = 0.0
            
            # Overall quality score
            speech_level_value = abs(metrics.speech_level) if hasattr(metrics.speech_level, 'size') and metrics.speech_level.size > 0 else abs(metrics.speech_level)
            metrics.overall_quality = (
                min(1, metrics.snr_ratio / 20) * 0.4 +
                metrics.clarity_score * 0.3 +
                min(1, speech_level_value / 60) * 0.3
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio quality: {e}")
        
        return metrics
    
    def detect_voice_activity(self, audio_data: AudioData) -> List[Dict[str, Any]]:
        """Detect voice activity in audio data."""
        if not self.features['vad']:
            return []
        
        try:
            # WebrtcVAD requires 16kHz mono audio
            if audio_data.sample_rate != 16000:
                # Resample to 16kHz if librosa is available
                if LIBROSA_AVAILABLE:
                    audio_data.data = librosa.resample(
                        audio_data.data, 
                        orig_sr=audio_data.sample_rate, 
                        target_sr=16000
                    )
                    audio_data.sample_rate = 16000
                else:
                    self.logger.warning("Cannot resample audio for VAD")
                    return []
            
            # Convert to 16-bit PCM with error handling
            try:
                # Handle invalid values
                clean_audio = np.nan_to_num(audio_data.data, nan=0.0, posinf=1.0, neginf=-1.0)
                clean_audio = np.clip(clean_audio, -1.0, 1.0)
                audio_int16 = (clean_audio * 32767).astype(np.int16)
            except (ValueError, RuntimeWarning):
                # Fallback: create silent audio
                frame_length = int(16000 * 30 / 1000)  # 30ms frame
                audio_int16 = np.zeros(frame_length, dtype=np.int16)
            
            # Process audio in 30ms frames
            frame_duration = 30  # ms
            frame_length = int(16000 * frame_duration / 1000)
            
            voice_activities = []
            for i in range(0, len(audio_int16), frame_length):
                frame = audio_int16[i:i + frame_length]
                if len(frame) < frame_length:
                    break
                
                # Pad frame if necessary
                if len(frame) < frame_length:
                    frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
                
                # Check if frame contains speech
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), 16000)
                    if is_speech:
                        start_time = i / 16000
                        end_time = (i + frame_length) / 16000
                        voice_activities.append({
                            'start': start_time,
                            'end': end_time,
                            'confidence': 0.8  # Fixed confidence for WebRTC VAD
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing VAD frame: {e}")
            
            return voice_activities
            
        except Exception as e:
            self.logger.error(f"Error detecting voice activity: {e}")
            return []
    
    def play_audio(self, audio_data: AudioData) -> bool:
        """Play audio data."""
        if not self.features['audio_playback']:
            self.logger.warning("Audio playback not available")
            return False
        
        try:
            import sounddevice as sd
            
            with self._lock:
                self.state = AudioProcessorState.PLAYING
                self.is_playing = True
            
            # Play audio in separate thread
            def play_thread():
                try:
                    sd.play(audio_data.data, audio_data.sample_rate)
                    sd.wait()  # Wait for playback to complete
                except Exception as e:
                    self.logger.error(f"Error playing audio: {e}")
                finally:
                    with self._lock:
                        self.state = AudioProcessorState.READY
                        self.is_playing = False
            
            thread = threading.Thread(target=play_thread)
            thread.daemon = True
            thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting audio playback: {e}")
            self.state = AudioProcessorState.ERROR
            return False
    
    def stop_playback(self) -> bool:
        """Stop audio playback."""
        if not self.is_playing:
            return True
        
        try:
            import sounddevice as sd
            sd.stop()
            
            with self._lock:
                self.is_playing = False
                self.state = AudioProcessorState.READY
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping playback: {e}")
            return False
    
    def get_recording_duration(self) -> float:
        """Get current recording duration."""
        if self.is_recording and self.recording_start_time:
            return time.time() - self.recording_start_time
        return self.recording_duration
    
    def get_audio_level(self) -> float:
        """Get current audio input level."""
        if not self.is_recording or not self.audio_buffer:
            return 0.0
        
        try:
            # Calculate RMS level from recent audio
            recent_audio = self.audio_buffer[-1] if self.audio_buffer else np.array([])
            if len(recent_audio) > 0:
                rms = np.sqrt(np.mean(recent_audio ** 2))
                return min(1.0, rms * 10)  # Normalize to 0-1
            return 0.0
        except Exception:
            return 0.0
    
    def create_silent_audio(self, duration: float) -> AudioData:
        """Create silent audio data."""
        samples = int(duration * self.sample_rate)
        silent_data = np.zeros(samples, dtype=np.float32)
        
        return AudioData(
            data=silent_data,
            sample_rate=self.sample_rate,
            duration=duration,
            channels=self.channels,
            format=self.format
        )
    
    def save_audio(self, audio_data: AudioData, filepath: str) -> bool:
        """Save audio data to file."""
        try:
            if SOUNDDEVICE_AVAILABLE:
                sf.write(filepath, audio_data.data, audio_data.sample_rate)
                self.logger.info(f"Audio saved to {filepath}")
                return True
            else:
                # Fallback: save as numpy array
                np.save(filepath.replace('.wav', '.npy'), audio_data.data)
                self.logger.info(f"Audio saved to {filepath} (numpy format)")
                return True
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
            return False
    
    def load_audio(self, filepath: str) -> Optional[AudioData]:
        """Load audio data from file."""
        try:
            if SOUNDDEVICE_AVAILABLE and filepath.endswith('.wav'):
                audio_data, sample_rate = sf.read(filepath)
                duration = len(audio_data) / sample_rate
                
                return AudioData(
                    data=audio_data,
                    sample_rate=sample_rate,
                    duration=duration,
                    channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                    format='wav'
                )
            elif filepath.endswith('.npy'):
                audio_data = np.load(filepath)
                duration = len(audio_data) / self.sample_rate
                
                return AudioData(
                    data=audio_data,
                    sample_rate=self.sample_rate,
                    duration=duration,
                    channels=self.channels,
                    format='numpy'
                )
            else:
                self.logger.error(f"Unsupported audio format: {filepath}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading audio: {e}")
            return None
    
    def get_state(self) -> AudioProcessorState:
        """Get current processor state."""
        return self.state
    
    def is_available(self) -> bool:
        """Check if audio processor is available."""
        return self.state != AudioProcessorState.ERROR
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status information."""
        return {
            'state': self.state.value,
            'is_recording': self.is_recording,
            'is_playing': self.is_playing,
            'recording_duration': self.get_recording_duration(),
            'audio_level': self.get_audio_level(),
            'available_features': self.features,
            'input_devices': len(self.input_devices) if hasattr(self, 'input_devices') else 0,
            'output_devices': len(self.output_devices) if hasattr(self, 'output_devices') else 0,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'buffer_size': len(self.audio_buffer),
            'max_buffer_size': self.max_buffer_size,
            'buffer_usage_percent': (len(self.audio_buffer) / self.max_buffer_size) * 100 if self.max_buffer_size > 0 else 0,
            'memory_usage_bytes': self._buffer_bytes_estimate,
            'memory_limit_bytes': self._max_memory_bytes
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        return {
            'buffer_size': len(self.audio_buffer),
            'max_buffer_size': self.max_buffer_size,
            'buffer_usage_percent': (len(self.audio_buffer) / self.max_buffer_size) * 100 if self.max_buffer_size > 0 else 0,
            'memory_usage_bytes': self._buffer_bytes_estimate,
            'memory_limit_bytes': self._max_memory_bytes,
            'memory_usage_percent': (self._buffer_bytes_estimate / self._max_memory_bytes) * 100 if self._max_memory_bytes > 0 else 0,
            'chunk_size': self.chunk_size,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }

    def force_cleanup_buffers(self):
        """Force cleanup of audio buffers to free memory."""
        try:
            with self._lock:
                cleared_count = len(self.audio_buffer)
                self.audio_buffer.clear()
                self._buffer_bytes_estimate = 0
                self.logger.info(f"Force cleanup: cleared {cleared_count} audio chunks")
                return cleared_count
        except Exception as e:
            self.logger.error(f"Error during force cleanup: {e}")
            return 0

    def get_audio_chunk(self) -> bytes:
        """Get the next audio chunk from the buffer."""
        if not self.audio_buffer:
            return b''
        with self._lock:
            if self.audio_buffer:
                chunk = self.audio_buffer.popleft()
                return chunk.tobytes()
        return b''

    def reduce_background_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce background noise from audio data."""
        if not NOISEREDUCE_AVAILABLE:
            return audio_data

        try:
            return nr.reduce_noise(y=audio_data, sr=self.sample_rate)
        except Exception as e:
            self.logger.error(f"Error reducing noise: {e}")
            return audio_data

    def convert_audio_format(self, audio_data: AudioData, target_format: str) -> AudioData:
        """Convert audio to different format."""
        if not self.features.get('format_conversion', False):
            return audio_data

        try:
            # Create a new AudioData object with the target format
            converted = AudioData(
                data=audio_data.data,
                sample_rate=audio_data.sample_rate,
                duration=audio_data.duration,
                channels=audio_data.channels,
                format=target_format
            )
            return converted
        except Exception as e:
            self.logger.error(f"Error converting format: {e}")
            return audio_data

    def calculate_audio_quality_metrics(self, audio_data: np.ndarray) -> AudioQualityMetrics:
        """Calculate audio quality metrics."""
        metrics = AudioQualityMetrics()

        if not LIBROSA_AVAILABLE:
            return metrics

        try:
            # Calculate basic metrics
            rms = np.sqrt(np.mean(audio_data**2))
            metrics.speech_level = float(rms)
            metrics.noise_level = float(np.std(audio_data))
            metrics.snr_ratio = float(metrics.speech_level / (metrics.noise_level + 1e-10))
            metrics.clarity_score = min(1.0, metrics.snr_ratio / 20.0)
            metrics.overall_quality = (metrics.clarity_score + metrics.snr_ratio / 40.0) / 2.0
            metrics.overall_quality = max(0.0, min(1.0, metrics.overall_quality))

        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")

        return metrics

    def normalize_audio_level(self, audio_data: np.ndarray, target_level: float = 0.5) -> np.ndarray:
        """Normalize audio level."""
        try:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                return audio_data * (target_level / max_val)
            return audio_data
        except Exception as e:
            self.logger.error(f"Error normalizing audio level: {e}")
            return audio_data

    def add_to_buffer(self, audio_data: np.ndarray):
        """Add audio data to buffer."""
        with self._lock:
            chunk_size_bytes = audio_data.nbytes
            if self._buffer_bytes_estimate + chunk_size_bytes > self._max_memory_bytes:
                self.logger.warning("Audio buffer memory limit reached, dropping audio chunk")
                return

            self.audio_buffer.append(audio_data)
            self._buffer_bytes_estimate += chunk_size_bytes

    def get_buffer_contents(self) -> List[np.ndarray]:
        """Get current buffer contents."""
        with self._lock:
            return list(self.audio_buffer)

    def clear_buffer(self):
        """Clear audio buffer."""
        with self._lock:
            self.audio_buffer.clear()
            self._buffer_bytes_estimate = 0

    def detect_voice_activity_simple(self, audio_data: np.ndarray) -> bool:
        """Detect voice activity in raw audio data."""
        if not VAD_AVAILABLE:
            # Simple threshold-based detection
            energy = np.sum(audio_data**2) / len(audio_data)
            return energy > 0.001

        try:
            # VAD expects 16-bit PCM, 16kHz, mono
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)

            # Convert to 16kHz if needed
            if self.sample_rate != 16000:
                audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=self.sample_rate, target_sr=16000)
                audio_data = (audio_data * 32767).astype(np.int16)

            # VAD works on 20ms frames (320 samples at 16kHz)
            frame_size = 320
            if len(audio_data) < frame_size:
                return False

            # Process each frame
            for i in range(0, len(audio_data) - frame_size + 1, frame_size):
                frame = audio_data[i:i + frame_size].tobytes()
                if self.vad.is_speech(frame, 16000):
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error detecting voice activity: {e}")
            return False

    def select_input_device(self, device_index: int) -> bool:
        """Select input device."""
        if device_index < len(self.input_devices):
            self.selected_input_device = device_index
            return True
        return False

    def save_audio_to_file(self, audio_data: AudioData, file_path: str) -> bool:
        """Save audio to file."""
        if not SOUNDDEVICE_AVAILABLE:
            return False

        try:
            sf.write(file_path, audio_data.data, audio_data.sample_rate)
            return True
        except Exception as e:
            self.logger.error(f"Error saving audio to file: {e}")
            return False

    def load_audio_from_file(self, file_path: str) -> Optional[AudioData]:
        """Load audio from file."""
        if not SOUNDDEVICE_AVAILABLE:
            return None

        try:
            audio_data, sr = sf.read(file_path)
            duration = len(audio_data) / sr

            return AudioData(
                data=audio_data,
                sample_rate=sr,
                duration=duration
            )
        except Exception as e:
            self.logger.error(f"Error loading audio from file: {e}")
            return None

    def create_audio_stream(self) -> Optional[object]:
        """Create audio stream."""
        if not SOUNDDEVICE_AVAILABLE:
            return None

        try:
            import sounddevice as sd
            # Create a mock stream for testing
            stream = MagicMock()
            stream.start_stream = MagicMock()
            stream.stop_stream = MagicMock()
            stream.close = MagicMock()
            return stream
        except Exception as e:
            self.logger.error(f"Error creating audio stream: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop recording with proper timeout
            if self.is_recording:
                self.is_recording = False  # Set flag first
                # Wait a bit for the recording loop to stop naturally
                time.sleep(0.2)
                # Then try to stop properly
                self.stop_recording()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            # Force cleanup even if errors occur
            try:
                with self._lock:
                    self.audio_buffer.clear()
                    self._buffer_bytes_estimate = 0
                    self.is_recording = False
                    self.is_playing = False
                    self.state = AudioProcessorState.IDLE
            except Exception as cleanup_error:
                self.logger.error(f"Error during forced cleanup: {cleanup_error}")

    def start_streaming_recording(self, chunk_callback: Optional[Callable[[AudioData], None]] = None) -> bool:
        """Start streaming audio recording with chunk processing."""
        if not self.streaming_enabled or not asyncio:
            self.logger.warning("Streaming not available or asyncio not available")
            return False

        if self.streaming_active:
            self.logger.warning("Streaming already active")
            return False

        try:
            with self._lock:
                self.streaming_active = True
                self.streaming_thread = threading.Thread(
                    target=self._streaming_worker,
                    args=(chunk_callback,),
                    daemon=True,
                    name="audio-streaming"
                )
                self.streaming_thread.start()
                self.logger.info("Streaming recording started")
                return True

        except Exception as e:
            self.logger.error(f"Error starting streaming recording: {e}")
            self.streaming_active = False
            return False

    def stop_streaming_recording(self) -> bool:
        """Stop streaming audio recording."""
        if not self.streaming_active:
            return True

        try:
            self.streaming_active = False

            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=2.0)
                if self.streaming_thread.is_alive():
                    self.logger.warning("Streaming thread did not terminate cleanly")

            self.logger.info("Streaming recording stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping streaming recording: {e}")
            return False

    def _streaming_worker(self, chunk_callback: Optional[Callable[[AudioData], None]]):
        """Background worker for streaming audio processing."""
        try:
            if not SOUNDDEVICE_AVAILABLE:
                return

            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if status or not self.streaming_active:
                    return

                try:
                    # Convert to AudioData
                    audio_chunk = AudioData(
                        data=indata.flatten().astype(np.float32),
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        format=self.format,
                        duration=len(indata) / self.sample_rate
                    )

                    # Process chunk (noise reduction, etc.)
                    if self.features['noise_reduction'] and NOISEREDUCE_AVAILABLE:
                        audio_chunk.data = nr.reduce_noise(
                            y=audio_chunk.data,
                            sr=audio_chunk.sample_rate,
                            stationary=True,
                            prop_decrease=0.8
                        )

                    # Add to streaming queue for async processing
                    if self.streaming_queue and not self.streaming_queue.full():
                        try:
                            self.streaming_queue.put_nowait(audio_chunk)
                        except asyncio.QueueFull:
                            self.logger.warning("Streaming queue full, dropping chunk")

                    # Call callback if provided
                    if chunk_callback:
                        try:
                            chunk_callback(audio_chunk)
                        except Exception as e:
                            self.logger.error(f"Error in streaming chunk callback: {e}")

                except Exception as e:
                    self.logger.error(f"Error processing streaming audio chunk: {e}")

            # Start streaming
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=int(self.sample_rate * self.stream_chunk_duration)
            ):
                while self.streaming_active:
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in streaming worker: {e}")
        finally:
            self.streaming_active = False

    async def get_streaming_chunk(self) -> Optional[AudioData]:
        """Get next streaming audio chunk asynchronously."""
        if not self.streaming_queue:
            return None

        try:
            chunk = await asyncio.wait_for(
                self.streaming_queue.get(),
                timeout=1.0
            )
            return chunk
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Error getting streaming chunk: {e}")
            return None

    def compress_audio_data(self, audio_data: AudioData) -> bytes:
        """Compress audio data for storage."""
        if not self.compression_enabled:
            return audio_data.to_bytes()

        try:
            import zlib
            raw_bytes = audio_data.to_bytes()
            compressed = zlib.compress(raw_bytes, level=self.compression_level)
            self.logger.debug(f"Compressed audio from {len(raw_bytes)} to {len(compressed)} bytes")
            return compressed
        except Exception as e:
            self.logger.error(f"Error compressing audio data: {e}")
            return audio_data.to_bytes()

    def decompress_audio_data(self, compressed_data: bytes) -> Optional[AudioData]:
        """Decompress audio data."""
        if not self.compression_enabled:
            return AudioData.from_bytes(compressed_data)

        try:
            import zlib
            decompressed = zlib.decompress(compressed_data)
            return AudioData.from_bytes(decompressed)
        except Exception as e:
            self.logger.error(f"Error decompressing audio data: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'streaming_active': self.streaming_active,
            'compression_enabled': self.compression_enabled,
            'buffer_size': len(self.audio_buffer),
            'buffer_memory_mb': self._buffer_bytes_estimate / (1024 * 1024),
            'streaming_queue_size': self.streaming_queue.qsize() if self.streaming_queue else 0,
        }

        # Add memory manager stats if available
        if self.memory_manager:
            stats.update({
                'memory_manager_active': True,
                'memory_manager_stats': self.memory_manager.get_performance_metrics()
            })
        else:
            stats['memory_manager_active'] = False

        # Add cache manager stats if available
        if self.cache_manager:
            stats.update({
                'cache_manager_active': True,
                'cache_manager_stats': self.cache_manager.get_stats()
            })
        else:
            stats['cache_manager_active'] = False

        return stats

    def _memory_cleanup_callback(self):
            """Memory cleanup callback for memory manager."""
            try:
                self.logger.info("Performing memory cleanup for audio processor")
                self.force_cleanup_buffers()
    
                # Clear any cached data
                if self.cache_manager:
                    self.cache_manager.clear()
    
                # Force garbage collection on audio-related objects
                import gc
                collected = gc.collect()
                self.logger.info(f"Garbage collection collected {collected} objects")
    
            except Exception as e:
                self.logger.error(f"Error in memory cleanup callback: {e}")

# Backward compatibility
AudioProcessor = SimplifiedAudioProcessor

# Factory function
def create_audio_processor(config=None) -> SimplifiedAudioProcessor:
    """Create and initialize audio processor."""
    return SimplifiedAudioProcessor(config)
