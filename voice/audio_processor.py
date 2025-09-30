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
from typing import Optional, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
import base64
import os
from pathlib import Path

try:
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

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
        
        # Audio buffers
        self.audio_buffer = []
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
        
        self.logger.info(f"Audio processor initialized with features: {self.features}")
    
    def _initialize_features(self):
        """Initialize available audio features."""
        try:
            # Initialize VAD if available
            if self.features['vad']:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
                self.logger.info("VAD initialized")
            
            # Initialize audio device info
            if self.features['audio_capture']:
                self._get_audio_devices()
            
            self.state = AudioProcessorState.READY
            
        except Exception as e:
            self.logger.error(f"Error initializing features: {e}")
            self.state = AudioProcessorState.ERROR
    
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
                self.audio_buffer = []
                
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
                        self.audio_buffer.append(indata.copy())
            
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=self.chunk_size
            ):
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Error in recording thread: {e}")
            self.state = AudioProcessorState.ERROR
    
    def stop_recording(self) -> Optional[AudioData]:
        """Stop audio recording and return recorded data."""
        if not self.is_recording:
            return None
        
        try:
            with self._lock:
                self.is_recording = False
                self.recording_duration = time.time() - self.recording_start_time
                
                # Wait for recording thread to finish
                if self._recording_thread:
                    self._recording_thread.join(timeout=1.0)
                
                # Process recorded audio
                if self.audio_buffer:
                    # Concatenate all audio chunks
                    audio_data = np.concatenate(self.audio_buffer, axis=0)
                    
                    # Convert to mono if needed
                    if audio_data.shape[1] > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    # Create AudioData object
                    result = AudioData(
                        data=audio_data,
                        sample_rate=self.sample_rate,
                        duration=self.recording_duration,
                        channels=self.channels,
                        format=self.format
                    )
                    
                    # Process audio (noise reduction, etc.)
                    result = self._process_audio(result)
                    
                    self.state = AudioProcessorState.READY
                    self.logger.info(f"Recording stopped, duration: {self.recording_duration:.2f}s")
                    return result
                
                self.state = AudioProcessorState.READY
                return None
                
        except Exception as e:
            self.logger.error(f"Error stopping recording: {e}")
            self.state = AudioProcessorState.ERROR
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
            zero_crossings = np.sum(np.diff(np.sign(audio_data.data)) != 0)
            zcr = zero_crossings / len(audio_data.data)
            metrics.clarity_score = max(0, 1 - zcr)
            
            # Overall quality score
            metrics.overall_quality = (
                min(1, metrics.snr_ratio / 20) * 0.4 +
                metrics.clarity_score * 0.3 +
                min(1, abs(metrics.speech_level) / 60) * 0.3
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
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data.data * 32767).astype(np.int16)
            
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
            'channels': self.channels
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop recording
            if self.is_recording:
                self.stop_recording()
            
            # Stop playback
            if self.is_playing:
                self.stop_playback()
            
            # Wait for threads to finish
            if self._recording_thread:
                self._recording_thread.join(timeout=1.0)
            
            self.state = AudioProcessorState.IDLE
            self.logger.info("Audio processor cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Backward compatibility
AudioProcessor = SimplifiedAudioProcessor

# Factory function
def create_audio_processor(config=None) -> SimplifiedAudioProcessor:
    """Create and initialize audio processor."""
    return SimplifiedAudioProcessor(config)
