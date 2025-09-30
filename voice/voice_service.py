"""
Voice Service Orchestration Module

This module provides the main voice service that coordinates all voice features:
- STT and TTS service coordination
- Voice command processing
- Session management
- Audio processing pipeline
- Error handling and fallback
- Performance optimization
"""

import asyncio
import time
import json
from typing import Optional, Dict, List, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import logging
import threading
from enum import Enum

from .config import VoiceConfig, VoiceProfile
from .audio_processor import SimplifiedAudioProcessor, AudioData
from .stt_service import STTService, STTResult
from .tts_service import TTSService, TTSResult
from .security import VoiceSecurity
from .commands import VoiceCommandProcessor

class VoiceSessionState(Enum):
    """Voice session states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class VoiceSession:
    """Voice session data."""
    session_id: str
    state: VoiceSessionState
    start_time: float
    last_activity: float
    conversation_history: List[Dict[str, Any]]
    current_voice_profile: str
    audio_buffer: List[AudioData]
    metadata: Dict[str, Any]

class VoiceService:
    """Main voice service that coordinates all voice features."""

    def __init__(self, config: VoiceConfig, security: VoiceSecurity):
        """Initialize voice service."""
        self.config = config
        self.security = security
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.audio_processor = SimplifiedAudioProcessor(config)
        self.stt_service = STTService(config)
        self.tts_service = TTSService(config)
        self.command_processor = VoiceCommandProcessor(config)

        # Session management
        self.sessions: Dict[str, VoiceSession] = {}
        self.current_session_id: Optional[str] = None
        self._sessions_lock = threading.RLock()  # Thread-safe session access

        # Service state
        self.is_running = False
        self.voice_thread = None
        self.voice_queue = asyncio.Queue()
        self._event_loop = None  # Will store the event loop reference

        # Callbacks
        self.on_text_received: Optional[Callable[[str, str], None]] = None
        self.on_audio_played: Optional[Callable[[AudioData], None]] = None
        self.on_command_executed: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None

        # Performance tracking
        self.metrics = {
            'sessions_created': 0,
            'total_interactions': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }

    def initialize(self) -> bool:
        """Initialize the voice service."""
        try:
            # Check if voice features are enabled
            if not self.config.voice_enabled:
                self.logger.info("Voice features are disabled")
                return False

            # Check service availability
            if not self._check_service_availability():
                self.logger.error("Voice services not available")
                return False

            # Initialize security
            if not self.security.initialize():
                self.logger.error("Security initialization failed")
                return False

            # Start voice service thread
            self.is_running = True
            self.voice_thread = threading.Thread(
                target=self._voice_service_worker,
                daemon=True
            )
            self.voice_thread.start()

            self.logger.info("Voice service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing voice service: {str(e)}")
            self.is_running = False
            return False

    def _check_service_availability(self) -> bool:
        """Check if required services are available."""
        issues = []

        # Check audio devices
        if not self.audio_processor.input_devices:
            issues.append("No input audio devices found")

        if not self.audio_processor.output_devices:
            issues.append("No output audio devices found")

        # Check STT service
        if not self.stt_service.is_available():
            issues.append("No STT service available")

        # Check TTS service
        if not self.tts_service.is_available():
            issues.append("No TTS service available")

        if issues:
            self.logger.error("Service availability issues: " + "; ".join(issues))
            return False

        return True

    def _voice_service_worker(self):
        """Worker thread for voice service."""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop  # Store reference for callbacks

            while self.is_running:
                try:
                    # Process voice queue
                    loop.run_until_complete(self._process_voice_queue())
                    time.sleep(0.01)

                except Exception as e:
                    self.logger.error(f"Error in voice service worker: {str(e)}")
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Fatal error in voice service worker: {str(e)}")
        finally:
            self.is_running = False

    async def _process_voice_queue(self):
        """Process items from the voice queue."""
        try:
            while not self.voice_queue.empty():
                item = await asyncio.wait_for(
                    self.voice_queue.get(),
                    timeout=0.01
                )

                command, data = item

                if command == "start_session":
                    await self._handle_start_session(data)
                elif command == "stop_session":
                    await self._handle_stop_session(data)
                elif command == "start_listening":
                    await self._handle_start_listening(data)
                elif command == "stop_listening":
                    await self._handle_stop_listening(data)
                elif command == "speak_text":
                    await self._handle_speak_text(data)
                elif command == "process_audio":
                    await self._handle_process_audio(data)
                else:
                    self.logger.warning(f"Unknown voice command: {command}")

        except asyncio.TimeoutError:
            pass
        except Exception as e:
            self.logger.error(f"Error processing voice queue: {str(e)}")

    def create_session(self, session_id: Optional[str] = None, voice_profile: Optional[str] = None) -> str:
        """Create a new voice session."""
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"

        with self._sessions_lock:
            if session_id in self.sessions:
                self.logger.warning(f"Session {session_id} already exists")
                return session_id

            try:
                # Create session
                session = VoiceSession(
                    session_id=session_id,
                    state=VoiceSessionState.IDLE,
                    start_time=time.time(),
                    last_activity=time.time(),
                    conversation_history=[],
                    current_voice_profile=voice_profile or self.config.default_voice_profile,
                    audio_buffer=[],
                    metadata={}
                )

                self.sessions[session_id] = session
                self.current_session_id = session_id

                self.metrics['sessions_created'] += 1

                self.logger.info(f"Created voice session: {session_id}")
                return session_id

            except Exception as e:
                self.logger.error(f"Error creating session {session_id}: {str(e)}")
                raise

    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get voice session by ID."""
        with self._sessions_lock:
            return self.sessions.get(session_id)

    def get_current_session(self) -> Optional[VoiceSession]:
        """Get current voice session."""
        with self._sessions_lock:
            if self.current_session_id:
                return self.sessions.get(self.current_session_id)
            return None

    def destroy_session(self, session_id: str):
        """Destroy a voice session."""
        with self._sessions_lock:
            try:
                if session_id in self.sessions:
                    session = self.sessions[session_id]

                    # Stop any ongoing operations
                    if session.state == VoiceSessionState.LISTENING:
                        self.stop_listening(session_id)

                    if session.state == VoiceSessionState.SPEAKING:
                        self.stop_speaking(session_id)

                    # Remove session
                    del self.sessions[session_id]

                    if self.current_session_id == session_id:
                        self.current_session_id = None

                    self.logger.info(f"Destroyed voice session: {session_id}")

            except Exception as e:
                self.logger.error(f"Error destroying session {session_id}: {str(e)}")

    def start_listening(self, session_id: Optional[str] = None) -> bool:
        """Start listening for voice input."""
        if session_id is None:
            session_id = self.current_session_id

        session = self.get_session(session_id)
        if not session:
            self.logger.error(f"Session {session_id} not found")
            return False

        try:
            # Update session state
            session.state = VoiceSessionState.LISTENING
            session.last_activity = time.time()

            # Start audio recording
            success = self.audio_processor.start_recording(self._audio_callback)

            if success:
                self.logger.info(f"Started listening for session: {session_id}")
                return True
            else:
                session.state = VoiceSessionState.ERROR
                return False

        except Exception as e:
            session.state = VoiceSessionState.ERROR
            self.logger.error(f"Error starting listening for session {session_id}: {str(e)}")
            return False

    def stop_listening(self, session_id: Optional[str] = None) -> AudioData:
        """Stop listening and return recorded audio."""
        if session_id is None:
            session_id = self.current_session_id

        session = self.get_session(session_id)
        if not session:
            self.logger.error(f"Session {session_id} not found")
            return AudioData(np.array([]), 16000)

        try:
            # Stop audio recording
            audio_data = self.audio_processor.stop_recording()

            # Update session state
            session.state = VoiceSessionState.IDLE
            session.last_activity = time.time()

            # Add to audio buffer
            if len(audio_data.data) > 0:
                session.audio_buffer.append(audio_data)

            self.logger.info(f"Stopped listening for session: {session_id}")
            return audio_data

        except Exception as e:
            session.state = VoiceSessionState.ERROR
            self.logger.error(f"Error stopping listening for session {session_id}: {str(e)}")
            return AudioData(np.array([]), 16000)

    def _audio_callback(self, audio_data: AudioData):
        """Callback for audio data processing."""
        try:
            session = self.get_current_session()
            if session and session.state == VoiceSessionState.LISTENING:
                # Add to async queue for processing using stored event loop
                if self._event_loop and self._event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.voice_queue.put(("process_audio", (session.session_id, audio_data))),
                        self._event_loop
                    )
                else:
                    self.logger.warning("Voice service event loop not available, dropping audio data")

        except Exception as e:
            self.logger.error(f"Error in audio callback: {str(e)}")

    async def _handle_process_audio(self, data: tuple):
        """Handle audio processing."""
        session_id, audio_data = data
        session = self.get_session(session_id)

        if not session:
            return

        try:
            # Update session state
            session.state = VoiceSessionState.PROCESSING
            session.last_activity = time.time()

            # Apply security processing
            processed_audio = await self.security.process_audio(audio_data)

            # Check for voice commands
            if self.config.voice_commands_enabled:
                command_result = await self.command_processor.process_audio(processed_audio)
                if command_result:
                    await self._handle_voice_command(session_id, command_result)
                    return

            # Transcribe audio
            stt_result = await self.stt_service.transcribe_audio(processed_audio)

            if stt_result.text.strip():
                # Add to conversation history
                session.conversation_history.append({
                    'type': 'user',
                    'text': stt_result.text,
                    'timestamp': time.time(),
                    'confidence': stt_result.confidence,
                    'provider': stt_result.provider
                })

                # Update metrics
                self.metrics['total_interactions'] += 1

                # Notify callback
                if self.on_text_received:
                    self.on_text_received(session_id, stt_result.text)

        except Exception as e:
            session.state = VoiceSessionState.ERROR
            self.logger.error(f"Error processing audio for session {session_id}: {str(e)}")

            if self.on_error:
                self.on_error(session_id, e)

        finally:
            if session.state == VoiceSessionState.PROCESSING:
                session.state = VoiceSessionState.IDLE

    async def _handle_voice_command(self, session_id: str, command_result: Dict[str, Any]):
        """Handle voice command."""
        try:
            session = self.get_session(session_id)
            if not session:
                return

            # Execute command
            execution_result = await self.command_processor.execute_command(command_result)

            # Add to conversation history
            session.conversation_history.append({
                'type': 'command',
                'command': command_result['command'],
                'result': execution_result,
                'timestamp': time.time()
            })

            # Notify callback
            if self.on_command_executed:
                self.on_command_executed(session_id, execution_result)

            # Provide voice feedback
            if execution_result.get('voice_feedback'):
                await self.speak_text(execution_result['voice_feedback'], session_id)

        except Exception as e:
            self.logger.error(f"Error handling voice command for session {session_id}: {str(e)}")

    async def speak_text(self, text: str, session_id: Optional[str] = None, voice_profile: Optional[str] = None) -> bool:
        """Speak text using TTS."""
        if session_id is None:
            session_id = self.current_session_id

        session = self.get_session(session_id)
        if not session:
            self.logger.error(f"Session {session_id} not found")
            return False

        try:
            # Update session state
            session.state = VoiceSessionState.SPEAKING
            session.last_activity = time.time()

            # Get voice profile
            if voice_profile is None:
                voice_profile = session.current_voice_profile

            # Synthesize speech
            tts_result = await self.tts_service.synthesize_speech(text, voice_profile)

            # Add to conversation history
            session.conversation_history.append({
                'type': 'assistant',
                'text': text,
                'timestamp': time.time(),
                'voice_profile': voice_profile,
                'provider': tts_result.provider,
                'duration': tts_result.duration
            })

            # Play audio
            success = self.audio_processor.play_audio(tts_result.audio_data)

            if success:
                # Notify callback
                if self.on_audio_played:
                    self.on_audio_played(tts_result.audio_data)

                self.logger.info(f"Spoke text for session {session_id}: {text[:50]}...")
                return True
            else:
                session.state = VoiceSessionState.ERROR
                return False

        except Exception as e:
            session.state = VoiceSessionState.ERROR
            self.logger.error(f"Error speaking text for session {session_id}: {str(e)}")
            return False

        finally:
            if session.state == VoiceSessionState.SPEAKING:
                session.state = VoiceSessionState.IDLE

    def stop_speaking(self, session_id: Optional[str] = None):
        """Stop speaking."""
        if session_id is None:
            session_id = self.current_session_id

        session = self.get_session(session_id)
        if session and session.state == VoiceSessionState.SPEAKING:
            session.state = VoiceSessionState.IDLE
            session.last_activity = time.time()

    def get_session_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get session statistics."""
        if session_id is None:
            session_id = self.current_session_id

        session = self.get_session(session_id)
        if not session:
            return {}

        try:
            duration = time.time() - session.start_time
            user_messages = len([m for m in session.conversation_history if m['type'] == 'user'])
            assistant_messages = len([m for m in session.conversation_history if m['type'] == 'assistant'])

            return {
                'session_id': session_id,
                'state': session.state.value,
                'duration': duration,
                'last_activity': time.time() - session.last_activity,
                'conversation_length': len(session.conversation_history),
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'voice_profile': session.current_voice_profile,
                'audio_buffer_size': len(session.audio_buffer)
            }

        except Exception as e:
            self.logger.error(f"Error getting session statistics for {session_id}: {str(e)}")
            return {}

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get voice service statistics."""
        try:
            stt_stats = self.stt_service.get_statistics()
            tts_stats = self.tts_service.get_statistics()

            return {
                'is_running': self.is_running,
                'sessions_count': len(self.sessions),
                'current_session_id': self.current_session_id,
                'stt_service': stt_stats,
                'tts_service': tts_stats,
                'metrics': self.metrics,
                'audio_devices': {
                    'input_devices': len(self.audio_processor.input_devices),
                    'output_devices': len(self.audio_processor.output_devices)
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting service statistics: {str(e)}")
            return {}

    def set_voice_profile(self, session_id: Optional[str] = None, voice_profile: Optional[str] = None):
        """Set voice profile for session."""
        if session_id is None:
            session_id = self.current_session_id

        session = self.get_session(session_id)
        if session and voice_profile:
            session.current_voice_profile = voice_profile
            self.logger.info(f"Set voice profile for session {session_id} to {voice_profile}")

    def cleanup(self):
        """Clean up voice service resources."""
        try:
            self.is_running = False

            # Clear event loop reference
            self._event_loop = None

            # Stop all sessions
            for session_id in list(self.sessions.keys()):
                self.destroy_session(session_id)

            # Clean up components
            self.audio_processor.cleanup()
            self.stt_service.cleanup()
            self.tts_service.cleanup()
            self.command_processor.cleanup()

            self.logger.info("Voice service cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up voice service: {str(e)}")

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()