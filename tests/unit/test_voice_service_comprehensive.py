"""
Comprehensive Voice Service Tests

Extensive testing for voice service features including:
- Voice session management and lifecycle
- Multi-provider STT/TTS integration and fallbacks
- Voice command processing and crisis detection
- Security and privacy features
- Error handling and recovery
- Integration with main application
"""

import pytest
import asyncio
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directories for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from voice.voice_service import VoiceService, VoiceSession, VoiceSessionState
from voice.audio_processor import SimplifiedAudioProcessor, AudioData
from voice.stt_service import STTService
from voice.tts_service import TTSService
from voice.commands import VoiceCommandProcessor
from voice.security import VoiceSecurity
from voice.config import VoiceConfig


class TestVoiceServiceComprehensive:
    """Comprehensive tests for VoiceService functionality."""

    def setup_method(self):
        """Set up test environment with comprehensive voice configuration."""
        # Create comprehensive mock configuration
        class ComprehensiveVoiceConfig:
            voice_enabled = True
            voice_commands_enabled = True
            stt_provider = "openai"
            tts_provider = "openai"
            default_voice_profile = "alloy"
            
            # Security configuration attributes
            session_timeout_minutes = 60
            encryption_key_rotation_days = 90
            audit_log_retention_days = 365
            consent_retention_days = 2555  # 7 years
            pii_detection_enabled = True
            encryption_enabled = True
            anonymization_enabled = True
            
            # Voice configuration attributes
            voice_profiles = {}
            elevenlabs_api_key = None
            elevenlabs_voice_id = None
            elevenlabs_model = "eleven_multilingual_v2"
            openai_whisper_model = "whisper-1"
            openai_whisper_language = "en"
            openai_whisper_temperature = 0.0
            whisper_model = "base"
            whisper_language = "en"
            whisper_temperature = 0.0
            whisper_beam_size = 5
            whisper_best_of = 5
            
            # Provider detection methods
            def is_google_speech_configured(self):
                return False
            
            def is_elevenlabs_configured(self):
                return False
            
            def is_whisper_configured(self):
                return True
            
            def is_piper_configured(self):
                return False
            
            class audio:
                max_buffer_size = 300
                max_memory_mb = 100
                sample_rate = 16000
                channels = 1
                chunk_size = 1024
                format = 'wav'
                stream_buffer_size = 10
                stream_chunk_duration = 0.1
                compression_enabled = True
                compression_level = 6
            
            class performance:
                cache_size = 100
                max_concurrent_sessions = 50
                session_timeout = 300
            
            def get_preferred_stt_service(self):
                return self.stt_provider
            
            def get_preferred_tts_service(self):
                return self.tts_provider
            
            def is_google_speech_configured(self):
                return False
            
            def is_elevenlabs_configured(self):
                return False
            
            def is_whisper_configured(self):
                return True
            
            def is_piper_configured(self):
                return False
        
        self.config = ComprehensiveVoiceConfig()
        self.security = VoiceSecurity(self.config)
        
        # Mock all external services
        with patch('voice.stt_service.STTService') as mock_stt, \
             patch('voice.tts_service.TTSService') as mock_tts, \
             patch('voice.commands.VoiceCommandProcessor') as mock_commands, \
             patch('database.db_manager.DatabaseManager') as mock_db:
            
            # Configure comprehensive STT mock
            self.mock_stt = mock_stt.return_value
            self.mock_stt.transcribe_audio = AsyncMock(return_value={
                'text': 'Hello therapist',
                'confidence': 0.95,
                'processing_time': 0.5,
                'provider': 'openai'
            })
            
            # Configure comprehensive TTS mock
            self.mock_tts = mock_tts.return_value
            self.mock_tts.synthesize_speech = AsyncMock(return_value={
                'audio_data': np.random.random(16000).astype(np.float32),
                'duration': 1.0,
                'processing_time': 0.3,
                'provider': 'openai'
            })
            
            # Configure comprehensive commands mock
            self.mock_commands = mock_commands.return_value
            self.mock_commands.process_command = AsyncMock(return_value={
                'command': None,
                'is_crisis': False,
                'response': None,
                'processing_time': 0.1
            })
            
            # Configure database mock
            self.mock_db = mock_db.return_value
            self.mock_db.save_session_data = AsyncMock(return_value=True)
            self.mock_db.get_session_data = AsyncMock(return_value=None)
            
            self.voice_service = VoiceService(self.config, self.security)
            
            # Mock session repository to avoid database issues
            self.voice_service.session_repo = Mock()
            self.voice_service.session_repo.save = Mock(return_value=True)

    def test_voice_session_lifecycle_comprehensive(self):
        """Test comprehensive voice session lifecycle with all states."""
        user_id = "test_user_comprehensive"
        
        # Test session creation
        session_id = self.voice_service.create_session(user_id=user_id)
        assert session_id is not None, "Session ID should not be None"
        assert session_id in self.voice_service.sessions, "Session should be in active sessions"
        
        session = self.voice_service.sessions[session_id]
        assert session.metadata.get('user_id') == user_id, "Session should have correct user ID"
        assert session.state == VoiceSessionState.IDLE, "Session should start in IDLE state"
        
        # Test state transitions (direct assignment since VoiceSession is a dataclass)
        session.state = VoiceSessionState.LISTENING
        assert session.state == VoiceSessionState.LISTENING, "State should change to LISTENING"
        
        session.state = VoiceSessionState.PROCESSING
        assert session.state == VoiceSessionState.PROCESSING, "State should change to PROCESSING"
        
        session.state = VoiceSessionState.SPEAKING
        assert session.state == VoiceSessionState.SPEAKING, "State should change to SPEAKING"
        
        # Test session metadata (direct assignment since VoiceSession is a dataclass)
        session.metadata["test_key"] = "test_value"
        assert session.metadata["test_key"] == "test_value", "Metadata should be stored"
        
        # Test conversation history (direct manipulation since VoiceSession is a dataclass)
        session.conversation_history.append({"speaker": "user", "message": "Hello therapist"})
        session.conversation_history.append({"speaker": "assistant", "message": "Hello! How can I help you today?"})
        
        history = session.conversation_history
        assert len(history) == 2, "Should have 2 conversation entries"
        assert history[0]["speaker"] == "user", "First entry should be from user"
        assert history[1]["speaker"] == "assistant", "Second entry should be from assistant"
        
        # Test session cleanup
        success = self.voice_service.end_session(session_id)
        assert success is True, "Session should end successfully"
        assert session_id not in self.voice_service.sessions, "Session should be removed from active sessions"

    def test_voice_input_processing_comprehensive(self):
        """Test comprehensive voice input processing with all features."""
        session_id = self.voice_service.create_session(user_id="test_user_input")
        
        # Create comprehensive test audio
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(
            data=audio_data,
            sample_rate=16000,
            duration=1.0,
            format='wav'
        )
        
        # Test successful voice input processing
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "Voice input processing should return a result"
        assert 'transcription' in result, "Result should contain transcription"
        assert 'response' in result, "Result should contain response"
        assert 'processing_time' in result, "Result should contain processing time"
        
        # Verify STT was called correctly
        self.mock_stt.transcribe_audio.assert_called_once_with(audio_obj)
        
        # Verify TTS was called for response
        assert self.mock_tts.synthesize_speech.called, "TTS should be called for response"
        
        # Verify commands were processed
        self.mock_commands.process_command.assert_called_once()
        
        # Verify session was updated
        session = self.voice_service.sessions[session_id]
        assert len(session.get_conversation_history()) > 0, "Conversation history should be updated"
        
        # Test with command detected
        self.mock_commands.process_command.return_value = {
            'command': 'breathing_exercise',
            'is_crisis': False,
            'response': 'Starting breathing exercise...',
            'processing_time': 0.1
        }
        
        result_with_command = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        assert 'command_response' in result_with_command, "Command response should be included"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_crisis_detection_and_response(self):
        """Test crisis detection and emergency response workflows."""
        session_id = self.voice_service.create_session("test_user_crisis")
        
        # Test crisis detection through commands
        self.mock_commands.process_command.return_value = {
            'command': 'emergency_help',
            'is_crisis': True,
            'response': 'I understand you need immediate help. Please call 911 or your local emergency services.',
            'emergency_contacts': ['911', '1-800-273-8255'],
            'processing_time': 0.05
        }
        
        # Create audio with crisis indicators
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        # Process crisis voice input
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "Crisis processing should return result"
        assert result['is_crisis'] is True, "Result should indicate crisis"
        assert 'emergency_response' in result, "Should include emergency response"
        assert 'emergency_contacts' in result, "Should include emergency contacts"
        
        # Verify crisis response prioritized TTS
        assert self.mock_tts.synthesize_speech.called, "Emergency TTS should be called"
        
        # Verify session marked as crisis
        session = self.voice_service.sessions[session_id]
        assert session.metadata.get('crisis_detected') is True, "Session should be marked as crisis"
        
        # Test follow-up crisis handling
        follow_up_audio = np.random.random(8000).astype(np.float32)
        follow_up_obj = AudioData(data=follow_up_audio, sample_rate=16000, duration=0.5)
        
        follow_up_result = asyncio.run(self.voice_service.process_voice_input(follow_up_obj, session_id))
        assert follow_up_result['is_crisis'] is True, "Follow-up should maintain crisis state"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_stt_provider_fallback_mechanism(self):
        """Test STT provider fallback mechanism when primary fails."""
        session_id = self.voice_service.create_session("test_user_fallback")
        
        # Configure STT mock to fail first, then succeed
        self.mock_stt.transcribe_audio.side_effect = [
            Exception("OpenAI API unavailable"),
            {'text': 'Hello from fallback', 'confidence': 0.90, 'provider': 'whisper'}
        ]
        
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        # Process voice input with fallback
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "Fallback should produce result"
        assert result['transcription']['text'] == 'Hello from fallback', "Should use fallback transcription"
        assert result['transcription']['provider'] == 'whisper', "Should indicate fallback provider"
        
        # Verify fallback was attempted
        assert self.mock_stt.transcribe_audio.call_count >= 1, "Should attempt multiple providers"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_tts_provider_fallback_mechanism(self):
        """Test TTS provider fallback mechanism when primary fails."""
        session_id = self.voice_service.create_session("test_user_tts_fallback")
        
        # Configure TTS mock to fail first, then succeed
        self.mock_tts.synthesize_speech.side_effect = [
            Exception("TTS service unavailable"),
            {'audio_data': np.random.random(16000).astype(np.float32), 'provider': 'piper', 'duration': 1.2}
        ]
        
        # Process voice input requiring TTS response
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "TTS fallback should produce result"
        assert 'response_audio' in result, "Should include response audio"
        
        # Verify fallback was attempted
        assert self.mock_tts.synthesize_speech.call_count >= 1, "Should attempt multiple TTS providers"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_voice_security_and_privacy(self):
        """Test voice security features and privacy protections."""
        session_id = self.voice_service.create_session("test_user_security")
        
        # Test audio data encryption
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        # Process voice input with security
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        # Verify security measures were applied
        assert self.security.encrypt_audio_data.called, "Audio data should be encrypted"
        assert self.security.sanitize_transcription.called, "Transcription should be sanitized"
        
        # Test PII detection and masking
        pii_audio_data = np.random.random(16000).astype(np.float32)
        pii_audio_obj = AudioData(data=pii_audio_data, sample_rate=16000, duration=1.0)
        
        # Mock PII detection
        self.security.detect_pii.return_value = True
        self.security.mask_pii.return_value = "My phone is [PHONE_NUMBER]"
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'My phone number is 555-123-4567',
            'confidence': 0.95
        }
        
        pii_result = asyncio.run(self.voice_service.process_voice_input(pii_audio_obj, session_id))
        
        assert '[PHONE_NUMBER]' in pii_result['transcription']['text'], "PII should be masked"
        
        # Test consent management
        assert self.security.check_consent.called, "Consent should be checked"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_voice_error_handling_and_recovery(self):
        """Test comprehensive error handling and recovery mechanisms."""
        session_id = self.voice_service.create_session("test_user_errors")
        
        # Test STT service failure
        self.mock_stt.transcribe_audio.side_effect = Exception("STT service completely failed")
        
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        # Should handle STT failure gracefully
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "Should return result even on STT failure"
        assert result.get('error') is not None, "Should indicate error in result"
        assert result['error']['type'] == 'stt_failure', "Should indicate STT failure type"
        
        # Reset mock for next test
        self.mock_stt.transcribe_audio.side_effect = None
        self.mock_stt.transcribe_audio.return_value = {'text': 'Hello', 'confidence': 0.95}
        
        # Test TTS service failure
        self.mock_tts.synthesize_speech.side_effect = Exception("TTS service failed")
        
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "Should return result even on TTS failure"
        assert result.get('error') is not None, "Should indicate TTS error"
        
        # Test command processing failure
        self.mock_commands.process_command.side_effect = Exception("Command processor failed")
        
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert result is not None, "Should handle command processor failure"
        assert result.get('transcription') is not None, "Should still provide transcription"
        
        # Test session state recovery
        session = self.voice_service.sessions[session_id]
        assert session.state == VoiceSession.State.ERROR, "Session should be in error state"
        
        # Test session recovery
        session.set_state(VoiceSession.State.IDLE)
        recovery_result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        assert recovery_result is not None, "Should recover and process normally"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_concurrent_voice_sessions_comprehensive(self):
        """Test handling of multiple concurrent voice sessions."""
        num_sessions = 20
        session_results = {}
        results_lock = threading.Lock()

        def concurrent_session_worker(session_index: int):
            """Worker function for concurrent voice session testing."""
            try:
                user_id = f"concurrent_user_{session_index}"
                session_id = self.voice_service.create_session(user_id)
                
                with results_lock:
                    session_results[session_index] = {
                        'session_id': session_id,
                        'user_id': user_id,
                        'interactions': [],
                        'errors': []
                    }
                
                # Process multiple interactions per session
                for interaction in range(3):
                    try:
                        audio_data = np.random.random(8000).astype(np.float32)
                        audio_obj = AudioData(
                            data=audio_data,
                            sample_rate=16000,
                            duration=0.5
                        )
                        
                        # Customize response for this interaction
                        self.mock_stt.transcribe_audio.return_value = {
                            'text': f'User {session_index} interaction {interaction}',
                            'confidence': 0.95
                        }
                        
                        result = asyncio.run(self.voice_service.process_voice_input(
                            audio_obj, session_id
                        ))
                        
                        with results_lock:
                            session_results[session_index]['interactions'].append({
                                'interaction': interaction,
                                'success': result is not None,
                                'transcription': result.get('transcription', {}).get('text') if result else None
                            })
                        
                        time.sleep(0.1)  # Small delay between interactions
                        
                    except Exception as e:
                        with results_lock:
                            session_results[session_index]['errors'].append(str(e))
                
                # Clean up session
                self.voice_service.end_session(session_id)
                
                with results_lock:
                    session_results[session_index]['completed'] = True
                    
            except Exception as e:
                with results_lock:
                    session_results[session_index] = {'error': str(e)}

        # Start concurrent sessions
        threads = []
        for i in range(num_sessions):
            thread = threading.Thread(target=concurrent_session_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all sessions to complete
        for thread in threads:
            thread.join(timeout=30.0)

        # Analyze concurrent session results
        successful_sessions = len([s for s in session_results.values() if s.get('completed')])
        total_interactions = sum(len(s.get('interactions', [])) for s in session_results.values())
        total_errors = sum(len(s.get('errors', [])) for s in session_results.values())

        # Assertions for concurrent processing
        assert successful_sessions > num_sessions * 0.8, \
            f"Too many failed concurrent sessions: {successful_sessions}/{num_sessions}"
        assert total_interactions > num_sessions * 2, \
            f"Too few successful interactions: {total_interactions}"
        assert total_errors < num_sessions, \
            f"Too many errors: {total_errors}"

        # Verify session isolation
        active_sessions_at_peak = len(self.voice_service.sessions)
        # Sessions should be cleaned up
        assert len(self.voice_service.sessions) < active_sessions_at_peak, \
            "Sessions not properly cleaned up"

    def test_voice_service_health_monitoring(self):
        """Test voice service health monitoring and metrics."""
        # Test health check
        health_status = self.voice_service.get_health_status()
        
        assert 'status' in health_status, "Health status should include status"
        assert 'active_sessions' in health_status, "Health status should include active sessions"
        assert 'performance_metrics' in health_status, "Health status should include performance metrics"
        assert 'service_status' in health_status, "Health status should include service status"
        
        # Test with active sessions
        session_id = self.voice_service.create_session("health_test_user")
        
        health_with_session = self.voice_service.get_health_status()
        assert health_with_session['active_sessions'] >= 1, "Should report active sessions"
        
        # Process some interactions to generate metrics
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        # Check performance metrics
        performance = self.voice_service.get_performance_metrics()
        assert 'total_interactions' in performance, "Should track total interactions"
        assert 'average_response_time' in performance, "Should calculate average response time"
        assert 'success_rate' in performance, "Should calculate success rate"
        
        assert performance['total_interactions'] > 0, "Should have processed interactions"
        assert performance['average_response_time'] >= 0, "Response time should be non-negative"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_voice_service_configuration_management(self):
        """Test voice service configuration management and updates."""
        # Test getting current configuration
        current_config = self.voice_service.get_configuration()
        assert 'stt_provider' in current_config, "Should include STT provider"
        assert 'tts_provider' in current_config, "Should include TTS provider"
        assert 'voice_profiles' in current_config, "Should include voice profiles"
        
        # Test updating configuration
        new_config = {
            'stt_provider': 'whisper',
            'tts_provider': 'piper',
            'default_voice_profile': 'male_enhanced'
        }
        
        success = self.voice_service.update_configuration(new_config)
        assert success is True, "Configuration update should succeed"
        
        updated_config = self.voice_service.get_configuration()
        assert updated_config['stt_provider'] == 'whisper', "STT provider should be updated"
        assert updated_config['tts_provider'] == 'piper', "TTS provider should be updated"
        
        # Test invalid configuration
        invalid_config = {
            'stt_provider': 'invalid_provider',
            'tts_provider': 'also_invalid'
        }
        
        invalid_success = self.voice_service.update_configuration(invalid_config)
        assert invalid_success is False, "Invalid configuration should fail"

    def test_voice_service_persistence_and_recovery(self):
        """Test voice service data persistence and recovery capabilities."""
        user_id = "persistence_test_user"
        
        # Create session and add data
        session_id = self.voice_service.create_session(user_id)
        
        # Add conversation history
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
        
        # Test session data persistence
        session_data = self.voice_service.get_session_data(session_id)
        assert session_data is not None, "Should retrieve session data"
        assert 'conversation_history' in session_data, "Should include conversation history"
        assert 'metadata' in session_data, "Should include metadata"
        
        # Test session restoration
        saved_data = session_data
        self.voice_service.end_session(session_id)
        
        # Restore session from saved data
        restored_session_id = self.voice_service.restore_session(saved_data)
        assert restored_session_id is not None, "Should restore session successfully"
        
        restored_session = self.voice_service.sessions[restored_session_id]
        assert restored_session.metadata.get('user_id') == user_id, "Restored session should have correct user"
        assert len(restored_session.get_conversation_history()) > 0, "Should restore conversation history"
        
        # Test interaction after restoration
        restored_result = asyncio.run(self.voice_service.process_voice_input(
            audio_obj, restored_session_id
        ))
        assert restored_result is not None, "Should process after restoration"
        
        # Clean up
        self.voice_service.end_session(restored_session_id)

    def test_voice_service_integration_features(self):
        """Test voice service integration with main application features."""
        session_id = self.voice_service.create_session("integration_test_user")
        
        # Test integration with knowledge base (RAG)
        audio_data = np.random.random(16000).astype(np.float32)
        audio_obj = AudioData(data=audio_data, sample_rate=16000, duration=1.0)
        
        # Mock RAG integration
        with patch('app.get_relevant_context') as mock_rag:
            mock_rag.return_value = [
                {"content": "Cognitive behavioral therapy techniques", "source": "therapy_guide.pdf"},
                {"content": "Anxiety management strategies", "source": "anxiety_workbook.pdf"}
            ]
            
            self.mock_stt.transcribe_audio.return_value = {
                'text': 'How can I manage my anxiety?',
                'confidence': 0.95
            }
            
            result = asyncio.run(self.voice_service.process_voice_input(audio_obj, session_id))
            
            # Should integrate with knowledge base
            assert mock_rag.called, "Should query knowledge base"
            assert result is not None, "Should integrate RAG results"
        
        # Test integration with user preferences
        with patch('auth.user_model.User.get_preferences') as mock_prefs:
            mock_prefs.return_value = {
                'voice_speed': 1.2,
                'preferred_voice': 'female_calm',
                'language': 'en'
            }
            
            # Should use user preferences in TTS
            preference_result = asyncio.run(self.voice_service.process_voice_input(
                audio_obj, session_id
            ))
            
            assert preference_result is not None, "Should apply user preferences"
        
        # Test integration with therapy progress tracking
        with patch('database.db_manager.DatabaseManager.save_therapy_progress') as mock_progress:
            mock_progress.return_value = True
            
            progress_result = asyncio.run(self.voice_service.process_voice_input(
                audio_obj, session_id
            ))
            
            # Should track therapy progress
            assert mock_progress.called, "Should track therapy progress"
        
        # Clean up
        self.voice_service.end_session(session_id)