"""
End-to-End Workflow Tests

Comprehensive testing of the complete AI Therapist application workflow including:
- Full user session lifecycle from start to finish
- Voice and text interaction workflows
- Crisis intervention and emergency response
- Multi-session concurrent usage
- Integration with all system components
- Real-world therapy scenarios
"""

import pytest
import asyncio
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import tempfile
import json

# Add parent directories for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import app
from voice.voice_service import VoiceService, VoiceSession
from voice.audio_processor import SimplifiedAudioProcessor, AudioData
from voice.stt_service import STTService
from voice.tts_service import TTSService
from voice.commands import VoiceCommandProcessor
from voice.security import VoiceSecurity
from voice.config import VoiceConfig
from auth.user_model import User
from database.db_manager import DatabaseManager


class TestEndToEndWorkflows:
    """End-to-end workflow tests for the complete AI Therapist application."""

    def setup_method(self):
        """Set up comprehensive test environment for end-to-end testing."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ELEVENLABS_API_KEY': 'test-elevenlabs-key',
            'JWT_SECRET_KEY': 'test-jwt-secret',
            'KNOWLEDGE_PATH': self.temp_dir + '/knowledge',
            'VECTORSTORE_PATH': self.temp_dir + '/vectorstore',
            'VOICE_ENABLED': 'true',
            'VOICE_INPUT_ENABLED': 'true',
            'VOICE_OUTPUT_ENABLED': 'true'
        })
        self.env_patcher.start()
        
        # Create comprehensive mock configuration
        self.mock_config = self._create_mock_config()
        self.mock_security = Mock()
        
        # Mock all external dependencies
        self._setup_mocks()
        
        # Create voice service with mocked dependencies
        self.voice_service = VoiceService(self.mock_config, self.mock_security)

    def teardown_method(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_config(self):
        """Create comprehensive mock configuration for testing."""
        class MockConfig:
            voice_enabled = True
            voice_commands_enabled = True
            stt_provider = "openai"
            tts_provider = "openai"
            default_voice_profile = "alloy"
            
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
        
        return MockConfig()

    def _setup_mocks(self):
        """Set up comprehensive mocks for all external dependencies."""
        # Mock STT service
        self.mock_stt = Mock()
        self.mock_stt.transcribe_audio = AsyncMock()
        
        # Mock TTS service
        self.mock_tts = Mock()
        self.mock_tts.synthesize_speech = AsyncMock()
        
        # Mock voice commands
        self.mock_commands = Mock()
        self.mock_commands.process_command = AsyncMock()
        
        # Mock database
        self.mock_db = Mock()
        self.mock_db.save_session_data = AsyncMock(return_value=True)
        self.mock_db.get_session_data = AsyncMock(return_value=None)
        self.mock_db.save_therapy_progress = AsyncMock(return_value=True)
        
        # Mock vector store
        self.mock_vectorstore = Mock()
        self.mock_vectorstore.similarity_search = Mock(return_value=[
            {"content": "CBT techniques for anxiety", "source": "cbt_guide.pdf"},
            {"content": "Mindfulness exercises", "source": "mindfulness.pdf"}
        ])
        
        # Mock Ollama
        self.mock_ollama = Mock()
        self.mock_ollama.generate = AsyncMock(return_value={
            'response': 'I understand you\'re feeling anxious. Let\'s work through some techniques together.',
            'model': 'llama3.2',
            'done': True
        })
        
        # Patch all external services
        patchers = [
            patch('voice.stt_service.STTService', return_value=self.mock_stt),
            patch('voice.tts_service.TTSService', return_value=self.mock_tts),
            patch('voice.commands.VoiceCommandProcessor', return_value=self.mock_commands),
            patch('database.db_manager.DatabaseManager', return_value=self.mock_db),
            patch('app.vectorstore', self.mock_vectorstore),
            patch('app.llm', self.mock_ollama)
        ]
        
        for patcher in patchers:
            patcher.start()

    def test_complete_voice_therapy_session(self):
        """Test a complete voice therapy session from start to finish."""
        user_id = "therapy_user_001"
        
        # Step 1: Initialize voice session
        session_id = self.voice_service.create_session(user_id)
        assert session_id is not None, "Session should be created successfully"
        
        session = self.voice_service.sessions[session_id]
        assert session.state == VoiceSession.State.IDLE, "Session should start idle"
        
        # Step 2: User introduces themselves via voice
        introduction_audio = np.random.random(16000).astype(np.float32)
        introduction_obj = AudioData(
            data=introduction_audio,
            sample_rate=16000,
            duration=1.0
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Hi, I\'m feeling anxious about work lately',
            'confidence': 0.95,
            'provider': 'openai'
        }
        
        self.mock_commands.process_command.return_value = {
            'command': None,
            'is_crisis': False,
            'response': None
        }
        
        # Mock RAG response for anxiety
        self.mock_vectorstore.similarity_search.return_value = [
            {"content": "Anxiety management techniques", "source": "anxiety_guide.pdf"}
        ]
        
        self.mock_ollama.generate.return_value = {
            'response': 'I understand work-related anxiety can be challenging. Let\'s explore some coping strategies together.',
            'model': 'llama3.2',
            'done': True
        }
        
        self.mock_tts.synthesize_speech.return_value = {
            'audio_data': np.random.random(16000).astype(np.float32),
            'duration': 3.2,
            'provider': 'openai'
        }
        
        # Process introduction
        intro_result = asyncio.run(self.voice_service.process_voice_input(
            introduction_obj, session_id
        ))
        
        assert intro_result is not None, "Introduction should be processed"
        assert intro_result['transcription']['text'] == 'Hi, I\'m feeling anxious about work lately'
        assert 'response_audio' in intro_result, "Should generate audio response"
        
        # Verify session state progression
        assert session.state == VoiceSession.State.IDLE, "Session should return to idle after processing"
        assert len(session.get_conversation_history()) == 2, "Should have user+assistant entries"
        
        # Step 3: User asks for specific help
        help_audio = np.random.random(24000).astype(np.float32)  # 1.5 seconds
        help_obj = AudioData(data=help_audio, sample_rate=16000, duration=1.5)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Can you teach me a breathing exercise?',
            'confidence': 0.92
        }
        
        self.mock_commands.process_command.return_value = {
            'command': 'breathing_exercise',
            'is_crisis': False,
            'response': 'I\'ll guide you through a calming breathing exercise.',
            'exercise_steps': ['Inhale for 4 counts', 'Hold for 4 counts', 'Exhale for 4 counts']
        }
        
        self.mock_tts.synthesize_speech.return_value = {
            'audio_data': np.random.random(32000).astype(np.float32),
            'duration': 8.5,
            'provider': 'openai'
        }
        
        # Process help request
        help_result = asyncio.run(self.voice_service.process_voice_input(
            help_obj, session_id
        ))
        
        assert help_result is not None, "Help request should be processed"
        assert help_result['transcription']['text'] == 'Can you teach me a breathing exercise?'
        assert 'command_response' in help_result, "Should include command response"
        
        # Step 4: User provides feedback
        feedback_audio = np.random.random(16000).astype(np.float32)
        feedback_obj = AudioData(data=feedback_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'That helped me feel calmer, thank you',
            'confidence': 0.96
        }
        
        self.mock_ollama.generate.return_value = {
            'response': 'I\'m glad the breathing exercise helped! Remember to use this technique whenever you feel anxious.',
            'model': 'llama3.2',
            'done': True
        }
        
        # Process feedback
        feedback_result = asyncio.run(self.voice_service.process_voice_input(
            feedback_obj, session_id
        ))
        
        assert feedback_result is not None, "Feedback should be processed"
        
        # Step 5: End session
        success = self.voice_service.end_session(session_id)
        assert success is True, "Session should end successfully"
        
        # Verify complete conversation history
        final_session_data = self.voice_service.get_session_data(session_id)
        conversation_history = final_session_data['conversation_history']
        
        assert len(conversation_history) >= 6, "Should have complete conversation history"
        assert conversation_history[0]['speaker'] == 'user', "Should start with user"
        assert conversation_history[-1]['speaker'] == 'assistant', "Should end with assistant"
        
        # Verify therapy progress was saved
        assert self.mock_db.save_therapy_progress.called, "Should save therapy progress"

    def test_crisis_intervention_workflow(self):
        """Test complete crisis intervention workflow."""
        user_id = "crisis_user_001"
        
        # Create session
        session_id = self.voice_service.create_session(user_id)
        
        # User expresses suicidal thoughts
        crisis_audio = np.random.random(16000).astype(np.float32)
        crisis_obj = AudioData(data=crisis_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I don\'t want to live anymore, I\'m thinking about ending it',
            'confidence': 0.94
        }
        
        # Crisis detection
        self.mock_commands.process_command.return_value = {
            'command': 'emergency_help',
            'is_crisis': True,
            'response': 'I\'m very concerned about what you\'re sharing. Please reach out for immediate help.',
            'emergency_contacts': ['911', '1-800-273-8255', 'Crisis Text Line: Text HOME to 741741'],
            'severity': 'high'
        }
        
        # Emergency response
        self.mock_tts.synthesize_speech.return_value = {
            'audio_data': np.random.random(24000).astype(np.float32),
            'duration': 6.0,
            'provider': 'openai'
        }
        
        # Process crisis input
        crisis_result = asyncio.run(self.voice_service.process_voice_input(
            crisis_obj, session_id
        ))
        
        assert crisis_result is not None, "Crisis should be processed"
        assert crisis_result['is_crisis'] is True, "Should detect crisis"
        assert 'emergency_response' in crisis_result, "Should include emergency response"
        assert 'emergency_contacts' in crisis_result, "Should include emergency contacts"
        
        # Verify session marked as crisis
        session = self.voice_service.sessions[session_id]
        assert session.metadata.get('crisis_detected') is True, "Session marked as crisis"
        assert session.metadata.get('crisis_severity') == 'high', "Severity recorded"
        
        # Follow-up crisis handling
        followup_audio = np.random.random(8000).astype(np.float32)
        followup_obj = AudioData(data=followup_audio, sample_rate=16000, duration=0.5)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I guess I could call the hotline',
            'confidence': 0.90
        }
        
        # Should maintain crisis response
        self.mock_commands.process_command.return_value = {
            'command': None,
            'is_crisis': True,
            'response': 'That\'s a good first step. The crisis counselors are trained to help.',
            'maintain_crisis_mode': True
        }
        
        followup_result = asyncio.run(self.voice_service.process_voice_input(
            followup_obj, session_id
        ))
        
        assert followup_result['is_crisis'] is True, "Should maintain crisis state"
        assert 'crisis_support' in followup_result, "Should provide continued support"
        
        # Verify emergency logging
        assert self.mock_db.save_session_data.called, "Should log crisis session"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_mixed_voice_and_text_interaction(self):
        """Test workflow with both voice and text interactions."""
        user_id = "mixed_interaction_user"
        
        # Create session
        session_id = self.voice_service.create_session(user_id)
        
        # Interaction 1: Voice greeting
        voice_greeting = np.random.random(16000).astype(np.float32)
        voice_obj = AudioData(data=voice_greeting, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Hello, I\'m here for my therapy session',
            'confidence': 0.96
        }
        
        voice_result = asyncio.run(self.voice_service.process_voice_input(
            voice_obj, session_id
        ))
        
        assert voice_result is not None, "Voice greeting should be processed"
        
        # Interaction 2: Text follow-up (simulating text input)
        text_message = "I\'ve been having trouble sleeping lately"
        
        # Mock text processing through voice service
        with patch.object(self.voice_service, 'process_text_input') as mock_text:
            mock_text.return_value = {
                'response': 'Sleep issues are very common. Let\'s explore what might be contributing.',
                'suggestions': ['Sleep hygiene techniques', 'Relaxation exercises'],
                'processing_time': 0.8
            }
            
            text_result = asyncio.run(self.voice_service.process_text_input(
                text_message, session_id
            ))
            
            assert text_result is not None, "Text input should be processed"
            assert 'suggestions' in text_result, "Should provide suggestions"
        
        # Interaction 3: Voice response to suggestions
        voice_response = np.random.random(24000).astype(np.float32)
        voice_response_obj = AudioData(data=voice_response, sample_rate=16000, duration=1.5)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'The sleep hygiene techniques sound helpful, can you explain more?',
            'confidence': 0.93
        }
        
        voice_response_result = asyncio.run(self.voice_service.process_voice_input(
            voice_response_obj, session_id
        ))
        
        assert voice_response_result is not None, "Voice response should be processed"
        
        # Verify mixed interaction history
        session_data = self.voice_service.get_session_data(session_id)
        conversation = session_data['conversation_history']
        
        # Should have both voice and text interactions properly logged
        assert len(conversation) >= 6, "Should record all interactions"
        
        # Verify interaction types are recorded
        for entry in conversation:
            assert 'speaker' in entry, "Should record speaker"
            assert 'message' in entry, "Should record message"
            assert 'timestamp' in entry, "Should record timestamp"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_concurrent_multi_user_therapy_sessions(self):
        """Test multiple concurrent therapy sessions for different users."""
        num_users = 10
        user_results = {}
        results_lock = threading.Lock()

        def user_session_worker(user_index: int):
            """Simulate a complete therapy session for one user."""
            user_id = f"concurrent_user_{user_index:03d}"
            
            try:
                # Create session
                session_id = self.voice_service.create_session(user_id)
                
                with results_lock:
                    user_results[user_index] = {
                        'user_id': user_id,
                        'session_id': session_id,
                        'interactions': [],
                        'errors': [],
                        'start_time': time.time()
                    }
                
                # Simulate therapy session with multiple interactions
                session_script = [
                    "I'm feeling stressed about my job",
                    "Can you help me with relaxation techniques?",
                    "That mindfulness exercise sounds helpful",
                    "Thank you, I feel better now"
                ]
                
                for interaction_num, script_text in enumerate(session_script):
                    try:
                        # Generate audio for this interaction
                        audio_data = np.random.random(16000).astype(np.float32)
                        audio_obj = AudioData(
                            data=audio_data,
                            sample_rate=16000,
                            duration=1.0
                        )
                        
                        # Configure response for this user and interaction
                        self.mock_stt.transcribe_audio.return_value = {
                            'text': script_text,
                            'confidence': 0.90 + (user_index * 0.01),  # Vary confidence slightly
                            'user_context': user_id
                        }
                        
                        # Generate personalized response
                        response_text = f"User {user_id}, I understand you're saying: {script_text[:20]}..."
                        
                        self.mock_ollama.generate.return_value = {
                            'response': response_text,
                            'model': 'llama3.2',
                            'done': True
                        }
                        
                        self.mock_tts.synthesize_speech.return_value = {
                            'audio_data': np.random.random(16000).astype(np.float32),
                            'duration': 2.0 + (user_index * 0.1),
                            'provider': 'openai'
                        }
                        
                        # Process interaction
                        start_time = time.time()
                        result = asyncio.run(self.voice_service.process_voice_input(
                            audio_obj, session_id
                        ))
                        end_time = time.time()
                        
                        with results_lock:
                            user_results[user_index]['interactions'].append({
                                'interaction_num': interaction_num,
                                'success': result is not None,
                                'processing_time': end_time - start_time,
                                'input_text': script_text,
                                'response_received': result.get('response') is not None if result else False
                            })
                        
                        # Small delay between interactions
                        time.sleep(0.05)
                        
                    except Exception as e:
                        with results_lock:
                            user_results[user_index]['errors'].append({
                                'interaction_num': interaction_num,
                                'error': str(e)
                            })
                
                # End session
                self.voice_service.end_session(session_id)
                
                with results_lock:
                    user_results[user_index]['end_time'] = time.time()
                    user_results[user_index]['total_duration'] = (
                        user_results[user_index]['end_time'] - 
                        user_results[user_index]['start_time']
                    )
                    user_results[user_index]['completed'] = True
                    
            except Exception as e:
                with results_lock:
                    user_results[user_index] = {
                        'user_id': user_id,
                        'error': str(e),
                        'failed': True
                    }

        # Start all concurrent sessions
        threads = []
        for i in range(num_users):
            thread = threading.Thread(target=user_session_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all sessions to complete
        for thread in threads:
            thread.join(timeout=60.0)

        # Analyze concurrent session results
        successful_users = len([u for u in user_results.values() if u.get('completed')])
        total_interactions = sum(len(u.get('interactions', [])) for u in user_results.values())
        total_errors = sum(len(u.get('errors', [])) for u in user_results.values())

        # Assertions for concurrent processing
        assert successful_users > num_users * 0.8, \
            f"Too many failed concurrent users: {successful_users}/{num_users}"
        assert total_interactions > num_users * 3, \
            f"Too few successful interactions: {total_interactions}"
        assert total_errors < num_users, \
            f"Too many errors in concurrent processing: {total_errors}"

        # Performance analysis
        completed_sessions = [u for u in user_results.values() if u.get('completed')]
        if completed_sessions:
            avg_duration = statistics.mean([u['total_duration'] for u in completed_sessions])
            assert avg_duration < 30.0, \
                f"Average session duration too long: {avg_duration:.1f}s"

            interaction_times = []
            for user_data in completed_sessions:
                interaction_times.extend([i['processing_time'] for i in user_data['interactions']])
            
            if interaction_times:
                avg_interaction_time = statistics.mean(interaction_times)
                assert avg_interaction_time < 3.0, \
                    f"Average interaction time too high: {avg_interaction_time:.2f}s"

        # Verify session isolation
        final_active_sessions = len(self.voice_service.sessions)
        assert final_active_sessions < num_users * 0.1, \
            "Too many sessions remain active (cleanup failed)"

    def test_voice_command_workflows(self):
        """Test comprehensive voice command workflows."""
        user_id = "command_user_001"
        session_id = self.voice_service.create_session(user_id)
        
        # Test 1: Breathing exercise command
        breathing_audio = np.random.random(8000).astype(np.float32)
        breathing_obj = AudioData(data=breathing_audio, sample_rate=16000, duration=0.5)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Start a breathing exercise',
            'confidence': 0.94
        }
        
        self.mock_commands.process_command.return_value = {
            'command': 'breathing_exercise',
            'is_crisis': False,
            'response': 'I\'ll guide you through a 4-7-8 breathing exercise.',
            'exercise_type': '4-7-8',
            'steps': [
                'Breathe in through your nose for 4 counts',
                'Hold your breath for 7 counts',
                'Exhale through your mouth for 8 counts'
            ],
            'duration_seconds': 120
        }
        
        breathing_result = asyncio.run(self.voice_service.process_voice_input(
            breathing_obj, session_id
        ))
        
        assert breathing_result is not None, "Breathing command should be processed"
        assert breathing_result['command_response']['command'] == 'breathing_exercise'
        assert 'steps' in breathing_result['command_response'], "Should include exercise steps"
        
        # Test 2: Reflection prompt command
        reflection_audio = np.random.random(12000).astype(np.float32)
        reflection_obj = AudioData(data=reflection_audio, sample_rate=16000, duration=0.75)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Give me a reflection prompt',
            'confidence': 0.92
        }
        
        self.mock_commands.process_command.return_value = {
            'command': 'reflection_prompt',
            'is_crisis': False,
            'response': 'Here\'s a reflection prompt for you:',
            'prompt': 'What are three things you\'re grateful for today?',
            'prompt_type': 'gratitude',
            'follow_up_suggestions': ['Take your time to think', 'Write down your thoughts']
        }
        
        reflection_result = asyncio.run(self.voice_service.process_voice_input(
            reflection_obj, session_id
        ))
        
        assert reflection_result is not None, "Reflection command should be processed"
        assert reflection_result['command_response']['prompt_type'] == 'gratitude'
        
        # Test 3: Session control command
        pause_audio = np.random.random(8000).astype(np.float32)
        pause_obj = AudioData(data=pause_audio, sample_rate=16000, duration=0.5)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Pause the session',
            'confidence': 0.95
        }
        
        self.mock_commands.process_command.return_value = {
            'command': 'pause_session',
            'is_crisis': False,
            'response': 'Session paused. Say "resume session" when you\'re ready to continue.',
            'session_state': 'paused',
            'timestamp': time.time()
        }
        
        pause_result = asyncio.run(self.voice_service.process_voice_input(
            pause_obj, session_id
        ))
        
        assert pause_result is not None, "Pause command should be processed"
        assert pause_result['command_response']['session_state'] == 'paused'
        
        # Verify session state is updated
        session = self.voice_service.sessions[session_id]
        assert session.metadata.get('session_state') == 'paused', "Session should be paused"
        
        # Test 4: Resume command
        resume_audio = np.random.random(8000).astype(np.float32)
        resume_obj = AudioData(data=resume_audio, sample_rate=16000, duration=0.5)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Resume session',
            'confidence': 0.96
        }
        
        self.mock_commands.process_command.return_value = {
            'command': 'resume_session',
            'is_crisis': False,
            'response': 'Session resumed. How are you feeling now?',
            'session_state': 'active',
            'timestamp': time.time()
        }
        
        resume_result = asyncio.run(self.voice_service.process_voice_input(
            resume_obj, session_id
        ))
        
        assert resume_result is not None, "Resume command should be processed"
        assert resume_result['command_response']['session_state'] == 'active'
        
        # Test 5: Unknown command handling
        unknown_audio = np.random.random(16000).astype(np.float32)
        unknown_obj = AudioData(data=unknown_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Do something random',
            'confidence': 0.88
        }
        
        self.mock_commands.process_command.return_value = {
            'command': None,
            'is_crisis': False,
            'response': None,
            'confidence': 0.3
        }
        
        # Should fall back to normal conversation processing
        self.mock_ollama.generate.return_value = {
            'response': 'I\'m not sure what specific command that is. Could you tell me more about what you\'d like help with?',
            'model': 'llama3.2',
            'done': True
        }
        
        unknown_result = asyncio.run(self.voice_service.process_voice_input(
            unknown_obj, session_id
        ))
        
        assert unknown_result is not None, "Unknown command should be handled gracefully"
        assert unknown_result['command_response']['command'] is None, "Should not recognize command"
        assert 'response' in unknown_result, "Should provide conversational response"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_therapy_progress_tracking_workflow(self):
        """Test therapy progress tracking throughout sessions."""
        user_id = "progress_tracking_user"
        
        # Session 1: Initial assessment
        session1_id = self.voice_service.create_session(user_id)
        
        initial_audio = np.random.random(16000).astype(np.float32)
        initial_obj = AudioData(data=initial_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I\'m new to therapy and feeling anxious about starting',
            'confidence': 0.94
        }
        
        self.mock_ollama.generate.return_value = {
            'response': 'It\'s completely normal to feel anxious about starting therapy. Let\'s take it one step at a time.',
            'model': 'llama3.2',
            'done': True
        }
        
        session1_result = asyncio.run(self.voice_service.process_voice_input(
            initial_obj, session1_id
        ))
        
        assert session1_result is not None, "Initial session should be processed"
        
        # End first session
        self.voice_service.end_session(session1_id)
        
        # Verify progress tracking
        assert self.mock_db.save_therapy_progress.called, "Should save initial progress"
        
        # Session 2: Follow-up after some time
        time.sleep(0.1)  # Simulate time passing
        
        session2_id = self.voice_service.create_session(user_id)
        
        followup_audio = np.random.random(16000).astype(np.float32)
        followup_obj = AudioData(data=followup_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I\'ve been practicing the breathing exercises and they help',
            'confidence': 0.96
        }
        
        # Mock progress data retrieval
        previous_progress = {
            'sessions_count': 1,
            'topics_discussed': ['anxiety', 'starting_therapy'],
            'techniques_practiced': [],
            'mood_ratings': [3],  # On 1-5 scale
            'last_session': time.time() - 86400  # Yesterday
        }
        
        self.mock_db.get_therapy_progress.return_value = previous_progress
        
        self.mock_ollama.generate.return_value = {
            'response': 'That\'s wonderful progress! Breathing exercises are a great foundation. How has your anxiety been overall?',
            'model': 'llama3.2',
            'done': True
        }
        
        session2_result = asyncio.run(self.voice_service.process_voice_input(
            followup_obj, session2_id
        ))
        
        assert session2_result is not None, "Follow-up session should be processed"
        
        # Progress update
        progress_audio = np.random.random(12000).astype(np.float32)
        progress_obj = AudioData(data=progress_audio, sample_rate=16000, duration=0.75)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'My anxiety is better, maybe a 4 out of 5 now',
            'confidence': 0.93
        }
        
        # Should update progress with new mood rating
        self.mock_ollama.generate.return_value = {
            'response': 'That\'s great improvement from when we started! The progress you\'re making is encouraging.',
            'model': 'llama3.2',
            'done': True
        }
        
        progress_result = asyncio.run(self.voice_service.process_voice_input(
            progress_obj, session2_id
        ))
        
        assert progress_result is not None, "Progress update should be processed"
        
        # Verify progress tracking calls
        assert self.mock_db.get_therapy_progress.called, "Should retrieve previous progress"
        assert self.mock_db.save_therapy_progress.call_count >= 2, "Should save updated progress"
        
        # Session 3: Long-term progress review
        session3_id = self.voice_service.create_session(user_id)
        
        review_audio = np.random.random(20000).astype(np.float32)
        review_obj = AudioData(data=review_audio, sample_rate=16000, duration=1.25)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Can we review my progress over the past few sessions?',
            'confidence': 0.95
        }
        
        # Mock comprehensive progress data
        comprehensive_progress = {
            'sessions_count': 3,
            'topics_discussed': ['anxiety', 'starting_therapy', 'breathing_exercises', 'progress'],
            'techniques_practiced': ['breathing_exercises'],
            'mood_ratings': [3, 4, 4],  # Improving trend
            'session_dates': [time.time() - 172800, time.time() - 86400, time.time()],
            'goals_achieved': ['learned_breathing_techniques'],
            'areas_of_improvement': ['anxiety_management']
        }
        
        self.mock_db.get_therapy_progress.return_value = comprehensive_progress
        
        self.mock_ollama.generate.return_value = {
            'response': 'Let\'s review your progress! You\'ve completed 3 sessions, learned breathing exercises, and your mood has improved from 3 to 4. That\'s excellent progress!',
            'model': 'llama3.2',
            'done': True
        }
        
        review_result = asyncio.run(self.voice_service.process_voice_input(
            review_obj, session3_id
        ))
        
        assert review_result is not None, "Progress review should be processed"
        
        # Verify comprehensive progress tracking
        progress_save_calls = self.mock_db.save_therapy_progress.call_args_list
        assert len(progress_save_calls) >= 3, "Should track progress across all sessions"
        
        # Check that progress data includes trends
        final_progress_call = progress_save_calls[-1]
        saved_progress = final_progress_call[0][0]  # First argument of the call
        
        assert 'mood_ratings' in saved_progress, "Should track mood trends"
        assert 'sessions_count' in saved_progress, "Should track session count"
        assert len(saved_progress['mood_ratings']) >= 3, "Should have multiple mood ratings"
        
        # Clean up
        self.voice_service.end_session(session2_id)
        self.voice_service.end_session(session3_id)

    def test_error_recovery_and_fallback_workflow(self):
        """Test comprehensive error recovery and fallback workflows."""
        user_id = "error_recovery_user"
        session_id = self.voice_service.create_session(user_id)
        
        # Test 1: STT service failure with fallback
        stt_failure_audio = np.random.random(16000).astype(np.float32)
        stt_failure_obj = AudioData(data=stt_failure_audio, sample_rate=16000, duration=1.0)
        
        # Primary STT fails
        self.mock_stt.transcribe_audio.side_effect = [
            Exception("OpenAI API unavailable"),
            {"text": "I'm having a hard time today", "confidence": 0.88, "provider": "whisper"}
        ]
        
        stt_result = asyncio.run(self.voice_service.process_voice_input(
            stt_failure_obj, session_id
        ))
        
        assert stt_result is not None, "Should recover from STT failure"
        assert stt_result['transcription']['provider'] == 'whisper', "Should use fallback provider"
        assert 'fallback_used' in stt_result, "Should indicate fallback was used"
        
        # Reset side effects
        self.mock_stt.transcribe_audio.side_effect = None
        
        # Test 2: TTS service failure with fallback
        tts_failure_audio = np.random.random(16000).astype(np.float32)
        tts_failure_obj = AudioData(data=tts_failure_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Can you help me understand my feelings?',
            'confidence': 0.92
        }
        
        # TTS fails
        self.mock_tts.synthesize_speech.side_effect = [
            Exception("TTS service unavailable"),
            {"audio_data": np.random.random(16000).astype(np.float32), "provider": "piper", "duration": 3.1}
        ]
        
        tts_result = asyncio.run(self.voice_service.process_voice_input(
            tts_failure_obj, session_id
        ))
        
        assert tts_result is not None, "Should recover from TTS failure"
        assert tts_result['response_audio']['provider'] == 'piper', "Should use TTS fallback"
        
        # Reset side effects
        self.mock_tts.synthesize_speech.side_effect = None
        
        # Test 3: Database failure with graceful degradation
        db_failure_audio = np.random.random(16000).astype(np.float32)
        db_failure_obj = AudioData(data=db_failure_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I want to track my progress',
            'confidence': 0.94
        }
        
        # Database fails
        self.mock_db.save_session_data.side_effect = Exception("Database connection lost")
        
        db_result = asyncio.run(self.voice_service.process_voice_input(
            db_failure_obj, session_id
        ))
        
        assert db_result is not None, "Should handle database failure gracefully"
        assert 'database_error' in db_result, "Should indicate database error"
        assert db_result['transcription']['text'] == 'I want to track my progress', "Should still process input"
        
        # Reset side effects
        self.mock_db.save_session_data.side_effect = None
        
        # Test 4: Network connectivity issues
        network_audio = np.random.random(16000).astype(np.float32)
        network_obj = AudioData(data=network_audio, sample_rate=16000, duration=1.0)
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'The exercises you recommended helped',
            'confidence': 0.91
        }
        
        # Simulate network timeout for LLM
        self.mock_ollama.generate.side_effect = [
            asyncio.TimeoutError("Network timeout"),
            {"response": "I'm glad the exercises helped! Would you like to try another technique?", "model": "llama3.2", "done": True}
        ]
        
        network_result = asyncio.run(self.voice_service.process_voice_input(
            network_obj, session_id
        ))
        
        assert network_result is not None, "Should recover from network timeout"
        assert network_result['response'] is not None, "Should provide fallback response"
        
        # Reset side effects
        self.mock_ollama.generate.side_effect = None
        
        # Test 5: Multiple simultaneous failures
        multi_failure_audio = np.random.random(16000).astype(np.float32)
        multi_failure_obj = AudioData(data=multi_failure_audio, sample_rate=16000, duration=1.0)
        
        # Multiple services fail
        self.mock_stt.transcribe_audio.side_effect = Exception("All STT services down")
        self.mock_tts.synthesize_speech.side_effect = Exception("All TTS services down")
        
        multi_failure_result = asyncio.run(self.voice_service.process_voice_input(
            multi_failure_obj, session_id
        ))
        
        assert multi_failure_result is not None, "Should handle multiple service failures"
        assert multi_failure_result.get('error') is not None, "Should indicate multiple errors"
        assert 'fallback_responses' in multi_failure_result, "Should provide fallback responses"
        
        # Verify session state after errors
        session = self.voice_service.sessions[session_id]
        assert session.state == VoiceSession.State.ERROR, "Session should be in error state"
        
        # Test 6: Session recovery after errors
        recovery_audio = np.random.random(16000).astype(np.float32)
        recovery_obj = AudioData(data=recovery_audio, sample_rate=16000, duration=1.0)
        
        # Reset all services to work normally
        self.mock_stt.transcribe_audio.side_effect = None
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I feel better now that things are working',
            'confidence': 0.95
        }
        
        self.mock_tts.synthesize_speech.side_effect = None
        self.mock_tts.synthesize_speech.return_value = {
            'audio_data': np.random.random(16000).astype(np.float32),
            'duration': 2.5,
            'provider': 'openai'
        }
        
        # Manually reset session state
        session.set_state(VoiceSession.State.IDLE)
        
        recovery_result = asyncio.run(self.voice_service.process_voice_input(
            recovery_obj, session_id
        ))
        
        assert recovery_result is not None, "Should recover and process normally"
        assert recovery_result.get('error') is None, "Should not have errors after recovery"
        assert session.state == VoiceSession.State.IDLE, "Session should be back to normal"
        
        # Clean up
        self.voice_service.end_session(session_id)

    def test_voice_quality_adaptation_workflow(self):
        """Test voice quality adaptation and optimization workflows."""
        user_id = "quality_adaptation_user"
        session_id = self.voice_service.create_session(user_id)
        
        # Test 1: Poor audio quality handling
        poor_quality_audio = np.random.normal(0, 0.5, 16000).astype(np.float32)  # Noisy audio
        poor_quality_obj = AudioData(
            data=poor_quality_audio,
            sample_rate=16000,
            duration=1.0,
            quality_metrics={'snr_db': 5.0, 'rms_level': 0.3}  # Poor quality
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I need help with stress',
            'confidence': 0.65,  # Low confidence due to poor audio
            'audio_quality': 'poor'
        }
        
        poor_result = asyncio.run(self.voice_service.process_voice_input(
            poor_quality_obj, session_id
        ))
        
        assert poor_result is not None, "Should handle poor quality audio"
        assert poor_result['transcription']['confidence'] < 0.8, "Should reflect low confidence"
        assert 'audio_quality_warning' in poor_result, "Should warn about audio quality"
        assert 'suggestions' in poor_result, "Should provide audio improvement suggestions"
        
        # Test 2: Audio enhancement processing
        enhancement_audio = np.random.normal(0, 0.2, 16000).astype(np.float32)  # Moderate noise
        enhancement_obj = AudioData(
            data=enhancement_audio,
            sample_rate=16000,
            duration=1.0,
            quality_metrics={'snr_db': 12.0, 'rms_level': 0.6}  # Moderate quality
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Can you speak more clearly?',
            'confidence': 0.78,
            'audio_quality': 'moderate',
            'enhanced': True
        }
        
        enhancement_result = asyncio.run(self.voice_service.process_voice_input(
            enhancement_obj, session_id
        ))
        
        assert enhancement_result is not None, "Should process enhanced audio"
        assert enhancement_result['transcription'].get('enhanced') is True, "Should indicate enhancement"
        
        # Test 3: Adaptive response based on audio quality
        adaptive_response_audio = np.random.random(16000).astype(np.float32)
        adaptive_response_obj = AudioData(
            data=adaptive_response_audio,
            sample_rate=16000,
            duration=1.0,
            quality_metrics={'snr_db': 20.0, 'rms_level': 0.8}  # Good quality
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'That\'s much clearer, thank you',
            'confidence': 0.94,
            'audio_quality': 'good'
        }
        
        # Response should adapt to improved audio quality
        self.mock_ollama.generate.return_value = {
            'response': 'Great! Your audio is much clearer now. I can better understand and help you.',
            'model': 'llama3.2',
            'done': True
        }
        
        adaptive_result = asyncio.run(self.voice_service.process_voice_input(
            adaptive_response_obj, session_id
        ))
        
        assert adaptive_result is not None, "Should provide adaptive response"
        assert 'clearer' in adaptive_result['response'], "Should acknowledge improved clarity"
        
        # Test 4: Real-time audio quality feedback
        realtime_feedback_audio = np.random.normal(0, 0.4, 8000).astype(np.float32)  # Short noisy clip
        realtime_feedback_obj = AudioData(
            data=realtime_feedback_audio,
            sample_rate=16000,
            duration=0.5,
            quality_metrics={'snr_db': 8.0, 'rms_level': 0.4}
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'Hello?',
            'confidence': 0.58,
            'audio_quality': 'poor',
            'realtime_feedback': True
        }
        
        realtime_result = asyncio.run(self.voice_service.process_voice_input(
            realtime_feedback_obj, session_id
        ))
        
        assert realtime_result is not None, "Should provide real-time feedback"
        assert 'realtime_quality_feedback' in realtime_result, "Should include real-time feedback"
        assert realtime_result['realtime_quality_feedback']['action_required'] is True, \
            "Should indicate action needed"
        
        # Test 5: Voice profile adaptation
        voice_profile_audio = np.random.random(16000).astype(np.float32)
        voice_profile_obj = AudioData(
            data=voice_profile_audio,
            sample_rate=16000,
            duration=1.0,
            voice_characteristics={'pitch': 'medium', 'speed': 'normal', 'accent': 'neutral'}
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': 'I usually speak with a deeper voice',
            'confidence': 0.91,
            'voice_profile_adapted': True,
            'adapted_profile': {'pitch_adjustment': -0.2, 'speed_adjustment': 0.0}
        }
        
        profile_result = asyncio.run(self.voice_service.process_voice_input(
            voice_profile_obj, session_id
        ))
        
        assert profile_result is not None, "Should adapt to voice profile"
        assert profile_result['transcription'].get('voice_profile_adapted') is True, \
            "Should indicate voice profile adaptation"
        
        # Test 6: Quality-based fallback decisions
        fallback_quality_audio = np.random.normal(0, 0.8, 16000).astype(np.float32)  # Very noisy
        fallback_quality_obj = AudioData(
            data=fallback_quality_audio,
            sample_rate=16000,
            duration=1.0,
            quality_metrics={'snr_db': 2.0, 'rms_level': 0.2}  # Very poor quality
        )
        
        self.mock_stt.transcribe_audio.return_value = {
            'text': '[UNCLEAR]',  # Unintelligible
            'confidence': 0.25,
            'audio_quality': 'unintelligible'
        }
        
        fallback_result = asyncio.run(self.voice_service.process_voice_input(
            fallback_quality_obj, session_id
        ))
        
        assert fallback_result is not None, "Should handle unintelligible audio"
        assert fallback_result['transcription']['text'] == '[UNCLEAR]', "Should indicate unclear audio"
        assert 'fallback_suggestion' in fallback_result, "Should suggest fallback approach"
        assert fallback_result['fallback_suggestion']['type'] == 'text_input', \
            "Should suggest text input fallback"
        
        # Clean up
        self.voice_service.end_session(session_id)


# Import statistics for performance analysis
import statistics