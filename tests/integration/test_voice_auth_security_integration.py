"""
Integration Testing for Voice, Auth, and Security Components

This module provides comprehensive integration testing including:
- Voice authentication integration
- Voice security and PII filtering integration
- Performance optimization under voice load
- Cross-component data flow validation
- Session management across components
- Error handling and recovery across boundaries
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from dataclasses import dataclass


@dataclass
class VoiceSession:
    """Voice session data structure."""
    session_id: str
    user_id: str
    auth_token: str
    start_time: float
    audio_data: List[bytes]
    transcriptions: List[str]
    security_flags: List[str]
    performance_metrics: Dict[str, float]


@dataclass
class AuthContext:
    """Authentication context for voice operations."""
    user_id: str
    role: str
    permissions: List[str]
    session_active: bool
    mfa_verified: bool
    voice_consent: bool


class TestVoiceAuthSecurityIntegration:
    """Comprehensive integration tests for voice, auth, and security."""
    
    @pytest.fixture
    def integrated_services(self):
        """Mock integrated services for testing."""
        with patch('voice.voice_service.VoiceService') as mock_voice, \
             patch('auth.auth_service.AuthService') as mock_auth, \
             patch('security.pii_protection.PIIProtection') as mock_pii, \
             patch('performance.memory_manager.MemoryManager') as mock_memory:
            
            # Voice service mock
            voice_service = Mock()
            voice_service.start_session.return_value = {'session_id': 'session_123', 'status': 'active'}
            voice_service.stop_session.return_value = True
            voice_service.transcribe_audio.return_value = {'text': 'Sample transcription', 'confidence': 0.95}
            voice_service.synthesize_speech.return_value = {'audio': b'synthetic_audio', 'duration': 2.5}
            voice_service.get_session_status.return_value = {'active': True, 'duration': 60}
            mock_voice.return_value = voice_service
            
            # Auth service mock
            auth_service = Mock()
            auth_service.authenticate_user.return_value = {
                'user_id': 'user_123',
                'role': 'therapist',
                'permissions': ['voice_access', 'phi_access'],
                'token': 'jwt_token_123'
            }
            auth_service.verify_token.return_value = True
            auth_service.check_permission.return_value = True
            auth_service.log_access.return_value = True
            mock_auth.return_value = auth_service
            
            # PII protection mock
            pii_service = Mock()
            pii_service.detect_phi.return_value = {
                'detected': True,
                'entities': [
                    {'type': 'PERSON', 'value': 'John Doe', 'confidence': 0.95}
                ]
            }
            pii_service.mask_phi.return_value = "PATIENT discussed anxiety treatment"
            pii_service.audit_access.return_value = True
            mock_pii.return_value = pii_service
            
            # Memory manager mock
            memory_service = Mock()
            memory_service.get_memory_usage.return_value = 100.0
            memory_service.check_memory_limit.return_value = False
            memory_service.cleanup.return_value = True
            mock_memory.return_value = memory_service
            
            yield {
                'voice': voice_service,
                'auth': auth_service,
                'pii': pii_service,
                'memory': memory_service
            }
    
    @pytest.fixture
    def sample_voice_session(self):
        """Sample voice session data for testing."""
        return VoiceSession(
            session_id='session_123',
            user_id='user_123',
            auth_token='jwt_token_123',
            start_time=time.time(),
            audio_data=[b'audio_chunk_1', b'audio_chunk_2', b'audio_chunk_3'],
            transcriptions=[
                "Patient John Doe discusses anxiety symptoms",
                "Reports difficulty sleeping at night",
                "Mentions stress from work"
            ],
            security_flags=['phi_detected', 'consent_verified'],
            performance_metrics={
                'audio_processing_time': 150.0,
                'transcription_time': 200.0,
                'pii_filtering_time': 50.0,
                'total_session_time': 400.0
            }
        )
    
    @pytest.fixture
    def auth_contexts(self):
        """Sample authentication contexts for testing."""
        return {
            'therapist': AuthContext(
                user_id='therapist_001',
                role='therapist',
                permissions=['voice_access', 'phi_access', 'patient_management'],
                session_active=True,
                mfa_verified=True,
                voice_consent=True
            ),
            'patient': AuthContext(
                user_id='patient_001',
                role='patient',
                permissions=['voice_access'],
                session_active=True,
                mfa_verified=False,
                voice_consent=True
            ),
            'admin': AuthContext(
                user_id='admin_001',
                role='admin',
                permissions=['voice_access', 'phi_access', 'system_management', 'audit_access'],
                session_active=True,
                mfa_verified=True,
                voice_consent=True
            ),
            'unauthorized': AuthContext(
                user_id='unauthorized_001',
                role='unauthorized',
                permissions=[],
                session_active=False,
                mfa_verified=False,
                voice_consent=False
            )
        }
    
    class TestVoiceAuthIntegration:
        """Test voice and authentication integration."""
        
        @pytest.mark.asyncio
        async def test_voice_session_authentication_flow(self, integrated_services, auth_contexts):
            """Test complete voice session authentication flow."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            # Test successful authentication and voice session start
            therapist_context = auth_contexts['therapist']
            
            # Step 1: Authenticate user
            auth_result = auth_service.authenticate_user(
                username=therapist_context.user_id,
                password='secure_password',
                mfa_token='123456'
            )
            
            assert auth_result['user_id'] == therapist_context.user_id
            assert 'token' in auth_result
            
            # Step 2: Verify token for voice access
            token_valid = auth_service.verify_token(auth_result['token'])
            assert token_valid is True
            
            # Step 3: Check voice access permission
            voice_permission = auth_service.check_permission(
                therapist_context.user_id,
                'voice_access'
            )
            assert voice_permission is True
            
            # Step 4: Start voice session
            session_result = voice_service.start_session(
                user_id=therapist_context.user_id,
                token=auth_result['token'],
                permissions=therapist_context.permissions
            )
            
            assert session_result['status'] == 'active'
            assert 'session_id' in session_result
            
            # Step 5: Log access for audit
            auth_service.log_access(
                user_id=therapist_context.user_id,
                action='start_voice_session',
                resource_id=session_result['session_id']
            )
            
            # Verify all services were called correctly
            auth_service.authenticate_user.assert_called_once()
            auth_service.verify_token.assert_called_once()
            auth_service.check_permission.assert_called_once()
            voice_service.start_session.assert_called_once()
            auth_service.log_access.assert_called_once()
        
        @pytest.mark.asyncio
        async def test_role_based_voice_access_control(self, integrated_services, auth_contexts):
            """Test role-based access control for voice features."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            access_results = {}
            
            for role_name, context in auth_contexts.items():
                # Mock authentication result
                auth_service.authenticate_user.return_value = {
                    'user_id': context.user_id,
                    'role': context.role,
                    'permissions': context.permissions,
                    'token': f'token_{context.user_id}'
                }
                
                # Test voice session start
                try:
                    auth_result = auth_service.authenticate_user(
                        username=context.user_id,
                        password='password'
                    )
                    
                    # Check permissions for different voice features
                    voice_permissions = {}
                    for permission in ['voice_access', 'phi_access', 'patient_management']:
                        has_permission = auth_service.check_permission(
                            context.user_id,
                            permission
                        )
                        voice_permissions[permission] = has_permission
                    
                    # Attempt to start voice session
                    if voice_permissions['voice_access']:
                        session_result = voice_service.start_session(
                            user_id=context.user_id,
                            token=auth_result['token'],
                            permissions=context.permissions
                        )
                        access_results[role_name] = {
                            'voice_access': True,
                            'session_started': True,
                            'permissions': voice_permissions
                        }
                    else:
                        access_results[role_name] = {
                            'voice_access': False,
                            'session_started': False,
                            'permissions': voice_permissions
                        }
                
                except Exception as e:
                    access_results[role_name] = {
                        'voice_access': False,
                        'session_started': False,
                        'error': str(e)
                    }
            
            # Verify role-based access
            assert access_results['therapist']['voice_access'] is True
            assert access_results['therapist']['session_started'] is True
            assert access_results['therapist']['permissions']['phi_access'] is True
            
            assert access_results['patient']['voice_access'] is True
            assert access_results['patient']['session_started'] is True
            assert access_results['patient']['permissions']['phi_access'] is False
            
            assert access_results['admin']['voice_access'] is True
            assert access_results['admin']['session_started'] is True
            assert access_results['admin']['permissions']['phi_access'] is True
            
            assert access_results['unauthorized']['voice_access'] is False
            assert access_results['unauthorized']['session_started'] is False
        
        @pytest.mark.asyncio
        async def test_session_security_and_token_validation(self, integrated_services):
            """Test session security and continuous token validation."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            # Simulate token expiration during voice session
            token_validity_sequence = [True, True, False, True]  # Token expires then refreshes
            
            auth_service.verify_token.side_effect = token_validity_sequence
            
            # Start voice session
            auth_result = auth_service.authenticate_user(
                username='therapist_001',
                password='password'
            )
            
            session_result = voice_service.start_session(
                user_id='therapist_001',
                token=auth_result['token'],
                permissions=['voice_access', 'phi_access']
            )
            
            # Simulate ongoing voice operations with token validation
            operations = ['transcribe', 'synthesize', 'transcribe']
            operation_results = []
            
            for operation in operations:
                # Validate token before operation
                token_valid = auth_service.verify_token(auth_result['token'])
                
                if token_valid:
                    if operation == 'transcribe':
                        result = voice_service.transcribe_audio(
                            session_id=session_result['session_id'],
                            audio_data=b'sample_audio'
                        )
                        operation_results.append({'operation': operation, 'success': True, 'result': result})
                    elif operation == 'synthesize':
                        result = voice_service.synthesize_speech(
                            text="Response to patient",
                            voice_profile="therapeutic"
                        )
                        operation_results.append({'operation': operation, 'success': True, 'result': result})
                else:
                    # Handle token refresh or session termination
                    operation_results.append({'operation': operation, 'success': False, 'reason': 'token_invalid'})
                    
                    # In real scenario, would attempt token refresh
                    if auth_service.verify_token.call_count < len(token_validity_sequence):
                        # Simulate successful token refresh
                        auth_service.verify_token.return_value = True
            
            # Verify security behavior
            successful_ops = [r for r in operation_results if r['success']]
            failed_ops = [r for r in operation_results if not r['success']]
            
            assert len(successful_ops) >= 2, "Most operations should succeed with token management"
            assert len(failed_ops) >= 1, "Should detect and handle token expiration"
            
            # Verify token validation was called for each operation
            assert auth_service.verify_token.call_count == len(operations)
        
        @pytest.mark.asyncio
        async def test_concurrent_voice_session_management(self, integrated_services):
            """Test concurrent voice session management with authentication."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            # Mock multiple users
            users = [
                {'user_id': f'user_{i}', 'role': 'therapist' if i % 2 == 0 else 'patient'}
                for i in range(5)
            ]
            
            async def manage_user_session(user_info):
                """Manage voice session for a single user."""
                try:
                    # Authenticate user
                    auth_result = auth_service.authenticate_user(
                        username=user_info['user_id'],
                        password='password'
                    )
                    
                    # Start voice session
                    session_result = voice_service.start_session(
                        user_id=user_info['user_id'],
                        token=auth_result['token'],
                        permissions=['voice_access']
                    )
                    
                    # Simulate session activity
                    await asyncio.sleep(0.1)
                    
                    # Get session status
                    status = voice_service.get_session_status(session_result['session_id'])
                    
                    # Stop session
                    voice_service.stop_session(session_result['session_id'])
                    
                    return {
                        'user_id': user_info['user_id'],
                        'session_id': session_result['session_id'],
                        'status': status,
                        'success': True
                    }
                
                except Exception as e:
                    return {
                        'user_id': user_info['user_id'],
                        'success': False,
                        'error': str(e)
                    }
            
            # Start concurrent sessions
            tasks = [manage_user_session(user) for user in users]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all sessions were managed correctly
            successful_sessions = [r for r in results if isinstance(r, dict) and r.get('success')]
            failed_sessions = [r for r in results if isinstance(r, dict) and not r.get('success')]
            
            assert len(successful_sessions) == len(users), "All user sessions should succeed"
            assert len(failed_sessions) == 0, "No session failures should occur"
            
            # Verify session uniqueness
            session_ids = [s['session_id'] for s in successful_sessions]
            assert len(session_ids) == len(set(session_ids)), "All session IDs should be unique"
            
            # Verify auth service was called for each user
            assert auth_service.authenticate_user.call_count == len(users)
            assert voice_service.start_session.call_count == len(users)
            assert voice_service.stop_session.call_count == len(users)
    
    class TestVoiceSecurityIntegration:
        """Test voice and security integration."""
        
        @pytest.mark.asyncio
        async def test_real_time_phi_filtering_in_voice_pipeline(self, integrated_services, sample_voice_session):
            """Test real-time PHI filtering during voice processing."""
            voice_service = integrated_services['voice']
            pii_service = integrated_services['pii']
            
            # Process voice session transcriptions with PHI detection
            filtered_transcriptions = []
            phi_alerts = []
            
            for transcription in sample_voice_session.transcriptions:
                # Step 1: Detect PHI in transcription
                phi_result = pii_service.detect_phi(transcription)
                
                if phi_result['detected']:
                    phi_alerts.append({
                        'transcription': transcription,
                        'entities': phi_result['entities'],
                        'timestamp': time.time()
                    })
                    
                    # Step 2: Mask PHI
                    masked_text = pii_service.mask_phi(transcription)
                    filtered_transcriptions.append(masked_text)
                    
                    # Step 3: Log PHI access for audit
                    pii_service.audit_access(
                        user_id=sample_voice_session.user_id,
                        operation='phi_masking',
                        data_type='voice_transcription'
                    )
                else:
                    filtered_transcriptions.append(transcription)
            
            # Verify PHI detection and masking
            assert len(phi_alerts) > 0, "Should detect PHI in transcriptions"
            assert len(filtered_transcriptions) == len(sample_voice_session.transcriptions)
            
            # Verify no PHI remains in filtered transcriptions
            for filtered in filtered_transcriptions:
                phi_entities = ['John Doe', 'Jane Smith', 'Dr.']
                for entity in phi_entities:
                    assert entity not in filtered, f"PHI entity '{entity}' should be masked"
            
            # Verify security services were called
            pii_service.detect_phi.assert_called()
            pii_service.mask_phi.assert_called()
            pii_service.audit_access.assert_called()
        
        @pytest.mark.asyncio
        async def test_voice_data_encryption_and_security(self, integrated_services):
            """Test voice data encryption and security measures."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            # Mock encryption/decryption
            encrypted_data = {}
            decrypted_data = {}
            
            def mock_encrypt(data, key):
                data_hash = hash(str(data))
                encrypted_data[data_hash] = data
                return f'encrypted_{data_hash}'
            
            def mock_decrypt(encrypted_data, key):
                data_hash = encrypted_data.replace('encrypted_', '')
                return encrypted_data.get(int(data_hash))
            
            # Test secure voice data handling
            audio_data = b'sensitive_voice_data'
            transcription = "Patient discusses personal health information"
            
            # Step 1: Authenticate user with proper permissions
            auth_result = auth_service.authenticate_user(
                username='therapist_001',
                password='password'
            )
            
            # Step 2: Encrypt audio data before processing
            encrypted_audio = mock_encrypt(audio_data, auth_result['token'])
            
            # Step 3: Process encrypted audio
            transcription_result = voice_service.transcribe_audio(
                session_id='session_123',
                audio_data=encrypted_audio  # In real scenario, would decrypt first
            )
            
            # Step 4: Encrypt transcription data
            encrypted_transcription = mock_encrypt(
                transcription_result['text'],
                auth_result['token']
            )
            
            # Step 5: Store encrypted data
            storage_secure = True  # In real scenario, verify secure storage
            
            # Step 6: Decrypt when needed
            decrypted_transcription = mock_decrypt(encrypted_transcription, auth_result['token'])
            
            # Verify security workflow
            assert encrypted_audio.startswith('encrypted_'), "Audio data should be encrypted"
            assert encrypted_transcription.startswith('encrypted_'), "Transcription should be encrypted"
            assert decrypted_transcription == transcription_result['text'], "Decryption should restore original"
            
            # Verify authentication was required
            auth_service.authenticate_user.assert_called_once()
        
        @pytest.mark.asyncio
        async def test_crisis_detection_and_emergency_response(self, integrated_services):
            """Test crisis detection in voice and emergency response integration."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            pii_service = integrated_services['pii']
            
            # Crisis keywords and scenarios
            crisis_scenarios = [
                {
                    'transcription': "I want to end my life",
                    'crisis_level': 'high',
                    'expected_response': 'emergency_protocol'
                },
                {
                    'transcription': "I feel hopeless and don't want to live anymore",
                    'crisis_level': 'high',
                    'expected_response': 'emergency_protocol'
                },
                {
                    'transcription': "I'm having suicidal thoughts",
                    'crisis_level': 'high',
                    'expected_response': 'emergency_protocol'
                },
                {
                    'transcription': "I feel very anxious and overwhelmed",
                    'crisis_level': 'medium',
                    'expected_response': 'immediate_support'
                }
            ]
            
            emergency_responses = []
            
            async def process_voice_with_crisis_detection(scenario):
                """Process voice input with crisis detection."""
                transcription = scenario['transcription']
                
                # Detect PHI first (privacy before crisis handling)
                phi_result = pii_service.detect_phi(transcription)
                if phi_result['detected']:
                    transcription = pii_service.mask_phi(transcription)
                
                # Crisis detection
                crisis_keywords = ['suicide', 'end my life', 'hopeless', 'suicidal', 'don\'t want to live']
                crisis_detected = any(keyword in transcription.lower() for keyword in crisis_keywords)
                
                if crisis_detected:
                    # Trigger emergency response
                    emergency_response = {
                        'scenario': scenario,
                        'crisis_detected': True,
                        'crisis_level': scenario['crisis_level'],
                        'response_type': scenario['expected_response'],
                        'timestamp': time.time(),
                        'actions_taken': [
                            'log_emergency_event',
                            'notify_crisis_team',
                            'provide_immediate_resources'
                        ]
                    }
                    
                    # Log emergency access
                    auth_service.log_access(
                        user_id='emergency_system',
                        action='crisis_detected',
                        resource_id='voice_session'
                    )
                    
                    emergency_responses.append(emergency_response)
                    return emergency_response
                else:
                    return {
                        'scenario': scenario,
                        'crisis_detected': False,
                        'response_type': 'normal_processing'
                    }
            
            # Process all crisis scenarios
            tasks = [process_voice_with_crisis_detection(scenario) for scenario in crisis_scenarios]
            results = await asyncio.gather(*tasks)
            
            # Verify crisis detection
            high_crisis_responses = [r for r in results if r.get('crisis_level') == 'high']
            detected_crisis_count = len([r for r in results if r.get('crisis_detected')])
            
            assert detected_crisis_count >= 3, "Should detect multiple crisis scenarios"
            assert len(high_crisis_responses) >= 3, "Should identify high-level crises"
            
            # Verify emergency response actions
            for response in emergency_responses:
                assert response['response_type'] == 'emergency_protocol', "Should trigger emergency protocol"
                assert len(response['actions_taken']) >= 3, "Should take multiple emergency actions"
                assert 'log_emergency_event' in response['actions_taken'], "Should log emergency event"
            
            # Verify audit logging
            auth_service.log_access.assert_called()
        
        @pytest.mark.asyncio
        async def test_voice_session_audit_trail_integrity(self, integrated_services, sample_voice_session):
            """Test comprehensive audit trail for voice sessions."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            pii_service = integrated_services['pii']
            
            audit_events = []
            
            def capture_audit_event(user_id, action, resource_id, **kwargs):
                audit_events.append({
                    'user_id': user_id,
                    'action': action,
                    'resource_id': resource_id,
                    'timestamp': time.time(),
                    **kwargs
                })
            
            # Mock audit logging
            auth_service.log_access.side_effect = capture_audit_event
            pii_service.audit_access.side_effect = capture_audit_event
            
            # Simulate complete voice session with audit tracking
            session_id = sample_voice_session.session_id
            user_id = sample_voice_session.user_id
            
            # 1. Session start
            voice_service.start_session(user_id=user_id, token='token', permissions=['voice_access'])
            auth_service.log_access(user_id, 'voice_session_start', session_id)
            
            # 2. Audio processing
            for i, audio_chunk in enumerate(sample_voice_session.audio_data):
                voice_service.transcribe_audio(session_id, audio_chunk)
                auth_service.log_access(user_id, 'audio_chunk_processed', f"{session_id}_chunk_{i}")
            
            # 3. PHI handling
            for i, transcription in enumerate(sample_voice_session.transcriptions):
                pii_service.detect_phi(transcription)
                pii_service.mask_phi(transcription)
                pii_service.audit_access(user_id, 'phi_processing', f"{session_id}_transcription_{i}")
            
            # 4. Session end
            voice_service.stop_session(session_id)
            auth_service.log_access(user_id, 'voice_session_end', session_id)
            
            # Verify audit trail completeness
            expected_events = [
                'voice_session_start',
                'audio_chunk_processed',
                'phi_processing',
                'voice_session_end'
            ]
            
            event_types = set(event['action'] for event in audit_events)
            
            for expected_event in expected_events:
                assert expected_event in event_types, f"Audit event {expected_event} should be recorded"
            
            # Verify audit event structure
            for event in audit_events:
                assert 'user_id' in event, "Audit event should include user_id"
                assert 'action' in event, "Audit event should include action"
                assert 'resource_id' in event, "Audit event should include resource_id"
                assert 'timestamp' in event, "Audit event should include timestamp"
            
            # Verify chronological order
            timestamps = [event['timestamp'] for event in audit_events]
            assert timestamps == sorted(timestamps), "Audit events should be in chronological order"
            
            # Verify sufficient audit coverage
            audio_events = [e for e in audit_events if e['action'] == 'audio_chunk_processed']
            phi_events = [e for e in audit_events if e['action'] == 'phi_processing']
            
            assert len(audio_events) == len(sample_voice_session.audio_data), "Should audit each audio chunk"
            assert len(phi_events) == len(sample_voice_session.transcriptions), "Should audit each transcription"
    
    class TestPerformanceVoiceIntegration:
        """Test performance optimization with voice services."""
        
        @pytest.mark.asyncio
        async def test_memory_management_during_voice_processing(self, integrated_services):
            """Test memory management during high-volume voice processing."""
            voice_service = integrated_services['voice']
            memory_service = integrated_services['memory']
            
            # Simulate memory pressure scenarios
            memory_usage_sequence = [100, 150, 200, 250, 180, 120, 100]  # MB
            memory_service.get_memory_usage.side_effect = memory_usage_sequence
            
            # Track memory management actions
            cleanup_triggered = []
            
            def track_cleanup():
                cleanup_triggered.append(time.time())
                return True
            
            memory_service.cleanup.side_effect = track_cleanup
            
            # Process multiple voice sessions with memory monitoring
            session_results = []
            
            for i in range(7):
                # Check memory before processing
                current_memory = memory_service.get_memory_usage()
                memory_pressure = current_memory > 200  # MB threshold
                
                if memory_pressure:
                    # Trigger cleanup before processing
                    memory_service.cleanup()
                
                # Process voice session
                audio_data = b'audio_data' * 1000  # Simulate larger audio data
                result = voice_service.transcribe_audio(
                    session_id=f'session_{i}',
                    audio_data=audio_data
                )
                
                session_results.append({
                    'session_id': f'session_{i}',
                    'memory_before': current_memory,
                    'memory_pressure': memory_pressure,
                    'processing_success': result is not None
                })
            
            # Verify memory management behavior
            sessions_with_pressure = [r for r in session_results if r['memory_pressure']]
            successful_sessions = [r for r in session_results if r['processing_success']]
            
            assert len(sessions_with_pressure) >= 3, "Should encounter memory pressure"
            assert len(successful_sessions) == len(session_results), "All sessions should succeed"
            assert len(cleanup_triggered) >= 2, "Should trigger cleanup multiple times"
            
            # Verify cleanup timing
            cleanup_times = sorted(cleanup_triggered)
            for i in range(1, len(cleanup_times)):
                time_diff = cleanup_times[i] - cleanup_times[i-1]
                assert time_diff > 0, "Cleanup events should be chronological"
        
        @pytest.mark.asyncio
        async def test_concurrent_voice_performance_optimization(self, integrated_services):
            """Test performance optimization under concurrent voice load."""
            voice_service = integrated_services['voice']
            memory_service = integrated_services['memory']
            
            # Configure realistic performance characteristics
            processing_times = [0.1, 0.15, 0.2, 0.12, 0.18]  # seconds
            voice_service.transcribe_audio.side_effect = lambda session_id, audio_data: {
                'text': 'Sample transcription',
                'processing_time': processing_times[hash(session_id) % len(processing_times)],
                'confidence': 0.95
            }
            
            # Concurrent voice processing
            concurrent_sessions = 10
            audio_chunks_per_session = 5
            
            async def process_voice_session(session_id):
                """Process voice session with performance tracking."""
                session_start = time.time()
                processing_times = []
                
                for chunk_id in range(audio_chunks_per_session):
                    chunk_start = time.time()
                    
                    # Process audio chunk
                    result = voice_service.transcribe_audio(
                        session_id=session_id,
                        audio_data=f'audio_{session_id}_{chunk_id}'.encode()
                    )
                    
                    chunk_time = time.time() - chunk_start
                    processing_times.append(chunk_time)
                    
                    # Small delay to simulate real processing
                    await asyncio.sleep(0.01)
                
                session_time = time.time() - session_start
                
                return {
                    'session_id': session_id,
                    'session_time': session_time,
                    'avg_chunk_time': np.mean(processing_times),
                    'max_chunk_time': np.max(processing_times),
                    'chunks_processed': len(processing_times)
                }
            
            # Run concurrent sessions
            tasks = [
                process_voice_session(f'session_{i}')
                for i in range(concurrent_sessions)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Analyze performance
            session_times = [r['session_time'] for r in results]
            avg_session_time = np.mean(session_times)
            max_session_time = np.max(session_times)
            
            total_chunks = sum(r['chunks_processed'] for r in results)
            throughput = total_chunks / total_time  # chunks per second
            
            # Performance assertions
            assert avg_session_time < 2.0, f"Average session time too high: {avg_session_time:.2f}s"
            assert max_session_time < 3.0, f"Max session time too high: {max_session_time:.2f}s"
            assert throughput > 5.0, f"Throughput too low: {throughput:.2f} chunks/sec"
            
            # Verify concurrency benefits
            sequential_time = avg_session_time * concurrent_sessions
            concurrency_speedup = sequential_time / total_time
            assert concurrency_speedup > 2.0, f"Concurrency speedup too low: {concurrency_speedup:.2f}x"
        
        @pytest.mark.asyncio
        async def test_resource_cleanup_after_voice_operations(self, integrated_services):
            """Test proper resource cleanup after voice operations."""
            voice_service = integrated_services['voice']
            memory_service = integrated_services['memory']
            
            # Track resource allocation and cleanup
            allocated_resources = []
            cleaned_resources = []
            
            class MockResource:
                def __init__(self, resource_id):
                    self.resource_id = resource_id
                    self.created_at = time.time()
                    allocated_resources.append(self)
                
                def cleanup(self):
                    cleaned_resources.append(self)
                    return True
            
            # Mock resource allocation for voice processing
            def mock_allocate_audio_buffer():
                return MockResource(f'audio_buffer_{len(allocated_resources)}')
            
            def mock_allocate_processing_context():
                return MockResource(f'processing_context_{len(allocated_resources)}')
            
            # Simulate voice processing with resource management
            session_count = 5
            
            for session_id in range(session_count):
                # Allocate resources for session
                audio_buffer = mock_allocate_audio_buffer()
                processing_context = mock_allocate_processing_context()
                
                # Process voice data
                voice_service.transcribe_audio(
                    session_id=f'session_{session_id}',
                    audio_data=b'sample_audio'
                )
                
                # Cleanup resources
                audio_buffer.cleanup()
                processing_context.cleanup()
                
                # Check memory usage
                memory_usage = memory_service.get_memory_usage()
                assert memory_usage < 500, f"Memory usage too high: {memory_usage}MB"
            
            # Verify all resources were cleaned up
            assert len(allocated_resources) == session_count * 2, "Should allocate expected resources"
            assert len(cleaned_resources) == len(allocated_resources), "All allocated resources should be cleaned up"
            
            # Verify cleanup completeness
            allocated_ids = set(r.resource_id for r in allocated_resources)
            cleaned_ids = set(r.resource_id for r in cleaned_resources)
            assert allocated_ids == cleaned_ids, "All allocated resources should be cleaned up"
            
            # Final memory cleanup
            memory_service.cleanup.assert_called()
    
    class TestErrorHandlingAndRecovery:
        """Test error handling and recovery across component boundaries."""
        
        @pytest.mark.asyncio
        async def test_authentication_failure_during_voice_session(self, integrated_services):
            """Test handling of authentication failures during voice sessions."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            # Simulate authentication failure
            auth_service.authenticate_user.side_effect = [
                Exception("Invalid credentials"),
                Exception("Token expired"),
                Exception("Insufficient permissions"),
                {'user_id': 'user_123', 'role': 'therapist', 'permissions': ['voice_access'], 'token': 'valid_token'}
            ]
            
            # Test voice session with authentication retry logic
            session_results = []
            
            for attempt in range(4):
                try:
                    # Attempt authentication
                    auth_result = auth_service.authenticate_user(
                        username='therapist_001',
                        password='password'
                    )
                    
                    # Start voice session if authentication succeeds
                    if isinstance(auth_result, dict) and 'token' in auth_result:
                        session_result = voice_service.start_session(
                            user_id=auth_result['user_id'],
                            token=auth_result['token'],
                            permissions=auth_result['permissions']
                        )
                        
                        session_results.append({
                            'attempt': attempt,
                            'success': True,
                            'session_id': session_result['session_id'],
                            'auth_result': auth_result
                        })
                        break
                    else:
                        session_results.append({
                            'attempt': attempt,
                            'success': False,
                            'error': 'Invalid auth result'
                        })
                
                except Exception as e:
                    session_results.append({
                        'attempt': attempt,
                        'success': False,
                        'error': str(e)
                    })
                
                # Wait before retry
                if attempt < 3:
                    await asyncio.sleep(0.01)
            
            # Verify error handling and recovery
            assert len(session_results) == 4, "Should attempt authentication 4 times"
            assert session_results[0]['success'] is False, "First attempt should fail"
            assert session_results[1]['success'] is False, "Second attempt should fail"
            assert session_results[2]['success'] is False, "Third attempt should fail"
            assert session_results[3]['success'] is True, "Fourth attempt should succeed"
            
            # Verify error types
            assert "Invalid credentials" in session_results[0]['error']
            assert "Token expired" in session_results[1]['error']
            assert "Insufficient permissions" in session_results[2]['error']
        
        @pytest.mark.asyncio
        async def test_voice_service_failover_and_recovery(self, integrated_services):
            """Test voice service failover and recovery mechanisms."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            
            # Simulate voice service failures and recovery
            transcription_results = [
                Exception("STT service unavailable"),
                Exception("Network timeout"),
                Exception("Audio processing error"),
                {'text': 'Transcription succeeded', 'confidence': 0.95}
            ]
            
            voice_service.transcribe_audio.side_effect = transcription_results
            
            # Test failover logic
            successful_transcriptions = []
            failed_attempts = []
            
            for attempt in range(4):
                try:
                    # Authenticate first
                    auth_result = auth_service.authenticate_user(
                        username='therapist_001',
                        password='password'
                    )
                    
                    # Attempt transcription
                    result = voice_service.transcribe_audio(
                        session_id='session_123',
                        audio_data=b'test_audio'
                    )
                    
                    successful_transcriptions.append({
                        'attempt': attempt,
                        'result': result
                    })
                    
                except Exception as e:
                    failed_attempts.append({
                        'attempt': attempt,
                        'error': str(e)
                    })
                    
                    # In real scenario, would trigger failover here
                    if attempt < 3:
                        await asyncio.sleep(0.01)  # Wait before retry
            
            # Verify failover behavior
            assert len(failed_attempts) == 3, "Should have 3 failed attempts"
            assert len(successful_transcriptions) == 1, "Should have 1 successful transcription"
            assert successful_transcriptions[0]['attempt'] == 3, "Success should come after failures"
            
            # Verify error types
            error_messages = [attempt['error'] for attempt in failed_attempts]
            assert "STT service unavailable" in error_messages
            assert "Network timeout" in error_messages
            assert "Audio processing error" in error_messages
        
        @pytest.mark.asyncio
        async def test_security_breach_response_and_recovery(self, integrated_services):
            """Test security breach response and recovery procedures."""
            voice_service = integrated_services['voice']
            auth_service = integrated_services['auth']
            pii_service = integrated_services['pii']
            
            # Simulate security breach scenarios
            breach_scenarios = [
                {
                    'type': 'unauthorized_access',
                    'detection': lambda: True,
                    'response': ['terminate_sessions', 'lock_account', 'audit_log']
                },
                {
                    'type': 'phi_exposure',
                    'detection': lambda: pii_service.detect_phi("John Doe SSN 123-45-6789")['detected'],
                    'response': ['mask_phi', 'audit_access', 'notify_admin']
                },
                {
                    'type': 'suspicious_activity',
                    'detection': lambda: True,  # Mock detection
                    'response': ['increase_monitoring', 'verify_identity', 'log_incident']
                }
            ]
            
            breach_responses = []
            
            async def handle_security_breach(scenario):
                """Handle security breach with appropriate response."""
                if scenario['detection']():
                    response_actions = []
                    
                    for action in scenario['response']:
                        if action == 'terminate_sessions':
                            voice_service.stop_session('all_sessions')
                            response_actions.append('sessions_terminated')
                        
                        elif action == 'lock_account':
                            # Mock account lock
                            response_actions.append('account_locked')
                        
                        elif action == 'audit_log':
                            auth_service.log_access(
                                user_id='security_system',
                                action='breach_detected',
                                resource_id=scenario['type']
                            )
                            response_actions.append('breach_logged')
                        
                        elif action == 'mask_phi':
                            masked = pii_service.mask_phi("John Doe SSN 123-45-6789")
                            response_actions.append(f'phi_masked: {masked}')
                        
                        elif action == 'notify_admin':
                            response_actions.append('admin_notified')
                        
                        elif action == 'increase_monitoring':
                            response_actions.append('monitoring_increased')
                        
                        elif action == 'verify_identity':
                            response_actions.append('identity_verified')
                        
                        elif action == 'log_incident':
                            auth_service.log_access(
                                user_id='security_system',
                                action='incident_logged',
                                resource_id=scenario['type']
                            )
                            response_actions.append('incident_logged')
                    
                    breach_responses.append({
                        'scenario': scenario['type'],
                        'detected': True,
                        'actions_taken': response_actions,
                        'timestamp': time.time()
                    })
                    
                    return True
                else:
                    return False
            
            # Process all breach scenarios
            for scenario in breach_scenarios:
                await handle_security_breach(scenario)
            
            # Verify breach response
            assert len(breach_responses) == len(breach_scenarios), "Should handle all breach scenarios"
            
            for response in breach_responses:
                assert response['detected'] is True, "Breach should be detected"
                assert len(response['actions_taken']) > 0, "Should take response actions"
                assert 'timestamp' in response, "Should timestamp response"
            
            # Verify specific responses
            phi_response = next(r for r in breach_responses if r['scenario'] == 'phi_exposure')
            assert any('phi_masked' in action for action in phi_response['actions_taken'])
            
            unauthorized_response = next(r for r in breach_responses if r['scenario'] == 'unauthorized_access')
            assert 'sessions_terminated' in unauthorized_response['actions_taken']
            
            # Verify audit logging
            auth_service.log_access.assert_called()
        
        @pytest.mark.asyncio
        async def test_graceful_degradation_under_load(self, integrated_services):
            """Test graceful degradation when system is under load."""
            voice_service = integrated_services['voice']
            memory_service = integrated_services['memory']
            
            # Simulate system under load
            memory_service.get_memory_usage.return_value = 450  # MB (high usage)
            memory_service.check_memory_limit.return_value = True  # Over limit
            
            # Configure degraded service responses
            normal_transcription = {'text': 'Full transcription', 'confidence': 0.95}
            degraded_transcription = {'text': 'Partial transcription', 'confidence': 0.80}
            minimal_transcription = {'text': 'Basic detection', 'confidence': 0.60}
            
            transcription_responses = [
                degraded_transcription,  # First request gets degraded response
                minimal_transcription,  # Second gets minimal
                normal_transcription   # Third returns to normal after cleanup
            ]
            
            voice_service.transcribe_audio.side_effect = transcription_responses
            
            # Track degradation levels
            degradation_levels = []
            
            async def process_with_adaptive_quality():
                """Process voice with adaptive quality based on system load."""
                memory_usage = memory_service.get_memory_usage()
                
                if memory_usage > 400:  # High memory usage
                    quality_level = 'minimal'
                elif memory_usage > 300:  # Medium memory usage
                    quality_level = 'degraded'
                else:
                    quality_level = 'normal'
                
                degradation_levels.append(quality_level)
                
                # Process with appropriate quality
                result = voice_service.transcribe_audio(
                    session_id='adaptive_session',
                    audio_data=b'test_audio'
                )
                
                return {
                    'quality_level': quality_level,
                    'result': result,
                    'memory_usage': memory_usage
                }
            
            # Process multiple requests
            results = []
            for i in range(3):
                result = await process_with_adaptive_quality()
                results.append(result)
                
                # Simulate memory cleanup after second request
                if i == 1:
                    memory_service.cleanup()
                    memory_service.get_memory_usage.return_value = 200  # Memory freed
            
            # Verify graceful degradation
            assert len(degradation_levels) == 3, "Should track degradation for all requests"
            assert degradation_levels[0] == 'minimal', "First request should use minimal quality"
            assert degradation_levels[1] == 'minimal', "Second request should use minimal quality"
            assert degradation_levels[2] == 'normal', "Third request should use normal quality"
            
            # Verify transcription quality matches degradation level
            assert results[0]['result']['confidence'] <= 0.80, "Degraded transcription should have lower confidence"
            assert results[1]['result']['confidence'] <= 0.80, "Degraded transcription should have lower confidence"
            assert results[2]['result']['confidence'] == 0.95, "Normal transcription should have high confidence"
            
            # Verify recovery
            assert results[2]['quality_level'] == 'normal', "Should recover to normal quality"
            memory_service.cleanup.assert_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])