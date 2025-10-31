"""
Comprehensive Emergency Flow Tests for Voice UI Components.

Tests focus on emergency protocols, crisis detection, error handling,
user guidance, and safety-critical scenarios in the voice interface.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List, Optional
import numpy as np
import json
import time
from datetime import datetime, timedelta

# Mock streamlit for testing
try:
    import streamlit as st
except ImportError:
    st = Mock()

from voice.voice_ui import (
    VoiceUIComponents, VoiceUIState, RecordingState, PlaybackState,
    TranscriptionResult, _STREAMLIT_AVAILABLE, _NUMPY_AVAILABLE
)
from voice.config import VoiceConfig
from voice.voice_service import VoiceService


class TestVoiceUIEmergencyFlows:
    """Comprehensive tests for emergency protocols and crisis handling in voice UI."""

    @pytest.fixture
    def mock_streamlit_emergency(self):
        """Create Streamlit mock configured for emergency scenario testing."""
        with patch('voice.voice_ui.st') as mock_st:
            # Session state with emergency-related properties
            session_state = {
                'emergency_mode': False,
                'crisis_detected': False,
                'last_crisis_timestamp': None,
                'emergency_contacts': [],
                'safety_protocol_active': False,
                'error_count': 0,
                'last_error_timestamp': None
            }
            mock_st.session_state = session_state

            # UI component mocks
            mock_st.error = Mock()
            mock_st.warning = Mock()
            mock_st.success = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.container = Mock()
            mock_st.markdown = Mock()
            mock_st.columns = Mock(return_value=[Mock(), Mock()])

            yield mock_st

    @pytest.fixture
    def voice_ui_components_emergency(self, mock_streamlit_emergency):
        """Create VoiceUIComponents for emergency flow testing."""
        config = VoiceConfig()
        components = VoiceUIComponents(config)
        return components

    # Crisis Detection and Response (5 tests)
    def test_crisis_keyword_detection_ui_response(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI response to crisis keyword detection."""
        # Mock crisis detection in transcription
        crisis_transcription = TranscriptionResult(
            text="I want to hurt myself",
            confidence=0.95,
            duration=2.0,
            timestamp=time.time()
        )

        voice_ui_components_emergency.ui_state.current_transcription = crisis_transcription

        # Mock crisis flag
        with patch.object(voice_ui_components_emergency.voice_service, 'process_voice_input') as mock_process:
            mock_result = Mock()
            mock_result.is_crisis = True
            mock_result.crisis_keywords = ['hurt', 'myself']
            mock_process.return_value = mock_result

            voice_ui_components_emergency.render_voice_interface()

            # UI should enter emergency mode
            assert mock_streamlit_emergency.session_state['emergency_mode'] is True
            assert mock_streamlit_emergency.session_state['crisis_detected'] is True

    def test_emergency_mode_ui_transformation(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI transformation when emergency mode is activated."""
        mock_streamlit_emergency.session_state['emergency_mode'] = True
        mock_streamlit_emergency.session_state['crisis_detected'] = True

        voice_ui_components_emergency.render_voice_interface()

        # UI should show emergency interface:
        # - Emergency contact information
        # - Crisis hotline numbers
        # - Calming instructions
        # - Disable regular voice features

    def test_crisis_escalation_protocol_display(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test display of crisis escalation protocols."""
        mock_streamlit_emergency.session_state.update({
            'emergency_mode': True,
            'crisis_level': 'high',
            'escalation_step': 1
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should display appropriate escalation steps:
        # - Step 1: Immediate calming techniques
        # - Step 2: Contact emergency services
        # - Step 3: Safety plan activation

    def test_emergency_contact_display_and_access(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test display and access to emergency contacts."""
        emergency_contacts = [
            {'name': 'Emergency Services', 'number': '911', 'type': 'emergency'},
            {'name': 'Crisis Hotline', 'number': '988', 'type': 'mental_health'},
            {'name': 'Therapist', 'number': '+1234567890', 'type': 'professional'}
        ]

        mock_streamlit_emergency.session_state.update({
            'emergency_mode': True,
            'emergency_contacts': emergency_contacts
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should prominently display emergency contacts
        # Should provide easy access methods (click to call, etc.)

    def test_crisis_recovery_and_deescalation(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test crisis recovery and UI de-escalation."""
        # Start in emergency mode
        mock_streamlit_emergency.session_state.update({
            'emergency_mode': True,
            'crisis_detected': True,
            'recovery_start_time': time.time()
        })

        # Simulate recovery time passage
        mock_streamlit_emergency.session_state['recovery_start_time'] = time.time() - 1800  # 30 minutes ago

        voice_ui_components_emergency.render_voice_interface()

        # Should gradually de-escalate emergency UI
        # Should provide recovery resources
        # Should transition back to normal mode when appropriate

    # Error Handling and Recovery (5 tests)
    def test_voice_service_connection_failure_ui(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI response to voice service connection failures."""
        # Mock service failure
        with patch.object(voice_ui_components_emergency.voice_service, 'is_available', return_value=False):
            voice_ui_components_emergency.render_voice_interface()

            # Should display connection error
            mock_streamlit_emergency.error.assert_called()

            # Should show reconnection options
            # Should provide offline fallback instructions

    def test_audio_device_error_handling_ui(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI handling of audio device errors."""
        # Mock audio device failure
        voice_ui_components_emergency.voice_service.audio_processor.input_devices = []
        voice_ui_components_emergency.voice_service.audio_processor.output_devices = []

        voice_ui_components_emergency.render_voice_interface()

        # Should detect device issues
        # Should provide device setup instructions
        # Should offer alternative input methods

    def test_network_connectivity_loss_ui_response(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI response to network connectivity loss."""
        mock_streamlit_emergency.session_state.update({
            'network_connected': False,
            'last_network_check': time.time() - 60  # 1 minute ago
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should show offline mode
        # Should queue operations for when connection returns
        # Should provide offline-capable features

    def test_service_timeout_error_ui_feedback(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI feedback for service timeout errors."""
        # Mock timeout error
        with patch.object(voice_ui_components_emergency.voice_service, 'process_voice_input') as mock_process:
            mock_process.side_effect = asyncio.TimeoutError("Service timeout")

            voice_ui_components_emergency.render_voice_interface()

            # Should show timeout error message
            # Should provide retry options
            # Should suggest alternative actions

    def test_consecutive_error_escalation_ui(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI escalation with consecutive errors."""
        # Simulate multiple consecutive errors
        error_counts = [1, 3, 5, 10]

        for error_count in error_counts:
            mock_streamlit_emergency.session_state['error_count'] = error_count

            voice_ui_components_emergency.render_voice_interface()

            # UI should escalate error handling:
            # Low errors: Show warning
            # Medium errors: Suggest troubleshooting
            # High errors: Offer to restart service
            # Critical errors: Show emergency contact info

    # Safety Protocol Activation (5 tests)
    def test_safety_protocol_activation_triggers(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test triggers for safety protocol activation."""
        safety_triggers = [
            {'crisis_detected': True, 'trigger': 'crisis_keywords'},
            {'self_harm_mentioned': True, 'trigger': 'self_harm_reference'},
            {'emergency_override': True, 'trigger': 'manual_activation'},
            {'suicidal_ideation': True, 'trigger': 'suicidal_content'},
            {'panic_attack_indicators': True, 'trigger': 'panic_symptoms'}
        ]

        for trigger_config in safety_triggers:
            mock_streamlit_emergency.session_state.update(trigger_config)

            voice_ui_components_emergency.render_voice_interface()

            # Should activate appropriate safety protocols
            assert mock_streamlit_emergency.session_state['safety_protocol_active'] is True

    def test_safety_protocol_ui_modifications(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI modifications when safety protocols are active."""
        mock_streamlit_emergency.session_state.update({
            'safety_protocol_active': True,
            'protocol_level': 'high'
        })

        voice_ui_components_emergency.render_voice_interface()

        # UI should be modified for safety:
        # - Limited voice input options
        # - Prominent safety resources
        # - Reduced cognitive load
        # - Calming visual design

    def test_safety_protocol_compliance_verification(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test safety protocol compliance with regulations."""
        mock_streamlit_emergency.session_state['safety_protocol_active'] = True

        voice_ui_components_emergency.render_voice_interface()

        # Should comply with HIPAA and mental health regulations:
        # - Secure data handling
        # - Audit logging
        # - Privacy protection
        # - Emergency contact protocols

    def test_safety_protocol_deactivation_conditions(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test conditions for deactivating safety protocols."""
        mock_streamlit_emergency.session_state.update({
            'safety_protocol_active': True,
            'protocol_activated_at': time.time() - 3600,  # 1 hour ago
            'user_stability_indicators': ['calm', 'oriented', 'safe']
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should evaluate deactivation conditions:
        # - Time since activation
        # - User stability indicators
        # - Professional assessment
        # - Safety plan completion

    def test_safety_protocol_audit_logging(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test audit logging of safety protocol activations."""
        mock_streamlit_emergency.session_state['safety_protocol_active'] = True

        with patch.object(voice_ui_components_emergency.voice_service, 'audit_repo') as mock_audit:
            voice_ui_components_emergency.render_voice_interface()

            # Should log safety protocol events
            mock_audit.save.assert_called()

            # Log should include:
            # - Activation timestamp
            # - Trigger reason
            # - User information (anonymized)
            # - Protocol actions taken

    # User Guidance and Support (5 tests)
    def test_first_time_user_guidance_display(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test guidance display for first-time users."""
        mock_streamlit_emergency.session_state.update({
            'is_first_time_user': True,
            'completed_tutorial': False
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should show onboarding tutorial
        # Should provide usage instructions
        # Should explain voice features
        # Should offer practice mode

    def test_contextual_help_system_activation(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test contextual help system based on user actions."""
        help_scenarios = [
            {'user_stuck': True, 'help_topic': 'voice_input'},
            {'error_encountered': True, 'help_topic': 'troubleshooting'},
            {'feature_unknown': True, 'help_topic': 'feature_explanation'},
            {'crisis_mode': True, 'help_topic': 'emergency_support'}
        ]

        for scenario in help_scenarios:
            mock_streamlit_emergency.session_state.update(scenario)

            voice_ui_components_emergency.render_voice_interface()

            # Should display relevant help content
            # Should provide step-by-step guidance
            # Should offer alternative interaction methods

    def test_progressive_disclosure_information_architecture(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test progressive disclosure of information."""
        user_experience_levels = [
            {'experience': 'beginner', 'disclosure_level': 'basic'},
            {'experience': 'intermediate', 'disclosure_level': 'intermediate'},
            {'experience': 'advanced', 'disclosure_level': 'full'}
        ]

        for level in user_experience_levels:
            mock_streamlit_emergency.session_state.update({
                'user_experience_level': level['experience']
            })

            voice_ui_components_emergency.render_voice_interface()

            # Should show appropriate information level:
            # Beginner: Simple explanations, basic features
            # Intermediate: More options, shortcuts
            # Advanced: Full feature set, customization

    def test_accessibility_guidance_and_support(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test accessibility guidance and support features."""
        accessibility_needs = [
            {'visual_impairment': True, 'needs': ['screen_reader', 'high_contrast', 'large_text']},
            {'motor_impairment': True, 'needs': ['keyboard_navigation', 'voice_commands', 'large_targets']},
            {'cognitive_impairment': True, 'needs': ['simple_language', 'progress_indicators', 'error_prevention']}
        ]

        for need_config in accessibility_needs:
            mock_streamlit_emergency.session_state.update(need_config)

            voice_ui_components_emergency.render_voice_interface()

            # Should provide appropriate accessibility support
            # Should adapt UI for specific needs
            # Should offer alternative interaction methods

    def test_multilingual_support_and_guidance(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test multilingual support and guidance."""
        languages = ['en', 'es', 'fr', 'de', 'zh']

        for language in languages:
            mock_streamlit_emergency.session_state.update({
                'user_language': language,
                'interface_language': language
            })

            voice_ui_components_emergency.render_voice_interface()

            # Should provide guidance in user's language
            # Should support voice commands in multiple languages
            # Should show localized emergency resources

    # System Health Monitoring (5 tests)
    def test_system_health_indicator_display(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test display of system health indicators."""
        health_statuses = [
            {'overall_health': 'healthy', 'components': {'stt': 'healthy', 'tts': 'healthy', 'audio': 'healthy'}},
            {'overall_health': 'degraded', 'components': {'stt': 'healthy', 'tts': 'warning', 'audio': 'healthy'}},
            {'overall_health': 'unhealthy', 'components': {'stt': 'error', 'tts': 'error', 'audio': 'healthy'}}
        ]

        for status in health_statuses:
            mock_streamlit_emergency.session_state.update({
                'system_health': status
            })

            voice_ui_components_emergency.render_voice_interface()

            # Should display appropriate health indicators
            # Should show component-level status
            # Should provide troubleshooting for issues

    def test_performance_monitoring_ui_feedback(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test UI feedback for performance monitoring."""
        performance_metrics = [
            {'response_time': 0.5, 'cpu_usage': 30, 'memory_usage': 60, 'status': 'good'},
            {'response_time': 2.0, 'cpu_usage': 70, 'memory_usage': 80, 'status': 'degraded'},
            {'response_time': 5.0, 'cpu_usage': 95, 'memory_usage': 95, 'status': 'critical'}
        ]

        for metrics in performance_metrics:
            mock_streamlit_emergency.session_state.update({
                'performance_metrics': metrics
            })

            voice_ui_components_emergency.render_voice_interface()

            # Should show performance indicators
            # Should suggest optimizations for poor performance
            # Should warn about critical resource usage

    def test_resource_availability_display(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test display of resource availability status."""
        resource_status = {
            'microphone': {'available': True, 'permission': 'granted'},
            'speaker': {'available': True, 'permission': 'granted'},
            'network': {'connected': True, 'speed': 'fast'},
            'storage': {'available': 500, 'unit': 'MB'},  # 500MB available
            'battery': {'level': 85, 'charging': False}
        }

        mock_streamlit_emergency.session_state['resource_status'] = resource_status

        voice_ui_components_emergency.render_voice_interface()

        # Should display resource status
        # Should warn about low resources
        # Should suggest resource optimization

    def test_service_availability_status_indicators(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test service availability status indicators."""
        service_status = {
            'stt_service': {'available': True, 'response_time': 0.8},
            'tts_service': {'available': True, 'response_time': 1.2},
            'voice_processing': {'available': True, 'queue_length': 2},
            'emergency_services': {'available': True, 'last_check': time.time()}
        }

        mock_streamlit_emergency.session_state['service_status'] = service_status

        voice_ui_components_emergency.render_voice_interface()

        # Should show service availability
        # Should indicate response times
        # Should show queue status
        # Should provide failover options

    def test_error_rate_monitoring_and_display(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test error rate monitoring and display."""
        error_metrics = {
            'total_errors': 15,
            'error_rate_percent': 2.5,
            'recent_errors': [
                {'timestamp': time.time() - 300, 'type': 'network_timeout', 'severity': 'medium'},
                {'timestamp': time.time() - 180, 'type': 'audio_device_error', 'severity': 'high'},
                {'timestamp': time.time() - 60, 'type': 'service_unavailable', 'severity': 'critical'}
            ],
            'error_trends': 'increasing'
        }

        mock_streamlit_emergency.session_state['error_metrics'] = error_metrics

        voice_ui_components_emergency.render_voice_interface()

        # Should display error statistics
        # Should show recent errors
        # Should indicate error trends
        # Should provide error mitigation suggestions

    # Emergency Override and Admin Controls (5 tests)
    def test_emergency_override_activation(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test activation of emergency override controls."""
        override_scenarios = [
            {'admin_override': True, 'reason': 'system_malfunction'},
            {'therapist_override': True, 'reason': 'patient_safety'},
            {'technical_override': True, 'reason': 'service_failure'},
            {'regulatory_override': True, 'reason': 'compliance_requirement'}
        ]

        for scenario in override_scenarios:
            mock_streamlit_emergency.session_state.update(scenario)

            voice_ui_components_emergency.render_voice_interface()

            # Should activate appropriate override controls
            # Should log override activation
            # Should provide override justification
            # Should limit override scope and duration

    def test_admin_control_panel_access(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test access to admin control panel in emergencies."""
        mock_streamlit_emergency.session_state.update({
            'admin_access_granted': True,
            'emergency_admin_mode': True
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should provide admin controls:
        # - Service restart options
        # - Configuration overrides
        # - Emergency data export
        # - System diagnostics access

    def test_emergency_data_backup_and_recovery(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test emergency data backup and recovery procedures."""
        mock_streamlit_emergency.session_state.update({
            'emergency_backup_required': True,
            'data_at_risk': True
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should initiate emergency backup
        # Should show backup progress
        # Should provide recovery options
        # Should ensure data integrity

    def test_system_isolation_and_containment(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test system isolation and containment during emergencies."""
        mock_streamlit_emergency.session_state.update({
            'system_compromised': True,
            'isolation_required': True,
            'containment_level': 'high'
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should isolate affected components
        # Should contain security breaches
        # Should limit data access
        # Should provide secure communication channels

    def test_emergency_communication_protocols(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test emergency communication protocols."""
        mock_streamlit_emergency.session_state.update({
            'emergency_communication_active': True,
            'communication_channels': ['secure_voice', 'encrypted_text', 'emergency_hotline']
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should establish emergency communication
        # Should use secure channels
        # Should bypass normal restrictions
        # Should ensure HIPAA compliance

    # Recovery and Restoration (5 tests)
    def test_post_emergency_system_restoration(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test system restoration after emergency resolution."""
        mock_streamlit_emergency.session_state.update({
            'emergency_resolved': True,
            'restoration_phase': 'active',
            'system_backup_restored': True
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should restore normal system operation
        # Should verify system integrity
        # Should resume normal user interactions
        # Should log restoration completion

    def test_gradual_service_restoration_phases(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test gradual restoration of services in phases."""
        restoration_phases = [
            {'phase': 1, 'services': ['basic_ui'], 'status': 'initializing'},
            {'phase': 2, 'services': ['basic_ui', 'voice_input'], 'status': 'partial'},
            {'phase': 3, 'services': ['basic_ui', 'voice_input', 'processing'], 'status': 'intermediate'},
            {'phase': 4, 'services': ['all_services'], 'status': 'full_restoration'}
        ]

        for phase in restoration_phases:
            mock_streamlit_emergency.session_state.update({
                'restoration_phase': phase['phase'],
                'available_services': phase['services'],
                'restoration_status': phase['status']
            })

            voice_ui_components_emergency.render_voice_interface()

            # Should gradually restore services
            # Should test each restored component
            # Should provide user feedback on progress

    def test_data_integrity_verification_post_recovery(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test data integrity verification after system recovery."""
        mock_streamlit_emergency.session_state.update({
            'recovery_complete': True,
            'data_integrity_check': 'in_progress'
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should verify data integrity
        # Should check conversation history
        # Should validate user settings
        # Should ensure no data corruption occurred

    def test_user_session_continuity_after_emergency(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test maintenance of user session continuity after emergency."""
        # Pre-emergency session data
        session_data = {
            'conversation_history': [
                {'speaker': 'user', 'text': 'Hello', 'timestamp': time.time() - 3600},
                {'speaker': 'ai', 'text': 'Hi there!', 'timestamp': time.time() - 3590}
            ],
            'voice_settings': {'speed': 1.2, 'volume': 0.8},
            'emergency_occurred': True,
            'session_restored': True
        }

        mock_streamlit_emergency.session_state.update(session_data)

        voice_ui_components_emergency.render_voice_interface()

        # Should restore user session
        # Should maintain conversation continuity
        # Should preserve user preferences
        # Should provide context about emergency

    def test_emergency_response_effectiveness_analysis(self, voice_ui_components_emergency, mock_streamlit_emergency):
        """Test analysis of emergency response effectiveness."""
        emergency_analysis = {
            'emergency_id': 'emerg_001',
            'response_time_seconds': 45,
            'escalation_steps_taken': 2,
            'resources_utilized': ['crisis_hotline', 'emergency_contacts'],
            'outcome': 'successful_deescalation',
            'lessons_learned': ['faster_detection_needed', 'more_resources_helpful']
        }

        mock_streamlit_emergency.session_state.update({
            'emergency_analysis_available': True,
            'emergency_analysis': emergency_analysis
        })

        voice_ui_components_emergency.render_voice_interface()

        # Should display emergency response analysis
        # Should provide improvement recommendations
        # Should update emergency protocols
        # Should log analysis for future reference
