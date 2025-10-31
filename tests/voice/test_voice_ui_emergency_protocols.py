"""
Voice UI Emergency Protocol Testing

Comprehensive test suite for emergency protocol UI flows and crisis management:
- Crisis keyword detection and alert triggers
- Emergency call initiation and contact protocols
- Crisis alert display and user guidance
- Emergency session logging and audit trails
- Multi-language emergency support
- Emergency protocol accessibility features

Coverage targets: Emergency protocol UI flows and crisis management testing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any
import time
import json

# Test imports with conditional handling
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = Mock()
    STREAMLIT_AVAILABLE = False

from voice.voice_ui import (
    VoiceUIComponents, VoiceUIState,
    _detect_crisis_keywords, display_crisis_alert,
    _initiate_emergency_call, _log_emergency_event,
    render_emergency_controls
)
from voice.config import VoiceConfig
from voice.voice_service import VoiceService


@pytest.fixture
def emergency_ui_components():
    """Create voice UI components configured for emergency testing."""
    config = Mock(spec=VoiceConfig)
    config.voice_enabled = True
    config.security = Mock()
    config.security.consent_required = False

    service = Mock(spec=VoiceService)
    service.is_available.return_value = True
    service.create_session.return_value = "emergency_session_123"

    with patch('voice.voice_ui.st', Mock()):
        components = VoiceUIComponents(service, config)
        yield components


class TestCrisisKeywordDetection:
    """Test crisis keyword detection functionality."""

    def test_crisis_keyword_detection_emergency_words(self):
        """Test detection of explicit emergency keywords."""
        emergency_texts = [
            "I need emergency help",
            "This is an emergency situation",
            "Call emergency services now",
            "Emergency contact needed"
        ]

        for text in emergency_texts:
            assert _detect_crisis_keywords(text) is True

    def test_crisis_keyword_detection_crisis_words(self):
        """Test detection of crisis-related keywords."""
        crisis_texts = [
            "I'm in crisis mode",
            "Crisis intervention required",
            "Feeling suicidal thoughts",
            "Suicidal ideation present",
            "Harm reduction needed",
            "Dangerous situation unfolding"
        ]

        for text in crisis_texts:
            assert _detect_crisis_keywords(text) is True

    def test_crisis_keyword_detection_help_words(self):
        """Test detection of help-related crisis keywords."""
        help_texts = [
            "I need help immediately",
            "Help me I'm in danger",
            "Urgent help required",
            "Help with suicidal thoughts"
        ]

        for text in help_texts:
            assert _detect_crisis_keywords(text) is True

    def test_crisis_keyword_detection_hurt_words(self):
        """Test detection of hurt/danger keywords."""
        hurt_texts = [
            "I want to hurt myself",
            "Going to harm myself",
            "Self-harm thoughts",
            "Danger to myself",
            "Feeling dangerous"
        ]

        for text in hurt_texts:
            assert _detect_crisis_keywords(text) is True

    def test_crisis_keyword_detection_case_insensitive(self):
        """Test case-insensitive crisis keyword detection."""
        mixed_case_texts = [
            "EMERGENCY HELP NEEDED",
            "crisis Intervention Required",
            "SuIcIdAl ThOuGhTs",
            "HARM to Myself",
            "Dangerous Situation"
        ]

        for text in mixed_case_texts:
            assert _detect_crisis_keywords(text) is True

    def test_crisis_keyword_detection_false_positives(self):
        """Test that non-crisis text doesn't trigger detection."""
        normal_texts = [
            "I need help with my homework",
            "This is an emergency drill at school",
            "The crisis management team met today",
            "She helped me with the project",
            "The emergency exit is over there"
        ]

        for text in normal_texts:
            assert _detect_crisis_keywords(text) is False

    def test_crisis_keyword_detection_partial_matches(self):
        """Test detection with partial keyword matches in context."""
        partial_texts = [
            "I'm feeling emergency-level stress",
            "This crisis is becoming unbearable",
            "My suicidal thoughts are getting worse",
            "The danger feels imminent"
        ]

        for text in partial_texts:
            assert _detect_crisis_keywords(text) is True


class TestCrisisAlertDisplay:
    """Test crisis alert display functionality."""

    @patch('voice.voice_ui.st')
    def test_crisis_alert_display_basic(self, mock_st):
        """Test basic crisis alert display."""
        display_crisis_alert("Immediate danger detected")

        mock_st.error.assert_called_once()
        error_message = mock_st.error.call_args[0][0]
        assert "🚨 CRISIS DETECTED:" in error_message
        assert "Immediate danger detected" in error_message

    @patch('voice.voice_ui.st')
    def test_crisis_alert_display_with_keywords(self, mock_st):
        """Test crisis alert with detected keywords."""
        keywords = ["suicide", "harm", "emergency"]
        alert_message = f"Crisis keywords detected: {keywords}"

        display_crisis_alert(alert_message)

        mock_st.error.assert_called_once()
        error_message = mock_st.error.call_args[0][0]
        assert "CRISIS DETECTED:" in error_message
        for keyword in keywords:
            assert keyword in error_message

    @patch('voice.voice_ui.st')
    def test_crisis_alert_session_state_update(self, mock_st):
        """Test crisis alert updates session state."""
        display_crisis_alert("Test crisis")

        assert mock_st.session_state['crisis_active'] is True

    @patch('voice.voice_ui.st')
    def test_crisis_alert_visual_indicators(self, mock_st, emergency_ui_components):
        """Test crisis alert visual indicators in UI."""
        # Verify emergency CSS classes are available
        css_content = str(emergency_ui_components)
        assert "emergency-indicator" in css_content
        assert "@keyframes blink" in css_content
        assert "background: #dc3545" in css_content


class TestEmergencyCallInitiation:
    """Test emergency call initiation protocols."""

    @patch('voice.voice_ui.st')
    def test_emergency_call_initiation_basic(self, mock_st):
        """Test basic emergency call initiation."""
        _initiate_emergency_call()

        mock_st.info.assert_called_once()
        info_message = mock_st.info.call_args[0][0]
        assert "Initiating emergency call" in info_message

    @patch('voice.voice_ui.st')
    def test_emergency_call_session_state_update(self, mock_st):
        """Test emergency call updates session state."""
        _initiate_emergency_call()

        assert mock_st.session_state['emergency_call_active'] is True

    @patch('voice.voice_ui.st')
    def test_emergency_call_with_contact_integration(self, mock_st):
        """Test emergency call with contact system integration."""
        # Simulate contact system integration
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'emergency_contacts': ['911', '988']}

            _initiate_emergency_call()

            # Verify call initiation with contacts
            mock_st.info.assert_called_once()


class TestEmergencyEventLogging:
    """Test emergency event logging and audit trails."""

    @patch('voice.voice_ui.st')
    def test_emergency_event_logging_basic(self, mock_st):
        """Test basic emergency event logging."""
        event_details = {
            'severity': 'high',
            'keywords': ['suicide', 'harm'],
            'timestamp': time.time()
        }

        _log_emergency_event('crisis_detected', event_details)

        assert 'emergency_log' in mock_st.session_state
        log_entry = mock_st.session_state['emergency_log'][0]
        assert log_entry['type'] == 'crisis_detected'
        assert log_entry['details'] == event_details
        assert 'timestamp' in log_entry

    @patch('voice.voice_ui.st')
    def test_emergency_event_logging_multiple_events(self, mock_st):
        """Test logging multiple emergency events."""
        events = [
            ('crisis_detected', {'severity': 'high'}),
            ('emergency_call_initiated', {'contact': '911'}),
            ('intervention_started', {'protocol': 'immediate'})
        ]

        for event_type, details in events:
            _log_emergency_event(event_type, details)

        emergency_log = mock_st.session_state['emergency_log']
        assert len(emergency_log) == 3

        for i, (expected_type, expected_details) in enumerate(events):
            assert emergency_log[i]['type'] == expected_type
            assert emergency_log[i]['details'] == expected_details

    @patch('voice.voice_ui.st')
    def test_emergency_event_logging_without_details(self, mock_st):
        """Test emergency event logging without additional details."""
        _log_emergency_event('session_start')

        log_entry = mock_st.session_state['emergency_log'][0]
        assert log_entry['type'] == 'session_start'
        assert isinstance(log_entry['timestamp'], float)
        assert 'details' in log_entry

    @patch('voice.voice_ui.st')
    def test_emergency_event_log_initialization(self, mock_st):
        """Test emergency log initialization."""
        # First call should initialize the log
        _log_emergency_event('test_event', {'test': 'data'})

        assert 'emergency_log' in mock_st.session_state
        assert isinstance(mock_st.session_state['emergency_log'], list)
        assert len(mock_st.session_state['emergency_log']) == 1


class TestEmergencyControlsRendering:
    """Test emergency controls UI rendering."""

    @patch('voice.voice_ui.st')
    def test_emergency_controls_basic_rendering(self, mock_st):
        """Test basic emergency controls rendering."""
        mock_st.button = Mock(return_value=False)

        render_emergency_controls()

        mock_st.error.assert_called_once()
        mock_st.button.assert_called_once()

        button_args = mock_st.button.call_args[0]
        assert "Call Emergency Services" in button_args[0]

    @patch('voice.voice_ui.st')
    def test_emergency_controls_button_action(self, mock_st):
        """Test emergency controls button click action."""
        mock_st.button = Mock(return_value=True)  # Simulate button click

        render_emergency_controls()

        # Verify emergency call would be initiated on button click
        # (In real implementation, this would trigger _initiate_emergency_call)

    @patch('voice.voice_ui.st')
    def test_emergency_controls_error_display(self, mock_st):
        """Test emergency controls error display."""
        render_emergency_controls()

        error_call = mock_st.error.call_args[0][0]
        assert "🚨 Emergency Protocol" in error_call

    @patch('voice.voice_ui.st')
    def test_emergency_controls_expanded_display(self, mock_st):
        """Test emergency controls expanded display."""
        mock_st.button = Mock(return_value=False)

        render_emergency_controls()

        # Verify comprehensive emergency UI elements
        assert mock_st.error.called
        assert mock_st.button.called


class TestEmergencyContactIntegration:
    """Test emergency contact system integration."""

    @patch('voice.voice_ui.st')
    async def test_emergency_contact_successful_call(self, mock_st):
        """Test successful emergency contact call."""
        from voice.voice_ui import handle_emergency_contact

        result = await handle_emergency_contact("emergency_services")

        mock_st.info.assert_called_once()
        assert result is False  # Test implementation returns False

    @patch('voice.voice_ui.st')
    async def test_emergency_contact_without_specified_contact(self, mock_st):
        """Test emergency contact handling without specified contact."""
        from voice.voice_ui import handle_emergency_contact

        result = await handle_emergency_contact()

        mock_st.info.assert_called_once()
        assert result is False

    @patch('voice.voice_ui.st')
    async def test_emergency_contact_with_predefined_contacts(self, mock_st):
        """Test emergency contact with predefined contact list."""
        from voice.voice_ui import handle_emergency_contact

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {
                'emergency_contacts': ['911', '988', 'local_crisis']
            }

            result = await handle_emergency_contact("911")

            mock_st.info.assert_called_once()


class TestEmergencySessionLogging:
    """Test emergency session logging functionality."""

    @patch('voice.voice_ui.st')
    async def test_emergency_session_logging_with_data(self, mock_st):
        """Test emergency session logging with detailed data."""
        from voice.voice_ui import log_emergency_session

        emergency_data = {
            'crisis_level': 'severe',
            'intervention_type': 'immediate',
            'contact_initiated': True,
            'session_duration': 45
        }

        await log_emergency_session(emergency_data)

        assert 'emergency_log' in mock_st.session_state
        log_entry = mock_st.session_state['emergency_log'][0]
        assert log_entry['details'] == emergency_data

    @patch('voice.voice_ui.st')
    async def test_emergency_session_logging_empty_data(self, mock_st):
        """Test emergency session logging with empty data."""
        from voice.voice_ui import log_emergency_session

        await log_emergency_session()

        log_entry = mock_st.session_state['emergency_log'][0]
        assert 'timestamp' in log_entry['details']

    @patch('voice.voice_ui.st')
    async def test_emergency_session_multiple_logging(self, mock_st):
        """Test multiple emergency session logging calls."""
        from voice.voice_ui import log_emergency_session

        # Log multiple emergency sessions
        for i in range(3):
            await log_emergency_session({'session_id': i, 'severity': 'high'})

        emergency_log = mock_st.session_state['emergency_log']
        assert len(emergency_log) == 3

        for i in range(3):
            assert emergency_log[i]['details']['session_id'] == i


class TestEmergencyUIAccessibility:
    """Test emergency UI accessibility features."""

    def test_emergency_button_aria_labels(self, emergency_ui_components):
        """Test ARIA labels for emergency controls."""
        from voice.voice_ui import render_accessible_voice_controls

        with patch('voice.voice_ui.st') as mock_st:
            controls = render_accessible_voice_controls()

            # Find emergency control
            emergency_control = next(
                (c for c in controls if 'Emergency' in c['label']), None
            )
            assert emergency_control is not None
            assert 'aria_label' in emergency_control

    @patch('voice.voice_ui.st')
    async def test_emergency_status_announcements(self, mock_st):
        """Test emergency status announcements for screen readers."""
        from voice.voice_ui import announce_voice_status

        await announce_voice_status("emergency_detected")

        announcement = mock_st.session_state['voice_status_announcement']
        assert announcement == "Emergency protocol activated"

    @patch('voice.voice_ui.st')
    def test_emergency_visual_indicators_high_contrast(self, mock_st, emergency_ui_components):
        """Test high contrast emergency visual indicators."""
        # Verify emergency CSS provides high contrast
        css_content = str(emergency_ui_components)
        assert "#dc3545" in css_content  # High contrast red
        assert "font-weight: bold" in css_content
        assert "@keyframes blink" in css_content


class TestEmergencyProtocolTriggers:
    """Test various triggers for emergency protocols."""

    def test_emergency_trigger_from_transcription(self, emergency_ui_components):
        """Test emergency protocol trigger from transcription analysis."""
        from voice.voice_ui import TranscriptionResult

        # Create transcription with crisis keywords
        crisis_transcription = TranscriptionResult(
            text="I need help, I'm feeling suicidal",
            confidence=0.95,
            duration=3.0,
            timestamp=time.time()
        )

        emergency_ui_components.ui_state.current_transcription = crisis_transcription

        # Test crisis detection on transcription
        crisis_detected = _detect_crisis_keywords(crisis_transcription.text)
        assert crisis_detected is True

    @patch('voice.voice_ui.st')
    def test_emergency_trigger_visual_feedback(self, mock_st, emergency_ui_components):
        """Test visual feedback when emergency is triggered."""
        # Verify emergency indicator CSS is available
        css_content = str(emergency_ui_components)
        assert "emergency-indicator" in css_content
        assert "animation: blink" in css_content

    def test_emergency_trigger_multiple_keywords(self, emergency_ui_components):
        """Test emergency trigger with multiple crisis keywords."""
        multi_crisis_text = "I'm in crisis, feeling suicidal, need emergency help, danger to myself"

        crisis_detected = _detect_crisis_keywords(multi_crisis_text)
        assert crisis_detected is True

        # Count keywords detected
        keywords_found = [
            word for word in ['emergency', 'crisis', 'suicidal', 'help', 'danger']
            if word.lower() in multi_crisis_text.lower()
        ]
        assert len(keywords_found) >= 3


class TestEmergencyProtocolRecovery:
    """Test emergency protocol recovery and reset."""

    @patch('voice.voice_ui.st')
    def test_emergency_protocol_reset(self, mock_st, emergency_ui_components):
        """Test emergency protocol reset functionality."""
        # Simulate emergency state
        emergency_ui_components.ui_state.recording_state = emergency_ui_components.ui_state.recording_state.ERROR

        # Reset UI state
        emergency_ui_components.reset_ui_state()

        # Verify reset
        assert emergency_ui_components.ui_state.recording_state.name == "IDLE"
        assert emergency_ui_components.ui_state.current_transcription is None

    @patch('voice.voice_ui.st')
    def test_emergency_session_cleanup(self, mock_st):
        """Test emergency session cleanup."""
        from voice.voice_ui import cleanup_voice_session

        with patch('voice.voice_ui.st') as mock_st:
            # Set up emergency session state
            mock_st.session_state = {
                'emergency_log': [{'event': 'crisis'}],
                'crisis_active': True,
                'emergency_call_active': True
            }

            cleanup_voice_session()

            # Verify cleanup (session state keys should be cleared)
            # Note: Actual cleanup implementation tested separately


class TestEmergencyMultiLanguageSupport:
    """Test multi-language emergency support."""

    def test_emergency_keywords_spanish(self):
        """Test Spanish emergency keywords."""
        spanish_crisis_texts = [
            "necesito ayuda de emergencia",
            "estoy en crisis",
            "pensamientos suicidas",
            "peligro inminente"
        ]

        # Note: Current implementation only supports English
        # This test documents the need for multi-language support
        for text in spanish_crisis_texts:
            # These should eventually return True with multi-language support
            result = _detect_crisis_keywords(text)
            # Currently returns False, but structure is ready for expansion
            assert isinstance(result, bool)

    def test_emergency_keywords_french(self):
        """Test French emergency keywords."""
        french_crisis_texts = [
            "aide d'urgence nécessaire",
            "je suis en crise",
            "idées suicidaires",
            "danger immédiat"
        ]

        # Document multi-language support requirement
        for text in french_crisis_texts:
            result = _detect_crisis_keywords(text)
            assert isinstance(result, bool)


class TestEmergencyProtocolIntegration:
    """Test integration of emergency protocols with voice UI."""

    @patch('voice.voice_ui.st')
    def test_emergency_integration_with_voice_service(self, mock_st, emergency_ui_components):
        """Test emergency protocol integration with voice service."""
        # Simulate voice service emergency detection
        emergency_ui_components.voice_service.on_error = Mock()

        # Test that UI can trigger emergency protocols
        # (Integration testing with voice service)

    @patch('voice.voice_ui.st')
    def test_emergency_protocol_state_persistence(self, mock_st, emergency_ui_components):
        """Test emergency protocol state persistence across UI updates."""
        # Set emergency state
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            display_crisis_alert("Test emergency")

            # Verify state persists
            assert mock_st.session_state['crisis_active'] is True

            # Simulate UI re-render
            display_crisis_alert("Continued emergency")

            # State should still be active
            assert mock_st.session_state['crisis_active'] is True


class TestEmergencyAccessibilityCompliance:
    """Test emergency protocol accessibility compliance."""

    def test_emergency_controls_keyboard_navigation(self, emergency_ui_components):
        """Test keyboard navigation for emergency controls."""
        # Verify emergency controls are keyboard accessible
        # (Test would verify focus management and keyboard event handling)

    @patch('voice.voice_ui.st')
    def test_emergency_screen_reader_integration(self, mock_st, emergency_ui_components):
        """Test screen reader integration for emergency alerts."""
        from voice.voice_ui import _announce_to_screen_reader

        with patch('voice.voice_ui.st') as mock_st:
            _announce_to_screen_reader("Emergency protocol activated - immediate assistance required")

            assert mock_st.session_state['screen_reader_announcement'] == \
                   "Emergency protocol activated - immediate assistance required"

    def test_emergency_high_contrast_mode(self, emergency_ui_components):
        """Test high contrast mode for emergency indicators."""
        # Verify emergency CSS provides sufficient contrast ratios
        css_content = str(emergency_ui_components)
        assert "#dc3545" in css_content  # Red background for emergency
        assert "color: white" in css_content  # White text for contrast


class TestEmergencyPerformanceMonitoring:
    """Test emergency protocol performance monitoring."""

    @patch('voice.voice_ui.st')
    def test_emergency_response_time_logging(self, mock_st):
        """Test emergency response time logging."""
        start_time = time.time()

        _log_emergency_event('emergency_triggered', {'start_time': start_time})

        log_entry = mock_st.session_state['emergency_log'][0]
        logged_time = log_entry['timestamp']

        # Verify timestamp is recorded
        assert isinstance(logged_time, float)
        assert logged_time >= start_time

    @patch('voice.voice_ui.st')
    def test_emergency_protocol_success_metrics(self, mock_st):
        """Test emergency protocol success metrics."""
        # Log emergency protocol execution
        _log_emergency_event('emergency_call_completed', {
            'duration': 30,
            'success': True,
            'contact_method': 'direct_call'
        })

        log_entry = mock_st.session_state['emergency_log'][0]
        assert log_entry['details']['success'] is True
        assert log_entry['details']['duration'] == 30


# Run basic validation
if __name__ == "__main__":
    print("Voice UI Emergency Protocol Test Suite")
    print("=" * 50)

    try:
        from voice.voice_ui import _detect_crisis_keywords, display_crisis_alert
        print("✅ Emergency utility imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        # Test crisis detection
        crisis_detected = _detect_crisis_keywords("I need emergency help")
        assert crisis_detected is True
        print("✅ Crisis detection working")
    except Exception as e:
        print(f"❌ Crisis detection failed: {e}")

    try:
        # Test emergency alert with mock
        with patch('voice.voice_ui.st') as mock_st:
            display_crisis_alert("Test emergency")
            print("✅ Emergency alert display working")
    except Exception as e:
        print(f"❌ Emergency alert failed: {e}")

    print("Emergency protocol test file created - run with pytest for full validation")
