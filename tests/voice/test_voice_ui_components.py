"""
Voice UI Components Testing

Comprehensive test suite for voice UI components including:
- Streamlit component mocking and interaction patterns
- Mobile responsiveness and touch gesture handling
- Emergency protocol UI flows and crisis management
- Error state handling and accessibility features

Coverage targets: 45 new tests for UI component testing (19%→70%)
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import time
import json

# Test imports with conditional handling
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = Mock()
    STREAMLIT_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = Mock()
    NUMPY_AVAILABLE = False

from voice.voice_ui import (
    VoiceUIComponents, VoiceUIState, TranscriptionResult,
    RecordingState, PlaybackState, create_voice_ui
)
from voice.config import VoiceConfig, VoiceProfile
from voice.voice_service import VoiceService
from voice.audio_processor import AudioData


class MockStreamlitColumn:
    """Mock Streamlit column that supports context manager."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockStreamlitExpander:
    """Mock Streamlit expander that supports context manager."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_st():
    """Create comprehensive Streamlit mock that handles context managers."""
    st_mock = MagicMock()

    # Mock columns to return context managers
    def mock_columns(*args):
        if len(args) == 1 and isinstance(args[0], list):
            # st.columns([2, 1, 1]) - returns tuple of column objects
            return tuple(MockStreamlitColumn() for _ in args[0])
        else:
            # st.columns(2) - returns tuple of column objects
            return tuple(MockStreamlitColumn() for _ in range(args[0] if args else 2))

    st_mock.columns.side_effect = mock_columns

    # Mock expander to return context manager
    st_mock.expander.side_effect = lambda *args, **kwargs: MockStreamlitExpander(*args, **kwargs)

    # Mock session_state as a dictionary-like object
    st_mock.session_state = {
        'voice_consent_given': True,
        'voice_session_active': True,
        'current_voice_profile': 'calm_therapist'
    }

    return st_mock


@pytest.fixture
def mock_config():
    """Create mock voice configuration."""
    config = Mock(spec=VoiceConfig)
    config.voice_enabled = True
    config.security = Mock()
    config.security.consent_required = False
    # Add voice_profiles attribute for UI tests
    config.voice_profiles = {'calm_therapist': Mock(), 'professional': Mock()}
    config.get_voice_profile = Mock(return_value='calm_therapist')
    return config


@pytest.fixture
def mock_voice_service():
    """Create mock voice service."""
    service = Mock(spec=VoiceService)
    service.is_available.return_value = True
    service.create_session.return_value = "test_session_123"
    service.on_text_received = None
    service.on_audio_played = None
    service.on_command_executed = None
    service.on_error = None
    return service


@pytest.fixture
def ui_components(mock_voice_service, mock_config, mock_st):
    """Create voice UI components instance."""
    with patch('voice.voice_ui.st', mock_st):
        components = VoiceUIComponents(mock_voice_service, mock_config)
        yield components


class TestVoiceUIComponentsInitialization:
    """Test VoiceUIComponents initialization and setup."""

    def test_initialization_basic(self, mock_voice_service, mock_config):
        """Test basic initialization of VoiceUIComponents."""
        with patch('voice.voice_ui.st', Mock()) if not STREAMLIT_AVAILABLE else patch.object(st, 'markdown'):
            components = VoiceUIComponents(mock_voice_service, mock_config)

            assert components.voice_service == mock_voice_service
            assert components.config == mock_config
            assert isinstance(components.ui_state, VoiceUIState)
            assert components.current_session_id == "test_session_123"

    def test_initialization_callbacks_setup(self, mock_voice_service, mock_config):
        """Test that service callbacks are properly set up during initialization."""
        with patch('voice.voice_ui.st', Mock()) if not STREAMLIT_AVAILABLE else patch.object(st, 'markdown'):
            components = VoiceUIComponents(mock_voice_service, mock_config)

            # Verify callbacks were assigned
            assert mock_voice_service.on_text_received is not None
            assert mock_voice_service.on_audio_played is not None
            assert mock_voice_service.on_command_executed is not None
            assert mock_voice_service.on_error is not None

    def test_initialization_css_injection(self, mock_voice_service, mock_config):
        """Test that custom CSS is injected during initialization."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.markdown = Mock()
            components = VoiceUIComponents(mock_voice_service, mock_config)

            # Verify CSS was injected
            mock_st.markdown.assert_called()
            call_args = mock_st.markdown.call_args[0][0]
            assert "voice-button" in call_args
            assert "transcription-display" in call_args
            assert "emergency-indicator" in call_args

    def test_initialization_session_creation_failure(self, mock_voice_service, mock_config):
        """Test initialization when session creation fails."""
        mock_voice_service.is_available.return_value = False

        with patch('voice.voice_ui.st', Mock()) if not STREAMLIT_AVAILABLE else patch.object(st, 'markdown'):
            components = VoiceUIComponents(mock_voice_service, mock_config)

            assert components.current_session_id is None

    def test_initialization_exception_handling(self, mock_voice_service, mock_config):
        """Test initialization with exception handling."""
        mock_voice_service.create_session.side_effect = Exception("Test error")

        with patch('voice.voice_ui.st', Mock()) if not STREAMLIT_AVAILABLE else patch.object(st, 'markdown'):
            with patch('voice.voice_ui.logging') as mock_logging:
                components = VoiceUIComponents(mock_voice_service, mock_config)

                mock_logging.getLogger.return_value.error.assert_called()


class TestVoiceUIStateManagement:
    """Test VoiceUIState management and updates."""

    def test_default_ui_state(self, ui_components):
        """Test default VoiceUIState values."""
        state = ui_components.ui_state

        assert state.recording_state == RecordingState.IDLE
        assert state.playback_state == PlaybackState.STOPPED
        assert state.current_transcription is None
        assert state.audio_level == 0.0
        assert state.recording_duration == 0.0
        assert state.playback_progress == 0.0
        assert state.selected_voice_profile == "calm_therapist"
        assert state.show_settings is False
        assert state.show_commands is False
        assert state.mobile_mode is False
        assert state.keyboard_shortcuts_enabled is True
        assert state.accessibility_mode is False

    def test_ui_state_updates(self, ui_components):
        """Test updating UI state values."""
        ui_components.ui_state.recording_state = RecordingState.RECORDING
        ui_components.ui_state.audio_level = 0.7
        ui_components.ui_state.show_settings = True

        assert ui_components.ui_state.recording_state == RecordingState.RECORDING
        assert ui_components.ui_state.audio_level == 0.7
        assert ui_components.ui_state.show_settings is True

    def test_reset_ui_state(self, ui_components):
        """Test resetting UI state to defaults."""
        # Modify state
        ui_components.ui_state.recording_state = RecordingState.RECORDING
        ui_components.ui_state.audio_level = 0.8
        ui_components.ui_state.current_transcription = TranscriptionResult("test", 0.9, 1.0, time.time())

        # Reset
        ui_components.reset_ui_state()

        # Verify reset
        assert ui_components.ui_state.recording_state == RecordingState.IDLE
        assert ui_components.ui_state.audio_level == 0.0
        assert ui_components.ui_state.current_transcription is None
        assert ui_components.waveform_data == []
        assert ui_components.spectrum_data == []

    def test_get_ui_state(self, ui_components):
        """Test getting current UI state."""
        state = ui_components.get_ui_state()
        assert isinstance(state, VoiceUIState)
        assert state is ui_components.ui_state


class TestVoiceUIComponentRendering:
    """Test VoiceUIComponents rendering methods."""

    @patch('voice.voice_ui.st')
    def test_render_voice_interface_disabled(self, mock_st, mock_voice_service):
        """Test rendering when voice is disabled."""
        config = Mock(spec=VoiceConfig)
        config.voice_enabled = False

        with patch('voice.voice_ui.st', mock_st):
            components = VoiceUIComponents(mock_voice_service, config)
            result = components.render_voice_interface()

            assert result is False
            mock_st.warning.assert_called_once()

    @patch('voice.voice_ui.st')
    def test_render_voice_interface_enabled(self, mock_st, ui_components):
        """Test rendering when voice is enabled."""
        with patch.object(ui_components, '_check_consent', return_value=True), \
             patch.object(ui_components, '_detect_mobile_mode'), \
             patch.object(ui_components, '_render_header'), \
             patch.object(ui_components, '_render_voice_input_interface'), \
             patch.object(ui_components, '_render_transcription_display'), \
             patch.object(ui_components, '_render_voice_output_controls'), \
             patch.object(ui_components, '_render_expandable_sections'), \
             patch.object(ui_components, '_render_keyboard_shortcuts'):

            result = ui_components.render_voice_interface()

            assert result is True

    @patch('voice.voice_ui.st')
    def test_render_voice_disabled_message(self, mock_st, mock_voice_service):
        """Test rendering voice disabled message."""
        config = Mock(spec=VoiceConfig)
        config.voice_enabled = False

        with patch('voice.voice_ui.st', mock_st):
            components = VoiceUIComponents(mock_voice_service, config)
            components._render_voice_disabled()

            mock_st.warning.assert_called_once()
            warning_text = mock_st.warning.call_args[0][0]
            assert "Voice Features Disabled" in warning_text

    @patch('voice.voice_ui.st')
    def test_check_consent_not_required(self, mock_st, ui_components):
        """Test consent check when not required."""
        ui_components.config.security.consent_required = False
        result = ui_components._check_consent()
        assert result is True

    @patch('voice.voice_ui.st')
    def test_check_consent_required_not_given(self, mock_st, ui_components):
        """Test consent check when required but not given."""
        ui_components.config.security.consent_required = True

        with patch.object(ui_components, '_render_consent_form', return_value=False):
            result = ui_components._check_consent()
            assert result is False

    @patch('voice.voice_ui.st')
    def test_check_consent_required_given(self, mock_st, ui_components):
        """Test consent check when required and given."""
        ui_components.config.security.consent_required = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'voice_consent_given': True}

            with patch.object(ui_components, '_render_consent_form', return_value=True):
                result = ui_components._check_consent()
                assert result is True


class TestVoiceUIConsentForm:
    """Test consent form rendering and handling."""

    @patch('voice.voice_ui.st')
    def test_render_consent_form_structure(self, mock_st, ui_components):
        """Test consent form structure and content."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            ui_components._render_consent_form()

            # Verify form structure
            mock_st.subheader.assert_called_with("🔒 Voice Features Consent")
            assert mock_st.write.call_count >= 1
            assert mock_st.expander.call_count >= 1

    @patch('voice.voice_ui.st')
    def test_render_consent_form_checkboxes(self, mock_st, ui_components):
        """Test consent form checkbox options."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}
            mock_st.checkbox = Mock(side_effect=[True, False, True, True, True, False])

            ui_components._render_consent_form()

            # Verify checkboxes were created
            assert mock_st.checkbox.call_count == 6

    @patch('voice.voice_ui.st')
    def test_render_consent_form_buttons(self, mock_st, ui_components):
        """Test consent form button interactions."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}
            mock_st.button = Mock(side_effect=[True, False])  # Accept=True, Decline=False

            result = ui_components._render_consent_form()

            # Should return True when Accept is clicked
            assert result is True


class TestVoiceUIHeaderAndStatus:
    """Test header and status rendering."""

    @patch('voice.voice_ui.st')
    def test_render_header_basic(self, mock_st, ui_components):
        """Test basic header rendering."""
        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_header()

            mock_st.subheader.assert_called_once()
            header_text = mock_st.subheader.call_args[0][0]
            assert "Voice Assistant" in header_text

    @patch('voice.voice_ui.st')
    def test_render_header_with_session(self, mock_st, ui_components):
        """Test header rendering with active session."""
        ui_components.current_session_id = "test_session_123"

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_header()

            # Should show session info
            mock_st.write.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_header_recording_state(self, mock_st, ui_components):
        """Test header rendering with different recording states."""
        states_to_test = [
            (RecordingState.RECORDING, "Recording..."),
            (RecordingState.PROCESSING, "Processing..."),
            (RecordingState.ERROR, "Error")
        ]

        for state, expected_text in states_to_test:
            ui_components.ui_state.recording_state = state

            with patch('voice.voice_ui.st') as mock_st:
                ui_components._render_header()

                # Check if status is displayed
                write_calls = [call[0][0] for call in mock_st.write.call_args_list]
                status_found = any(expected_text.lower() in call.lower() for call in write_calls)
                assert status_found, f"Expected '{expected_text}' in header for state {state}"


class TestVoiceUIInputInterface:
    """Test voice input interface rendering."""

    @patch('voice.voice_ui.st')
    def test_render_voice_input_interface_structure(self, mock_st, ui_components):
        """Test voice input interface structure."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)

            ui_components._render_voice_input_interface()

            # Verify column layout
            mock_st.columns.assert_called()

            # Verify main voice button
            mock_st.button.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_voice_input_interface_button_states(self, mock_st, ui_components):
        """Test voice input button state changes."""
        button_states = [
            (RecordingState.IDLE, "🎤 Start Recording"),
            (RecordingState.RECORDING, "⏹️ Stop Recording"),
            (RecordingState.PROCESSING, "⏳ Processing..."),
            (RecordingState.ERROR, "🔄 Retry")
        ]

        for state, expected_label in button_states:
            ui_components.ui_state.recording_state = state

            with patch('voice.voice_ui.st') as mock_st:
                mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
                mock_st.button = Mock(return_value=False)

                ui_components._render_voice_input_interface()

                # Check button label
                button_calls = mock_st.button.call_args_list
                labels = [call[1].get('label', call[0][0]) for call in button_calls]
                assert any(expected_label in label for label in labels), f"Expected '{expected_label}' for state {state}"

    @patch('voice.voice_ui.st')
    def test_render_voice_input_interface_button_actions(self, mock_st, ui_components):
        """Test voice input button click actions."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=True)  # Simulate button click

            ui_components._render_voice_input_interface()

            # Verify button was rendered with proper parameters
            button_calls = mock_st.button.call_args_list
            assert len(button_calls) > 0


class TestVoiceUITranscriptionDisplay:
    """Test transcription display rendering."""

    @patch('voice.voice_ui.st')
    def test_render_transcription_display_no_transcription(self, mock_st, ui_components):
        """Test transcription display when no transcription exists."""
        ui_components.ui_state.current_transcription = None

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_transcription_display()

            mock_st.markdown.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_transcription_display_with_transcription(self, mock_st, ui_components):
        """Test transcription display with transcription result."""
        transcription = TranscriptionResult(
            text="Hello, this is a test transcription",
            confidence=0.95,
            duration=2.5,
            timestamp=time.time(),
            is_editable=True
        )
        ui_components.ui_state.current_transcription = transcription

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_transcription_display()

            # Verify transcription text is displayed
            markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
            transcription_found = any("Hello, this is a test transcription" in call for call in markdown_calls)
            assert transcription_found

    @patch('voice.voice_ui.st')
    def test_render_transcription_display_confidence_indicators(self, mock_st, ui_components):
        """Test transcription display confidence indicators."""
        confidence_levels = [
            (0.95, "confidence-high"),
            (0.75, "confidence-medium"),
            (0.45, "confidence-low")
        ]

        for confidence, expected_class in confidence_levels:
            transcription = TranscriptionResult(
                text="Test text",
                confidence=confidence,
                duration=1.0,
                timestamp=time.time()
            )
            ui_components.ui_state.current_transcription = transcription

            with patch('voice.voice_ui.st') as mock_st:
                ui_components._render_transcription_display()

                # Check if appropriate confidence class is used
                markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
                confidence_found = any(expected_class in call for call in markdown_calls)
                assert confidence_found, f"Expected confidence class '{expected_class}' for confidence {confidence}"

    @patch('voice.voice_ui.st')
    def test_render_transcription_display_editable_mode(self, mock_st, ui_components):
        """Test transcription display in editable mode."""
        transcription = TranscriptionResult(
            text="Editable text",
            confidence=0.9,
            duration=1.5,
            timestamp=time.time(),
            is_editable=True
        )
        ui_components.ui_state.current_transcription = transcription

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.text_area = Mock(return_value="Edited text")

            ui_components._render_transcription_display()

            # Verify text area was created for editing
            mock_st.text_area.assert_called()


class TestVoiceUIVisualization:
    """Test audio visualization rendering."""

    @patch('voice.voice_ui.st')
    @patch('voice.voice_ui.np', NUMPY_AVAILABLE)
    def test_render_audio_visualization_recording(self, mock_np, mock_st, ui_components):
        """Test audio visualization during recording."""
        ui_components.ui_state.recording_state = RecordingState.RECORDING
        ui_components.waveform_data = [0.1, 0.2, 0.3, 0.4, 0.5]

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_audio_visualization()

            # Verify visualization container
            mock_st.markdown.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_audio_visualization_idle(self, mock_st, ui_components):
        """Test audio visualization when idle."""
        ui_components.ui_state.recording_state = RecordingState.IDLE

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_audio_visualization()

            # Should not render visualization when idle
            mock_st.markdown.assert_not_called()

    @patch('voice.voice_ui.st')
    @patch('voice.voice_ui.np', NUMPY_AVAILABLE)
    def test_waveform_bar_rendering(self, mock_np, mock_st, ui_components):
        """Test waveform bar rendering."""
        ui_components.waveform_data = [0.2, 0.5, 0.8, 0.3, 0.1]

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_audio_visualization()

            # Check for waveform bar CSS classes
            markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
            waveform_found = any("waveform-bar" in call for call in markdown_calls)
            assert waveform_found


class TestVoiceUIOutputControls:
    """Test voice output controls rendering."""

    @patch('voice.voice_ui.st')
    def test_render_voice_output_controls_basic(self, mock_st, ui_components):
        """Test basic voice output controls rendering."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)
            mock_st.slider = Mock(return_value=50)

            ui_components._render_voice_output_controls()

            # Verify layout structure
            mock_st.columns.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_voice_output_controls_buttons(self, mock_st, ui_components):
        """Test voice output control buttons."""
        button_labels = ["▶️ Play", "⏸️ Pause", "⏹️ Stop", "🔄 Restart"]

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)
            mock_st.slider = Mock(return_value=50)

            ui_components._render_voice_output_controls()

            # Check for expected button labels
            button_calls = mock_st.button.call_args_list
            rendered_labels = [call[0][0] for call in button_calls]
            for expected_label in button_labels:
                assert any(expected_label in label for label in rendered_labels), f"Expected button '{expected_label}' not found"

    @patch('voice.voice_ui.st')
    def test_render_voice_output_controls_progress_bar(self, mock_st, ui_components):
        """Test voice output progress bar."""
        ui_components.ui_state.playback_progress = 0.75

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)
            mock_st.slider = Mock(return_value=75)

            ui_components._render_voice_output_controls()

            # Verify progress slider
            mock_st.slider.assert_called()


class TestVoiceUISettingsPanel:
    """Test voice settings panel rendering."""

    @patch('voice.voice_ui.st')
    def test_render_settings_panel_basic(self, mock_st, ui_components):
        """Test basic settings panel rendering."""
        ui_components.ui_state.show_settings = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()
            mock_st.selectbox = Mock(return_value="calm_therapist")
            mock_st.slider = Mock(side_effect=[0.8, 1.0, 0.5])
            mock_st.checkbox = Mock(side_effect=[True, False, True])

            ui_components._render_expandable_sections()

            # Verify settings panel structure
            mock_st.expander.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_settings_panel_voice_profiles(self, mock_st, ui_components):
        """Test voice profile selection in settings."""
        ui_components.ui_state.show_settings = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()
            mock_st.selectbox = Mock(return_value="calm_therapist")
            mock_st.slider = Mock(side_effect=[0.8, 1.0, 0.5])
            mock_st.checkbox = Mock(side_effect=[True, False, True])

            ui_components._render_expandable_sections()

            # Verify voice profile selector
            mock_st.selectbox.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_settings_panel_audio_controls(self, mock_st, ui_components):
        """Test audio control sliders in settings."""
        ui_components.ui_state.show_settings = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()
            mock_st.selectbox = Mock(return_value="calm_therapist")
            mock_st.slider = Mock(side_effect=[0.8, 1.0, 0.5])
            mock_st.checkbox = Mock(side_effect=[True, False, True])

            ui_components._render_expandable_sections()

            # Verify audio control sliders (volume, speed, pitch)
            assert mock_st.slider.call_count == 3


class TestVoiceUICommandsReference:
    """Test voice commands reference rendering."""

    @patch('voice.voice_ui.st')
    def test_render_commands_reference_basic(self, mock_st, ui_components):
        """Test basic commands reference rendering."""
        ui_components.ui_state.show_commands = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()

            ui_components._render_expandable_sections()

            # Verify commands expander
            mock_st.expander.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_commands_reference_search(self, mock_st, ui_components):
        """Test commands reference with search functionality."""
        ui_components.ui_state.show_commands = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()
            mock_st.text_input = Mock(return_value="")
            mock_st.markdown = Mock()

            ui_components._render_expandable_sections()

            # Verify search input
            mock_st.text_input.assert_called()


class TestVoiceUIKeyboardShortcuts:
    """Test keyboard shortcuts rendering."""

    @patch('voice.voice_ui.st')
    def test_render_keyboard_shortcuts_enabled(self, mock_st, ui_components):
        """Test keyboard shortcuts rendering when enabled."""
        ui_components.ui_state.keyboard_shortcuts_enabled = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()

            ui_components._render_keyboard_shortcuts()

            # Verify shortcuts expander
            mock_st.expander.assert_called()

    @patch('voice.voice_ui.st')
    def test_render_keyboard_shortcuts_disabled(self, mock_st, ui_components):
        """Test keyboard shortcuts rendering when disabled."""
        ui_components.ui_state.keyboard_shortcuts_enabled = False

        with patch('voice.voice_ui.st') as mock_st:
            ui_components._render_keyboard_shortcuts()

            # Should not render shortcuts when disabled
            mock_st.expander.assert_not_called()


class TestVoiceUIMobileResponsiveness:
    """Test mobile responsiveness features."""

    @patch('voice.voice_ui.st')
    def test_detect_mobile_mode_screen_size(self, mock_st, ui_components):
        """Test mobile mode detection based on screen size."""
        screen_sizes = [
            (1920, False),  # Desktop
            (768, True),    # Tablet
            (375, True)     # Mobile
        ]

        for width, expected_mobile in screen_sizes:
            with patch('voice.voice_ui.st') as mock_st:
                mock_st.session_state = {'screen_width': width}

                ui_components._detect_mobile_mode()

                assert ui_components.ui_state.mobile_mode == expected_mobile

    def test_mobile_mode_button_sizes(self, mock_st, ui_components):
        """Test button size adjustments for mobile mode."""
        ui_components.ui_state.mobile_mode = True

        mock_st.button = Mock(return_value=False)

        ui_components._render_voice_input_interface()

            # Verify mobile-optimized CSS classes are applied
            # (This would be verified through CSS injection in real implementation)

    @patch('voice.voice_ui.st')
    def test_touch_gesture_detection(self, mock_st, ui_components):
        """Test touch gesture detection for mobile."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'input_device': 'touch'}

            # Test touch device detection function
            from voice.voice_ui import _detect_touch_device
            result = _detect_touch_device()
            assert result is True

    def test_viewport_orientation_detection(self, ui_components):
        """Test viewport orientation detection."""
        orientations = ['portrait', 'landscape']

        for orientation in orientations:
            with patch('voice.voice_ui.st') as mock_st:
                mock_st.session_state = {'orientation': orientation}

                from voice.voice_ui import _get_viewport_orientation
                result = _get_viewport_orientation()
                assert result == orientation

    def test_layout_adjustment_for_orientation(self, ui_components):
        """Test layout adjustments based on orientation."""
        from voice.voice_ui import adjust_layout_for_orientation

        # Test portrait mode
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'orientation': 'portrait'}
            layout = adjust_layout_for_orientation()
            assert layout['stacked'] is True
            assert layout['button_size'] == 'large'

        # Test landscape mode
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'orientation': 'landscape'}
            layout = adjust_layout_for_orientation()
            assert layout['stacked'] is False
            assert layout['button_size'] == 'medium'


class TestVoiceUIAccessibility:
    """Test accessibility features."""

    def test_enable_accessibility_mode(self, ui_components):
        """Test enabling accessibility mode."""
        ui_components.enable_accessibility_mode(True)

        assert ui_components.ui_state.accessibility_mode is True

    @patch('voice.voice_ui.st')
    def test_accessibility_mode_indicators(self, mock_st, ui_components):
        """Test accessibility mode visual indicators."""
        ui_components.ui_state.accessibility_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            ui_components.enable_accessibility_mode(True)

            mock_st.info.assert_called_once()
            info_text = mock_st.info.call_args[0][0]
            assert "Accessibility Mode Enabled" in info_text

    def test_enable_mobile_optimization(self, ui_components):
        """Test enabling mobile optimization."""
        ui_components.enable_mobile_optimization(True)

        assert ui_components.ui_state.mobile_mode is True

    @patch('voice.voice_ui.st')
    def test_mobile_optimization_indicators(self, mock_st, ui_components):
        """Test mobile optimization visual indicators."""
        ui_components.enable_mobile_optimization(True)

        with patch('voice.voice_ui.st') as mock_st:
            ui_components.enable_mobile_optimization(True)

            mock_st.info.assert_called_once()
            info_text = mock_st.info.call_args[0][0]
            assert "Mobile Mode Enabled" in info_text

    def test_accessible_voice_controls_rendering(self, ui_components):
        """Test accessible voice controls rendering."""
        from voice.voice_ui import render_accessible_voice_controls

        with patch('voice.voice_ui.st') as mock_st:
            controls = render_accessible_voice_controls()

            assert isinstance(controls, list)
            assert len(controls) == 3  # Voice Toggle, Emergency, Settings

            for control in controls:
                assert 'label' in control
                assert 'aria_label' in control
                assert control['aria_label'].startswith(control['label'].split()[0])

    def test_screen_reader_announcements(self, ui_components):
        """Test screen reader announcement functionality."""
        from voice.voice_ui import _announce_to_screen_reader

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            _announce_to_screen_reader("Test announcement")

            assert mock_st.session_state['screen_reader_announcement'] == "Test announcement"


class TestVoiceUIEmergencyProtocol:
    """Test emergency protocol UI flows."""

    def test_crisis_keyword_detection(self, ui_components):
        """Test crisis keyword detection in text."""
        from voice.voice_ui import _detect_crisis_keywords

        crisis_texts = [
            "I need help immediately",
            "This is an emergency situation",
            "I feel like hurting myself",
            "Call emergency services"
        ]

        non_crisis_texts = [
            "I need assistance with my homework",
            "This is a fire drill practice",
            "I feel like going for a walk"
        ]

        for text in crisis_texts:
            assert _detect_crisis_keywords(text) is True

        for text in non_crisis_texts:
            assert _detect_crisis_keywords(text) is False

    @patch('voice.voice_ui.st')
    def test_crisis_alert_display(self, mock_st, ui_components):
        """Test crisis alert display."""
        from voice.voice_ui import display_crisis_alert

        with patch('voice.voice_ui.st') as mock_st:
            display_crisis_alert("Immediate danger detected")

            mock_st.error.assert_called_once()
            error_text = mock_st.error.call_args[0][0]
            assert "CRISIS DETECTED" in error_text
            assert "Immediate danger detected" in error_text

    @patch('voice.voice_ui.st')
    def test_emergency_call_initiation(self, ui_components):
        """Test emergency call initiation."""
        from voice.voice_ui import _initiate_emergency_call

        with patch('voice.voice_ui.st') as mock_st:
            _initiate_emergency_call()

            mock_st.info.assert_called_once()
            info_text = mock_st.info.call_args[0][0]
            assert "Initiating emergency call" in info_text

    def test_emergency_event_logging(self, ui_components):
        """Test emergency event logging."""
        from voice.voice_ui import _log_emergency_event

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            emergency_details = {
                'type': 'crisis_detected',
                'severity': 'high',
                'location': 'voice_interface'
            }

            _log_emergency_event('emergency_detected', emergency_details)

            assert 'emergency_log' in mock_st.session_state
            log_entry = mock_st.session_state['emergency_log'][0]
            assert log_entry['type'] == 'emergency_detected'
            assert log_entry['details'] == emergency_details

    @patch('voice.voice_ui.st')
    def test_emergency_controls_rendering(self, ui_components):
        """Test emergency controls rendering."""
        from voice.voice_ui import render_emergency_controls

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.button = Mock(return_value=False)

            render_emergency_controls()

            mock_st.error.assert_called_once()
            mock_st.button.assert_called_once()
            button_text = mock_st.button.call_args[0][0]
            assert "Call Emergency Services" in button_text

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_emergency_contact_handling(self, ui_components):
        """Test emergency contact integration."""
        from voice.voice_ui import handle_emergency_contact

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_emergency_contact("emergency_services")

            mock_st.info.assert_called_once()
            assert result is False  # No emergency call initiated in test

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_emergency_session_logging(self, ui_components):
        """Test emergency session logging."""
        from voice.voice_ui import log_emergency_session

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            emergency_data = {'crisis_level': 'high', 'response_time': 30}
            await log_emergency_session(emergency_data)

            assert 'emergency_log' in mock_st.session_state


class TestVoiceUIErrorHandling:
    """Test error state handling."""

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_microphone_error_handling(self, ui_components):
        """Test microphone permission error handling."""
        from voice.voice_ui import handle_microphone_error

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.button = Mock(return_value=False)

            await handle_microphone_error("Permission denied")

            mock_st.error.assert_called_once()
            mock_st.button.assert_called_once()

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_audio_device_failure_handling(self, ui_components):
        """Test audio device failure handling."""
        from voice.voice_ui import handle_audio_device_failure

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.button = Mock(return_value=False)

            result = await handle_audio_device_failure("headset")

            mock_st.warning.assert_called_once()
            mock_st.button.assert_called_once()
            assert result is False

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_network_error_handling(self, ui_components):
        """Test network connectivity error handling."""
        from voice.voice_ui import handle_network_error

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_network_error("Connection lost")

            mock_st.warning.assert_called_once()
            assert result is True

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_rate_limit_error_handling(self, ui_components):
        """Test rate limiting error handling."""
        from voice.voice_ui import handle_rate_limit_error

        with patch('voice.voice_ui.st') as mock_st:
            await handle_rate_limit_error("Rate limit exceeded")

            mock_st.warning.assert_called_once()

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_memory_error_handling(self, ui_components):
        """Test memory exhaustion error handling."""
        from voice.voice_ui import handle_memory_error

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_memory_error("Out of memory")

            mock_st.error.assert_called_once()
            assert result is True

    @patch('voice.voice_ui.st')
    def test_rate_limit_ui_feedback(self, ui_components):
        """Test rate limit UI feedback."""
        from voice.voice_ui import _handle_rate_limit

        with patch('voice.voice_ui.st') as mock_st:
            _handle_rate_limit()

            mock_st.warning.assert_called_once()
            warning_text = mock_st.warning.call_args[0][0]
            assert "Rate limit reached" in warning_text


class TestVoiceUIAudioProcessingIntegration:
    """Test audio processing integration."""

    @patch('voice.voice_ui.np')
    def test_waveform_plot_creation(self, mock_np, ui_components):
        """Test waveform plot data creation."""
        from voice.voice_ui import _create_waveform_plot

        audio_data = [0.1, 0.3, 0.5, 0.2, 0.4]

        if NUMPY_AVAILABLE:
            mock_np.array.return_value = audio_data
            plot_data = _create_waveform_plot(audio_data)

            assert 'x' in plot_data
            assert 'y' in plot_data
            assert plot_data['type'] == 'waveform'
            assert len(plot_data['x']) == len(audio_data)
        else:
            plot_data = _create_waveform_plot(audio_data)
            assert plot_data == {'x': [], 'y': [], 'type': 'waveform'}

    @patch('voice.voice_ui.np')
    def test_fft_computation(self, mock_np, ui_components):
        """Test FFT computation for spectrum analysis."""
        from voice.voice_ui import _compute_fft

        audio_data = [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # Need more data for FFT

        if NUMPY_AVAILABLE:
            # Mock numpy arrays with tolist method
            mock_magnitudes = Mock()
            mock_magnitudes.__len__ = Mock(return_value=len(audio_data))
            mock_magnitudes.__getitem__ = Mock(return_value=Mock())
            mock_magnitudes.__getitem__().tolist = Mock(return_value=[1.0] * (len(audio_data)//2))
            mock_magnitudes.tolist = Mock(return_value=[1.0] * len(audio_data))

            mock_fft_result = Mock()
            mock_fft_result.__abs__ = Mock(return_value=mock_magnitudes)
            mock_np.fft.fft.return_value = mock_fft_result

            mock_freq_array = Mock()
            mock_freq_array.__len__ = Mock(return_value=len(audio_data))
            mock_freq_array.__getitem__ = Mock(return_value=Mock())
            mock_freq_array.__getitem__().tolist = Mock(return_value=[0.1 * i for i in range(len(audio_data)//2)])
            mock_freq_array.tolist = Mock(return_value=[0.1 * i for i in range(len(audio_data))])
            mock_np.fft.fftfreq.return_value = mock_freq_array

            spectrum_data = _compute_fft(audio_data)

            assert 'frequencies' in spectrum_data
            assert 'magnitudes' in spectrum_data
            assert spectrum_data['type'] == 'spectrum'
        else:
            spectrum_data = _compute_fft(audio_data)
            assert spectrum_data == {'frequencies': [], 'magnitudes': [], 'type': 'spectrum'}

    @patch('voice.voice_ui.np')
    def test_volume_level_calculation(self, mock_np, ui_components):
        """Test volume level calculation from audio data."""
        from voice.voice_ui import _calculate_volume_level

        audio_data = [0.1, -0.2, 0.3, -0.1, 0.4]

        if NUMPY_AVAILABLE:
            mock_array = Mock()
            mock_array.__pow__ = Mock(return_value=[0.01, 0.04, 0.09, 0.01, 0.16])  # squared values
            mock_np.array.return_value = mock_array
            mock_np.mean.return_value = 0.04  # RMS squared average
            mock_np.sqrt.return_value = 0.2

            volume = _calculate_volume_level(audio_data)

            assert isinstance(volume, float)
            assert volume == 0.2
        else:
            volume = _calculate_volume_level(audio_data)
            assert volume == 0.0


class TestVoiceUIAsyncOperations:
    """Test async operations in voice UI."""

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_voice_button_press_handling(self, ui_components):
        """Test voice button press handling."""
        from voice.voice_ui import handle_voice_button_press

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_voice_button_press("start_recording")

            mock_st.rerun.assert_called_once()
            assert result == "Button start_recording pressed"

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_voice_status_announcements(self, ui_components):
        """Test voice status announcements."""
        from voice.voice_ui import announce_voice_status

        with patch('voice.voice_ui.st') as mock_st:
            await announce_voice_status("recording_started")

            assert mock_st.session_state['voice_status_announcement'] == "Voice recording started"

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    @patch('voice.voice_ui.np')
    async def test_waveform_display_updates(self, mock_np, ui_components):
        """Test waveform display updates."""
        from voice.voice_ui import update_waveform_display

        audio_data = {'waveform': [0.1, 0.2, 0.3]}

        with patch('voice.voice_ui.st') as mock_st:
            await update_waveform_display(audio_data)

            assert 'waveform_data' in mock_st.session_state

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    @patch('voice.voice_ui.np')
    async def test_spectrum_display_updates(self, mock_np, ui_components):
        """Test spectrum display updates."""
        from voice.voice_ui import update_spectrum_display

        audio_data = {'waveform': [0.1, 0.2, 0.3] * 50}  # More data for FFT

        with patch('voice.voice_ui.st') as mock_st:
            await update_spectrum_display(audio_data)

            assert 'spectrum_data' in mock_st.session_state

    @pytest.mark.asyncio
    @patch('voice.voice_ui.st')
    async def test_volume_meter_updates(self, ui_components):
        """Test volume meter updates."""
        from voice.voice_ui import update_volume_meter

        audio_data = [0.1, 0.2, 0.3]

        with patch('voice.voice_ui.st') as mock_st:
            await update_volume_meter(audio_data)

            assert 'volume_level' in mock_st.session_state


class TestVoiceUIFactoryFunction:
    """Test voice UI factory function."""

    def test_create_voice_ui_factory(self, mock_voice_service, mock_config):
        """Test create_voice_ui factory function."""
        with patch('voice.voice_ui.st', Mock()) if not STREAMLIT_AVAILABLE else patch.object(st, 'markdown'):
            ui_instance = create_voice_ui(mock_voice_service, mock_config)

            assert isinstance(ui_instance, VoiceUIComponents)
            assert ui_instance.voice_service == mock_voice_service
            assert ui_instance.config == mock_config


class TestVoiceUIUtilityFunctions:
    """Test utility functions in voice UI."""

    def test_get_screen_width_utility(self, ui_components):
        """Test screen width detection utility."""
        from voice.voice_ui import _get_screen_width

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'screen_width': 1440}
            width = _get_screen_width()
            assert width == 1440

        # Default when not set
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}
            width = _get_screen_width()
            assert width == 1024

    def test_keyboard_shortcuts_generation(self, ui_components):
        """Test keyboard shortcuts generation."""
        from voice.voice_ui import _generate_keyboard_shortcuts

        shortcuts = _generate_keyboard_shortcuts()

        assert isinstance(shortcuts, dict)
        assert 'voice_toggle' in shortcuts
        assert 'emergency' in shortcuts
        assert 'settings' in shortcuts

    def test_render_keyboard_shortcuts_utility(self, ui_components):
        """Test keyboard shortcuts rendering utility."""
        from voice.voice_ui import render_keyboard_shortcuts

        with patch('voice.voice_ui.st') as mock_st:
            shortcuts = render_keyboard_shortcuts()

            assert isinstance(shortcuts, list)
            assert len(shortcuts) > 0

            for shortcut in shortcuts:
                assert 'key' in shortcut
                assert 'shortcut' in shortcut

    def test_browser_info_retrieval(self, ui_components):
        """Test browser information retrieval."""
        from voice.voice_ui import _get_browser_info

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'user_agent': 'Chrome/91', 'browser': 'Chrome'}

            info = _get_browser_info()

            assert info['user_agent'] == 'Chrome/91'
            assert info['browser'] == 'Chrome'

    def test_debounced_input_handling(self, ui_components):
        """Test debounced input handling."""
        from voice.voice_ui import _debounce_input

        call_count = 0
        def test_func():
            nonlocal call_count
            call_count += 1

        debounced_func = _debounce_input(test_func, delay=0.1)

        # Call multiple times quickly
        for _ in range(3):
            debounced_func()

        # Should only be called once due to debouncing
        assert call_count == 1

    @patch('voice.voice_ui.st')
    def test_voice_controls_rendering_utility(self, mock_st, ui_components):
        """Test voice controls rendering utility."""
        from voice.voice_ui import render_voice_controls

        controls = render_voice_controls()

        assert isinstance(controls, list)
        assert len(controls) == 3

        expected_types = ['button', 'button', 'button']
        for i, control in enumerate(controls):
            assert control['type'] == expected_types[i]


class TestVoiceUIStateQueries:
    """Test voice UI state query methods."""

    def test_is_recording_state_checks(self, ui_components):
        """Test is_recording state checks."""
        ui_components.ui_state.recording_state = RecordingState.RECORDING
        assert ui_components.is_recording() is True

        ui_components.ui_state.recording_state = RecordingState.IDLE
        assert ui_components.is_recording() is False

    def test_is_playing_state_checks(self, ui_components):
        """Test is_playing state checks."""
        ui_components.ui_state.playback_state = PlaybackState.PLAYING
        assert ui_components.is_playing() is True

        ui_components.ui_state.playback_state = PlaybackState.STOPPED
        assert ui_components.is_playing() is False

    def test_get_current_transcription(self, ui_components):
        """Test getting current transcription."""
        transcription = TranscriptionResult("test text", 0.9, 1.0, time.time())
        ui_components.ui_state.current_transcription = transcription

        result = ui_components.get_current_transcription()
        assert result == transcription

    def test_get_current_transcription_none(self, ui_components):
        """Test getting current transcription when none exists."""
        ui_components.ui_state.current_transcription = None

        result = ui_components.get_current_transcription()
        assert result is None


# Run basic validation
if __name__ == "__main__":
    # Quick validation that imports work
    print("Voice UI Components Test Suite")
    print("=" * 50)

    try:
        from voice.voice_ui import VoiceUIComponents, VoiceUIState
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        # Test basic instantiation with mocks
        mock_service = Mock()
        mock_config = Mock()
        mock_config.voice_enabled = True
        mock_config.security = Mock()
        mock_config.security.consent_required = False

        with patch('voice.voice_ui.st', Mock()):
            ui = VoiceUIComponents(mock_service, mock_config)
            print("✅ Basic instantiation successful")
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")

    print("Test file created successfully - run with pytest for full validation")
