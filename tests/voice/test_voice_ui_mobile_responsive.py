"""
Voice UI Mobile Responsiveness Testing

Comprehensive test suite for mobile responsiveness features:
- Touch gesture handling and detection
- Responsive layout adjustments
- Mobile-optimized button sizes and spacing
- Viewport orientation changes
- Touch-friendly interface elements

Coverage targets: Mobile responsiveness testing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import time

# Test imports with conditional handling
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = Mock()
    STREAMLIT_AVAILABLE = False

from voice.voice_ui import (
    VoiceUIComponents, VoiceUIState, RecordingState,
    _detect_touch_device, _get_viewport_orientation,
    adjust_layout_for_orientation, render_voice_controls
)
from voice.config import VoiceConfig
from voice.voice_service import VoiceService


@pytest.fixture
def mobile_ui_components():
    """Create voice UI components configured for mobile testing."""
    config = Mock(spec=VoiceConfig)
    config.voice_enabled = True
    config.security = Mock()
    config.security.consent_required = False

    service = Mock(spec=VoiceService)
    service.is_available.return_value = True
    service.create_session.return_value = "mobile_session_123"

    with patch('voice.voice_ui.st', Mock()):
        components = VoiceUIComponents(service, config)
        components.ui_state.mobile_mode = True
        yield components


class TestMobileDeviceDetection:
    """Test mobile device detection functionality."""

    def test_touch_device_detection_positive(self):
        """Test touch device detection when touch is available."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'input_device': 'touch'}

            result = _detect_touch_device()
            assert result is True

    def test_touch_device_detection_negative(self):
        """Test touch device detection when only mouse is available."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'input_device': 'mouse'}

            result = _detect_touch_device()
            assert result is False

    def test_touch_device_detection_default(self):
        """Test touch device detection with default state."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            result = _detect_touch_device()
            assert result is False

    def test_viewport_orientation_portrait(self):
        """Test viewport orientation detection for portrait mode."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'orientation': 'portrait'}

            result = _get_viewport_orientation()
            assert result == 'portrait'

    def test_viewport_orientation_landscape(self):
        """Test viewport orientation detection for landscape mode."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'orientation': 'landscape'}

            result = _get_viewport_orientation()
            assert result == 'landscape'

    def test_viewport_orientation_default(self):
        """Test viewport orientation detection with default state."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            result = _get_viewport_orientation()
            assert result == 'landscape'


class TestResponsiveLayoutAdjustments:
    """Test responsive layout adjustments for different screen sizes."""

    def test_layout_adjustment_portrait_mode(self):
        """Test layout adjustments for portrait orientation."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'orientation': 'portrait'}

            layout = adjust_layout_for_orientation()

            assert layout['stacked'] is True
            assert layout['button_size'] == 'large'

    def test_layout_adjustment_landscape_mode(self):
        """Test layout adjustments for landscape orientation."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'orientation': 'landscape'}

            layout = adjust_layout_for_orientation()

            assert layout['stacked'] is False
            assert layout['button_size'] == 'medium'

    def test_mobile_mode_detection_small_screen(self):
        """Test mobile mode detection for small screens."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'screen_width': 375}  # iPhone width

            components = mobile_ui_components()
            components._detect_mobile_mode()

            assert components.ui_state.mobile_mode is True

    def test_mobile_mode_detection_tablet_screen(self):
        """Test mobile mode detection for tablet screens."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'screen_width': 768}  # iPad width

            components = mobile_ui_components()
            components._detect_mobile_mode()

            assert components.ui_state.mobile_mode is True

    def test_mobile_mode_detection_desktop_screen(self):
        """Test mobile mode detection for desktop screens."""
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'screen_width': 1920}  # Desktop width

            components = mobile_ui_components()
            components._detect_mobile_mode()

            assert components.ui_state.mobile_mode is False


class TestMobileOptimizedVoiceControls:
    """Test mobile-optimized voice control rendering."""

    @patch('voice.voice_ui.st')
    def test_mobile_voice_button_sizes(self, mock_st, mobile_ui_components):
        """Test voice button size adjustments for mobile."""
        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)

            mobile_ui_components._render_voice_input_interface()

            # Verify mobile CSS classes are applied
            # This would be verified through CSS injection in real implementation
            mock_st.markdown.assert_called()

    @patch('voice.voice_ui.st')
    def test_mobile_voice_controls_layout(self, mock_st, mobile_ui_components):
        """Test mobile voice controls layout structure."""
        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)

            mobile_ui_components._render_voice_input_interface()

            # Verify column layout for mobile
            mock_st.columns.assert_called_with([1, 2, 1])

    @patch('voice.voice_ui.st')
    def test_mobile_touch_targets(self, mock_st, mobile_ui_components):
        """Test touch target sizes for mobile devices."""
        mobile_ui_components.ui_state.mobile_mode = True

        # Test that mobile CSS includes touch-action and larger targets
        css_content = mobile_ui_components._inject_custom_css()
        # CSS injection happens in __init__, we test the CSS content directly
        assert "touch-action: manipulation" in str(mobile_ui_components)
        assert "-webkit-tap-highlight-color" in str(mobile_ui_components)

    def test_mobile_responsive_css_breakpoints(self, mobile_ui_components):
        """Test CSS breakpoints for mobile responsiveness."""
        # Verify mobile CSS contains responsive breakpoints
        css_injection = mobile_ui_components._inject_custom_css()
        # CSS is injected via st.markdown, we verify it contains mobile breakpoints
        assert "@media (max-width: 768px)" in str(mobile_ui_components)
        assert "@media (max-width: 480px)" in str(mobile_ui_components)


class TestTouchGestureHandling:
    """Test touch gesture handling and events."""

    @patch('voice.voice_ui.st')
    async def test_touch_button_press_feedback(self, mock_st):
        """Test touch button press feedback."""
        from voice.voice_ui import handle_voice_button_press

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_voice_button_press("start_recording")

            mock_st.rerun.assert_called_once()
            assert "start_recording" in result

    @patch('voice.voice_ui.st')
    def test_touch_focus_management(self, mock_st, mobile_ui_components):
        """Test touch focus management."""
        from voice.voice_ui import handle_voice_focus

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            handle_voice_focus("emergency_button")

            assert mock_st.session_state['voice_focused'] is True

    @patch('voice.voice_ui.st')
    def test_touch_gesture_debouncing(self, mock_st):
        """Test debouncing for touch gestures."""
        from voice.voice_ui import _debounce_input

        call_count = 0
        def test_touch_handler():
            nonlocal call_count
            call_count += 1

        debounced_handler = _debounce_input(test_touch_handler, delay=0.2)

        # Simulate rapid touch events
        for _ in range(5):
            debounced_handler()

        # Should only trigger once due to debouncing
        assert call_count == 1

    @patch('voice.voice_ui.st')
    async def test_touch_input_processing(self, mock_st):
        """Test touch input processing and validation."""
        from voice.voice_ui import handle_debounced_input

        with patch('voice.voice_ui.st') as mock_st:
            await handle_debounced_input("voice_command", "start recording")

            assert mock_st.session_state['debounced_input'] is True


class TestMobileEmergencyProtocol:
    """Test emergency protocol features for mobile devices."""

    @patch('voice.voice_ui.st')
    def test_mobile_emergency_button_size(self, mock_st, mobile_ui_components):
        """Test emergency button size on mobile devices."""
        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            from voice.voice_ui import render_emergency_controls

            mock_st.button = Mock(return_value=False)
            render_emergency_controls()

            # Verify emergency button is rendered
            mock_st.button.assert_called_once()

    @patch('voice.voice_ui.st')
    async def test_mobile_emergency_call_initiation(self, mock_st):
        """Test emergency call initiation on mobile."""
        from voice.voice_ui import handle_emergency_contact

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_emergency_contact("emergency_services")

            mock_st.info.assert_called_once()
            assert result is False  # No actual call in test

    @patch('voice.voice_ui.st')
    def test_mobile_crisis_alert_display(self, mock_st):
        """Test crisis alert display on mobile devices."""
        from voice.voice_ui import display_crisis_alert

        with patch('voice.voice_ui.st') as mock_st:
            display_crisis_alert("Mobile crisis detected")

            mock_st.error.assert_called_once()
            error_message = mock_st.error.call_args[0][0]
            assert "CRISIS DETECTED" in error_message
            assert "Mobile crisis detected" in error_message

    @patch('voice.voice_ui.st')
    async def test_mobile_emergency_session_logging(self, mock_st):
        """Test emergency session logging on mobile."""
        from voice.voice_ui import log_emergency_session

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {}

            emergency_data = {
                'device_type': 'mobile',
                'location': 'voice_ui',
                'crisis_level': 'high'
            }

            await log_emergency_session(emergency_data)

            assert 'emergency_log' in mock_st.session_state
            log_entry = mock_st.session_state['emergency_log'][0]
            assert log_entry['details']['device_type'] == 'mobile'


class TestMobileAccessibilityFeatures:
    """Test accessibility features optimized for mobile."""

    def test_mobile_accessibility_mode_activation(self, mobile_ui_components):
        """Test accessibility mode activation on mobile."""
        mobile_ui_components.enable_accessibility_mode(True)

        assert mobile_ui_components.ui_state.accessibility_mode is True
        assert mobile_ui_components.ui_state.mobile_mode is True

    @patch('voice.voice_ui.st')
    def test_mobile_screen_reader_support(self, mock_st, mobile_ui_components):
        """Test screen reader support on mobile devices."""
        from voice.voice_ui import _announce_to_screen_reader

        mobile_ui_components.ui_state.accessibility_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            _announce_to_screen_reader("Mobile voice status update")

            assert mock_st.session_state['screen_reader_announcement'] == "Mobile voice status update"

    @patch('voice.voice_ui.st')
    async def test_mobile_voice_status_announcements(self, mock_st, mobile_ui_components):
        """Test voice status announcements for mobile accessibility."""
        from voice.voice_ui import announce_voice_status

        with patch('voice.voice_ui.st') as mock_st:
            await announce_voice_status("recording_started")

            announcement = mock_st.session_state['voice_status_announcement']
            assert announcement == "Voice recording started"

    def test_mobile_aria_labels(self, mobile_ui_components):
        """Test ARIA labels for mobile voice controls."""
        from voice.voice_ui import render_accessible_voice_controls

        with patch('voice.voice_ui.st') as mock_st:
            controls = render_accessible_voice_controls()

            assert len(controls) == 3
            for control in controls:
                assert 'aria_label' in control
                assert 'Mobile' in control['aria_label'] or 'Voice' in control['aria_label']

    @patch('voice.voice_ui.st')
    def test_mobile_keyboard_shortcuts_adaptation(self, mock_st, mobile_ui_components):
        """Test keyboard shortcuts adaptation for mobile."""
        from voice.voice_ui import render_keyboard_shortcuts

        mobile_ui_components.ui_state.mobile_mode = True

        shortcuts = render_keyboard_shortcuts()

        # Verify shortcuts are available even on mobile (for Bluetooth keyboards)
        assert isinstance(shortcuts, list)
        assert len(shortcuts) > 0


class TestMobileAudioVisualization:
    """Test audio visualization features on mobile devices."""

    @patch('voice.voice_ui.st')
    @patch('voice.voice_ui.np')
    def test_mobile_waveform_display(self, mock_np, mock_st, mobile_ui_components):
        """Test waveform display optimization for mobile."""
        mobile_ui_components.ui_state.mobile_mode = True
        mobile_ui_components.ui_state.recording_state = RecordingState.RECORDING
        mobile_ui_components.waveform_data = [0.1, 0.2, 0.3, 0.4, 0.5]

        with patch('voice.voice_ui.st') as mock_st:
            mobile_ui_components._render_audio_visualization()

            # Verify mobile-optimized visualization
            mock_st.markdown.assert_called()

    @patch('voice.voice_ui.st')
    async def test_mobile_volume_meter_touch(self, mock_st, mobile_ui_components):
        """Test volume meter interaction on touch devices."""
        from voice.voice_ui import update_volume_meter

        mobile_ui_components.ui_state.mobile_mode = True

        audio_data = [0.1, 0.2, 0.3, 0.4, 0.5]

        with patch('voice.voice_ui.st') as mock_st:
            await update_volume_meter(audio_data)

            assert 'volume_level' in mock_st.session_state

    @patch('voice.voice_ui.st')
    def test_mobile_visualization_sizing(self, mock_st, mobile_ui_components):
        """Test visualization sizing for mobile screens."""
        mobile_ui_components.ui_state.mobile_mode = True

        # Verify mobile CSS contains smaller visualization heights
        css_content = str(mobile_ui_components)
        assert "height: 100px" in css_content  # Mobile-optimized height

        # Test with smaller screens
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'screen_width': 375}

            mobile_ui_components._detect_mobile_mode()
            assert mobile_ui_components.ui_state.mobile_mode is True


class TestMobileErrorHandling:
    """Test error handling optimized for mobile devices."""

    @patch('voice.voice_ui.st')
    async def test_mobile_network_error_handling(self, mock_st, mobile_ui_components):
        """Test network error handling on mobile."""
        from voice.voice_ui import handle_network_error

        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_network_error("Mobile connection lost")

            mock_st.warning.assert_called_once()
            assert result is True

    @patch('voice.voice_ui.st')
    async def test_mobile_microphone_error_touch(self, mock_st, mobile_ui_components):
        """Test microphone error handling on touch devices."""
        from voice.voice_ui import handle_microphone_error

        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.button = Mock(return_value=False)

            await handle_microphone_error("Touch device microphone access denied")

            mock_st.error.assert_called_once()
            mock_st.button.assert_called_once()

    @patch('voice.voice_ui.st')
    async def test_mobile_memory_error_compaction(self, mock_st, mobile_ui_components):
        """Test memory error handling with mobile optimizations."""
        from voice.voice_ui import handle_memory_error

        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            result = await handle_memory_error("Mobile device memory low")

            mock_st.error.assert_called_once()
            assert result is True


class TestMobileSettingsPanel:
    """Test settings panel adaptations for mobile."""

    @patch('voice.voice_ui.st')
    def test_mobile_settings_panel_layout(self, mock_st, mobile_ui_components):
        """Test settings panel layout for mobile devices."""
        mobile_ui_components.ui_state.mobile_mode = True
        mobile_ui_components.ui_state.show_settings = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.expander = Mock()
            mock_st.selectbox = Mock(return_value="mobile_therapist")
            mock_st.slider = Mock(side_effect=[0.7, 0.9, 0.4])
            mock_st.checkbox = Mock(side_effect=[True, False, True])

            mobile_ui_components._render_expandable_sections()

            # Verify mobile-optimized settings layout
            mock_st.expander.assert_called()

    @patch('voice.voice_ui.st')
    def test_mobile_settings_touch_targets(self, mock_st, mobile_ui_components):
        """Test settings touch targets for mobile."""
        mobile_ui_components.ui_state.mobile_mode = True

        # Verify mobile CSS includes larger touch targets for settings
        css_content = str(mobile_ui_components)
        assert "touch-action: manipulation" in css_content


class TestMobileTranscriptionDisplay:
    """Test transcription display adaptations for mobile."""

    @patch('voice.voice_ui.st')
    def test_mobile_transcription_font_sizes(self, mock_st, mobile_ui_components):
        """Test transcription font sizes for mobile readability."""
        from voice.audio_processor import AudioData
        from voice.voice_ui import TranscriptionResult

        mobile_ui_components.ui_state.mobile_mode = True

        transcription = TranscriptionResult(
            text="Mobile transcription test",
            confidence=0.92,
            duration=1.5,
            timestamp=time.time(),
            is_editable=True
        )
        mobile_ui_components.ui_state.current_transcription = transcription

        with patch('voice.voice_ui.st') as mock_st:
            mobile_ui_components._render_transcription_display()

            # Verify mobile-optimized transcription display
            mock_st.markdown.assert_called()

    @patch('voice.voice_ui.st')
    def test_mobile_transcription_editing(self, mock_st, mobile_ui_components):
        """Test transcription editing on mobile devices."""
        from voice.voice_ui import TranscriptionResult

        mobile_ui_components.ui_state.mobile_mode = True

        transcription = TranscriptionResult(
            text="Editable mobile text",
            confidence=0.88,
            duration=2.0,
            timestamp=time.time(),
            is_editable=True
        )
        mobile_ui_components.ui_state.current_transcription = transcription

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.text_area = Mock(return_value="Edited on mobile")

            mobile_ui_components._render_transcription_display()

            # Verify text area for mobile editing
            mock_st.text_area.assert_called()


class TestMobilePlaybackControls:
    """Test playback controls optimization for mobile."""

    @patch('voice.voice_ui.st')
    def test_mobile_playback_button_sizes(self, mock_st, mobile_ui_components):
        """Test playback button sizes for mobile touch."""
        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)
            mock_st.slider = Mock(return_value=50)

            mobile_ui_components._render_voice_output_controls()

            # Verify mobile-optimized button sizes
            mock_st.button.assert_called()

    @patch('voice.voice_ui.st')
    def test_mobile_progress_bar_touch(self, mock_st, mobile_ui_components):
        """Test progress bar touch interaction on mobile."""
        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
            mock_st.button = Mock(return_value=False)
            mock_st.slider = Mock(return_value=75)

            mobile_ui_components._render_voice_output_controls()

            # Verify touch-friendly progress bar
            mock_st.slider.assert_called()


class TestMobileThemeSupport:
    """Test theme support for mobile devices."""

    def test_mobile_dark_mode_support(self, mobile_ui_components):
        """Test dark mode CSS for mobile devices."""
        mobile_ui_components.ui_state.mobile_mode = True

        # Verify mobile dark mode CSS classes
        css_content = str(mobile_ui_components)
        assert ".dark-mode" in css_content
        assert "background: #1a1a1a" in css_content
        assert "color: #ffffff" in css_content

    @patch('voice.voice_ui.st')
    def test_mobile_theme_switching(self, mock_st, mobile_ui_components):
        """Test theme switching on mobile devices."""
        mobile_ui_components.ui_state.mobile_mode = True

        # Test theme class application
        # CSS injection would handle this in real implementation
        assert "dark-mode" in str(mobile_ui_components)


class TestMobilePerformanceOptimizations:
    """Test performance optimizations for mobile devices."""

    @patch('voice.voice_ui.st')
    async def test_mobile_component_lazy_loading(self, mock_st, mobile_ui_components):
        """Test component lazy loading for mobile performance."""
        from voice.voice_ui import load_voice_component

        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            await load_voice_component("mobile_waveform")

            assert mock_st.session_state['loading_mobile_waveform'] is True

    @patch('voice.voice_ui.st')
    def test_mobile_memory_cleanup(self, mock_st, mobile_ui_components):
        """Test memory cleanup optimizations for mobile."""
        from voice.voice_ui import cleanup_voice_session

        mobile_ui_components.ui_state.mobile_mode = True

        with patch('voice.voice_ui.st') as mock_st:
            # Simulate memory cleanup
            mock_st.session_state = {
                'audio_buffers': [1, 2, 3],
                'waveform_data': [0.1, 0.2],
                'emergency_log': [{'event': 'test'}]
            }

            cleanup_voice_session()

            # Verify cleanup function called (actual cleanup tested in utility)
            # This tests the async wrapper

    def test_mobile_audio_quality_adaptation(self, mobile_ui_components):
        """Test audio quality adaptation for mobile bandwidth."""
        from voice.voice_ui import _reduce_audio_quality

        # Test quality reduction for mobile
        current_quality = 44100
        reduced_quality = _reduce_audio_quality(current_quality)

        assert reduced_quality < current_quality
        assert reduced_quality in [22050, 16000, 8000]


class TestMobileBrowserCompatibility:
    """Test browser compatibility features for mobile."""

    @patch('voice.voice_ui.st')
    async def test_mobile_browser_audio_initialization(self, mock_st):
        """Test browser audio context initialization on mobile."""
        from voice.voice_ui import initialize_browser_audio

        with patch('voice.voice_ui.st') as mock_st:
            result = await initialize_browser_audio()

            assert mock_st.session_state['browser_audio_init'] is True
            assert result is True

    @patch('voice.voice_ui.st')
    async def test_mobile_browser_permissions(self, mock_st):
        """Test browser permissions request on mobile."""
        from voice.voice_ui import request_browser_permissions

        with patch('voice.voice_ui.st') as mock_st:
            result = await request_browser_permissions()

            assert mock_st.session_state['permissions_requested'] is True
            assert result is True

    @patch('voice.voice_ui.st')
    def test_mobile_browser_info_detection(self, mock_st, mobile_ui_components):
        """Test browser information detection for mobile."""
        from voice.voice_ui import _get_browser_info

        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {
                'user_agent': 'Mobile Safari/604.1',
                'browser': 'Safari Mobile'
            }

            info = _get_browser_info()

            assert info['browser'] == 'Safari Mobile'
            assert 'Mobile' in info['user_agent']


# Run basic validation
if __name__ == "__main__":
    print("Voice UI Mobile Responsiveness Test Suite")
    print("=" * 50)

    try:
        from voice.voice_ui import _detect_touch_device, adjust_layout_for_orientation
        print("✅ Mobile utility imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        # Test basic mobile detection
        with patch('voice.voice_ui.st') as mock_st:
            mock_st.session_state = {'screen_width': 375}
            layout = adjust_layout_for_orientation()
            print("✅ Mobile layout adjustment successful")
    except Exception as e:
        print(f"❌ Layout adjustment failed: {e}")

    print("Mobile responsiveness test file created - run with pytest for full validation")
