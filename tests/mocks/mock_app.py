"""
Mock app module for testing without streamlit dependency.

This module provides mock implementations of app.py functions that are
used in integration tests, allowing them to run without requiring
streamlit to be installed.
"""

import hashlib
import asyncio
from typing import Dict, Any, List


class MockStreamlitState:
    """Mock Streamlit session state for testing."""

    def __init__(self):
        self._state = {
            'messages': [],
            'conversation_chain': None,
            'vectorstore': None,
            'cache_hits': 0,
            'total_requests': 0,
            'voice_enabled': False,
            'voice_config': None,
            'voice_security': None,
            'voice_service': None,
            'voice_ui': None,
            'voice_command_processor': None,
            'voice_consent_given': False,
            'voice_setup_complete': False,
            'voice_setup_step': 0
        }

    def get(self, key, default=None):
        return self._state.get(key, default)

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def __contains__(self, key):
        return key in self._state


# Global mock session state
_mock_session_state = MockStreamlitState()


def initialize_session_state():
    """Initializes mock session state variables."""
    global _mock_session_state
    _mock_session_state = MockStreamlitState()
    return _mock_session_state


def handle_voice_text_received(session_id: str, text: str):
    """Handle text received from voice input."""
    global _mock_session_state

    if text.strip():
        # Add voice text to conversation
        _mock_session_state['messages'].append({"role": "user", "content": f"🎤 {text}"})

        # Process voice commands first
        if _mock_session_state.get('voice_command_processor'):
            try:
                # Check for voice commands
                result = asyncio.run(_mock_session_state['voice_command_processor'].process_text(text, session_id=session_id))
                if result and result.is_command:
                    return handle_voice_command_executed(session_id, str(result))
            except Exception as e:
                # Handle command processing errors gracefully
                print(f"Voice command processing error: {e}")

        # Trigger conversation processing
        return True
    return False


def handle_voice_command_executed(session_id: str, command_result: str):
    """Handle executed voice command."""
    print(f"Voice command executed: {command_result}")
    return True

def handle_assistant_response(session_id: str, text: str):
    """Handle assistant response to voice output."""
    global _mock_session_state
    
    if text.strip():
        # Add assistant response to conversation
        _mock_session_state['messages'].append({"role": "assistant", "content": text})
    
    return True


def detect_crisis_content(text):
    """Detect crisis situations requiring immediate intervention."""
    # Enhanced crisis keywords from voice command system
    # More specific crisis-related phrases to reduce false positives
    crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'self-harm',
        'hurt myself', 'want to die', 'no reason to live',
        'better off dead', 'can\'t go on', 'end it all',
        'suicidal', 'depressed', 'hopeless', 'worthless',
        'crisis', 'emergency', 'overwhelmed', 'desperate',
        'alone', 'isolated'
    ]

    text_lower = text.lower()
    detected_keywords = []

    for keyword in crisis_keywords:
        if keyword in text_lower:
            detected_keywords.append(keyword)

    is_crisis = len(detected_keywords) > 0
    return is_crisis, detected_keywords


def generate_crisis_response():
    """Generate appropriate crisis response with resources."""
    crisis_message = """
    🚨 **IMMEDIATE HELP NEEDED** 🚨
    I'm concerned about your safety. Please reach out for immediate help:
    **National Suicide Prevention Lifeline: 988**
    **Crisis Text Line: Text HOME to 741741**

    You can also:
    - Call 911 if you're in immediate danger
    - Go to your nearest emergency room
    - Contact a trusted friend, family member, or mental health professional

    Remember, help is available and you don't have to go through this alone.
    """
    return crisis_message


class ResponseCache:
    """Mock response cache for testing."""

    def __init__(self):
        self.cache = {}
        self.max_size = 100

    def get_cache_key(self, question, context_hash):
        """Generate cache key for question and context."""
        return f"{hashlib.sha256(question.encode()).hexdigest()}_{context_hash}"

    def get(self, question, context_hash):
        """Get cached response."""
        key = self.get_cache_key(question, context_hash)
        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, question, context_hash, response):
        """Cache response."""
        key = self.get_cache_key(question, context_hash)
        self.cache[key] = response

        # Limit cache size
        if len(self.cache) > self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]


class EmbeddingCache:
    """Mock embedding cache for testing."""

    def __init__(self):
        self.cache = {}
        self.cache_dir = "./embedding_cache"

    def get_embedding_key(self, text):
        """Generate embedding cache key."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text):
        """Get cached embedding."""
        key = self.get_embedding_key(text)
        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, text, embedding):
        """Cache embedding."""
        key = self.get_embedding_key(text)
        self.cache[key] = embedding


# Mock streamlit module
class MockStreamlit:
    """Mock streamlit module for testing."""

    class session_state_class:
        """Mock session state that behaves like a dictionary."""

        def __init__(self):
            self._data = MockStreamlitState()._state

        def __getattr__(self, name):
            if name in self._data:
                return self._data[name]
            raise AttributeError(f"'MockSessionState' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            if name.startswith('_'):
                super().__setattr__(name, value)
            else:
                self._data[name] = value

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def __contains__(self, key):
            return key in self._data

    session_state = session_state_class()

    @staticmethod
    def info(message):
        """Mock st.info() function."""
        print(f"INFO: {message}")

    @staticmethod
    def error(message):
        """Mock st.error() function."""
        print(f"ERROR: {message}")

    @staticmethod
    def success(message):
        """Mock st.success() function."""
        print(f"SUCCESS: {message}")

    @staticmethod
    def warning(message):
        """Mock st.warning() function."""
        print(f"WARNING: {message}")


# Function to get mock session state for tests
def get_mock_session_state():
    """Get the current mock session state."""
    return _mock_session_state


# Function to reset mock session state for tests
def reset_mock_session_state():
    """Reset the mock session state."""
    global _mock_session_state
    _mock_session_state = MockStreamlitState()