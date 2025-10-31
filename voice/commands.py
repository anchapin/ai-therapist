"""
Voice Commands Module

This module handles comprehensive voice command processing including:
- Command pattern matching and natural language processing
- Command execution with confidence scoring
- Wake word detection and context-aware interpretation
- Emergency response and crisis detection
- Navigation, session control, and feature access
- Command help and feedback systems
- Custom command registration and extensibility
- Command validation, security, and comprehensive logging
"""

import re
import time
import asyncio
from typing import Optional, Dict, List, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
from datetime import datetime
from enum import Enum
import hashlib
from collections import defaultdict

from .config import VoiceConfig
from .audio_processor import AudioData
from .stt_service import STTResult

# Import main app crisis detection for consistency
try:
    from app import detect_crisis_content, generate_crisis_response
except ImportError:
    # Fallback crisis detection if app module not available
    def detect_crisis_content(text):
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'self-harm',
            'hurt myself', 'want to die', 'no reason to live',
            'better off dead', 'can\'t go on', 'end it all'
        ]
        text_lower = text.lower()
        detected_keywords = [keyword for keyword in crisis_keywords if keyword in text_lower]
        return bool(detected_keywords), detected_keywords

    def generate_crisis_response():
        return """
        🚨 **IMMEDIATE HELP NEEDED** 🚨

        I'm concerned about your safety. Please reach out for immediate help:

        **National Suicide Prevention Lifeline: 988**
        **Crisis Text Line: Text HOME to 741741**

        Your life matters, and there are people who want to help you right now.
        """

class CommandCategory(Enum):
    """Voice command categories."""
    NAVIGATION = "navigation"
    SESSION_CONTROL = "session_control"
    EMERGENCY = "emergency"
    FEATURE_ACCESS = "feature_access"
    VOICE_CONTROL = "voice_control"
    SETTINGS = "settings"
    HELP = "help"
    MEDITATION = "meditation"
    JOURNAL = "journal"
    RESOURCES = "resources"

class SecurityLevel(Enum):
    """Security levels for voice commands."""
    LOW = "low"           # Basic commands, no sensitive access
    MEDIUM = "medium"     # User settings and preferences
    HIGH = "high"         # Emergency and crisis response
    CRITICAL = "critical" # System-level operations

@dataclass
class VoiceCommand:
    """Enhanced voice command definition."""
    name: str
    description: str
    category: CommandCategory
    patterns: List[str]
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.LOW
    enabled: bool = True
    cooldown: int = 0
    priority: int = 0  # Higher priority commands are matched first
    require_wake_word: bool = True
    context_aware: bool = False
    confidence_threshold: float = 0.6
    emergency_keywords: List[str] = field(default_factory=list)
    response_templates: List[str] = field(default_factory=list)
    fallback_commands: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommandMatch:
    """Enhanced command match result."""
    command: VoiceCommand
    confidence: float
    parameters: Dict[str, Any]
    matched_text: str
    timestamp: float
    context_score: float = 0.0
    alternative_matches: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    is_emergency: bool = False
    crisis_keywords_detected: List[str] = field(default_factory=list)
    match_method: str = "pattern"  # pattern, semantic, hybrid

class VoiceCommandProcessor:
    """Enhanced voice command processor with comprehensive emergency detection."""

    def __init__(self, config: VoiceConfig):
        """Initialize voice command processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Command definitions
        self.commands: Dict[str, VoiceCommand] = {}
        self.commands_by_category: Dict[CommandCategory, List[str]] = defaultdict(list)
        self.command_history: List[CommandMatch] = []

        # Emergency and crisis detection
        self.emergency_keywords = [
            'suicide', 'kill myself', 'end my life', 'self-harm', 'hurt myself',
            'want to die', 'don\'t want to live', 'end it all', 'no reason to live',
            'overdose', 'jump off', 'hang myself', 'shoot myself', 'cut myself',
            'emergency', 'crisis', 'help me', 'I need help', 'urgent help',
            'immediate help', 'save me', 'can\'t go on', 'give up', 'hopeless',
            'depressed', 'anxiety attack', 'panic attack', 'breakdown', 'meltdown',
            'abuse', 'violence', 'danger', 'threat', 'harassment', 'assault'
        ]

        self.crisis_resources = {
            'national_suicide_prevention': '988',
            'crisis_text_line': 'Text HOME to 741741',
            'emergency_services': '911',
            'veterans_crisis': '988 then press 1',
            'lifeline': '1-800-273-8255',
            'trevor_project': '1-866-488-7386',
            'domestic_violence': '1-800-799-7233'
        }

        # Command execution handlers
        self.command_handlers: Dict[str, Callable] = {}
        self.emergency_handlers: Dict[str, Callable] = {}

        # Wake word detection
        self.wake_word_enabled = config.voice_commands_enabled
        self.wake_word = getattr(config, 'voice_command_wake_word', 'hey therapist').lower()
        self.wake_word_detected = False
        self.last_wake_word_time = 0
        self.wake_word_timeout = getattr(config, 'voice_command_timeout', 30000) / 1000

        # Command cooldown tracking
        self.last_command_times: Dict[str, float] = {}

        # Context awareness
        self.conversation_context: Dict[str, Any] = {}
        self.session_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}

        # Performance metrics
        self.metrics = {
            'total_commands_processed': 0,
            'successful_commands': 0,
            'emergency_commands_triggered': 0,
            'average_confidence': 0.0,
            'average_response_time': 0.0
        }

        # Logging and auditing
        self.audit_log: List[Dict[str, Any]] = []
        self.command_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'executions': 0,
            'success_rate': 0.0,
            'average_confidence': 0.0,
            'last_used': None
        })

        # Initialize commands
        self._initialize_default_commands()
        self._register_command_handlers()
        self._register_emergency_handlers()

        self.logger.info(f"Voice command processor initialized with {len(self.commands)} commands")

    def _register_emergency_handlers(self):
        """Register emergency-specific handlers."""
        self.emergency_handlers = {
            "suicide_prevention": self._handle_suicide_prevention,
            "crisis_intervention": self._handle_crisis_intervention,
            "emergency_contact": self._handle_emergency_contact,
            "immediate_danger": self._handle_immediate_danger
        }

    def _initialize_default_commands(self):
        """Initialize comprehensive default voice commands."""
        default_commands = [
            # Navigation Commands
            VoiceCommand(
                name="go_home",
                description="Navigate to the home page",
                category=CommandCategory.NAVIGATION,
                patterns=[
                    r"(go to|navigate to|take me to) (the )?home",
                    r"(go|back|return) home",
                    r"(show|display) home page",
                    r"(main|start) page"
                ],
                action="navigate_home",
                examples=["Go home", "Take me to home", "Back to home page"],
                priority=8,
                require_wake_word=False
            ),
            VoiceCommand(
                name="get_help",
                description="Get help with voice commands and app features",
                category=CommandCategory.HELP,
                patterns=[
                    r"(help|what can you do)",
                    r"(what are the|list) (commands|voice commands)",
                    r"(how do|how to) (use|work with) voice commands",
                    r"(tell me|show me) (the )?commands",
                    r"(I need|i need) (assistance|help) (with|using) (the )?app"
                ],
                action="show_help",
                examples=["Help", "What can you do?", "List voice commands", "How do I use voice commands?"],
                priority=9,
                require_wake_word=False
            ),
            VoiceCommand(
                name="open_settings",
                description="Open settings and preferences",
                category=CommandCategory.SETTINGS,
                patterns=[
                    r"(open|go to|show) settings",
                    r"(settings|preferences|configuration)",
                    r"(change|modify) settings",
                    r"(voice|audio) settings"
                ],
                action="open_settings",
                examples=["Open settings", "Show preferences", "Change voice settings"],
                security_level=SecurityLevel.MEDIUM,
                priority=7
            ),

            # Session Control Commands
            VoiceCommand(
                name="start_session",
                description="Start a new therapy session",
                category=CommandCategory.SESSION_CONTROL,
                patterns=[
                    r"(start|begin) (a )?(new )?(therapy )?session",
                    r"(let's|let us) (talk|chat)",
                    r"(i need|I need) (to talk|to chat)",
                    r"hello therapist",
                    r"hi therapist",
                    r"(new|fresh) (conversation|session)"
                ],
                action="start_session",
                examples=["Start a new session", "Let's talk", "I need to talk to someone", "Hello therapist"],
                response_templates=[
                    "I'm here to listen. How are you feeling today?",
                    "I'm ready to help. What's on your mind?",
                    "Welcome! Let's start our session together."
                ],
                priority=10
            ),
            VoiceCommand(
                name="end_session",
                description="End the current therapy session",
                category=CommandCategory.SESSION_CONTROL,
                patterns=[
                    r"(end|stop|finish) (the )?(therapy )?session",
                    r"(that's|that is) (all|everything) (for now|today)",
                    r"(goodbye|bye|see you) (therapist|later)",
                    r"(thanks|thank you) (for|for your) help",
                    r"(we're|we are) (done|finished)",
                    r"(close|end) (our )?(conversation|session)"
                ],
                action="end_session",
                examples=["End the session", "That's all for today", "Goodbye therapist", "Thank you for your help"],
                response_templates=[
                    "Thank you for sharing with me today. Remember, I'm here whenever you need to talk.",
                    "It was good talking with you. Take care of yourself.",
                    "Our session is ending, but I'm here for you anytime."
                ],
                priority=8
            ),
            VoiceCommand(
                name="clear_conversation",
                description="Clear the current conversation history",
                category=CommandCategory.SESSION_CONTROL,
                patterns=[
                    r"(clear|delete|remove) (the )?(conversation|chat|history)",
                    r"(start|begin) (over|fresh|new)",
                    r"(erase|wipe) (our )?(conversation|history)",
                    r"(new|clean) slate"
                ],
                action="clear_conversation",
                examples=["Clear conversation", "Delete history", "Start over", "Clear chat"],
                security_level=SecurityLevel.MEDIUM,
                priority=6
            ),

            # Emergency Commands
            VoiceCommand(
                name="emergency_help",
                description="Get immediate emergency help and crisis resources",
                category=CommandCategory.EMERGENCY,
                patterns=[
                    r"(emergency|crisis|help me|I need help)",
                    r"(i'm|i am) (in crisis|having a crisis|feeling suicidal)",
                    r"(call|contact) (help|emergency|someone)",
                    r"(need|require) (immediate|urgent) (help|assistance)",
                    r"(i want|i want to) (die|kill myself|end my life)",
                    r"(i can't|i cannot) (go on|take it anymore)"
                ],
                action="emergency_response",
                examples=["Emergency help", "I'm in crisis", "I need immediate help", "Call emergency services"],
                security_level=SecurityLevel.HIGH,
                priority=100,  # Highest priority
                require_wake_word=False,
                emergency_keywords=self.emergency_keywords,
                cooldown=0  # No cooldown for emergencies
            ),
            VoiceCommand(
                name="call_crisis_line",
                description="Get crisis hotline numbers and resources",
                category=CommandCategory.EMERGENCY,
                patterns=[
                    r"(call|dial) (crisis line|hotline|help line)",
                    r"(crisis|suicide|emergency) (hotline|line|number)",
                    r"(need|want) crisis (resources|help)",
                    r"(show|give me) crisis (numbers|contacts)"
                ],
                action="provide_crisis_resources",
                examples=["Call crisis line", "Give me crisis hotline numbers", "Show crisis resources"],
                security_level=SecurityLevel.HIGH,
                priority=90,
                require_wake_word=False
            ),

            # Feature Access Commands
            VoiceCommand(
                name="start_meditation",
                description="Start a guided meditation session",
                category=CommandCategory.MEDITATION,
                patterns=[
                    r"(start|begin) (a )?(guided )?meditation",
                    r"(meditate|meditation|mindfulness)",
                    r"(breathing|relaxation) exercise",
                    r"(calm|relax) (down|me|my mind)",
                    r"(stress|anxiety) relief"
                ],
                action="start_meditation",
                examples=["Start meditation", "Guided meditation", "Breathing exercise", "Help me relax"],
                response_templates=[
                    "Let's begin a meditation exercise. Find a comfortable position and close your eyes.",
                    "I'll guide you through a relaxation exercise. Take a deep breath in...",
                    "Let's practice mindfulness together. Focus on your breathing..."
                ],
                priority=15
            ),
            VoiceCommand(
                name="open_journal",
                description="Open the journal feature for self-reflection",
                category=CommandCategory.JOURNAL,
                patterns=[
                    r"(open|start|begin) journal",
                    r"(journal|diary|log) (my|our) thoughts",
                    r"(write|record) (in|to) journal",
                    r"(reflect|reflection) time",
                    r"(thoughts|feelings) journal"
                ],
                action="open_journal",
                examples=["Open journal", "Start journaling", "Write in my journal", "Reflection time"],
                response_templates=[
                    "I'll open your journal. What thoughts or feelings would you like to record today?",
                    "Let's do some journaling together. What's on your mind?",
                    "Your journal is ready. Take a moment to reflect on your experiences."
                ],
                priority=12
            ),
            VoiceCommand(
                name="show_resources",
                description="Show therapy resources and educational materials",
                category=CommandCategory.RESOURCES,
                patterns=[
                    r"(show|display|view) resources",
                    r"(therapy|mental health) resources",
                    r"(educational|learning) materials",
                    r"(helpful|useful) (resources|tools)",
                    r"(reading|articles|worksheets)"
                ],
                action="show_resources",
                examples=["Show resources", "Therapy resources", "Educational materials", "Helpful tools"],
                priority=10
            ),

            # Voice Control Commands
            VoiceCommand(
                name="pause_conversation",
                description="Pause the conversation temporarily",
                category=CommandCategory.VOICE_CONTROL,
                patterns=[
                    r"(pause|hold on|wait) (a )?(moment|second|minute)",
                    r"(give me|let me) (think|a moment)",
                    r"(one )?moment please",
                    r"can you (wait|hold on)",
                    r"(stop|pause) (for|here)"
                ],
                action="pause_conversation",
                examples=["Pause for a moment", "Hold on a second", "Let me think", "Can you wait a moment?"],
                priority=20
            ),
            VoiceCommand(
                name="resume_conversation",
                description="Resume the paused conversation",
                category=CommandCategory.VOICE_CONTROL,
                patterns=[
                    r"(resume|continue) (the )?(conversation|session)",
                    r"(i'm|i am) (ready|back)",
                    r"(okay|ok) (let's|let us) (continue|go on)",
                    r"(you can|you can) (continue|go on)",
                    r"(start|begin) again"
                ],
                action="resume_conversation",
                examples=["Resume the conversation", "I'm ready to continue", "Okay, let's continue", "You can go on now"],
                priority=20
            ),
            VoiceCommand(
                name="repeat_last_response",
                description="Repeat the last AI response",
                category=CommandCategory.VOICE_CONTROL,
                patterns=[
                    r"(repeat|say again) (that|what you said)",
                    r"(can you|could you) (repeat|say that again)",
                    r"(what did you say|what was that)",
                    r"(i didn't|I did not) (hear|catch) (that|you)",
                    r"(again|once more)"
                ],
                action="repeat_last_response",
                examples=["Repeat that please", "Can you say that again?", "What did you say?", "I didn't catch that"],
                priority=25
            ),
            VoiceCommand(
                name="speak_slower",
                description="Ask the AI to speak more slowly",
                category=CommandCategory.VOICE_CONTROL,
                patterns=[
                    r"speak (more )?(slowly|slower)",
                    r"(can you|could you) (talk|speak) (more )?(slowly|slower)",
                    r"(slow|slow down) (a bit|please)",
                    r"(too fast|you're too fast)",
                    r"(reduce|lower) (your )?speed"
                ],
                action="adjust_speech_speed",
                parameters={"speed": "slower"},
                examples=["Speak more slowly please", "Can you talk slower?", "Slow down a bit", "You're talking too fast"],
                priority=15
            ),
            VoiceCommand(
                name="speak_faster",
                description="Ask the AI to speak more quickly",
                category=CommandCategory.VOICE_CONTROL,
                patterns=[
                    r"speak (more )?(quickly|faster)",
                    r"(can you|could you) (talk|speak) (more )?(quickly|faster)",
                    r"(speed up|faster) (a bit|please)",
                    r"(too slow|you're too slow)",
                    r"(increase|raise) (your )?speed"
                ],
                action="adjust_speech_speed",
                parameters={"speed": "faster"},
                examples=["Speak more quickly", "Can you talk faster?", "Speed up a bit", "You're talking too slow"],
                priority=15
            ),
            VoiceCommand(
                name="adjust_volume",
                description="Adjust the audio volume",
                category=CommandCategory.VOICE_CONTROL,
                patterns=[
                    r"(volume|sound) (up|down|louder|softer)",
                    r"(turn|make) it (louder|softer|quieter)",
                    r"(increase|decrease) (the )?volume",
                    r"(can you|could you) (speak|talk) (louder|softer)",
                    r"(volume) (level|control)"
                ],
                action="adjust_volume",
                examples=["Volume up", "Make it louder", "Turn it down", "Speak softer"],
                priority=12
            ),

            # Voice Profile Commands
            VoiceCommand(
                name="change_voice",
                description="Change the AI voice profile",
                category=CommandCategory.SETTINGS,
                patterns=[
                    r"(change|switch) (your )?voice",
                    r"(can you|could you) (use|speak with) (a )?(different|new) voice",
                    r"(i'd|I would) (like|prefer) (a )?(different|new) voice",
                    r"(voice|voice profile) (change|switch)",
                    r"(male|female|calm|professional|empathetic) voice"
                ],
                action="change_voice_profile",
                examples=["Change your voice", "Can you use a different voice?", "I'd prefer a different voice", "Switch voice profile"],
                security_level=SecurityLevel.MEDIUM,
                priority=8
            ),

            # System Commands
            VoiceCommand(
                name="status_check",
                description="Check system status and health",
                category=CommandCategory.SETTINGS,
                patterns=[
                    r"(status|how are you|are you working)",
                    r"(system|service) status",
                    r"(everything|all) (okay|working|fine)",
                    r"(check|test) (system|status)",
                    r"(are you|you) (online|available|working)"
                ],
                action="check_status",
                examples=["Status check", "How are you working?", "System status", "Is everything okay?"],
                priority=5
            )
        ]

        # Register default commands
        for command in default_commands:
            self.register_command(command)

    def _register_command_handlers(self):
        """Register command execution handlers."""
        self.command_handlers = {
            # Navigation commands
            "navigate_home": self._handle_navigate_home,
            "open_settings": self._handle_open_settings,

            # Session control commands
            "start_session": self._handle_start_session,
            "end_session": self._handle_end_session,
            "clear_conversation": self._handle_clear_conversation,
            "pause_conversation": self._handle_pause_conversation,
            "resume_conversation": self._handle_resume_conversation,

            # Emergency commands
            "emergency_response": self._handle_emergency_response,
            "provide_crisis_resources": self._handle_provide_crisis_resources,

            # Feature access commands
            "start_meditation": self._handle_start_meditation,
            "open_journal": self._handle_open_journal,
            "show_resources": self._handle_show_resources,

            # Voice control commands
            "repeat_last_response": self._handle_repeat_last_response,
            "adjust_speech_speed": self._handle_adjust_speech_speed,
            "adjust_volume": self._handle_adjust_volume,
            "change_voice_profile": self._handle_change_voice_profile,

            # Help and system commands
            "show_help": self._handle_show_help,
            "check_status": self._handle_check_status
        }

    def register_command(self, command: VoiceCommand):
        """Register a new voice command."""
        try:
            self.commands[command.name] = command
            self.commands_by_category[command.category].append(command.name)
            self.logger.info(f"Registered voice command: {command.name} in category {command.category.value}")
        except Exception as e:
            self.logger.error(f"Error registering command {command.name}: {str(e)}")

    def unregister_command(self, command_name: str):
        """Unregister a voice command."""
        try:
            if command_name in self.commands:
                del self.commands[command_name]
                self.logger.info(f"Unregistered voice command: {command_name}")
        except Exception as e:
            self.logger.error(f"Error unregistering command {command_name}: {str(e)}")

    def register_command_handler(self, command_name: str, handler: Callable):
        """Register a custom command handler."""
        try:
            self.command_handlers[command_name] = handler
            self.logger.info(f"Registered handler for command: {command_name}")
        except Exception as e:
            self.logger.error(f"Error registering handler for {command_name}: {str(e)}")

    def get_available_commands(self) -> List[Dict[str, Any]]:
        """Get list of available commands."""
        commands_list = []
        for command in self.commands.values():
            if command.enabled:
                commands_list.append({
                    'name': command.name,
                    'description': command.description,
                    'patterns': command.patterns,
                    'action': command.action,
                    'examples': command.examples or [],
                    'security_level': command.security_level
                })
        return commands_list

    async def process_audio(self, audio_data: AudioData) -> Optional[CommandMatch]:
        """Process audio data for voice commands."""
        if not self.config.voice_commands_enabled:
            return None

        try:
            # First, check for wake word if needed
            if self.wake_word_enabled and not self.wake_word_detected:
                if await self._detect_wake_word(audio_data):
                    self.wake_word_detected = True
                    self.last_wake_word_time = time.time()
                    return CommandMatch(
                        command=VoiceCommand(
                            name="wake_word",
                            description="Wake word detected",
                            patterns=[],
                            action="wake_word_detected"
                        ),
                        confidence=0.9,
                        parameters={},
                        matched_text=self.wake_word,
                        timestamp=time.time()
                    )

            # Check if wake word is still active
            if self.wake_word_enabled and self.wake_word_detected:
                if time.time() - self.last_wake_word_time > self.wake_word_timeout:
                    self.wake_word_detected = False
                    return None

            # Process for commands
            # This would typically involve running STT on the audio
            # For now, we'll return None (in a real implementation, this would call STT)
            return None

        except Exception as e:
            self.logger.error(f"Error processing audio for commands: {str(e)}")
            return None

    async def process_text(self, text: str, session_id: str = None) -> Optional[CommandMatch]:
        """Enhanced text processing for voice commands with context awareness."""
        if not self.config.voice_commands_enabled:
            return None

        try:
            # Check for emergency keywords immediately (bypass wake word requirement)
            detected_emergency_keywords = self._detect_emergency_keywords(text)
            if detected_emergency_keywords:
                self.logger.warning(f"Emergency keywords detected: {detected_emergency_keywords}")
                # Emergency commands don't require wake word
                return await self._match_command(text, session_id)

            # Check for wake word if needed
            if self.wake_word_enabled and not self.wake_word_detected:
                if self._check_wake_word_in_text(text):
                    self.wake_word_detected = True
                    self.last_wake_word_time = time.time()
                    return CommandMatch(
                        command=VoiceCommand(
                            name="wake_word",
                            description="Wake word detected",
                            category=CommandCategory.HELP,
                            patterns=[],
                            action="wake_word_detected",
                            security_level=SecurityLevel.LOW,
                            priority=5
                        ),
                        confidence=0.9,
                        parameters={},
                        matched_text=self.wake_word,
                        timestamp=time.time(),
                        session_id=session_id
                    )

            # Check if wake word is still active
            if self.wake_word_enabled and self.wake_word_detected:
                if time.time() - self.last_wake_word_time > self.wake_word_timeout:
                    self.wake_word_detected = False

            # Match commands with enhanced processing
            return await self._match_command(text, session_id)

        except Exception as e:
            self.logger.error(f"Error processing text for commands: {str(e)}")
            return None

    async def _detect_wake_word(self, audio_data: AudioData) -> bool:
        """Detect wake word in audio data."""
        # This would use actual wake word detection (like Porcupine or similar)
        # For now, we'll implement a simple text-based check that would work with STT
        return False

    def _check_wake_word_in_text(self, text: str) -> bool:
        """Check if wake word is in text."""
        return self.wake_word in text.lower()

    async def _match_command(self, text: str, session_id: str = None) -> Optional[CommandMatch]:
        """Enhanced command matching with context awareness and NLP."""
        start_time = time.time()
        text_lower = text.lower()
        best_match = None
        best_confidence = 0.0
        alternative_matches = []

        # Check for emergency keywords first (highest priority)
        detected_emergency_keywords = self._detect_emergency_keywords(text)
        is_emergency = len(detected_emergency_keywords) > 0

        # Sort commands by priority and emergency status
        sorted_commands = sorted(
            self.commands.values(),
            key=lambda cmd: (
                0 if cmd.category == CommandCategory.EMERGENCY and is_emergency else 1,
                -cmd.priority,
                cmd.name
            )
        )

        for command in sorted_commands:
            if not command.enabled:
                continue

            # Check cooldown (except for emergencies)
            if command.cooldown > 0 and command.category != CommandCategory.EMERGENCY:
                last_time = self.last_command_times.get(command.name, 0)
                if time.time() - last_time < command.cooldown:
                    continue

            # Match against patterns
            for pattern in command.patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    # Get context score if session_id is provided
                    context_score = 0.0
                    if session_id and command.context_aware:
                        context_score = await self._get_context_score(session_id, command)

                    # Calculate enhanced confidence
                    confidence = await self._calculate_enhanced_confidence(
                        match, text_lower, command, context_score
                    )

                    # Emergency bonus for emergency commands with detected keywords
                    if command.category == CommandCategory.EMERGENCY and is_emergency:
                        confidence = min(confidence + 0.3, 1.0)

                    if confidence >= command.confidence_threshold:
                        # Extract enhanced parameters
                        parameters = await self._extract_enhanced_parameters(match, command, text)

                        command_match = CommandMatch(
                            command=command,
                            confidence=confidence,
                            parameters=parameters,
                            matched_text=match.group(),
                            timestamp=time.time(),
                            context_score=context_score,
                            processing_time=time.time() - start_time,
                            session_id=session_id,
                            is_emergency=is_emergency,
                            crisis_keywords_detected=detected_emergency_keywords,
                            match_method="hybrid"
                        )

                        if confidence > best_confidence:
                            # Demote previous best match to alternatives
                            if best_match:
                                alternative_matches.append({
                                    'command': best_match.command.name,
                                    'confidence': best_match.confidence,
                                    'reason': 'lower confidence'
                                })

                            best_confidence = confidence
                            best_match = command_match
                        else:
                            alternative_matches.append({
                                'command': command.name,
                                'confidence': confidence,
                                'reason': 'lower confidence than best match'
                            })

        if best_match:
            # Update alternative matches
            best_match.alternative_matches = alternative_matches[:5]  # Keep top 5 alternatives

            # Update last command time (except for emergencies which can be repeated)
            if best_match.command.category != CommandCategory.EMERGENCY:
                self.last_command_times[best_match.command.name] = time.time()

            # Add to history
            self.command_history.append(best_match)

            # Keep history manageable
            if len(self.command_history) > 100:
                self.command_history = self.command_history[-50:]

            # Update metrics
            self.metrics['total_commands_processed'] += 1
            self.metrics['average_confidence'] = (
                (self.metrics['average_confidence'] * (self.metrics['total_commands_processed'] - 1) + best_confidence) /
                self.metrics['total_commands_processed']
            )

        return best_match

    def _calculate_confidence(self, match: re.Match, text: str, pattern: str) -> float:
        """Calculate confidence score for command match."""
        try:
            # Base confidence from match quality
            matched_length = len(match.group())
            text_length = len(text)
            coverage = matched_length / text_length if text_length > 0 else 0

            # Pattern specificity (longer patterns are more specific)
            pattern_specificity = min(len(pattern) / 20, 1.0)  # Normalize to 0-1

            # Word boundary matches are more confident
            has_word_boundaries = bool(match.re.pattern.startswith(r'\b') or
                                    match.re.pattern.endswith(r'\b'))

            # Calculate overall confidence
            confidence = coverage * 0.5 + pattern_specificity * 0.3 + (0.2 if has_word_boundaries else 0)

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _extract_parameters(self, match: re.Match, command: VoiceCommand) -> Dict[str, Any]:
        """Extract parameters from command match."""
        parameters = {}

        try:
            # Extract named groups from regex match
            if match.groupdict():
                parameters.update(match.groupdict())

            # Add default parameters from command definition
            if command.parameters:
                parameters.update(command.parameters)

            # Extract specific parameters based on command type
            if command.name == "change_voice":
                # Try to extract voice preference
                voice_match = re.search(r'(male|female|calm|empathetic|professional)', match.group(), re.IGNORECASE)
                if voice_match:
                    parameters['voice_type'] = voice_match.group(1).lower()

            elif command.name == "adjust_volume":
                # Extract direction
                if any(word in match.group().lower() for word in ['up', 'louder', 'increase']):
                    parameters['direction'] = 'up'
                elif any(word in match.group().lower() for word in ['down', 'softer', 'decrease']):
                    parameters['direction'] = 'down'

            elif command.name == "adjust_speech_speed":
                # Speed parameter already set in command definition
                pass

        except Exception as e:
            self.logger.error(f"Error extracting parameters: {str(e)}")

        return parameters

    async def execute_command(self, command_match: CommandMatch) -> Dict[str, Any]:
        """Enhanced command execution with comprehensive logging and analytics."""
        start_time = time.time()
        execution_success = False
        result = None

        try:
            command = command_match.command
            handler = self.command_handlers.get(command.action)

            # Enhanced logging with session context
            log_data = {
                'command_name': command.name,
                'action': command.action,
                'category': command.category.value,
                'parameters': command_match.parameters,
                'confidence': command_match.confidence,
                'session_id': command_match.session_id,
                'is_emergency': command_match.is_emergency,
                'crisis_keywords': command_match.crisis_keywords_detected,
                'timestamp': start_time
            }

            if handler:
                # Log command execution start
                self.logger.info(f"Executing command: {command.name} (category: {command.category.value})")

                # For emergency commands, log with higher severity
                if command.category == CommandCategory.EMERGENCY:
                    self.logger.warning(f"EMERGENCY COMMAND EXECUTED: {command.name} - {command_match.crisis_keywords_detected}")

                # Execute handler
                result = await handler(command_match.parameters)
                execution_success = True

                # Update metrics and statistics
                self.metrics['successful_commands'] += 1
                self._update_command_statistics(command.name, True, command_match.confidence)

                # Log successful execution
                log_data['success'] = True
                log_data['processing_time'] = time.time() - start_time
                self.audit_log.append(log_data)

                return {
                    'command': command.name,
                    'action': command.action,
                    'category': command.category.value,
                    'parameters': command_match.parameters,
                    'result': result,
                    'success': True,
                    'confidence': command_match.confidence,
                    'processing_time': time.time() - start_time,
                    'is_emergency': command_match.is_emergency,
                    'crisis_keywords_detected': command_match.crisis_keywords_detected,
                    'timestamp': time.time()
                }
            else:
                self.logger.warning(f"No handler found for command: {command.action}")
                self._update_command_statistics(command.name, False, command_match.confidence)

                # Log handler not found
                log_data['success'] = False
                log_data['error'] = 'No handler found'
                log_data['processing_time'] = time.time() - start_time
                self.audit_log.append(log_data)

                return {
                    'command': command.name,
                    'action': command.action,
                    'category': command.category.value,
                    'parameters': command_match.parameters,
                    'result': None,
                    'success': False,
                    'error': 'No handler found',
                    'confidence': command_match.confidence,
                    'processing_time': time.time() - start_time,
                    'timestamp': time.time()
                }

        except Exception as e:
            self.logger.error(f"Error executing command {command.name}: {str(e)}")
            self._update_command_statistics(command.name, False, command_match.confidence)

            # Log exception
            log_data['success'] = False
            log_data['error'] = str(e)
            log_data['processing_time'] = time.time() - start_time
            self.audit_log.append(log_data)

            # For emergency commands, ensure error is logged with highest severity
            if command_match.is_emergency:
                self.logger.critical(f"EMERGENCY COMMAND FAILED: {command.name} - Error: {str(e)}")

            return {
                'command': command.name,
                'action': command.action,
                'category': command.category.value,
                'parameters': command_match.parameters,
                'result': None,
                'success': False,
                'error': str(e),
                'confidence': command_match.confidence,
                'processing_time': time.time() - start_time,
                'is_emergency': command_match.is_emergency,
                'timestamp': time.time()
            }

    # Command handlers
    async def _handle_start_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start session command."""
        return {
            'message': 'Starting new therapy session',
            'action': 'start_session',
            'voice_feedback': "I'm here to listen. How are you feeling today?"
        }

    async def _handle_end_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle end session command."""
        return {
            'message': 'Ending therapy session',
            'action': 'end_session',
            'voice_feedback': "Thank you for sharing with me today. Remember, I'm here whenever you need to talk."
        }

    async def _handle_pause_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pause conversation command."""
        return {
            'message': 'Conversation paused',
            'action': 'pause_conversation',
            'voice_feedback': "I'll pause here. Take your time."
        }

    async def _handle_resume_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resume conversation command."""
        return {
            'message': 'Conversation resumed',
            'action': 'resume_conversation',
            'voice_feedback': "Welcome back. Where would you like to continue?"
        }

    async def _handle_repeat_last_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repeat last response command."""
        return {
            'message': 'Repeating last response',
            'action': 'repeat_last_response',
            'voice_feedback': None  # Will be handled by the voice service
        }

    async def _handle_adjust_speech_speed(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle adjust speech speed command."""
        speed = parameters.get('speed', 'normal')
        return {
            'message': f'Adjusted speech speed to {speed}',
            'action': 'adjust_speech_speed',
            'parameters': parameters,
            'voice_feedback': f"I'll speak more {speed} for you."
        }

    async def _handle_change_voice_profile(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle change voice profile command."""
        voice_type = parameters.get('voice_type', 'default')
        return {
            'message': f'Changed voice profile to {voice_type}',
            'action': 'change_voice_profile',
            'parameters': parameters,
            'voice_feedback': f"I've changed my voice to {voice_type}. How does this sound?"
        }

    async def _handle_show_help(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle show help command."""
        commands = self.get_available_commands()
        command_list = "\n".join([f"• {cmd['name']}: {cmd['description']}" for cmd in commands[:5]])

        return {
            'message': 'Showing available commands',
            'action': 'show_help',
            'voice_feedback': f"Here are some commands I can understand: {command_list}"
        }

  
    async def _handle_adjust_volume(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle adjust volume command."""
        direction = parameters.get('direction', 'up')
        return {
            'message': f'Adjusted volume {direction}',
            'action': 'adjust_volume',
            'parameters': parameters,
            'voice_feedback': f"I've turned the volume {direction}."
        }

    async def _handle_check_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle check status command."""
        return {
            'message': 'System status check',
            'action': 'check_status',
            'voice_feedback': "I'm working normally and ready to help you."
        }

    # New command handlers for enhanced functionality
    async def _handle_navigate_home(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigate home command."""
        return {
            'message': 'Navigating to home page',
            'action': 'navigate_home',
            'voice_feedback': "Taking you to the home page."
        }

    async def _handle_open_settings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle open settings command."""
        return {
            'message': 'Opening settings',
            'action': 'open_settings',
            'voice_feedback': "Opening settings page where you can customize voice and app preferences."
        }

    async def _handle_clear_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear conversation command."""
        return {
            'message': 'Clearing conversation history',
            'action': 'clear_conversation',
            'voice_feedback': "I've cleared our conversation history. We can start fresh whenever you're ready."
        }

    async def _handle_provide_crisis_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle provide crisis resources command."""
        resources_text = "Here are important crisis resources:\n"
        for name, contact in self.crisis_resources.items():
            resources_text += f"• {name.replace('_', ' ').title()}: {contact}\n"

        return {
            'message': 'Providing crisis resources',
            'action': 'provide_crisis_resources',
            'voice_feedback': "I'll provide you with important crisis resources. These are available 24/7 when you need immediate help.",
            'resources': self.crisis_resources
        }

    async def _handle_start_meditation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start meditation command."""
        meditation_guidance = "Let's begin a mindfulness exercise. Find a comfortable position and close your eyes. Take a deep breath in through your nose, hold for a moment, and exhale slowly through your mouth. Focus on the sensation of your breathing. Notice how your body feels with each breath. If your mind wanders, gently bring it back to your breathing. Continue this for a few minutes, allowing yourself to be present in this moment."

        return {
            'message': 'Starting guided meditation',
            'action': 'start_meditation',
            'voice_feedback': meditation_guidance,
            'duration': '5-10 minutes'
        }

    async def _handle_open_journal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle open journal command."""
        return {
            'message': 'Opening journal feature',
            'action': 'open_journal',
            'voice_feedback': "Your journal is ready. What thoughts or feelings would you like to record today? You can speak naturally, and I'll help you document your reflections."
        }

    async def _handle_show_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle show resources command."""
        return {
            'message': 'Showing therapy resources',
            'action': 'show_resources',
            'voice_feedback': "I'll show you our collection of therapy resources including worksheets, articles, and educational materials to support your mental health journey."
        }

    # Enhanced emergency response handlers
    async def _handle_emergency_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency response using app's crisis detection system."""
        # Log emergency event
        await self._log_emergency_event("emergency_command_triggered", parameters)

        # Use the main app's crisis response for consistency
        try:
            crisis_response = generate_crisis_response()
            return {
                'message': 'Emergency response activated',
                'action': 'emergency_response',
                'voice_feedback': crisis_response,
                'resources': self.crisis_resources,
                'severity': 'critical'
            }
        except Exception:
            # Fallback response
            return {
                'message': 'Emergency response activated',
                'action': 'emergency_response',
                'voice_feedback': "I understand you need immediate help. Please call the National Suicide Prevention Lifeline at 988, or dial 911 for emergency services. Your safety is important.",
                'resources': self.crisis_resources,
                'severity': 'critical'
            }

    async def _handle_suicide_prevention(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle suicide prevention emergency."""
        await self._log_emergency_event("suicide_prevention_activated", parameters)

        response = "I'm deeply concerned about your safety right now. Your life is valuable and there are people who want to help you. Please call the National Suicide Prevention Lifeline at 988, or dial 911 for immediate emergency services. You can also text HOME to 741741 to connect with a crisis counselor. You are not alone, and help is available 24/7. Please stay on the line with me until you've connected with help."

        return {
            'voice_feedback': response,
            'resources': {
                'national_suicide_prevention': '988',
                'crisis_text_line': 'Text HOME to 741741',
                'emergency_services': '911'
            },
            'immediate_actions': [
                'Call 988 or 911 immediately',
                'Stay on the line until connected with help',
                'Remove any harmful items from your area'
            ]
        }

    async def _handle_crisis_intervention(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general crisis intervention."""
        await self._log_emergency_event("crisis_intervention_activated", parameters)

        response = "I understand you're going through a difficult time. Help is available. Please reach out to the Crisis Text Line by texting HOME to 741741, or call 988 to speak with someone who can support you. You don't have to go through this alone."

        return {
            'voice_feedback': response,
            'resources': self.crisis_resources,
            'immediate_actions': [
                'Text HOME to 741741',
                'Call 988 for support',
                'Reach out to a trusted friend or family member'
            ]
        }

    async def _handle_emergency_contact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency contact request."""
        await self._log_emergency_event("emergency_contact_requested", parameters)

        response = "For immediate emergency assistance, please call 911. For mental health crisis support, call 988. These services are available 24/7 and can provide immediate help."

        return {
            'voice_feedback': response,
            'resources': {
                'emergency_services': '911',
                'crisis_line': '988'
            },
            'immediate_actions': [
                'Call 911 for life-threatening emergencies',
                'Call 988 for mental health crises'
            ]
        }

    async def _handle_immediate_danger(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle immediate danger situations."""
        await self._log_emergency_event("immediate_danger_detected", parameters)

        response = "If you are in immediate danger, please call 911 right away. Your safety is the top priority. Do not wait - call emergency services immediately."

        return {
            'voice_feedback': response,
            'resources': {'emergency_services': '911'},
            'immediate_actions': [
                'Call 911 immediately',
                'Get to a safe location if possible',
                'Stay on the line until help arrives'
            ]
        }

    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history."""
        recent_history = self.command_history[-limit:]
        return [
            {
                'command': match.command.name,
                'confidence': match.confidence,
                'matched_text': match.matched_text,
                'timestamp': match.timestamp
            }
            for match in recent_history
        ]

    def clear_command_history(self):
        """Clear command history."""
        self.command_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get command processor statistics."""
        return {
            'total_commands': len(self.commands),
            'enabled_commands': len([c for c in self.commands.values() if c.enabled]),
            'command_history_size': len(self.command_history),
            'wake_word_enabled': self.wake_word_enabled,
            'wake_word_detected': self.wake_word_detected,
            'last_wake_word_time': self.last_wake_word_time,
            'min_confidence': self.config.voice_command_min_confidence
        }

    # Enhanced emergency detection and logging methods
    def _detect_emergency_keywords(self, text: str) -> List[str]:
        """Detect emergency keywords in text using app's crisis detection."""
        try:
            # Use the main app's crisis detection for consistency
            is_crisis, crisis_keywords = detect_crisis_content(text)
            return crisis_keywords
        except Exception:
            # Fallback to internal detection
            detected = []
            text_lower = text.lower()
            for keyword in self.emergency_keywords:
                if keyword in text_lower:
                    detected.append(keyword)
            return detected

    def _classify_emergency_type(self, keywords: List[str]) -> str:
        """Classify emergency type based on detected keywords."""
        if not keywords:
            return "crisis_intervention"

        suicide_keywords = ['suicide', 'kill myself', 'end my life', 'want to die',
                          'don\'t want to live', 'end it all', 'no reason to live']
        danger_keywords = ['danger', 'violence', 'threat', 'abuse', 'assault', 'harassment']

        if any(keyword in keywords for keyword in suicide_keywords):
            return "suicide_prevention"
        elif any(keyword in keywords for keyword in danger_keywords):
            return "immediate_danger"
        else:
            return "crisis_intervention"

    async def _log_emergency_event(self, event_type: str, parameters: Dict[str, Any]):
        """Log emergency event with comprehensive details."""
        try:
            event = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'parameters': parameters,
                'emergency_keywords': self._detect_emergency_keywords(parameters.get('matched_text', '')),
                'session_id': parameters.get('session_id'),
                'user_id': parameters.get('user_id'),
                'severity': 'critical'
            }

            # Add to audit log
            self.audit_log.append(event)

            # Log to system logger
            self.logger.warning(f"EMERGENCY EVENT: {event_type} - {parameters}")

            # Update metrics
            self.metrics['emergency_commands_triggered'] += 1

        except Exception as e:
            self.logger.error(f"Error logging emergency event: {str(e)}")

    # Enhanced natural language processing methods
    async def _calculate_enhanced_confidence(self, match: re.Match, text: str, command: VoiceCommand,
                                           context_score: float = 0.0) -> float:
        """Calculate enhanced confidence score using multiple factors."""
        try:
            # Base confidence from match quality
            matched_length = len(match.group())
            text_length = len(text)
            coverage = matched_length / text_length if text_length > 0 else 0

            # Pattern specificity
            pattern_specificity = min(len(match.re.pattern) / 20, 1.0)

            # Word boundary matches
            has_word_boundaries = bool(match.re.pattern.startswith(r'\b') or
                                    match.re.pattern.endswith(r'\b'))

            # Command priority bonus
            priority_bonus = command.priority / 100.0

            # Context awareness bonus
            context_bonus = context_score * 0.3

            # Emergency detection bonus
            emergency_bonus = 0.2 if command.category == CommandCategory.EMERGENCY and any(
                keyword in text.lower() for keyword in self.emergency_keywords
            ) else 0.0

            # Historical accuracy for this command
            historical_accuracy = self.command_stats[command.name]['success_rate']

            # Calculate overall confidence
            confidence = (
                coverage * 0.3 +
                pattern_specificity * 0.2 +
                (0.15 if has_word_boundaries else 0) +
                priority_bonus * 0.1 +
                context_bonus +
                emergency_bonus +
                historical_accuracy * 0.15
            )

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {str(e)}")
            return 0.5

    async def _extract_enhanced_parameters(self, match: re.Match, command: VoiceCommand,
                                         text: str) -> Dict[str, Any]:
        """Extract enhanced parameters with natural language understanding."""
        parameters = {}

        try:
            # Extract named groups from regex match
            if match.groupdict():
                parameters.update(match.groupdict())

            # Add default parameters from command definition
            if command.parameters:
                parameters.update(command.parameters)

            # Enhanced parameter extraction based on command category
            if command.category == CommandCategory.VOICE_CONTROL:
                if command.name == "adjust_volume":
                    # Extract volume direction and magnitude
                    direction_params = self._extract_volume_parameters(text)
                    parameters.update(direction_params)

                elif command.name == "change_voice_profile":
                    # Extract voice characteristics
                    voice_params = self._extract_voice_parameters(text)
                    parameters.update(voice_params)

            elif command.category == CommandCategory.EMERGENCY:
                # Extract emergency context
                emergency_params = self._extract_emergency_parameters(text)
                parameters.update(emergency_params)

            elif command.category == CommandCategory.FEATURE_ACCESS:
                # Extract feature preferences
                if command.name == "start_meditation":
                    meditation_params = self._extract_meditation_parameters(text)
                    parameters.update(meditation_params)

        except Exception as e:
            self.logger.error(f"Error extracting enhanced parameters: {str(e)}")

        return parameters

    def _extract_volume_parameters(self, text: str) -> Dict[str, Any]:
        """Extract volume adjustment parameters."""
        params = {}
        text_lower = text.lower()

        # Direction
        if any(word in text_lower for word in ['up', 'louder', 'increase', 'higher']):
            params['direction'] = 'up'
        elif any(word in text_lower for word in ['down', 'softer', 'decrease', 'lower', 'quieter']):
            params['direction'] = 'down'

        # Magnitude
        if any(word in text_lower for word in ['a lot', 'much', 'way', 'really']):
            params['magnitude'] = 'large'
        elif any(word in text_lower for word in ['a bit', 'little', 'slightly']):
            params['magnitude'] = 'small'
        else:
            params['magnitude'] = 'medium'

        return params

    def _extract_voice_parameters(self, text: str) -> Dict[str, Any]:
        """Extract voice profile parameters."""
        params = {}
        text_lower = text.lower()

        # Voice type
        voice_types = {
            'male': 'male',
            'female': 'female',
            'calm': 'calm',
            'empathetic': 'empathetic',
            'professional': 'professional',
            'warm': 'warm',
            'soft': 'soft',
            'clear': 'clear'
        }

        for voice_type, param_value in voice_types.items():
            if voice_type in text_lower:
                params['voice_type'] = param_value
                break

        return params

    def _extract_emergency_parameters(self, text: str) -> Dict[str, Any]:
        """Extract emergency-related parameters."""
        params = {}
        text_lower = text.lower()

        # Detected emergency keywords
        params['emergency_keywords'] = self._detect_emergency_keywords(text)

        # Urgency level
        if any(word in text_lower for word in ['immediate', 'right now', 'urgent', 'asap']):
            params['urgency'] = 'high'
        elif any(word in text_lower for word in ['help', 'need', 'support']):
            params['urgency'] = 'medium'
        else:
            params['urgency'] = 'low'

        return params

    def _extract_meditation_parameters(self, text: str) -> Dict[str, Any]:
        """Extract meditation session parameters."""
        params = {}
        text_lower = text.lower()

        # Duration preference
        if any(word in text_lower for word in ['quick', 'short', 'brief']):
            params['duration'] = 'short'
        elif any(word in text_lower for word in ['long', 'extended', 'deep']):
            params['duration'] = 'long'
        else:
            params['duration'] = 'medium'

        # Type
        if any(word in text_lower for word in ['breathing', 'breath']):
            params['type'] = 'breathing'
        elif any(word in text_lower for word in ['mindfulness', 'mindful']):
            params['type'] = 'mindfulness'
        else:
            params['type'] = 'general'

        return params

    # Context awareness methods
    async def _update_conversation_context(self, session_id: str, context: Dict[str, Any]):
        """Update conversation context for better command understanding."""
        self.conversation_context[session_id] = context

    async def _get_context_score(self, session_id: str, command: VoiceCommand) -> float:
        """Calculate context relevance score for a command."""
        try:
            if session_id not in self.conversation_context:
                return 0.0

            context = self.conversation_context[session_id]
            score = 0.0

            # Check if command is relevant to current conversation state
            if 'current_activity' in context:
                activity = context['current_activity']
                if (activity == 'meditation' and command.category == CommandCategory.MEDITATION) or \
                   (activity == 'journaling' and command.category == CommandCategory.JOURNAL) or \
                   (activity == 'crisis' and command.category == CommandCategory.EMERGENCY):
                    score += 0.5

            # Check user preferences
            if 'user_preferences' in context:
                prefs = context['user_preferences']
                if command.name in prefs.get('frequent_commands', []):
                    score += 0.3

            # Time-based context
            if 'session_duration' in context:
                duration = context['session_duration']
                if duration > 300 and command.category == CommandCategory.SESSION_CONTROL:  # Long session
                    score += 0.2

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating context score: {str(e)}")
            return 0.0

    # Audit and analytics methods
    def _update_command_statistics(self, command_name: str, success: bool, confidence: float):
        """Update command execution statistics."""
        try:
            stats = self.command_stats[command_name]
            stats['executions'] += 1
            stats['last_used'] = time.time()

            # Update success rate
            if stats['executions'] > 1:
                stats['success_rate'] = ((stats['success_rate'] * (stats['executions'] - 1)) +
                                       (1 if success else 0)) / stats['executions']
            else:
                stats['success_rate'] = 1 if success else 0

            # Update average confidence
            if stats['executions'] > 1:
                stats['average_confidence'] = ((stats['average_confidence'] * (stats['executions'] - 1)) +
                                             confidence) / stats['executions']
            else:
                stats['average_confidence'] = confidence

        except Exception as e:
            self.logger.error(f"Error updating command statistics: {str(e)}")

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self.audit_log[-limit:]

    def get_command_analytics(self) -> Dict[str, Any]:
        """Get comprehensive command analytics."""
        try:
            analytics = {
                'total_commands': self.metrics['total_commands_processed'],
                'success_rate': self.metrics['successful_commands'] / max(self.metrics['total_commands_processed'], 1),
                'emergency_incidents': self.metrics['emergency_commands_triggered'],
                'average_confidence': self.metrics['average_confidence'],
                'average_response_time': self.metrics['average_response_time'],
                'top_commands': sorted(
                    [(name, stats['executions']) for name, stats in self.command_stats.items()],
                    key=lambda x: x[1], reverse=True
                )[:10],
                'category_usage': {
                    category.value: len(commands) for category, commands in self.commands_by_category.items()
                },
                'recent_activity': self.get_command_history(5)
            }

            return analytics

        except Exception as e:
            self.logger.error(f"Error generating command analytics: {str(e)}")
            return {}

    def cleanup(self):
        """Clean up command processor resources."""
        try:
            self.clear_command_history()
            self.commands.clear()
            self.command_handlers.clear()
            self.emergency_handlers.clear()
            self.conversation_context.clear()
            self.session_history.clear()
            self.audit_log.clear()
            self.logger.info("Voice command processor cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up command processor: {str(e)}")

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()