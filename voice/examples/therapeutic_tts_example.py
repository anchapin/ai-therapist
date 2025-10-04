#!/usr/bin/env python3
"""
Therapeutic TTS Example

This example demonstrates how to use the TTS service for therapeutic applications,
showcasing voice profiles, emotion control, and therapeutic response patterns.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the voice module to the path
sys.path.append(str(Path(__file__).parent.parent))

from voice.tts_service import TTSService, EmotionType
from voice.config import VoiceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapeuticVoiceAssistant:
    """Example therapeutic voice assistant using the TTS service."""

    def __init__(self):
        """Initialize the voice assistant."""
        self.config = VoiceConfig()
        self.tts_service = TTSService(self.config)
        self.session_context = {}

    async def respond_therapeutically(self, user_input: str, emotion_context: str = "neutral") -> str:
        """
        Generate a therapeutic response and convert it to speech.

        Args:
            user_input: What the user said
            emotion_context: Detected emotional state of the user

        Returns:
            Generated speech response
        """
        # Generate therapeutic response based on user input and emotion
        response = self._generate_therapeutic_response(user_input, emotion_context)

        # Determine appropriate voice profile and emotion
        voice_profile, tts_emotion = self._select_voice_parameters(emotion_context)

        # Convert to speech
        try:
            tts_result = await self.tts_service.synthesize_speech(
                text=response,
                voice_profile=voice_profile,
                emotion=tts_emotion,
                ssml_enabled=True
            )

            # Save the audio for this session
            session_id = self.session_context.get('session_id', 'default')
            audio_file = f"session_{session_id}_response_{len(self.session_context.get('responses', []))}.wav"
            self.tts_service.save_audio(tts_result.audio_data, audio_file)

            # Update session context
            if 'responses' not in self.session_context:
                self.session_context['responses'] = []
            self.session_context['responses'].append({
                'user_input': user_input,
                'response': response,
                'audio_file': audio_file,
                'voice_profile': voice_profile,
                'emotion': tts_emotion.value,
                'duration': tts_result.duration
            })

            return response

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return response  # Return text even if speech fails

    def _generate_therapeutic_response(self, user_input: str, emotion_context: str) -> str:
        """Generate a therapeutic response based on user input."""
        user_input_lower = user_input.lower()

        # Common therapeutic response patterns
        if any(word in user_input_lower for word in ['anxious', 'worried', 'stress', 'panic']):
            return "I notice you're feeling anxious. That's completely valid. Let's take a moment to acknowledge those feelings. Would you like to try a breathing exercise together?"

        elif any(word in user_input_lower for word in ['sad', 'depressed', 'down', 'hopeless']):
            return "It sounds like you're carrying a heavy emotional weight right now. Your feelings are important and valid. Would you like to explore what's contributing to these feelings?"

        elif any(word in user_input_lower for word in ['angry', 'frustrated', 'mad', 'upset']):
            return "I can hear the frustration in your voice. Anger is often a signal that something important to us is being threatened. What do you think might be underneath this anger?"

        elif any(word in user_input_lower for word in ['confused', 'lost', 'unsure', 'overwhelmed']):
            return "It makes sense that you're feeling overwhelmed right now. When things feel too big, it can help to break them down into smaller, more manageable pieces. What's one small step we could focus on?"

        elif any(word in user_input_lower for word in ['tired', 'exhausted', 'burnout', 'fatigue']):
            return "Burnout and exhaustion are real challenges, especially when you're trying to balance many responsibilities. Your body and mind are telling you they need rest. What would it look like to give yourself permission to rest?"

        else:
            return "Thank you for sharing that with me. I'm here to listen and support you. Could you tell me more about what's on your mind?"

    def _select_voice_parameters(self, emotion_context: str) -> tuple:
        """Select appropriate voice profile and emotion based on context."""
        emotion_context_lower = emotion_context.lower()

        if emotion_context_lower in ['anxious', 'worried', 'stressed']:
            return 'calm_therapist', EmotionType.CALM
        elif emotion_context_lower in ['sad', 'depressed', 'hopeless']:
            return 'empathetic_guide', EmotionType.EMPATHETIC
        elif emotion_context_lower in ['angry', 'frustrated']:
            return 'professional_counselor', EmotionType.PROFESSIONAL
        elif emotion_context_lower in ['tired', 'exhausted']:
            return 'empathetic_guide', EmotionType.SUPPORTIVE
        else:
            return 'calm_therapist', EmotionType.NEUTRAL

    async def run_therapy_session(self):
        """Run an example therapy session."""
        print("🎤 Therapeutic Voice Assistant Demo")
        print("=" * 50)
        print("This demo shows how the TTS service can be used in therapeutic contexts.")
        print("The assistant will respond to different emotional states with appropriate voices.\n")

        # Set up session
        import time
        session_id = f"session_{int(time.time())}"
        self.session_context['session_id'] = session_id

        # Example user inputs and their emotional contexts
        therapy_scenarios = [
            ("I've been feeling really anxious about work lately", "anxious"),
            ("Sometimes I just feel so sad and don't know why", "sad"),
            ("I'm so frustrated with everything happening right now", "angry"),
            ("I'm completely overwhelmed with all my responsibilities", "overwhelmed"),
            ("I feel like I'm burning out and can't keep going", "exhausted")
        ]

        for i, (user_input, emotion) in enumerate(therapy_scenarios, 1):
            print(f"🗣️  User: '{user_input}' (Emotion: {emotion})")

            # Generate therapeutic response
            response = await self.respond_therapeutically(user_input, emotion)
            print(f"🎵 Assistant: '{response}'")

            # Get the last response details
            if self.session_context.get('responses'):
                last_response = self.session_context['responses'][-1]
                print(f"   🎭 Voice Profile: {last_response['voice_profile']}")
                print(f"   😊 Emotion: {last_response['emotion']}")
                print(f"   ⏱️  Duration: {last_response['duration']:.2f}s")
                print(f"   💾 Audio saved: {last_response['audio_file']}")

            print()

        # Session summary
        print("📊 Session Summary:")
        print("-" * 30)
        responses = self.session_context.get('responses', [])
        print(f"Total responses: {len(responses)}")

        # Calculate session statistics
        total_duration = sum(r['duration'] for r in responses)
        print(f"Total audio duration: {total_duration:.2f}s")

        # Show voice profile usage
        voice_usage = {}
        for response in responses:
            profile = response['voice_profile']
            voice_usage[profile] = voice_usage.get(profile, 0) + 1

        print("Voice profiles used:")
        for profile, count in voice_usage.items():
            print(f"  {profile}: {count} time(s)")

        # Show emotion distribution
        emotion_usage = {}
        for response in responses:
            emotion = response['emotion']
            emotion_usage[emotion] = emotion_usage.get(emotion, 0) + 1

        print("Emotions conveyed:")
        for emotion, count in emotion_usage.items():
            print(f"  {emotion}: {count} time(s)")

        # TTS service statistics
        print("\n📈 TTS Service Statistics:")
        print("-" * 30)
        stats = self.tts_service.get_statistics()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key.replace('_', ' ').title()}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\n🎉 Therapy session demo complete!")
        print(f"Session files saved with prefix: session_{session_id}_")

        # Cleanup
        self.tts_service.cleanup()

async def main():
    """Run the therapeutic TTS example."""
    assistant = TherapeuticVoiceAssistant()

    if not assistant.tts_service.is_available():
        print("❌ TTS service not available. Please configure API keys.")
        print("Required:")
        print("- OPENAI_API_KEY (for OpenAI TTS)")
        print("- ELEVENLABS_API_KEY (for ElevenLabs TTS)")
        return

    await assistant.run_therapy_session()

if __name__ == "__main__":
    asyncio.run(main())