# Enhanced Text-to-Speech (TTS) Service

A comprehensive, therapeutic-quality TTS service designed specifically for AI therapy applications with multiple provider support and advanced voice customization.

## Features

### 🎭 Multiple TTS Providers
- **OpenAI TTS API** (Primary provider) - High-quality natural voices
- **ElevenLabs API** (Premium alternative) - Ultra-realistic emotional voices
- **Piper TTS** (Local offline) - Privacy-focused offline processing

### 🧠 Therapeutic Voice Profiles
- **Calm Therapist** - Soothing voice for anxiety and stress relief
- **Empathetic Guide** - Warm and supportive for emotional discussions
- **Professional Counselor** - Authoritative voice for structured therapy
- **Encouraging Coach** - Motivational voice for positive reinforcement

### 😊 Advanced Emotion Control
- **6 emotion types**: Calm, Empathetic, Professional, Encouraging, Supportive, Neutral
- **Fine-grained prosody control**: Pitch, speed, volume, emphasis
- **Natural emotional expression** with therapeutic appropriateness

### 🎨 SSML Support
- **Prosody attributes** for natural intonation
- **Emphasis tags** for therapeutic keywords
- **Break tags** for therapeutic pacing
- **Custom pronunciation** support

### ⚡ Performance Features
- **Voice caching** for rapid response times
- **Streaming synthesis** for real-time playback
- **Pre-voice generation** for common responses
- **LRU cache management** with configurable size

### 🔒 Privacy & Security
- **HIPAA-compliant** design principles
- **Local processing** option with Piper TTS
- **Data encryption** for voice data
- **No cloud data transmission** when using local providers

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp template.env .env
# Edit .env with your API keys
```

### 2. Basic Usage

```python
import asyncio
from voice.tts_service import TTSService
from voice.config import VoiceConfig

async def main():
    # Initialize TTS service
    config = VoiceConfig()
    tts = TTSService(config)

    # Check availability
    if not tts.is_available():
        print("Please configure API keys")
        return

    # Generate speech
    result = await tts.synthesize_speech(
        text="Hello, I'm your AI therapist. How can I help you today?",
        voice_profile="calm_therapist"
    )

    # Save audio
    tts.save_audio(result.audio_data, "greeting.wav")
    print(f"Generated {result.duration:.2f}s of audio")

asyncio.run(main())
```

### 3. Voice Profile Selection

```python
# Available therapeutic voice profiles
profiles = tts.voice_profiles.keys()
print(list(profilees))
# ['calm_therapist', 'empathetic_guide', 'professional_counselor', 'encouraging_coach']

# Use specific profile
result = await tts.synthesize_speech(
    text="I understand you're feeling overwhelmed.",
    voice_profile="empathetic_guide"
)
```

### 4. Emotion Control

```python
from voice.tts_service import EmotionType

# Apply specific emotion
result = await tts.synthesize_speech(
    text="You're doing great by sharing this with me.",
    emotion=EmotionType.ENCOURAGING
)
```

### 5. Streaming Synthesis

```python
# Real-time streaming for longer content
async for chunk in tts.synthesize_stream(
    text="This is a longer therapeutic response that will be streamed in real-time..."
):
    # Process audio chunk immediately
    play_audio(chunk.data, chunk.sample_rate)
```

## Configuration

### Environment Variables

#### OpenAI TTS (Primary Provider)
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_TTS_MODEL=tts-1  # or tts-1-hd for higher quality
OPENAI_TTS_VOICE=alloy  # alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_SPEED=1.0
```

#### ElevenLabs TTS (Premium Alternative)
```bash
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL=eleven_multilingual_v2
ELEVENLABS_VOICE_SPEED=1.0
ELEVENLABS_VOICE_STABILITY=0.5
ELEVENLABS_VOICE_SIMILARITY_BOOST=0.8
```

#### Piper TTS (Local Offline)
```bash
PIPER_TTS_MODEL_PATH=./models/piper/voices/en_US-lessac-medium.onnx
PIPER_TTS_SPEAKER_ID=0
```

### Voice Profile Configuration

Voice profiles are JSON files in `./voice_profiles/` directory:

```json
{
  "name": "calm_therapist",
  "description": "Calm and soothing voice for therapy sessions",
  "voice_id": "alloy",
  "language": "en-US",
  "gender": "female",
  "pitch": 0.9,
  "speed": 0.85,
  "volume": 0.8,
  "emotion": "calm",
  "style": "conversational",
  "elevenlabs_settings": {
    "stability": 0.7,
    "similarity_boost": 0.8,
    "style": 0.1
  }
}
```

## Advanced Features

### Custom Voice Profiles

```python
# Create custom voice profile
custom_profile = tts.create_custom_voice_profile(
    name="my_therapeutic_voice",
    base_profile="calm_therapist",
    modifications={
        "pitch": 1.1,
        "speed": 0.9,
        "volume": 0.85,
        "description": "My custom therapeutic voice"
    }
)
```

### SSML Enhancement

```python
# Text with SSML markup
therapeutic_text = """
It's <emphasis level='strong'>important</emphasis> to acknowledge your feelings.
Taking time to <prosody rate='slow' pitch='-10%'>breathe and reflect</prosody> can help.
"""

result = await tts.synthesize_speech(
    text=therapeutic_text,
    ssml_enabled=True
)
```

### Performance Optimization

```python
# Preload common therapeutic responses
common_responses = [
    "I understand how you're feeling.",
    "That sounds really challenging.",
    "Let's explore this together.",
    "You're doing great by sharing this."
]

tts.preload_common_responses(common_responses)

# Check cache performance
stats = tts.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Average processing time: {stats['average_processing_time']:.3f}s")
```

### Provider Fallback

The service automatically falls back to available providers:

```python
# Try providers in order: OpenAI -> ElevenLabs -> Piper
result = await tts.synthesize_speech(
    text="This will use the best available provider."
)

# Explicit provider selection
result = await tts.synthesize_speech(
    text="Using ElevenLabs specifically.",
    provider="elevenlabs"
)
```

## Therapeutic Use Cases

### Anxiety Management
```python
anxiety_response = await tts.synthesize_speech(
    text="I notice you're feeling anxious. Let's take a moment to breathe together.",
    voice_profile="calm_therapist",
    emotion=EmotionType.CALM
)
```

### Emotional Support
```python
support_response = await tts.synthesize_speech(
    text="It's completely valid to feel this way. I'm here to support you.",
    voice_profile="empathetic_guide",
    emotion=EmotionType.EMPATHETIC
)
```

### Motivational Encouragement
```python
encouragement = await tts.synthesize_speech(
    text="You've made important progress by being here today.",
    voice_profile="encouraging_coach",
    emotion=EmotionType.ENCOURAGING
)
```

## Testing

Run the comprehensive test suite:

```bash
# Basic TTS functionality test
python test_tts_service.py

# Therapeutic context example
python voice/examples/therapeutic_tts_example.py
```

## File Structure

```
voice/
├── tts_service.py          # Main TTS service implementation
├── config.py              # Configuration management
├── audio_processor.py     # Audio processing utilities
├── examples/
│   └── therapeutic_tts_example.py
├── __init__.py
└── README_TTS.md          # This file
```

## Security & Privacy

### Data Protection
- **No audio storage** by default (configurable)
- **Local processing** option with Piper TTS
- **Encrypted transmission** for cloud providers
- **HIPAA-compliant** design principles

### Compliance Features
- **Consent recording** for voice interactions
- **Data retention policies** with automatic cleanup
- **Anonymization options** for sensitive content
- **Emergency protocols** for crisis situations

## Troubleshooting

### Common Issues

**No TTS providers available:**
```bash
# Check API keys
echo $OPENAI_API_KEY
echo $ELEVENLABS_API_KEY

# Verify Piper installation
piper-tts --help
```

**Audio quality issues:**
- Try different voice models (`tts-1-hd` for higher quality)
- Adjust voice profile settings
- Check audio hardware configuration

**Performance problems:**
- Increase cache size in configuration
- Enable streaming for longer content
- Use local Piper TTS for offline scenarios

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in environment
export VOICE_LOG_LEVEL=DEBUG
```

## Integration Examples

### With Main Application
```python
# In app.py
from voice.tts_service import TTSService
from voice.config import VoiceConfig

class AITherapistApp:
    def __init__(self):
        self.config = VoiceConfig()
        self.tts = TTSService(self.config)

    async def respond_with_voice(self, text: str, emotional_context: str):
        # Select appropriate voice based on context
        voice_profile = self.select_therapeutic_voice(emotional_context)

        # Generate speech
        result = await self.tts.synthesize_speech(
            text=text,
            voice_profile=voice_profile
        )

        # Play audio and return text
        self.play_audio(result.audio_data)
        return result.text
```

## Contributing

This TTS service is designed specifically for therapeutic applications. When contributing:

1. **Prioritize therapeutic appropriateness** over technical features
2. **Maintain HIPAA compliance** in all implementations
3. **Test with therapeutic content** to ensure emotional appropriateness
4. **Document privacy implications** of any new features

## License

This implementation is part of the AI Therapist project and follows the same licensing terms.

## Support

For issues specific to the TTS service:
1. Check the troubleshooting section above
2. Run the test scripts to verify functionality
3. Review configuration requirements
4. Ensure all dependencies are properly installed