# TTSService Branch Coverage Test Summary

## Overview
Added comprehensive branch coverage tests for TTSService in `tests/voice/test_tts_service_branches.py` covering previously untested code paths.

## Tests Added (30 total)

### 1. SSML Generation Tests (7 tests)
Tests for `_generate_ssml()` method with different configuration toggles:

- **test_generate_ssml_all_disabled**: Verifies SSML is completely disabled and returns plain text
- **test_generate_ssml_prosody_enabled**: Tests prosody tags enabled, emphasis disabled
- **test_generate_ssml_emphasis_enabled**: Tests emphasis tags enabled, prosody disabled
- **test_generate_ssml_all_enabled**: Tests all SSML features enabled together
- **test_generate_ssml_prosody_attributes**: Verifies correct pitch, rate, and volume attributes
- **test_generate_ssml_default_prosody_values**: Tests prosody with default 1.0 values uses "medium"
- **test_generate_ssml_therapeutic_keywords_case_insensitive**: Verifies emphasis applied case-insensitively

**Coverage:**
- SSMLSettings option toggles (enabled, prosody_attributes, emphasis_tags)
- Prosody tag generation and closing
- Emphasis tag generation for therapeutic keywords
- Case-insensitive keyword matching

### 2. Caching Tests (11 tests)
Tests for cache key generation, LRU eviction, and audio saving:

#### Cache Key Tests
- **test_get_cache_key_basic**: Basic cache key generation
- **test_get_cache_key_with_emotion**: Cache key with/without emotion parameter
- **test_get_cache_key_different_emotions**: Different emotions generate different keys

#### Cache Result & LRU Tests
- **test_cache_result_basic**: Basic cache storage
- **test_cache_result_lru_eviction**: LRU eviction when cache is full (critical path)

#### Audio Saving Tests
- **test_save_audio_success**: Successful audio file save
- **test_save_audio_creates_directory**: Parent directory creation
- **test_save_audio_format_mapping**: Different format mappings (wav, mp3, flac, ogg)
- **test_save_audio_soundfile_import_error**: Handles soundfile import error (error path)
- **test_save_audio_write_error**: Handles write errors (error path)
- **test_save_audio_permission_error**: Handles permission errors (error path)

**Coverage:**
- `_get_cache_key()` with and without emotion
- `_cache_result()` LRU eviction logic
- `save_audio()` success and all error paths

### 3. Provider Availability Tests (12 tests)
Tests for `get_supported_*()` methods when providers are unavailable:

#### Voice Support Tests
- **test_get_supported_voices_no_providers**: No providers available returns empty list
- **test_get_supported_voices_only_openai**: Only OpenAI voices returned
- **test_get_supported_voices_only_elevenlabs**: Only ElevenLabs voices returned
- **test_get_supported_voices_only_piper**: Only Piper voices returned
- **test_get_supported_voices_all_providers**: All provider voices returned

#### Model & Language Tests
- **test_get_supported_models_no_providers**: No providers returns empty list
- **test_get_supported_models_openai_available**: OpenAI models returned
- **test_get_supported_languages_always_returns_list**: Languages always available

#### Provider Availability Tests
- **test_get_available_providers_none_available**: No providers returns empty list
- **test_get_available_providers_partial**: Partial provider availability
- **test_is_available_with_no_providers**: is_available() returns False
- **test_is_available_with_any_provider**: is_available() returns True

**Coverage:**
- `get_supported_voices()` conditional branches for each provider
- `get_supported_models()` conditional branches
- `get_supported_languages()` always returns list
- `get_available_providers()` conditional logic
- `is_available()` with different provider states

## Test Execution

### Run all new tests:
```bash
python3 -m pytest tests/voice/test_tts_service_branches.py -v
```

### Run specific test classes:
```bash
# SSML tests only
python3 -m pytest tests/voice/test_tts_service_branches.py::TestSSMLGeneration -v

# Caching tests only
python3 -m pytest tests/voice/test_tts_service_branches.py::TestCaching -v

# Provider tests only
python3 -m pytest tests/voice/test_tts_service_branches.py::TestGetSupportedMethods -v
```

### Results
✅ All 30 tests pass successfully
✅ Test execution time: ~7 seconds
✅ No external dependencies required (fully mocked)

## Branch Coverage Improvements

### Methods with Improved Coverage:
1. **_generate_ssml()**: All conditional branches for SSML settings
2. **_get_cache_key()**: With/without emotion parameter paths
3. **_cache_result()**: LRU eviction logic when cache is full
4. **save_audio()**: All error handling paths (import error, write error, permission error)
5. **get_supported_voices()**: All provider conditional branches
6. **get_supported_models()**: Provider availability branches
7. **get_available_providers()**: Provider check branches
8. **is_available()**: Multiple provider state branches

### Critical Paths Tested:
- ✅ LRU cache eviction (removes oldest entry when full)
- ✅ SSML tag opening and closing consistency
- ✅ Error handling for missing dependencies (soundfile)
- ✅ Graceful degradation when providers unavailable
- ✅ Case-insensitive text matching in SSML generation

## Code Quality

### Test Patterns Used:
- Proper fixture isolation with `@pytest.fixture`
- Comprehensive mocking with `unittest.mock`
- Temporary directory usage for file tests
- Patch context managers for safe mocking
- Clear test naming following conventions
- Minimal setup/teardown with proper cleanup

### Mock Strategy:
- Mock config with required attributes
- Patch service initialization to avoid API calls
- Mock provider clients (openai_client, elevenlabs_client, piper_tts)
- Mock SSML settings for configuration testing

## Integration
These tests integrate seamlessly with the existing test suite and follow the same patterns used in:
- `tests/unit/test_tts_service_comprehensive.py`
- `tests/unit/test_tts_service.py`
- `tests/voice/` directory structure

## Notes
- Tests are fully isolated and don't require actual TTS providers
- All file operations use temporary directories
- No external network calls are made
- Tests run quickly (~0.2s per test on average)
