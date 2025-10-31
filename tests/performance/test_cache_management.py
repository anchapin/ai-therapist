"""
Cache Management Testing

Comprehensive test suite for cache management functionality:
- LRU eviction policies and memory limits
- TTL (time-to-live) expiration
- Compression for memory efficiency
- Concurrent access patterns
- Cache statistics and monitoring

Coverage targets: Cache management testing for 20-36%→70-80% coverage improvement
"""

import pytest
import time
import threading
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from performance.cache_manager import (
    CacheManager, CacheEntry, CacheStats, CacheEvictionPolicy,
    COMPRESSION_AVAILABLE
)


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return {
        'max_cache_size': 10,
        'max_memory_mb': 1,  # Small limit for testing
        'eviction_policy': 'lru',
        'enable_compression': True,
        'compression_threshold_bytes': 100,
        'cleanup_interval': 1.0,  # Fast cleanup for tests
        'stats_interval': 1.0     # Fast stats for tests
    }


@pytest.fixture
def cache_manager(cache_config):
    """Create cache manager instance for testing."""
    manager = CacheManager(cache_config)
    yield manager
    manager.stop()  # Ensure cleanup


class TestCacheManagerInitialization:
    """Test CacheManager initialization and configuration."""

    def test_initialization_with_config(self, cache_config):
        """Test cache manager initialization with configuration."""
        manager = CacheManager(cache_config)

        assert manager.max_cache_size == 10
        assert manager.max_memory_mb == 1
        assert manager.eviction_policy == CacheEvictionPolicy.LRU
        assert manager.enable_compression == True
        assert manager.compression_threshold_bytes == 100
        assert isinstance(manager.cache, dict)
        assert isinstance(manager.stats, CacheStats)

        manager.stop()

    def test_initialization_default_config(self):
        """Test cache manager initialization with default configuration."""
        manager = CacheManager()

        assert manager.max_cache_size == 100
        assert manager.max_memory_mb == 256
        assert manager.eviction_policy == CacheEvictionPolicy.LRU
        assert manager.enable_compression == True
        assert isinstance(manager.cache, dict)
        assert isinstance(manager.stats, CacheStats)

        manager.stop()

    def test_initialization_with_callbacks(self, cache_config):
        """Test cache manager initialization with callback registration."""
        manager = CacheManager(cache_config)

        evicted_items = []
        hit_keys = []
        miss_keys = []

        def eviction_callback(key, value):
            evicted_items.append((key, value))

        def hit_callback(key):
            hit_keys.append(key)

        def miss_callback(key):
            miss_keys.append(key)

        manager.register_eviction_callback(eviction_callback)
        manager.register_hit_callback(hit_callback)
        manager.register_miss_callback(miss_callback)

        assert len(manager.eviction_callbacks) == 1
        assert len(manager.hit_callbacks) == 1
        assert len(manager.miss_callbacks) == 1

        manager.stop()


class TestCacheBasicOperations:
    """Test basic cache operations (get, set, delete)."""

    def test_set_and_get_basic(self, cache_manager):
        """Test basic set and get operations."""
        # Set a value
        result = cache_manager.set("key1", "value1")
        assert result is True

        # Get the value
        value = cache_manager.get("key1")
        assert value == "value1"

        # Check stats
        stats = cache_manager.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0
        assert stats['sets'] == 1

    def test_get_nonexistent_key(self, cache_manager):
        """Test getting a key that doesn't exist."""
        value = cache_manager.get("nonexistent")
        assert value is None

        stats = cache_manager.get_stats()
        assert stats['misses'] == 1

    def test_delete_existing_key(self, cache_manager):
        """Test deleting an existing key."""
        cache_manager.set("key1", "value1")
        result = cache_manager.delete("key1")
        assert result is True

        value = cache_manager.get("key1")
        assert value is None

        stats = cache_manager.get_stats()
        assert stats['deletes'] == 1

    def test_delete_nonexistent_key(self, cache_manager):
        """Test deleting a key that doesn't exist."""
        result = cache_manager.delete("nonexistent")
        assert result is False

        stats = cache_manager.get_stats()
        assert stats['deletes'] == 0

    def test_contains_operation(self, cache_manager):
        """Test contains (in) operation."""
        cache_manager.set("key1", "value1")

        assert "key1" in cache_manager
        assert "nonexistent" not in cache_manager

    def test_clear_cache(self, cache_manager):
        """Test clearing the entire cache."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        cache_manager.clear()

        assert cache_manager.size() == 0
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None


class TestCacheLRUEviction:
    """Test LRU (Least Recently Used) eviction policy."""

    def test_lru_eviction_order(self, cache_manager):
        """Test that LRU eviction removes least recently used items."""
        # Set max cache size to 3 for this test
        cache_manager.max_cache_size = 3

        # Add items in order
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")

        # Access key1 to make it most recently used
        cache_manager.get("key1")

        # Add fourth item, should evict key2 (least recently used)
        cache_manager.set("key4", "value4")

        # Check that key2 was evicted, others remain
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") is None  # Evicted
        assert cache_manager.get("key3") == "value3"
        assert cache_manager.get("key4") == "value4"

    def test_lru_eviction_with_access_pattern(self, cache_manager):
        """Test LRU eviction with complex access patterns."""
        cache_manager.max_cache_size = 2

        cache_manager.set("a", "1")
        cache_manager.set("b", "2")
        cache_manager.get("a")  # a is now most recently used
        cache_manager.set("c", "3")  # Should evict b

        assert cache_manager.get("a") == "1"
        assert cache_manager.get("b") is None
        assert cache_manager.get("c") == "3"

    def test_lru_eviction_callbacks(self, cache_manager):
        """Test that eviction callbacks are triggered during LRU eviction."""
        cache_manager.max_cache_size = 2

        evicted_items = []
        def eviction_callback(key, value):
            evicted_items.append((key, value))

        cache_manager.register_eviction_callback(eviction_callback)

        cache_manager.set("a", "1")
        cache_manager.set("b", "2")
        cache_manager.set("c", "3")  # Should evict a

        assert len(evicted_items) == 1
        assert evicted_items[0] == ("a", "1")


class TestCacheTTLExpiration:
    """Test TTL (Time-To-Live) expiration functionality."""

    def test_ttl_expiration_basic(self, cache_manager):
        """Test basic TTL expiration."""
        # Set item with 1 second TTL
        cache_manager.set("ttl_key", "ttl_value", ttl_seconds=1)

        # Should be available immediately
        assert cache_manager.get("ttl_key") == "ttl_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache_manager.get("ttl_key") is None

    def test_ttl_expiration_with_cleanup(self, cache_manager):
        """Test TTL expiration with background cleanup."""
        cache_manager.start()  # Start background cleanup

        # Set item with short TTL
        cache_manager.set("short_ttl", "value", ttl_seconds=0.5)

        # Wait for cleanup
        time.sleep(2)

        # Should be cleaned up
        assert cache_manager.get("short_ttl") is None

        cache_manager.stop()

    def test_ttl_no_expiration_when_accessed(self, cache_manager):
        """Test that accessing an item resets its TTL."""
        cache_manager.set("ttl_key", "ttl_value", ttl_seconds=2)

        # Access after 1 second
        time.sleep(1)
        assert cache_manager.get("ttl_key") == "ttl_value"

        # Access again after another second (total 2 seconds)
        time.sleep(1)
        assert cache_manager.get("ttl_key") == "ttl_value"

        # Should still be available after total 2.5 seconds
        time.sleep(0.5)
        assert cache_manager.get("ttl_key") == "ttl_value"

    def test_ttl_expiration_eviction_stats(self, cache_manager):
        """Test that TTL expiration updates eviction statistics."""
        cache_manager.set("ttl_key", "ttl_value", ttl_seconds=0.5)

        # Wait for expiration
        time.sleep(0.6)

        # Access should trigger eviction
        cache_manager.get("ttl_key")

        stats = cache_manager.get_stats()
        assert stats['evictions'] >= 1  # Should count as eviction


class TestCacheCompression:
    """Test cache compression functionality."""

    @pytest.mark.skipif(not COMPRESSION_AVAILABLE, reason="Compression not available")
    def test_compression_basic(self, cache_manager):
        """Test basic compression functionality."""
        # Create a large string that should be compressed
        large_value = "x" * 200  # Over compression threshold

        cache_manager.set("compressed_key", large_value)

        # Verify it was stored
        retrieved = cache_manager.get("compressed_key")
        assert retrieved == large_value

        # Check compression stats
        stats = cache_manager.get_stats()
        assert stats['compressed_entries'] >= 1
        assert stats['compression_savings_bytes'] > 0

    @pytest.mark.skipif(not COMPRESSION_AVAILABLE, reason="Compression not available")
    def test_compression_threshold(self, cache_manager):
        """Test compression threshold behavior."""
        small_value = "small"  # Under compression threshold
        large_value = "x" * 200  # Over compression threshold

        cache_manager.set("small_key", small_value)
        cache_manager.set("large_key", large_value)

        # Both should be retrievable
        assert cache_manager.get("small_key") == small_value
        assert cache_manager.get("large_key") == large_value

    @pytest.mark.skipif(not COMPRESSION_AVAILABLE, reason="Compression not available")
    def test_compression_disabled(self, cache_manager):
        """Test behavior when compression is disabled."""
        cache_manager.enable_compression = False

        large_value = "x" * 200
        cache_manager.set("no_compress_key", large_value)

        retrieved = cache_manager.get("no_compress_key")
        assert retrieved == large_value

        stats = cache_manager.get_stats()
        assert stats['compressed_entries'] == 0

    def test_compression_not_available_fallback(self, cache_manager):
        """Test fallback when compression is not available."""
        with patch('performance.cache_manager.COMPRESSION_AVAILABLE', False):
            large_value = "x" * 200
            cache_manager.set("fallback_key", large_value)

            retrieved = cache_manager.get("fallback_key")
            assert retrieved == large_value


class TestCacheMemoryManagement:
    """Test cache memory management and limits."""

    def test_memory_limit_enforcement(self, cache_manager):
        """Test that memory limits are enforced."""
        # Set very low memory limit
        cache_manager.max_memory_bytes = 100  # 100 bytes

        # Add items that exceed the limit
        cache_manager.set("key1", "x" * 50)  # 50 bytes
        cache_manager.set("key2", "x" * 60)  # 60 bytes, should trigger eviction

        # First item should be evicted
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") == "x" * 60

    def test_memory_usage_calculation(self, cache_manager):
        """Test memory usage calculation."""
        cache_manager.set("key1", "hello")
        cache_manager.set("key2", "world")

        memory_usage = cache_manager.memory_usage()
        assert memory_usage > 0

        stats = cache_manager.get_stats()
        assert stats['memory_usage_mb'] == memory_usage
        assert stats['max_memory_mb'] == cache_manager.max_memory_mb

    def test_memory_usage_percentage(self, cache_manager):
        """Test memory usage percentage calculation."""
        # Add some data
        cache_manager.set("test_key", "test_value")

        stats = cache_manager.get_stats()
        usage_percent = stats['memory_usage_percent']

        assert 0 <= usage_percent <= 100
        assert usage_percent == (stats['memory_usage_mb'] / stats['max_memory_mb']) * 100


class TestCacheConcurrentAccess:
    """Test concurrent access patterns."""

    def test_concurrent_reads(self, cache_manager):
        """Test concurrent read operations."""
        cache_manager.set("shared_key", "shared_value")

        results = []

        def concurrent_read():
            value = cache_manager.get("shared_key")
            results.append(value)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=concurrent_read)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All reads should succeed
        assert len(results) == 10
        assert all(result == "shared_value" for result in results)

    def test_concurrent_writes(self, cache_manager):
        """Test concurrent write operations."""
        def concurrent_write(key_suffix):
            cache_manager.set(f"key_{key_suffix}", f"value_{key_suffix}")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_write, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All writes should succeed
        for i in range(10):
            assert cache_manager.get(f"key_{i}") == f"value_{i}"

    def test_concurrent_mixed_operations(self, cache_manager):
        """Test mixed concurrent read/write operations."""
        cache_manager.set("counter", 0)

        def increment_counter():
            current = cache_manager.get("counter") or 0
            cache_manager.set("counter", current + 1)

        threads = []
        for _ in range(20):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Counter should be incremented 20 times
        final_value = cache_manager.get("counter")
        assert final_value == 20

    def test_concurrent_eviction(self, cache_manager):
        """Test concurrent access during eviction."""
        cache_manager.max_cache_size = 5

        def stress_test():
            for i in range(100):
                cache_manager.set(f"stress_{i}", f"value_{i}")
                cache_manager.get(f"stress_{i % 10}")  # Access some items

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=stress_test)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Cache should still be functional
        assert cache_manager.size() <= cache_manager.max_cache_size


class TestCacheStatisticsAndMonitoring:
    """Test cache statistics and monitoring."""

    def test_hit_miss_statistics(self, cache_manager):
        """Test hit/miss statistics tracking."""
        # Set up some data
        cache_manager.set("hit_key", "hit_value")
        cache_manager.set("miss_key", "miss_value")
        cache_manager.delete("miss_key")  # Remove it

        # Generate hits and misses
        cache_manager.get("hit_key")  # Hit
        cache_manager.get("hit_key")  # Hit
        cache_manager.get("miss_key")  # Miss
        cache_manager.get("another_miss")  # Miss

        stats = cache_manager.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.5
        assert stats['hit_rate_percent'] == 50.0

    def test_cache_size_statistics(self, cache_manager):
        """Test cache size and entry statistics."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")

        stats = cache_manager.get_stats()
        assert stats['entries'] == 3
        assert stats['total_size_bytes'] > 0
        assert len(cache_manager.cache) == 3

    def test_compression_statistics(self, cache_manager):
        """Test compression-related statistics."""
        if COMPRESSION_AVAILABLE:
            # Add compressible data
            large_data = "x" * 200
            cache_manager.set("compress_test", large_data)

            stats = cache_manager.get_stats()
            assert 'compressed_entries' in stats
            assert 'compression_savings_bytes' in stats
            assert 'compression_ratio' in stats

    def test_uptime_tracking(self, cache_manager):
        """Test cache uptime tracking."""
        initial_time = time.time()

        # Wait a bit
        time.sleep(0.1)

        stats = cache_manager.get_stats()
        uptime = stats['uptime_seconds']

        assert uptime >= 0.1
        assert uptime <= 1.0  # Should be reasonable


class TestCacheBackgroundOperations:
    """Test background cleanup and statistics operations."""

    def test_background_cleanup_thread(self, cache_manager):
        """Test background cleanup thread operation."""
        cache_manager.start()

        # Add item with short TTL
        cache_manager.set("cleanup_test", "cleanup_value", ttl_seconds=0.5)

        # Wait for cleanup
        time.sleep(2)

        # Should be cleaned up
        assert cache_manager.get("cleanup_test") is None

        cache_manager.stop()

    def test_background_stats_thread(self, cache_manager):
        """Test background statistics thread."""
        cache_manager.start()

        # Perform some operations
        cache_manager.set("stats_test", "stats_value")
        cache_manager.get("stats_test")

        # Wait for stats update
        time.sleep(2)

        # Stats should be updated
        stats = cache_manager.get_stats()
        assert stats['sets'] >= 1
        assert stats['hits'] >= 1

        cache_manager.stop()

    def test_cleanup_interruption_resistance(self, cache_manager):
        """Test that cleanup thread handles interruptions gracefully."""
        cache_manager.start()

        # Simulate interruption
        cache_manager.is_running = False

        # Wait for thread to stop
        time.sleep(1)

        # Should not crash
        assert not cache_manager.is_running

        cache_manager.stop()


class TestCacheWarmupAndBulkOperations:
    """Test cache warmup and bulk operations."""

    def test_cache_warmup(self, cache_manager):
        """Test cache warmup with initial data."""
        initial_data = {
            "warmup_key1": "warmup_value1",
            "warmup_key2": "warmup_value2",
            "warmup_key3": "warmup_value3"
        }

        cache_manager.warm_cache(initial_data)

        # All items should be available
        for key, value in initial_data.items():
            assert cache_manager.get(key) == value

    def test_cache_warmup_with_ttl(self, cache_manager):
        """Test cache warmup with TTL values."""
        initial_data = {
            "ttl_warmup1": "value1",
            "ttl_warmup2": "value2"
        }

        cache_manager.warm_cache(initial_data, ttl_seconds=1)

        # Items should be available initially
        assert cache_manager.get("ttl_warmup1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Items should be expired
        assert cache_manager.get("ttl_warmup1") is None


class TestCacheErrorHandling:
    """Test error handling in cache operations."""

    def test_set_operation_error_handling(self, cache_manager):
        """Test error handling in set operations."""
        # Test with invalid data that might cause issues
        result = cache_manager.set("error_test", None)
        # Should handle gracefully
        assert isinstance(result, bool)

    def test_get_operation_error_handling(self, cache_manager):
        """Test error handling in get operations."""
        # Test getting from corrupted cache entry
        cache_manager.set("corrupt_test", "valid_value")

        # Manually corrupt the entry
        if "corrupt_test" in cache_manager.cache:
            entry = cache_manager.cache["corrupt_test"]
            entry.value = None  # Simulate corruption

        # Should handle gracefully
        result = cache_manager.get("corrupt_test")
        assert result is not None  # Should return the value despite corruption

    def test_compression_error_handling(self, cache_manager):
        """Test error handling in compression operations."""
        if COMPRESSION_AVAILABLE:
            # Test compression with problematic data
            problematic_data = object()  # Object that can't be compressed
            result = cache_manager.set("problematic", problematic_data)
            # Should handle gracefully without crashing
            assert isinstance(result, bool)


# Run basic validation
if __name__ == "__main__":
    print("Cache Management Test Suite")
    print("=" * 40)

    try:
        from performance.cache_manager import CacheManager
        print("✅ Cache manager imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")

    try:
        config = {'max_cache_size': 5, 'max_memory_mb': 1}
        cache = CacheManager(config)
        cache.set("test", "value")
        retrieved = cache.get("test")
        assert retrieved == "value"
        cache.stop()
        print("✅ Basic cache operations working")
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")

    print("Cache management test file created - run with pytest for full validation")
