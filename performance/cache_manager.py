"""
Performance Cache Management Module

This module provides enhanced caching capabilities with memory limits,
LRU eviction, and comprehensive cache monitoring for the AI Therapist
voice services.

Features:
- LRU cache with configurable memory bounds
- Automatic cache eviction when memory thresholds reached
- Cache hit/miss statistics and monitoring
- Memory-aware caching with size limits
- Cache compression for large objects
- TTL (time-to-live) support
- Cache warming capabilities
- Performance metrics collection
"""

import time
import threading
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import weakref
import sys
import os
from enum import Enum

try:
    import zlib
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    SIZE_BASED = "size_based"
    TTL = "ttl"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    created_time: float = 0.0
    last_access_time: float = 0.0
    ttl_seconds: Optional[float] = None
    compressed: bool = False
    original_size: int = 0

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    deletes: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    compression_ratio: float = 1.0
    hit_rate: float = 0.0
    uptime_seconds: float = 0.0

class CacheManager:
    """
    Enhanced cache manager with memory limits and LRU eviction.

    Features:
    - Configurable memory limits with automatic eviction
    - LRU eviction policy with access tracking
    - TTL support for cache expiration
    - Compression for memory efficiency
    - Comprehensive statistics and monitoring
    - Thread-safe operations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.max_cache_size = self.config.get('max_cache_size', 100)  # Number of entries
        self.max_memory_mb = self.config.get('max_memory_mb', 256)  # Memory limit in MB
        self.max_memory_bytes = self.max_memory_mb * 1024 * 1024
        self.eviction_policy = CacheEvictionPolicy(self.config.get('eviction_policy', 'lru'))
        self.enable_compression = self.config.get('enable_compression', True) and COMPRESSION_AVAILABLE
        self.compression_threshold_bytes = self.config.get('compression_threshold_bytes', 1024)
        self.cleanup_interval = self.config.get('cleanup_interval', 300.0)  # 5 minutes
        self.stats_interval = self.config.get('stats_interval', 60.0)  # 1 minute

        # Cache storage - using OrderedDict for LRU ordering
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_lock = threading.RLock()

        # Statistics
        self.stats = CacheStats(max_size_bytes=self.max_memory_bytes)
        self.start_time = time.time()

        # Background tasks
        self.cleanup_thread = None
        self.stats_thread = None
        self.is_running = False

        # Callbacks
        self.eviction_callbacks: List[Callable[[str, Any], None]] = []
        self.hit_callbacks: List[Callable[[str], None]] = []
        self.miss_callbacks: List[Callable[[str], None]] = []

        # Size estimation cache for performance
        self._size_cache: Dict[int, int] = {}
        self._size_cache_lock = threading.Lock()

        self.logger.info(f"Cache manager initialized: max_size={self.max_cache_size}, max_memory={self.max_memory_mb}MB")

    def start(self):
        """Start background maintenance threads."""
        if self.is_running:
            return

        self.is_running = True

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="cache-cleanup"
        )
        self.cleanup_thread.start()

        # Start stats thread
        self.stats_thread = threading.Thread(
            target=self._stats_worker,
            daemon=True,
            name="cache-stats"
        )
        self.stats_thread.start()

        self.logger.info("Cache manager background tasks started")

    def stop(self):
        """Stop background maintenance threads."""
        self.is_running = False

        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)

        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=5.0)

        self.logger.info("Cache manager background tasks stopped")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check TTL
                if entry.ttl_seconds and (time.time() - entry.created_time) > entry.ttl_seconds:
                    self._evict_entry(key)
                    self.stats.misses += 1
                    self._trigger_miss_callbacks(key)
                    return None

                # Update access tracking
                entry.access_count += 1
                entry.last_access_time = time.time()

                # Move to end for LRU
                self.cache.move_to_end(key)

                # Decompress if needed
                value = self._decompress_value(entry) if entry.compressed else entry.value

                self.stats.hits += 1
                self._trigger_hit_callbacks(key)

                return value
            else:
                self.stats.misses += 1
                self._trigger_miss_callbacks(key)
                return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            # Estimate size
            size_bytes = self._estimate_size(value)

            # Check if we need to compress
            should_compress = (
                self.enable_compression and
                size_bytes > self.compression_threshold_bytes and
                self._is_compressible(value)
            )

            # Compress if needed
            if should_compress:
                compressed_value, compressed_size = self._compress_value(value)
                if compressed_value is not None:
                    value = compressed_value
                    original_size = size_bytes
                    size_bytes = compressed_size
                    compressed = True
                else:
                    compressed = False
                    original_size = size_bytes
            else:
                compressed = False
                original_size = size_bytes

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_time=time.time(),
                last_access_time=time.time(),
                ttl_seconds=ttl_seconds,
                compressed=compressed,
                original_size=original_size
            )

            with self.cache_lock:
                # Check if key already exists
                existing_size = 0
                if key in self.cache:
                    existing_size = self.cache[key].size_bytes

                # Check memory limits before adding
                if self.stats.total_size_bytes - existing_size + size_bytes > self.max_memory_bytes:
                    # Evict entries to make room
                    self._evict_to_make_room(size_bytes)

                # Remove existing entry if present
                if key in self.cache:
                    self.cache.pop(key)

                # Add new entry
                self.cache[key] = entry
                self.cache.move_to_end(key)  # Mark as recently used

                # Update statistics
                self.stats.total_size_bytes += size_bytes
                self.stats.sets += 1

                # Check entry count limit
                if len(self.cache) > self.max_cache_size:
                    self._evict_lru()

                return True

        except Exception as e:
            self.logger.error(f"Error setting cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.deletes += 1
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self.cache_lock:
            self.stats.total_size_bytes = 0
            self.cache.clear()
            self.logger.info("Cache cleared")

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.cache_lock:
            return key in self.cache

    def size(self) -> int:
        """Get number of entries in cache."""
        with self.cache_lock:
            return len(self.cache)

    def memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.stats.total_size_bytes / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.cache_lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (self.stats.hits / total_requests) if total_requests > 0 else 0.0

            compression_savings = 0
            compressed_entries = 0
            for entry in self.cache.values():
                if entry.compressed:
                    compressed_entries += 1
                    compression_savings += entry.original_size - entry.size_bytes

            return {
                'entries': len(self.cache),
                'memory_usage_mb': self.memory_usage(),
                'max_memory_mb': self.max_memory_mb,
                'memory_usage_percent': (self.memory_usage() / self.max_memory_mb) * 100,
                'hit_rate': hit_rate,
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'sets': self.stats.sets,
                'deletes': self.stats.deletes,
                'compression_enabled': self.enable_compression,
                'compressed_entries': compressed_entries,
                'compression_savings_bytes': compression_savings,
                'compression_ratio': self.stats.compression_ratio,
                'eviction_policy': self.eviction_policy.value,
                'uptime_seconds': time.time() - self.start_time,
                'cache_hit_rate_percent': hit_rate * 100
            }

    def warm_cache(self, items: Dict[str, Any], ttl_seconds: Optional[float] = None):
        """Warm cache with initial data."""
        for key, value in items.items():
            self.set(key, value, ttl_seconds)
        self.logger.info(f"Cache warmed with {len(items)} items")

    def register_eviction_callback(self, callback: Callable[[str, Any], None]):
        """Register callback for cache evictions."""
        self.eviction_callbacks.append(callback)

    def register_hit_callback(self, callback: Callable[[str], None]):
        """Register callback for cache hits."""
        self.hit_callbacks.append(callback)

    def register_miss_callback(self, callback: Callable[[str], None]):
        """Register callback for cache misses."""
        self.miss_callbacks.append(callback)

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return

        # Get least recently used item (first in OrderedDict)
        lru_key, lru_entry = next(iter(self.cache.items()))

        # Trigger eviction callbacks
        self._trigger_eviction_callbacks(lru_key, lru_entry.value)

        # Remove entry
        self.stats.total_size_bytes -= lru_entry.size_bytes
        del self.cache[lru_key]
        self.stats.evictions += 1

        self.logger.debug(f"Evicted LRU entry: {lru_key}")

    def _evict_to_make_room(self, required_bytes: int):
        """Evict entries to make room for new entry."""
        target_size = self.max_memory_bytes - required_bytes

        while self.stats.total_size_bytes > target_size and self.cache:
            self._evict_lru()

    def _evict_entry(self, key: str):
        """Evict a specific entry."""
        if key in self.cache:
            entry = self.cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.cache[key]
            self.stats.evictions += 1

    def _cleanup_worker(self):
        """Background cleanup worker for expired entries."""
        while self.is_running:
            try:
                self._cleanup_expired_entries()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                time.sleep(30.0)

    def _stats_worker(self):
        """Background statistics worker."""
        last_update = time.time()

        while self.is_running:
            try:
                current_time = time.time()
                if current_time - last_update >= self.stats_interval:
                    self._update_stats()
                    last_update = current_time

                time.sleep(10.0)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in stats worker: {e}")
                time.sleep(30.0)

    def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        with self.cache_lock:
            for key, entry in self.cache.items():
                if entry.ttl_seconds and (current_time - entry.created_time) > entry.ttl_seconds:
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                if key in self.cache:  # Double check in case it was removed
                    entry = self.cache[key]
                    self.stats.total_size_bytes -= entry.size_bytes
                    del self.cache[key]
                    self.stats.evictions += 1

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _update_stats(self):
        """Update cache statistics."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests

        # Update compression ratio
        compressed_entries = [entry for entry in self.cache.values() if entry.compressed]
        if compressed_entries:
            total_original = sum(entry.original_size for entry in compressed_entries)
            total_compressed = sum(entry.size_bytes for entry in compressed_entries)
            if total_original > 0:
                self.stats.compression_ratio = total_compressed / total_original

        self.stats.uptime_seconds = time.time() - self.start_time

    def _estimate_size(self, obj: Any) -> int:
        """Estimate the memory size of an object."""
        obj_id = id(obj)

        # Check cache first
        with self._size_cache_lock:
            if obj_id in self._size_cache:
                return self._size_cache[obj_id]

        try:
            # Use sys.getsizeof for basic size
            size = sys.getsizeof(obj)

            # Add size of contained objects for common types
            if isinstance(obj, (list, tuple)):
                # Sample first few items to estimate
                sample_size = min(len(obj), 10)
                if sample_size > 0:
                    sample_items = obj[:sample_size]
                    avg_item_size = sum(sys.getsizeof(item) for item in sample_items) / sample_size
                    size += int(avg_item_size * len(obj))
            elif isinstance(obj, dict):
                # Estimate based on key-value pairs
                sample_size = min(len(obj), 10)
                if sample_size > 0:
                    sample_items = list(obj.items())[:sample_size]
                    avg_item_size = sum(
                        sys.getsizeof(k) + sys.getsizeof(v) for k, v in sample_items
                    ) / sample_size
                    size += int(avg_item_size * len(obj))
            elif isinstance(obj, str):
                # String size is usually accurate
                pass
            elif hasattr(obj, '__dict__'):
                # For objects with __dict__, include attribute sizes
                for attr_name, attr_value in obj.__dict__.items():
                    size += sys.getsizeof(attr_name) + sys.getsizeof(attr_value)

            # Cache the result
            with self._size_cache_lock:
                self._size_cache[obj_id] = size

            return size

        except Exception:
            # Fallback to basic size
            return sys.getsizeof(obj)

    def _is_compressible(self, obj: Any) -> bool:
        """Check if an object is suitable for compression."""
        # Only compress strings, bytes, and certain container types
        return isinstance(obj, (str, bytes, list, dict)) and len(str(obj)) > 100

    def _compress_value(self, value: Any) -> Tuple[Optional[bytes], int]:
        """Compress a value and return compressed data and size."""
        if not COMPRESSION_AVAILABLE:
            return None, 0

        try:
            # Convert to bytes for compression
            if isinstance(value, str):
                data = value.encode('utf-8')
            elif isinstance(value, (list, dict)):
                import json
                data = json.dumps(value).encode('utf-8')
            elif isinstance(value, bytes):
                data = value
            else:
                return None, 0

            # Compress
            compressed = zlib.compress(data, level=6)
            return compressed, len(compressed)

        except Exception as e:
            self.logger.debug(f"Compression failed for value: {e}")
            return None, 0

    def _decompress_value(self, entry: CacheEntry) -> Any:
        """Decompress a cached value."""
        if not entry.compressed or not COMPRESSION_AVAILABLE:
            return entry.value

        try:
            # Decompress
            decompressed = zlib.decompress(entry.value)

            # Convert back to original type if needed
            # For now, return as bytes since we compressed to bytes
            return decompressed

        except Exception as e:
            self.logger.error(f"Decompression failed for cache entry {entry.key}: {e}")
            return entry.value

    def _trigger_eviction_callbacks(self, key: str, value: Any):
        """Trigger eviction callbacks."""
        for callback in self.eviction_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                self.logger.error(f"Error in eviction callback: {e}")

    def _trigger_hit_callbacks(self, key: str):
        """Trigger hit callbacks."""
        for callback in self.hit_callbacks:
            try:
                callback(key)
            except Exception as e:
                self.logger.error(f"Error in hit callback: {e}")

    def _trigger_miss_callbacks(self, key: str):
        """Trigger miss callbacks."""
        for callback in self.miss_callbacks:
            try:
                callback(key)
            except Exception as e:
                self.logger.error(f"Error in miss callback: {e}")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.contains(key)

    def __len__(self) -> int:
        """Get number of entries in cache."""
        return self.size()

    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.stop()
        except:
            pass