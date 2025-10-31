"""
Shared utility functions for voice services.
"""

import hashlib


def generate_cache_key(text: str) -> str:
    """
    Generate a secure SHA-256 hash key for caching purposes.
    
    Args:
        text: The text to hash
        
    Returns:
        SHA-256 hash of the input text
    """
    return hashlib.sha256(text.encode()).hexdigest()
