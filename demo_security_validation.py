#!/usr/bin/env python3
"""
Security Validation Demo

This script demonstrates the security validation patterns and testing approaches
used in the comprehensive security test suite, without requiring the full voice module.

Usage: python demo_security_validation.py
"""

import re
import json
import tempfile
import threading
import time
from pathlib import Path
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class SecurityValidationDemo:
    """Demonstrates security validation patterns."""

    def __init__(self):
        # Patterns from voice/security.py
        self.USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')
        self.IP_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
        self.ALLOWED_CONSENT_TYPES = {
            'voice_processing', 'data_storage', 'transcription',
            'analysis', 'all_consent', 'emergency_protocol'
        }

    def validate_user_id(self, user_id):
        """Validate user ID format (from voice/security.py)."""
        if not isinstance(user_id, str):
            return False
        return bool(self.USER_ID_PATTERN.match(user_id))

    def validate_ip_address(self, ip_address):
        """Validate IP address format (from voice/security.py)."""
        if not isinstance(ip_address, str) or not ip_address:
            return True  # Empty IP is allowed for local contexts
        return bool(self.IP_PATTERN.match(ip_address))

    def validate_user_agent(self, user_agent):
        """Validate and sanitize user agent string (from voice/security.py)."""
        if not isinstance(user_agent, str):
            return False

        # Length limit
        if len(user_agent) > 500:
            return False

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';&]', '', user_agent)
        return len(sanitized) == len(user_agent)  # No dangerous chars found

    def validate_consent_type(self, consent_type):
        """Validate consent type against allowed values (from voice/security.py)."""
        if not isinstance(consent_type, str):
            return False
        return consent_type in self.ALLOWED_CONSENT_TYPES

    def demo_input_validation(self):
        """Demonstrate input validation security."""
        print("=== Input Validation Demo ===\n")

        # Test valid inputs
        print("✓ Valid User IDs:")
        valid_user_ids = ["user123", "test_user", "user-with-dash", "U", "a" * 50]
        for user_id in valid_user_ids:
            result = self.validate_user_id(user_id)
            print(f"  '{user_id}': {'✓' if result else '✗'}")

        print("\n✗ Invalid User IDs:")
        invalid_user_ids = ["", None, "user@domain.com", "user#123", "user space", "a" * 51]
        for user_id in invalid_user_ids:
            result = self.validate_user_id(user_id) if user_id is not None else False
            print(f"  '{user_id}': {'✓' if result else '✗'}")

        print("\n✓ Valid IP Addresses:")
        valid_ips = ["192.168.1.1", "10.0.0.1", "127.0.0.1", ""]
        for ip in valid_ips:
            result = self.validate_ip_address(ip)
            print(f"  '{ip}': {'✓' if result else '✗'}")

        print("\n✗ Invalid IP Addresses:")
        invalid_ips = ["999.999.999.999", "192.168.1", "192.168.1.1.1", "192.168.1.a"]
        for ip in invalid_ips:
            result = self.validate_ip_address(ip)
            print(f"  '{ip}': {'✓' if result else '✗'}")

        print("\n✓ Valid User Agents:")
        valid_agents = ["Mozilla/5.0", "Chrome/91.0", "TestAgent/1.0", ""]
        for agent in valid_agents:
            result = self.validate_user_agent(agent)
            print(f"  '{agent[:20]}...': {'✓' if result else '✗'}")

        print("\n✗ Invalid User Agents:")
        invalid_agents = ["<script>alert('xss')</script>", "Agent\"quotes\"", "Agent;semicolon&", "a" * 501]
        for agent in invalid_agents:
            result = self.validate_user_agent(agent)
            print(f"  '{agent[:20]}...': {'✓' if result else '✗'}")

        print("\n✓ Valid Consent Types:")
        for consent_type in self.ALLOWED_CONSENT_TYPES:
            print(f"  '{consent_type}': ✓")

        print("\n✗ Invalid Consent Types:")
        invalid_types = ["invalid_consent", "ALL", "none", "", None]
        for consent_type in invalid_types:
            result = self.validate_consent_type(consent_type) if consent_type is not None else False
            print(f"  '{consent_type}': {'✓' if result else '✗'}")

    def demo_memory_safety(self):
        """Demonstrate memory safety patterns."""
        print("\n=== Memory Safety Demo ===\n")

        # Simulate bounded deque (from voice/audio_processor.py)
        max_buffer_size = 50
        audio_buffer = deque(maxlen=max_buffer_size)
        memory_estimate = 0
        max_memory_bytes = 10 * 1024 * 1024  # 10MB limit

        print(f"Buffer Configuration:")
        print(f"  Max buffer size: {max_buffer_size}")
        print(f"  Memory limit: {max_memory_bytes:,} bytes")

        # Fill buffer beyond capacity
        print(f"\nTesting buffer size enforcement...")
        for i in range(100):
            chunk_size = 1024  # 1KB chunks
            test_data = np.random.rand(chunk_size // 4, 1).astype(np.float32)  # ~1KB

            # Check memory limit
            if memory_estimate + test_data.nbytes > max_memory_bytes:
                print(f"  Memory limit reached at chunk {i}, stopping")
                break

            # Add to buffer
            audio_buffer.append(test_data)
            memory_estimate += test_data.nbytes

        print(f"  Buffer size after 100 additions: {len(audio_buffer)} (max: {max_buffer_size})")
        print(f"  Memory usage: {memory_estimate:,} bytes")

        # Test cleanup
        print(f"\nTesting buffer cleanup...")
        cleared_count = len(audio_buffer)
        audio_buffer.clear()
        memory_estimate = 0
        print(f"  Cleared {cleared_count} items")
        print(f"  Buffer size after cleanup: {len(audio_buffer)}")
        print(f"  Memory after cleanup: {memory_estimate} bytes")

    def demo_thread_safety(self):
        """Demonstrate thread safety patterns."""
        print("\n=== Thread Safety Demo ===\n")

        # Simulate session management with lock (from voice/voice_service.py)
        sessions_lock = threading.RLock()
        sessions = {}
        operations = []
        errors = []

        def session_operations(worker_id):
            """Simulate concurrent session operations."""
            try:
                for i in range(5):
                    session_id = f"session_{worker_id}_{i}"

                    # Create session (thread-safe)
                    with sessions_lock:
                        sessions[session_id] = {
                            'id': session_id,
                            'created_by': worker_id,
                            'created_at': time.time()
                        }
                        operations.append(f"create_{session_id}")

                    # Access session (thread-safe)
                    with sessions_lock:
                        session = sessions.get(session_id)
                        operations.append(f"access_{session_id}" if session else f"missing_{session_id}")

                    # Destroy session (thread-safe)
                    with sessions_lock:
                        if session_id in sessions:
                            del sessions[session_id]
                            operations.append(f"destroy_{session_id}")

                    time.sleep(0.001)  # Small delay to increase interleaving

            except Exception as e:
                errors.append(e)

        print(f"Running concurrent session operations...")
        start_time = time.time()

        # Run operations concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(session_operations, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        duration = time.time() - start_time

        print(f"Completed in {duration:.3f} seconds")
        print(f"Operations performed: {len(operations)}")
        print(f"Errors encountered: {len(errors)}")
        print(f"Sessions remaining: {len(sessions)}")

        if errors:
            print(f"Errors: {errors}")
        else:
            print("✓ No thread safety errors detected")

    def demo_attack_prevention(self):
        """Demonstrate attack prevention."""
        print("\n=== Attack Prevention Demo ===\n")

        # SQL injection attempts
        print("SQL Injection Prevention:")
        sql_attacks = [
            "'; DROP TABLE users; --",
            "user' OR '1'='1",
            "admin'; INSERT INTO users VALUES('hacker', 'password'); --"
        ]

        for attack in sql_attacks:
            # Test as user_id
            user_id_safe = not self.validate_user_id(attack)
            print(f"  '{attack[:30]}...': {'✓ Blocked' if user_id_safe else '✗ Allowed'}")

        # XSS attempts
        print("\nXSS Prevention:")
        xss_attacks = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>"
        ]

        for attack in xss_attacks:
            # Test as user agent
            user_agent_safe = not self.validate_user_agent(attack)
            print(f"  '{attack[:30]}...': {'✓ Blocked' if user_agent_safe else '✗ Allowed'}")

        # Path traversal attempts
        print("\nPath Traversal Prevention:")
        path_attacks = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "${HOME}/.ssh/id_rsa",
            "file:///etc/passwd"
        ]

        for attack in path_attacks:
            # Test as user_id
            user_id_safe = not self.validate_user_id(attack)
            print(f"  '{attack[:30]}...': {'✓ Blocked' if user_id_safe else '✗ Allowed'}")

        # Unicode attacks
        print("\nUnicode Attack Prevention:")
        unicode_attacks = [
            "\x00user\x00",  # Null bytes
            "user\ufeff",    # BOM character
            "用户123",       # Chinese characters
            "🤖user",       # Emoji
        ]

        for attack in unicode_attacks:
            # Test as user_id
            user_id_safe = not self.validate_user_id(attack)
            print(f"  '{repr(attack)}': {'✓ Blocked' if user_id_safe else '✗ Allowed'}")

    def demo_audit_logging(self):
        """Demonstrate audit logging."""
        print("\n=== Audit Logging Demo ===\n")

        # Simulate audit logs
        audit_logs = []

        def log_security_event(event_type, user_id, action, result, details=None):
            """Simulate security audit logging."""
            log_entry = {
                'timestamp': time.time(),
                'event_type': event_type,
                'user_id': user_id,
                'action': action,
                'result': result,
                'details': details or {}
            }
            audit_logs.append(log_entry)

        # Log various security events
        events = [
            ("consent_update", "user123", "grant_consent", "success", {"consent_type": "voice_processing"}),
            ("consent_update", "user123", "revoke_consent", "success", {"consent_type": "voice_processing"}),
            ("login_attempt", "admin", "login", "failed", {"reason": "invalid_password"}),
            ("emergency", "user456", "emergency_protocol", "triggered", {"type": "crisis"}),
            ("access_denied", "hacker", "access_sensitive_data", "denied", {"reason": "invalid_permissions"}),
        ]

        print("Simulating security events...")
        for event_type, user_id, action, result, details in events:
            log_security_event(event_type, user_id, action, result, details)
            print(f"  {event_type}: {user_id} -> {action} ({result})")

        print(f"\nAudit Summary:")
        print(f"  Total events logged: {len(audit_logs)}")
        print(f"  Successful operations: {sum(1 for log in audit_logs if log['result'] == 'success')}")
        print(f"  Failed operations: {sum(1 for log in audit_logs if log['result'] in ['failed', 'denied'])}")
        print(f"  Emergency events: {sum(1 for log in audit_logs if log['event_type'] == 'emergency')}")

        # Show sample log entry
        if audit_logs:
            print(f"\nSample audit log entry:")
            sample_log = audit_logs[0]
            print(f"  Timestamp: {sample_log['timestamp']}")
            print(f"  Event Type: {sample_log['event_type']}")
            print(f"  User ID: {sample_log['user_id']}")
            print(f"  Action: {sample_log['action']}")
            print(f"  Result: {sample_log['result']}")
            print(f"  Details: {sample_log['details']}")

    def run_demo(self):
        """Run all security validation demonstrations."""
        print("AI Therapist Voice Security Validation Demo")
        print("=" * 50)
        print("This demo demonstrates the security validation patterns")
        print("used in the comprehensive security test suite.\n")

        self.demo_input_validation()
        self.demo_memory_safety()
        self.demo_thread_safety()
        self.demo_attack_prevention()
        self.demo_audit_logging()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("These patterns are implemented and tested in:")
        print("- test_voice_security_comprehensive.py")
        print("- voice/security.py")
        print("- voice/audio_processor.py")
        print("- voice/voice_service.py")
        print("\nTo run the full test suite:")
        print("  python run_security_tests.py")
        print("  pytest test_voice_security_comprehensive.py -v")


if __name__ == "__main__":
    demo = SecurityValidationDemo()
    demo.run_demo()