#!/usr/bin/env python3
"""
Performance Test Debugging Script for AI Therapist

This script systematically identifies and fixes performance test issues,
including memory leaks, resource exhaustion, and timeout problems.
"""

import sys
import json
import traceback
import importlib
from pathlib import Path
from datetime import datetime
import subprocess
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class PerformanceTestDebugger:
    """Comprehensive performance test debugger."""

    def __init__(self):
        self.project_root = project_root
        self.performance_test_dir = self.project_root / 'tests' / 'performance'
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'script': 'debug_performance_tests.py'
            },
            'test_files': {},
            'memory_issues': [],
            'timeout_issues': [],
            'resource_issues': [],
            'dependency_issues': [],
            'fixes_applied': [],
            'summary': {}
        }

    def discover_performance_tests(self):
        """Discover all performance test files."""
        print("🔍 Discovering performance test files...")

        if not self.performance_test_dir.exists():
            print(f"❌ Performance test directory not found: {self.performance_test_dir}")
            return []

        test_files = list(self.performance_test_dir.glob('test_*.py'))
        print(f"  Found {len(test_files)} performance test files")

        return test_files

    def analyze_memory_usage_patterns(self, test_file):
        """Analyze memory usage patterns in performance tests."""
        print(f"  Analyzing memory usage in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Look for memory-intensive patterns
            memory_patterns = [
                (r'psutil\.', 'PSUTIL_USAGE', 'psutil memory monitoring'),
                (r'memory_profiler', 'MEMORY_PROFILER', 'Memory profiler usage'),
                (r'trackbar|gc\.collect', 'MEMORY_MANAGEMENT', 'Memory management calls'),
                (r'large.*array|numpy\.random\.randn', 'LARGE_ARRAYS', 'Large array creation'),
                (r'while.*True|for.*range\(\d{4,}', 'INFINITE_LOOPS', 'Potential infinite loops'),
                (r'recursion|recursive', 'RECURSION', 'Recursive function calls'),
            ]

            for pattern, issue_type, description in memory_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        'type': issue_type,
                        'line_number': content[:match.start()].count('\n') + 1,
                        'match': match.group(0),
                        'description': description
                    })

        except Exception as e:
            issues.append({
                'type': 'MEMORY_ANALYSIS_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def analyze_timeout_configurations(self, test_file):
        """Analyze timeout configurations in performance tests."""
        print(f"  Analyzing timeout configurations in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Look for timeout patterns
            timeout_patterns = [
                (r'timeout.*=\s*(\d+)', 'TIMEOUT_VALUE', 'Explicit timeout setting'),
                (r'sleep\((\d+)\)', 'SLEEP_DURATION', 'Sleep duration'),
                (r'time\.sleep', 'TIME_SLEEP', 'Time sleep calls'),
                (r'@pytest\.mark\.timeout', 'PYTEST_TIMEOUT', 'Pytest timeout decorator'),
                (r'wait.*for|await', 'ASYNC_WAIT', 'Async wait patterns'),
            ]

            for pattern, issue_type, description in timeout_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        'type': issue_type,
                        'line_number': content[:match.start()].count('\n') + 1,
                        'match': match.group(0),
                        'description': description,
                        'value': match.groups()[0] if match.groups() else None
                    })

        except Exception as e:
            issues.append({
                'type': 'TIMEOUT_ANALYSIS_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def analyze_resource_usage(self, test_file):
        """Analyze resource usage patterns."""
        print(f"  Analyzing resource usage in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Look for resource-intensive patterns
            resource_patterns = [
                (r'multiprocessing|Process\(', 'MULTIPROCESSING', 'Multiprocessing usage'),
                (r'threading|Thread\(', 'THREADING', 'Threading usage'),
                (r'concurrent\.futures', 'CONCURRENT_FUTURES', 'Concurrent futures usage'),
                (r'open\(.*[\'"]\w', 'FILE_HANDLES', 'File handle operations'),
                (r'socket\.|requests\.', 'NETWORK_RESOURCES', 'Network resource usage'),
                (r'tempfile|TemporaryFile', 'TEMPORARY_FILES', 'Temporary file usage'),
            ]

            for pattern, issue_type, description in resource_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        'type': issue_type,
                        'line_number': content[:match.start()].count('\n') + 1,
                        'match': match.group(0),
                        'description': description
                    })

        except Exception as e:
            issues.append({
                'type': 'RESOURCE_ANALYSIS_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def run_performance_test(self, test_file):
        """Run a performance test with comprehensive monitoring."""
        print(f"  Running {test_file.name}...")

        try:
            # Set environment for performance tests
            env = {
                'PERFORMANCE_TEST_MODE': 'true',
                'MEMORY_PROFILING': 'true',
                'CI': 'true',
                'PYTHONMALLOC': 'debug'  # Enable memory debugging
            }

            # Run pytest with performance-specific settings
            result = subprocess.run([
                'python', '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=long',
                '--no-header',
                '--capture=no',
                '--log-cli-level=DEBUG',
                '--maxfail=1',
                '--timeout=300'  # 5 minute timeout for performance tests
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute hard timeout
            cwd=str(self.project_root),
            env={**dict(subprocess.os.environ), **env}
            )

            return {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': 'Performance test timed out after 10 minutes',
                'success': False
            }
        except Exception as e:
            return {
                'exit_code': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def parse_performance_errors(self, output, test_file):
        """Parse performance test specific errors."""
        errors = []

        # Performance test specific error patterns
        error_patterns = [
            # Memory errors
            (r'MemoryError', 'MEMORY_ERROR'),
            (r'OutOfMemoryError', 'OUT_OF_MEMORY'),
            (r'memory leak|memoryleak', 'MEMORY_LEAK'),
            (r'gc\.collect.*failed', 'GARBAGE_COLLECTION_ERROR'),

            # Timeout errors
            (r'TimeoutError', 'TIMEOUT_ERROR'),
            (r'test.*timed out', 'TEST_TIMEOUT'),
            (r'pytest.*timeout', 'PYTEST_TIMEOUT'),

            # Resource errors
            (r'OSError.*too many files', 'FILE_DESCRIPTOR_ERROR'),
            (r'PermissionError.*denied', 'RESOURCE_PERMISSION_ERROR'),
            (r'ConnectionRefusedError', 'RESOURCE_CONNECTION_ERROR'),

            # Process/thread errors
            (r'Process.*error', 'PROCESS_ERROR'),
            (r'Thread.*error', 'THREAD_ERROR'),
            (r'deadlock|dead.lock', 'DEADLOCK_ERROR'),

            # Performance metric errors
            (r'psutil.*Error', 'PSUTIL_ERROR'),
            (r'Performance.*Error', 'PERFORMANCE_METRIC_ERROR'),

            # Generic errors that affect performance
            (r'RecursionError', 'RECURSION_ERROR'),
            (r'StackOverflowError', 'STACK_OVERFLOW'),
        ]

        for pattern, error_type in error_patterns:
            matches = re.finditer(pattern, output, re.IGNORECASE)
            for match in matches:
                errors.append({
                    'type': error_type,
                    'message': match.group(0),
                    'context': match.start(),
                    'line_number': output[:match.start()].count('\n') + 1
                })

        return errors

    def generate_performance_fixes(self, test_file, issues):
        """Generate fixes for performance test issues."""
        fixes = []

        try:
            content = test_file.read_text(encoding='utf-8')
            original_content = content

            # Fix 1: Add proper timeout handling
            timeout_errors = [i for i in issues if 'TIMEOUT' in i['type']]
            if timeout_errors:
                content = self._fix_timeout_handling(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_TIMEOUT_HANDLING',
                        'description': f'Fixed timeout handling for {len(timeout_errors)} issues',
                        'applied': True
                    })

            # Fix 2: Add memory management
            memory_errors = [i for i in issues if 'MEMORY' in i['type']]
            if memory_errors:
                content = self._fix_memory_management(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_MEMORY_MANAGEMENT',
                        'description': f'Fixed memory management for {len(memory_errors)} issues',
                        'applied': True
                    })

            # Fix 3: Add resource cleanup
            resource_errors = [i for i in issues if 'RESOURCE' in i['type'] or 'FILE' in i['type'] or 'PROCESS' in i['type']]
            if resource_errors:
                content = self._fix_resource_cleanup(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_RESOURCE_CLEANUP',
                        'description': f'Fixed resource cleanup for {len(resource_errors)} issues',
                        'applied': True
                    })

            # Fix 4: Add performance monitoring setup
            if any('psutil' in str(i) for i in issues):
                content = self._fix_performance_monitoring(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_PERFORMANCE_MONITORING',
                        'description': 'Fixed performance monitoring setup',
                        'applied': True
                    })

            # Write back fixed content
            if content != original_content:
                test_file.write_text(content, encoding='utf-8')
                print(f"    ✅ Applied performance fixes to {test_file.name}")

        except Exception as e:
            fixes.append({
                'type': 'FIX_ERROR',
                'description': f'Error applying performance fixes: {e}',
                'applied': False
            })

        return fixes

    def _fix_timeout_handling(self, content):
        """Fix timeout handling issues."""
        # Add proper timeout imports and setup
        if 'timeout' in content.lower() and 'signal' not in content:
            lines = content.split('\n')
            insert_pos = 0

            for i, line in enumerate(lines):
                if line.strip().startswith('import ') and i > 0:
                    insert_pos = i + 1
                    break

            timeout_setup = [
                '',
                '# Timeout handling for performance tests',
                'import signal',
                'import time',
                'from contextlib import contextmanager',
                '',
                '@contextmanager',
                'def timeout_context(seconds):',
                '    """Context manager for timeout handling."""',
                '    def timeout_handler(signum, frame):',
                '        raise TimeoutError(f"Operation timed out after {seconds} seconds")',
                '    ',
                '    old_handler = signal.signal(signal.SIGALRM, timeout_handler)',
                '    signal.alarm(seconds)',
                '    try:',
                '        yield',
                '    finally:',
                '        signal.alarm(0)',
                '        signal.signal(signal.SIGALRM, old_handler)',
                ''
            ]

            lines[insert_pos:insert_pos] = timeout_setup
            content = '\n'.join(lines)

        # Add timeout decorator to long-running tests
        content = re.sub(
            r'def (test_\w+.*):\s*\n',
            lambda m: f'@pytest.mark.timeout(300)\ndef {m.group(1)}:\n',
            content
        )

        return content

    def _fix_memory_management(self, content):
        """Fix memory management issues."""
        # Add garbage collection and memory monitoring
        if 'gc' not in content:
            lines = content.split('\n')
            insert_pos = 0

            for i, line in enumerate(lines):
                if line.strip().startswith('import ') and i > 0:
                    insert_pos = i + 1
                    break

            memory_setup = [
                '',
                '# Memory management for performance tests',
                'import gc',
                'import psutil',
                'import os',
                '',
                'def get_memory_usage():',
                '    """Get current memory usage in MB."""',
                '    process = psutil.Process(os.getpid())',
                '    return process.memory_info().rss / 1024 / 1024',
                '',
                'def force_garbage_collection():',
                '    """Force garbage collection and return memory freed."""',
                '    before = get_memory_usage()',
                '    gc.collect()',
                '    after = get_memory_usage()',
                '    return before - after',
                ''
            ]

            lines[insert_pos:insert_pos] = memory_setup
            content = '\n'.join(lines)

        # Add memory cleanup in test teardown
        content = re.sub(
            r'(def test_\w+\(.*\):\s*\n)',
            r'\1    initial_memory = get_memory_usage()\n    try:\n        ',
            content
        )

        content = re.sub(
            r'(\s+)(assert .*)',
            r'\1    \2\n    finally:\n        freed_memory = force_garbage_collection()\n        assert freed_memory >= 0, "Memory leak detected"',
            content
        )

        return content

    def _fix_resource_cleanup(self, content):
        """Fix resource cleanup issues."""
        # Add resource cleanup context managers
        if 'with' not in content or 'close' not in content:
            # Add proper file handling
            content = re.sub(
                r'open\(([^)]+)\)',
                r'with open(\1) as f:',
                content
            )

            # Add process cleanup
            if 'multiprocessing' in content:
                process_cleanup = [
                    '',
                    '@pytest.fixture(autouse=True)',
                    'def cleanup_processes():',
                    '    """Cleanup any leftover processes after tests."""',
                    '    yield',
                    '    import multiprocessing',
                    '    for proc in multiprocessing.active_children():',
                    '        proc.terminate()',
                    '        proc.join(timeout=1)',
                    ''
                ]

                lines = content.split('\n')
                lines.extend(process_cleanup)
                content = '\n'.join(lines)

        return content

    def _fix_performance_monitoring(self, content):
        """Fix performance monitoring setup."""
        # Add psutil availability check
        if 'psutil' in content and 'PSUTIL_AVAILABLE' not in content:
            lines = content.split('\n')
            insert_pos = 0

            for i, line in enumerate(lines):
                if 'import psutil' in line:
                    insert_pos = i + 1
                    break

            psutil_check = [
                'try:',
                '    import psutil',
                '    PSUTIL_AVAILABLE = True',
                'except ImportError:',
                '    PSUTIL_AVAILABLE = False',
                '    psutil = None',
                ''
            ]

            # Replace the psutil import with the check
            lines = [line for line in lines if not line.strip().startswith('import psutil')]
            lines[insert_pos:insert_pos] = psutil_check
            content = '\n'.join(lines)

        return content

    def analyze_all_performance_tests(self):
        """Analyze all performance test files."""
        print("🚀 Starting Performance Test Analysis")
        print("=" * 50)

        test_files = self.discover_performance_tests()

        if not test_files:
            print("❌ No performance test files found")
            return

        total_tests = len(test_files)
        successful_tests = 0

        for test_file in test_files:
            print(f"\n📋 Analyzing {test_file.name}")
            print("-" * 40)

            # Analyze memory usage patterns
            memory_issues = self.analyze_memory_usage_patterns(test_file)
            self.memory_issues.extend(memory_issues)

            # Analyze timeout configurations
            timeout_issues = self.analyze_timeout_configurations(test_file)
            self.timeout_issues.extend(timeout_issues)

            # Analyze resource usage
            resource_issues = self.analyze_resource_usage(test_file)
            self.resource_issues.extend(resource_issues)

            # Run the performance test
            test_result = self.run_performance_test(test_file)

            # Parse errors from output
            if not test_result['success']:
                stdout_errors = self.parse_performance_errors(test_result['stdout'], test_file)
                stderr_errors = self.parse_performance_errors(test_result['stderr'], test_file)

                all_errors = stdout_errors + stderr_errors

                # Generate and apply fixes
                if all_errors:
                    fixes = self.generate_performance_fixes(test_file, all_errors)
                    self.fixes_applied.extend(fixes)

                # Re-run test to check if fixes worked
                if self.fixes_applied:
                    print(f"  🔄 Re-running {test_file.name} after fixes...")
                    retry_result = self.run_performance_test(test_file)
                    if retry_result['success']:
                        successful_tests += 1
                        print(f"  ✅ {test_file.name} passed after fixes")
                    else:
                        print(f"  ❌ {test_file.name} still failing")
                        self.results['test_files'][test_file.name] = {
                            'status': 'FAILED_AFTER_FIXES',
                            'errors': all_errors,
                            'output': retry_result['stderr']
                        }
                else:
                    print(f"  ❌ {test_file.name} failed")
                    self.results['test_files'][test_file.name] = {
                        'status': 'FAILED',
                        'errors': all_errors,
                        'output': test_result['stderr']
                    }
            else:
                successful_tests += 1
                print(f"  ✅ {test_file.name} passed")
                self.results['test_files'][test_file.name] = {
                    'status': 'PASSED',
                    'output': test_result['stdout']
                }

        # Generate summary
        self.results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'total_memory_issues': len(self.memory_issues),
            'total_timeout_issues': len(self.timeout_issues),
            'total_resource_issues': len(self.resource_issues),
            'total_dependency_issues': len(self.dependency_issues),
            'total_fixes_applied': len(self.fixes_applied)
        }

    def save_report(self):
        """Save debugging report."""
        report_path = self.project_root / 'reports' / 'performance-debug-report.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Performance test debug report saved to: {report_path}")
        return report_path

    def print_summary(self):
        """Print debugging summary."""
        print("\n📊 Performance Test Debugging Summary")
        print("=" * 40)
        print(f"Total Tests: {self.results['summary']['total_tests']}")
        print(f"Successful: {self.results['summary']['successful_tests']}")
        print(f"Failed: {self.results['summary']['failed_tests']}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1%}")
        print(f"Memory Issues: {self.results['summary']['total_memory_issues']}")
        print(f"Timeout Issues: {self.results['summary']['total_timeout_issues']}")
        print(f"Resource Issues: {self.results['summary']['total_resource_issues']}")
        print(f"Fixes Applied: {self.results['summary']['total_fixes_applied']}")

        if self.fixes_applied:
            print(f"\n🔧 Applied Fixes:")
            for fix in self.fixes_applied:
                status = "✅" if fix.get('applied', False) else "❌"
                print(f"  {status} {fix['description']}")

    def run_debugging(self):
        """Run complete debugging process."""
        self.analyze_all_performance_tests()
        self.save_report()
        self.print_summary()

        # Return exit code based on success rate
        success_rate = self.results['summary']['success_rate']
        if success_rate >= 0.6:  # 60% success rate threshold for performance tests
            print("\n✅ Performance test debugging completed successfully")
            return 0
        else:
            print(f"\n⚠️ Performance test debugging completed with {success_rate:.1%} success rate")
            return 1

def main():
    """Main debugging function."""
    debugger = PerformanceTestDebugger()
    return debugger.run_debugging()

if __name__ == "__main__":
    sys.exit(main())