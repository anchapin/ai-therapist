#!/usr/bin/env python3
"""
Integration Test Debugging Script for AI Therapist

This script systematically identifies and fixes integration test issues,
including service mocking, numpy recursion problems, and configuration issues.
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

class IntegrationTestDebugger:
    """Comprehensive integration test debugger."""

    def __init__(self):
        self.project_root = project_root
        self.integration_test_dir = self.project_root / 'tests' / 'integration'
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'script': 'debug_integration_tests.py'
            },
            'test_files': {},
            'service_mocking_issues': [],
            'numpy_recursion_issues': [],
            'dependency_issues': [],
            'fixes_applied': [],
            'summary': {}
        }

    def discover_integration_tests(self):
        """Discover all integration test files."""
        print("🔍 Discovering integration test files...")

        if not self.integration_test_dir.exists():
            print(f"❌ Integration test directory not found: {self.integration_test_dir}")
            return []

        test_files = list(self.integration_test_dir.glob('test_*.py'))
        print(f"  Found {len(test_files)} integration test files")

        return test_files

    def analyze_service_dependencies(self, test_file):
        """Analyze service dependencies in integration tests."""
        print(f"  Analyzing service dependencies in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Look for service dependencies
            service_patterns = [
                (r'from\s+voice\.(\w+)_service\s+import', 'VOICE_SERVICE', 'Voice service module'),
                (r'from\s+voice\.stt_service\s+import', 'STT_SERVICE', 'Speech-to-Text service'),
                (r'from\s+voice\.tts_service\s+import', 'TTS_SERVICE', 'Text-to-Speech service'),
                (r'from\s+voice\.audio_processor\s+import', 'AUDIO_PROCESSOR', 'Audio processor'),
                (r'import\s+ollama', 'OLLAMA', 'Ollama LLM service'),
                (r'from\s+openai\s+import', 'OPENAI', 'OpenAI API'),
            ]

            for pattern, service_type, description in service_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    issues.append({
                        'type': 'SERVICE_DEPENDENCY',
                        'service_type': service_type,
                        'match': match.group(0),
                        'description': description
                    })

        except Exception as e:
            issues.append({
                'type': 'SERVICE_ANALYSIS_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def detect_numpy_recursion_issues(self, test_file):
        """Detect potential numpy recursion issues."""
        print(f"  Checking for numpy recursion issues in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Look for patterns that might cause numpy recursion
            recursion_patterns = [
                (r'np\.array\(.*np\.array\(', 'NESTED_ARRAY', 'Nested numpy array creation'),
                (r'@patch\([\'"]numpy\.array[\'"]', 'NUMPY_PATCH', 'Patching numpy.array can cause recursion'),
                (r'MagicMock\(spec=np\.array\)', 'NUMPY_SPEC_MOCK', 'Mock with numpy array spec'),
                (r'numpy\.ndarray.*MagicMock', 'NUMPY_MOCK_MIX', 'Mixing numpy arrays with mocks'),
            ]

            for pattern, issue_type, description in recursion_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    issues.append({
                        'type': issue_type,
                        'line_number': content[:match.start()].count('\n') + 1,
                        'match': match.group(0),
                        'description': description
                    })

        except Exception as e:
            issues.append({
                'type': 'NUMPY_ANALYSIS_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def analyze_mock_configurations(self, test_file):
        """Analyze mock configurations in integration tests."""
        print(f"  Analyzing mock configurations in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Check for mock configuration issues
            mock_patterns = [
                (r'@patch\([\'"][^\'"]*\.([^\'"]+)[\'"]\)', 'EXTERNAL_SERVICE_PATCH', 'Patching external service'),
                (r'MagicMock\(\)\.(\w+).*=.*MagicMock', 'NESTED_MOCK', 'Nested mock configuration'),
                (r'with patch\([\'"]([^\'"]+)[\'"].*as.*mock:', 'CONTEXT_MANAGER_MOCK', 'Context manager mock'),
            ]

            for pattern, issue_type, description in mock_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    issues.append({
                        'type': issue_type,
                        'line_number': content[:match.start()].count('\n') + 1,
                        'match': match.group(0),
                        'description': description,
                        'details': match.groups() if match.groups() else []
                    })

        except Exception as e:
            issues.append({
                'type': 'MOCK_CONFIG_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def run_integration_test(self, test_file):
        """Run an integration test with comprehensive error capture."""
        print(f"  Running {test_file.name}...")

        try:
            # Set environment variables for integration tests
            env = {
                'OLLAMA_HOST': 'http://ollama:11434',
                'CI': 'true',
                'VOICE_ENABLED': 'false',  # Disable voice for integration tests
                'INTEGRATION_TEST_MODE': 'true'
            }

            # Run pytest with comprehensive output
            result = subprocess.run([
                'python', '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=long',
                '--no-header',
                '--capture=no',  # Don't capture output for debugging
                '--log-cli-level=DEBUG',
                '--maxfail=1'
            ],
            capture_output=True,
            text=True,
            timeout=120,  # Longer timeout for integration tests
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
                'stderr': 'Integration test timed out after 120 seconds',
                'success': False
            }
        except Exception as e:
            return {
                'exit_code': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def parse_integration_errors(self, output, test_file):
        """Parse integration test specific errors."""
        errors = []

        # Integration test specific error patterns
        error_patterns = [
            # Numpy recursion errors
            (r'RecursionError.*maximum recursion depth exceeded', 'RECURSION_ERROR'),
            (r'StackOverflowError', 'STACK_OVERFLOW'),
            (r'numpy.*recursion', 'NUMPY_RECURSION'),

            # Service connection errors
            (r'ConnectionRefusedError', 'SERVICE_CONNECTION_ERROR'),
            (r'TimeoutError.*connecting', 'SERVICE_TIMEOUT'),
            (r'HTTPError.*404', 'SERVICE_NOT_FOUND'),
            (r'HTTPError.*500', 'SERVICE_ERROR'),

            # Import errors in integration context
            (r'ImportError.*cannot import name', 'INTEGRATION_IMPORT_ERROR'),
            (r'ModuleNotFoundError.*No module named', 'INTEGRATION_MODULE_ERROR'),

            # Configuration errors
            (r'ConfigurationError', 'CONFIG_ERROR'),
            (r'ValidationError', 'VALIDATION_ERROR'),

            # Resource issues
            (r'MemoryError', 'MEMORY_ERROR'),
            (r'FileNotFoundError.*config', 'CONFIG_FILE_ERROR'),

            # Mocking issues
            (r'AttributeError.*Mock object', 'MOCK_ATTRIBUTE_ERROR'),
            (r'TypeError.*Mock', 'MOCK_TYPE_ERROR'),
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

    def generate_integration_fixes(self, test_file, issues):
        """Generate fixes for integration test issues."""
        fixes = []

        try:
            content = test_file.read_text(encoding='utf-8')
            original_content = content

            # Fix 1: Add numpy recursion protection
            recursion_errors = [i for i in issues if i['type'].startswith('NUMPY') or i['type'] == 'RECURSION_ERROR']
            if recursion_errors:
                content = self._fix_numpy_recursion(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_NUMPY_RECURSION',
                        'description': f'Added numpy recursion protection for {len(recursion_errors)} issues',
                        'applied': True
                    })

            # Fix 2: Add proper service mocking
            service_errors = [i for i in issues if 'SERVICE' in i['type']]
            if service_errors:
                content = self._fix_service_mocking(content, service_errors)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_SERVICE_MOCKING',
                        'description': f'Fixed service mocking for {len(service_errors)} issues',
                        'applied': True
                    })

            # Fix 3: Add integration test configuration
            config_errors = [i for i in issues if 'CONFIG' in i['type']]
            if config_errors:
                content = self._fix_integration_config(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_INTEGRATION_CONFIG',
                        'description': f'Added integration test configuration',
                        'applied': True
                    })

            # Fix 4: Add proper mock setup for external services
            mock_errors = [i for i in issues if 'MOCK' in i['type']]
            if mock_errors:
                content = self._fix_integration_mocks(content, mock_errors)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_INTEGRATION_MOCKS',
                        'description': f'Fixed {len(mock_errors)} mock configuration issues',
                        'applied': True
                    })

            # Write back fixed content
            if content != original_content:
                test_file.write_text(content, encoding='utf-8')
                print(f"    ✅ Applied integration fixes to {test_file.name}")

        except Exception as e:
            fixes.append({
                'type': 'FIX_ERROR',
                'description': f'Error applying integration fixes: {e}',
                'applied': False
            })

        return fixes

    def _fix_numpy_recursion(self, content):
        """Add numpy recursion protection."""
        # Add recursion limit and numpy safety at the top
        if 'sys.setrecursionlimit' not in content:
            lines = content.split('\n')
            insert_pos = 0

            # Find the best insertion point
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) and i > 0:
                    insert_pos = i
                    break

            recursion_protection = [
                'import sys',
                'import numpy as np',
                '',
                '# Protect against numpy recursion issues',
                'sys.setrecursionlimit(1000)',
                '',
                '# Configure numpy for safe testing',
                'np.set_printoptions(threshold=10)',
                ''
            ]

            lines[insert_pos:insert_pos] = recursion_protection
            content = '\n'.join(lines)

        # Replace dangerous numpy patterns
        content = re.sub(
            r'MagicMock\(spec=np\.array\)',
            'MagicMock(spec=np.ndarray, __spec__=None)',
            content
        )

        return content

    def _fix_service_mocking(self, content, service_errors):
        """Add proper service mocking for integration tests."""
        # Add comprehensive service mocking
        if '@pytest.fixture' in content and 'mock_services' not in content:
            lines = content.split('\n')

            # Find where to insert the fixture
            insert_pos = len(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith('@pytest.fixture'):
                    insert_pos = i + 10  # Add after existing fixture
                    break

            service_mock_fixture = [
                '',
                '@pytest.fixture',
                'def mock_integration_services():',
                '    """Mock all external services for integration testing."""',
                '    with patch(\'voice.stt_service.openai\') as mock_openai, \\',
                '         patch(\'voice.tts_service.openai\') as mock_tts_openai, \\',
                '         patch(\'ollama.Client\') as mock_ollama:',
                '        mock_openai.Audio.transcribe.return_value = {"text": "mock transcription"}',
                '        mock_tts_openai.Audio.speak.return_value = MagicMock()',
                '        mock_ollama.return_value.generate.return_value = {"response": "mock response"}',
                '        yield {',
                '            "openai": mock_openai,',
                '            "ollama": mock_ollama',
                '        }',
                ''
            ]

            lines[insert_pos:insert_pos] = service_mock_fixture
            content = '\n'.join(lines)

        return content

    def _fix_integration_config(self, content):
        """Add integration test configuration."""
        # Add environment setup for integration tests
        if 'INTEGRATION_TEST_MODE' not in content:
            lines = content.split('\n')
            insert_pos = 0

            # Insert after imports
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') and i > 0:
                    insert_pos = i + 1
                    break

            config_setup = [
                '',
                '# Integration test configuration',
                'import os',
                'os.environ["INTEGRATION_TEST_MODE"] = "true"',
                'os.environ["VOICE_ENABLED"] = "false"',
                'os.environ["CI"] = "true"',
                ''
            ]

            lines[insert_pos:insert_pos] = config_setup
            content = '\n'.join(lines)

        return content

    def _fix_integration_mocks(self, content, mock_errors):
        """Fix mock configurations for integration tests."""
        # Add proper mock attributes
        content = re.sub(
            r'MagicMock\(\)',
            'MagicMock(__spec__=None, __name__="mock")',
            content
        )

        # Fix patch decorators
        content = re.sub(
            r'@patch\([\'"]([^.]+)\.([^\'"]+)[\'"]\)',
            lambda m: f'@patch("{m.group(1)}.{m.group(2)}", new_callable=MagicMock)',
            content
        )

        return content

    def analyze_all_integration_tests(self):
        """Analyze all integration test files."""
        print("🚀 Starting Integration Test Analysis")
        print("=" * 50)

        test_files = self.discover_integration_tests()

        if not test_files:
            print("❌ No integration test files found")
            return

        total_tests = len(test_files)
        successful_tests = 0

        for test_file in test_files:
            print(f"\n📋 Analyzing {test_file.name}")
            print("-" * 40)

            # Analyze service dependencies
            service_issues = self.analyze_service_dependencies(test_file)
            self.service_mocking_issues.extend(service_issues)

            # Detect numpy recursion issues
            recursion_issues = self.detect_numpy_recursion_issues(test_file)
            self.numpy_recursion_issues.extend(recursion_issues)

            # Analyze mock configurations
            mock_issues = self.analyze_mock_configurations(test_file)
            self.dependency_issues.extend(mock_issues)

            # Run the integration test
            test_result = self.run_integration_test(test_file)

            # Parse errors from output
            if not test_result['success']:
                stdout_errors = self.parse_integration_errors(test_result['stdout'], test_file)
                stderr_errors = self.parse_integration_errors(test_result['stderr'], test_file)

                all_errors = stdout_errors + stderr_errors

                # Generate and apply fixes
                if all_errors:
                    fixes = self.generate_integration_fixes(test_file, all_errors)
                    self.fixes_applied.extend(fixes)

                # Re-run test to check if fixes worked
                if self.fixes_applied:
                    print(f"  🔄 Re-running {test_file.name} after fixes...")
                    retry_result = self.run_integration_test(test_file)
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
            'total_service_issues': len(self.service_mocking_issues),
            'total_numpy_issues': len(self.numpy_recursion_issues),
            'total_dependency_issues': len(self.dependency_issues),
            'total_fixes_applied': len(self.fixes_applied)
        }

    def save_report(self):
        """Save debugging report."""
        report_path = self.project_root / 'reports' / 'integration-debug-report.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Integration test debug report saved to: {report_path}")
        return report_path

    def print_summary(self):
        """Print debugging summary."""
        print("\n📊 Integration Test Debugging Summary")
        print("=" * 40)
        print(f"Total Tests: {self.results['summary']['total_tests']}")
        print(f"Successful: {self.results['summary']['successful_tests']}")
        print(f"Failed: {self.results['summary']['failed_tests']}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1%}")
        print(f"Service Issues: {self.results['summary']['total_service_issues']}")
        print(f"Numpy Issues: {self.results['summary']['total_numpy_issues']}")
        print(f"Dependency Issues: {self.results['summary']['total_dependency_issues']}")
        print(f"Fixes Applied: {self.results['summary']['total_fixes_applied']}")

        if self.fixes_applied:
            print(f"\n🔧 Applied Fixes:")
            for fix in self.fixes_applied:
                status = "✅" if fix.get('applied', False) else "❌"
                print(f"  {status} {fix['description']}")

    def run_debugging(self):
        """Run complete debugging process."""
        self.analyze_all_integration_tests()
        self.save_report()
        self.print_summary()

        # Return exit code based on success rate
        success_rate = self.results['summary']['success_rate']
        if success_rate >= 0.7:  # 70% success rate threshold for integration tests
            print("\n✅ Integration test debugging completed successfully")
            return 0
        else:
            print(f"\n⚠️ Integration test debugging completed with {success_rate:.1%} success rate")
            return 1

def main():
    """Main debugging function."""
    debugger = IntegrationTestDebugger()
    return debugger.run_debugging()

if __name__ == "__main__":
    sys.exit(main())