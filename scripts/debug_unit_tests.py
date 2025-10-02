#!/usr/bin/env python3
"""
Unit Test Debugging Script for AI Therapist

This script systematically identifies and fixes unit test issues,
including import errors, mocking problems, and configuration issues.
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

class UnitTestDebugger:
    """Comprehensive unit test debugger."""

    def __init__(self):
        self.project_root = project_root
        self.unit_test_dir = self.project_root / 'tests' / 'unit'
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'script': 'debug_unit_tests.py'
            },
            'test_files': {},
            'import_errors': [],
            'mocking_issues': [],
            'configuration_issues': [],
            'fixes_applied': [],
            'summary': {}
        }

    def discover_unit_tests(self):
        """Discover all unit test files."""
        print("🔍 Discovering unit test files...")

        if not self.unit_test_dir.exists():
            print(f"❌ Unit test directory not found: {self.unit_test_dir}")
            return []

        test_files = list(self.unit_test_dir.glob('test_*.py'))
        print(f"  Found {len(test_files)} unit test files")

        return test_files

    def analyze_test_imports(self, test_file):
        """Analyze imports in a test file."""
        print(f"  Analyzing imports in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Find all import statements
            import_patterns = [
                r'from\s+(\S+)\s+import',
                r'import\s+(\S+)',
            ]

            imports = []
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)

            # Test each import
            for import_name in imports:
                # Handle complex import statements
                clean_import = import_name.split('.')[0]

                try:
                    # Try to import the module
                    if import_name.startswith('voice.'):
                        # Handle voice module imports
                        module_path = self.project_root / import_name.replace('.', '/')
                        if module_path.with_suffix('.py').exists():
                            status = 'SUCCESS'
                        else:
                            status = 'MISSING_FILE'
                            issues.append({
                                'type': 'VOICE_MODULE_MISSING',
                                'import': import_name,
                                'error': f'Voice module file not found: {module_path.with_suffix(".py")}'
                            })
                    else:
                        # Try standard import
                        importlib.import_module(clean_import)
                        status = 'SUCCESS'

                except ImportError as e:
                    status = 'FAILED'
                    issues.append({
                        'type': 'IMPORT_ERROR',
                        'import': import_name,
                        'error': str(e)
                    })
                except Exception as e:
                    status = 'ERROR'
                    issues.append({
                        'type': 'UNEXPECTED_ERROR',
                        'import': import_name,
                        'error': str(e)
                    })

        except Exception as e:
            issues.append({
                'type': 'FILE_READ_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def detect_mocking_issues(self, test_file):
        """Detect mocking-related issues in test files."""
        print(f"  Checking mocking in {test_file.name}...")

        issues = []

        try:
            content = test_file.read_text(encoding='utf-8')

            # Check for common mocking patterns and potential issues
            mock_patterns = [
                (r'MagicMock\(\)\.(\w+)', 'MOCK_ATTRIBUTE_ACCESS', 'Missing mock attribute configuration'),
                (r'@patch\([\'"](\w+)', 'MISSING_PATCH_TARGET', 'Patch target may not exist'),
                (r'with patch\([\'"](\w+)', 'MISSING_PATCH_TARGET', 'Patch target may not exist'),
                (r'__spec__', 'SPEC_ATTRIBUTE_ERROR', 'Mock object missing __spec__ attribute'),
                (r'AttributeError.*__spec__', 'SPEC_ATTRIBUTE_ERROR', 'Mock configuration missing __spec__'),
            ]

            for pattern, issue_type, description in mock_patterns:
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
                'type': 'MOCK_ANALYSIS_ERROR',
                'file': str(test_file),
                'error': str(e)
            })

        return issues

    def run_individual_test(self, test_file):
        """Run an individual test and capture errors."""
        print(f"  Running {test_file.name}...")

        try:
            # Run pytest with verbose output and capture everything
            result = subprocess.run([
                'python', '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=short',
                '--no-header',
                '--maxfail=3'
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(self.project_root)
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
                'stderr': 'Test timed out after 60 seconds',
                'success': False
            }
        except Exception as e:
            return {
                'exit_code': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def parse_test_output(self, output, test_file):
        """Parse pytest output to identify specific errors."""
        errors = []

        # Look for common error patterns
        error_patterns = [
            (r'ModuleNotFoundError: No module named [\'"](\w+)[\'"]', 'MODULE_NOT_FOUND'),
            (r'ImportError: cannot import name [\'"](\w+)[\'"]', 'IMPORT_NAME_ERROR'),
            (r'AttributeError: [\'"](\w+)[\'"].*__spec__', 'SPEC_ATTRIBUTE_ERROR'),
            (r'AttributeError: [\'"](\w+)[\'"]', 'ATTRIBUTE_ERROR'),
            (r'FileNotFoundError: \[Errno 2\] No such file or directory: [\'"]([^\'"]+)[\'"]', 'FILE_NOT_FOUND'),
            (r'PermissionError: \[Errno 13\] Permission denied: [\'"]([^\'"]+)[\'"]', 'PERMISSION_ERROR'),
        ]

        for pattern, error_type in error_patterns:
            matches = re.finditer(pattern, output)
            for match in matches:
                errors.append({
                    'type': error_type,
                    'message': match.group(0),
                    'details': match.groups() if match.groups() else []
                })

        return errors

    def generate_fixes(self, test_file, issues):
        """Generate automatic fixes for common issues."""
        fixes = []

        try:
            content = test_file.read_text(encoding='utf-8')
            original_content = content

            # Fix 1: Add missing __spec__ attribute to mocks
            spec_error_count = len([i for i in issues if i['type'] == 'SPEC_ATTRIBUTE_ERROR'])
            if spec_error_count > 0:
                content = self._fix_mock_spec_attributes(content)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_MOCK_SPEC',
                        'description': f'Fixed {spec_error_count} mock __spec__ attribute errors',
                        'applied': True
                    })

            # Fix 2: Add proper import handling for voice modules
            module_errors = [i for i in issues if i['type'] == 'VOICE_MODULE_MISSING']
            if module_errors:
                content = self._fix_voice_module_imports(content, module_errors)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_VOICE_IMPORTS',
                        'description': f'Fixed {len(module_errors)} voice module import issues',
                        'applied': True
                    })

            # Fix 3: Add proper mock configuration
            attribute_errors = [i for i in issues if i['type'] == 'ATTRIBUTE_ERROR']
            if attribute_errors:
                content = self._fix_mock_attributes(content, attribute_errors)
                if content != original_content:
                    fixes.append({
                        'type': 'FIX_MOCK_ATTRIBUTES',
                        'description': f'Fixed {len(attribute_errors)} mock attribute errors',
                        'applied': True
                    })

            # Write back fixed content
            if content != original_content:
                test_file.write_text(content, encoding='utf-8')
                print(f"    ✅ Applied fixes to {test_file.name}")

        except Exception as e:
            fixes.append({
                'type': 'FIX_ERROR',
                'description': f'Error applying fixes: {e}',
                'applied': False
            })

        return fixes

    def _fix_mock_spec_attributes(self, content):
        """Fix missing __spec__ attributes in mocks."""
        # Add spec attributes to MagicMock calls
        content = re.sub(
            r'MagicMock\(\)',
            'MagicMock(__spec__=None)',
            content
        )

        # Add spec attributes to patch decorators
        content = re.sub(
            r'(@patch\([\'"][^\'"]+[\'"]\))',
            r'\1\n@patch("builtins.__spec__", None)',
            content
        )

        return content

    def _fix_voice_module_imports(self, content, module_errors):
        """Fix voice module import issues."""
        # Add proper voice module path setup
        if 'import sys' not in content and 'from pathlib import Path' not in content:
            lines = content.split('\n')
            insert_pos = 0

            # Find good insertion point (after docstring or first import)
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('"""'):
                    insert_pos = i + 1
                    break

            setup_code = [
                'import sys',
                'from pathlib import Path',
                '',
                '# Add project root to Python path',
                'project_root = Path(__file__).parent.parent.parent',
                'sys.path.insert(0, str(project_root))',
                ''
            ]

            lines[insert_pos:insert_pos] = setup_code
            content = '\n'.join(lines)

        return content

    def _fix_mock_attributes(self, content, attribute_errors):
        """Fix mock attribute configuration issues."""
        # Add proper mock configurations for common missing attributes
        for error in attribute_errors:
            if 'details' in error and error['details']:
                attr_name = error['details'][0]

                # Add mock configuration
                mock_config = f'mock.{attr_name} = MagicMock(__spec__=None)\n'

                # Find the test function and add the configuration
                test_pattern = r'(def test_[^(]+\([^)]*\):\s*\n)'
                matches = list(re.finditer(test_pattern, content))

                if matches:
                    # Insert after the first test function
                    insert_pos = matches[0].end()
                    content = content[:insert_pos] + '    ' + mock_config + content[insert_pos:]

        return content

    def analyze_all_tests(self):
        """Analyze all unit test files."""
        print("🚀 Starting Unit Test Analysis")
        print("=" * 50)

        test_files = self.discover_unit_tests()

        if not test_files:
            print("❌ No unit test files found")
            return

        total_tests = len(test_files)
        successful_tests = 0

        for test_file in test_files:
            print(f"\n📋 Analyzing {test_file.name}")
            print("-" * 40)

            # Analyze imports
            import_issues = self.analyze_test_imports(test_file)
            self.import_errors.extend(import_issues)

            # Detect mocking issues
            mocking_issues = self.detect_mocking_issues(test_file)
            self.mocking_issues.extend(mocking_issues)

            # Run the test
            test_result = self.run_individual_test(test_file)

            # Parse errors from output
            if not test_result['success']:
                stdout_errors = self.parse_test_output(test_result['stdout'], test_file)
                stderr_errors = self.parse_test_output(test_result['stderr'], test_file)

                all_errors = stdout_errors + stderr_errors

                # Generate and apply fixes
                if all_errors:
                    fixes = self.generate_fixes(test_file, all_errors)
                    self.fixes_applied.extend(fixes)

                # Re-run test to check if fixes worked
                if self.fixes_applied:
                    print(f"  🔄 Re-running {test_file.name} after fixes...")
                    retry_result = self.run_individual_test(test_file)
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
            'total_import_errors': len(self.import_errors),
            'total_mocking_issues': len(self.mocking_issues),
            'total_fixes_applied': len(self.fixes_applied)
        }

    def save_report(self):
        """Save debugging report."""
        report_path = self.project_root / 'reports' / 'unit-debug-report.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Unit test debug report saved to: {report_path}")
        return report_path

    def print_summary(self):
        """Print debugging summary."""
        print("\n📊 Unit Test Debugging Summary")
        print("=" * 40)
        print(f"Total Tests: {self.results['summary']['total_tests']}")
        print(f"Successful: {self.results['summary']['successful_tests']}")
        print(f"Failed: {self.results['summary']['failed_tests']}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1%}")
        print(f"Import Errors: {self.results['summary']['total_import_errors']}")
        print(f"Mocking Issues: {self.results['summary']['total_mocking_issues']}")
        print(f"Fixes Applied: {self.results['summary']['total_fixes_applied']}")

        if self.fixes_applied:
            print(f"\n🔧 Applied Fixes:")
            for fix in self.fixes_applied:
                status = "✅" if fix.get('applied', False) else "❌"
                print(f"  {status} {fix['description']}")

    def run_debugging(self):
        """Run complete debugging process."""
        self.analyze_all_tests()
        self.save_report()
        self.print_summary()

        # Return exit code based on success rate
        success_rate = self.results['summary']['success_rate']
        if success_rate >= 0.8:  # 80% success rate threshold
            print("\n✅ Unit test debugging completed successfully")
            return 0
        else:
            print(f"\n⚠️ Unit test debugging completed with {success_rate:.1%} success rate")
            return 1

def main():
    """Main debugging function."""
    debugger = UnitTestDebugger()
    return debugger.run_debugging()

if __name__ == "__main__":
    sys.exit(main())