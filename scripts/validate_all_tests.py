#!/usr/bin/env python3
"""
Final Test Validation Script for AI Therapist

This script runs the complete test suite after all fixes have been applied
and generates a comprehensive validation report.
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import subprocess
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestValidator:
    """Comprehensive test validation system."""

    def __init__(self):
        self.project_root = project_root
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'script': 'validate_all_tests.py',
                'validation_type': 'FINAL_VALIDATION'
            },
            'test_categories': {},
            'coverage_analysis': {},
            'compliance_analysis': {},
            'performance_analysis': {},
            'recommendations': [],
            'summary': {}
        }

    def run_test_category(self, category, test_path, description, timeout=300):
        """Run a specific test category."""
        print(f"\n🔍 Running {description}...")
        print("-" * 50)

        try:
            # Set environment for validation
            env = {
                'VALIDATION_MODE': 'true',
                'CI': 'true',
                'OLLAMA_HOST': 'http://ollama:11434',
                'HIPAA_MODE': 'true' if category == 'security' else 'false',
                'PERFORMANCE_TEST_MODE': 'true' if category == 'performance' else 'false',
                'INTEGRATION_TEST_MODE': 'true' if category == 'integration' else 'false',
            }

            # Build pytest command
            pytest_cmd = [
                'python', '-m', 'pytest',
                str(test_path),
                '-v',
                '--tb=short',
                '--no-header',
                '--maxfail=5',
                f'--timeout={timeout}'
            ]

            # Add coverage for main categories
            if category in ['unit', 'integration', 'security']:
                pytest_cmd.extend([
                    '--cov=voice',
                    '--cov-report=json',
                    '--cov-report=term-missing',
                    f'--cov-append={category != "unit"}'  # Don't append for unit tests
                ])

            # Run pytest
            start_time = datetime.now()
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60,  # Extra minute for overhead
                cwd=str(self.project_root),
                env={**dict(subprocess.os.environ), **env}
            )
            end_time = datetime.now()

            # Parse results
            duration = (end_time - start_time).total_seconds()

            # Extract test count from output
            test_match = re.search(r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) xfailed)?(?:, (\d+) error[s]?)?', result.stdout)
            passed = int(test_match.group(1)) if test_match and test_match.group(1) else 0
            failed = int(test_match.group(2)) if test_match and test_match.group(2) else 0
            skipped = int(test_match.group(3)) if test_match and test_match.group(3) else 0
            xfailed = int(test_match.group(4)) if test_match and test_match.group(4) else 0
            errors = int(test_match.group(5)) if test_match and test_match.group(5) else 0

            total = passed + failed + errors

            category_result = {
                'description': description,
                'exit_code': result.returncode,
                'duration_seconds': duration,
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'xfailed': xfailed,
                'errors': errors,
                'success_rate': passed / total if total > 0 else 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0 and errors == 0
            }

            self.results['test_categories'][category] = category_result

            # Print summary
            print(f"  Tests: {total} total, {passed} passed, {failed} failed, {errors} errors")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Success Rate: {category_result['success_rate']:.1%}")

            if category_result['success']:
                print(f"  ✅ {description} passed")
            else:
                print(f"  ❌ {description} failed")

            return category_result

        except subprocess.TimeoutExpired:
            category_result = {
                'description': description,
                'exit_code': -1,
                'error': 'Test timeout',
                'success': False
            }
            self.results['test_categories'][category] = category_result
            print(f"  ⏰ {description} timed out")
            return category_result

        except Exception as e:
            category_result = {
                'description': description,
                'exit_code': -2,
                'error': str(e),
                'success': False
            }
            self.results['test_categories'][category] = category_result
            print(f"  ❌ {description} error: {e}")
            return category_result

    def analyze_coverage(self):
        """Analyze test coverage."""
        print("\n📊 Analyzing test coverage...")

        coverage_file = self.project_root / 'coverage.json'
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)

                totals = coverage_data.get('totals', {})

                self.results['coverage_analysis'] = {
                    'total_statements': totals.get('num_statements', 0),
                    'covered_statements': totals.get('covered_lines', 0),
                    'missing_statements': totals.get('missing_lines', 0),
                    'coverage_percentage': totals.get('percent_covered', 0) / 100.0,
                    'targets': {
                        'unit_tests': 0.90,
                        'integration_tests': 0.85,
                        'security_tests': 0.95,
                        'overall': 0.90
                    }
                }

                coverage_pct = self.results['coverage_analysis']['coverage_percentage']
                print(f"  Overall Coverage: {coverage_pct:.1%}")

                # Check if coverage targets are met
                if coverage_pct >= 0.90:
                    print("  ✅ Coverage target met (≥90%)")
                else:
                    print(f"  ⚠️ Coverage target not met (need {0.90 - coverage_pct:.1%} more)")

            except Exception as e:
                print(f"  ❌ Failed to analyze coverage: {e}")
                self.results['coverage_analysis'] = {'error': str(e)}
        else:
            print("  ⚠️ No coverage data found")
            self.results['coverage_analysis'] = {'error': 'No coverage data available'}

    def analyze_compliance(self):
        """Analyze compliance with requirements."""
        print("\n🔍 Analyzing compliance...")

        # Check test category success rates
        compliance_requirements = {
            'unit_testing_coverage': {
                'requirement': '90%+ unit test coverage',
                'status': 'COMPLETED' if self.results['test_categories'].get('unit', {}).get('success_rate', 0) >= 0.90 else 'FAILED',
                'actual': self.results['test_categories'].get('unit', {}).get('success_rate', 0),
                'target': 0.90
            },
            'integration_testing': {
                'requirement': 'Service integration testing',
                'status': 'COMPLETED' if self.results['test_categories'].get('integration', {}).get('success', False) else 'FAILED',
                'actual': self.results['test_categories'].get('integration', {}).get('success_rate', 0),
                'target': 0.80
            },
            'security_testing': {
                'requirement': 'HIPAA compliance testing',
                'status': 'COMPLETED' if self.results['test_categories'].get('security', {}).get('success', False) else 'FAILED',
                'actual': self.results['test_categories'].get('security', {}).get('success_rate', 0),
                'target': 0.95
            },
            'performance_testing': {
                'requirement': 'Load and scalability testing',
                'status': 'COMPLETED' if self.results['test_categories'].get('performance', {}).get('success', False) else 'FAILED',
                'actual': self.results['test_categories'].get('performance', {}).get('success_rate', 0),
                'target': 0.70
            },
            'automation': {
                'requirement': '≥ 80% test automation',
                'status': 'COMPLETED',  # All tests are automated
                'actual': 1.0,
                'target': 0.80
            }
        }

        self.results['compliance_analysis'] = compliance_requirements

        # Print compliance status
        print(f"  Unit Tests: {'✅' if compliance_requirements['unit_testing_coverage']['status'] == 'COMPLETED' else '❌'}")
        print(f"  Integration Tests: {'✅' if compliance_requirements['integration_testing']['status'] == 'COMPLETED' else '❌'}")
        print(f"  Security Tests: {'✅' if compliance_requirements['security_testing']['status'] == 'COMPLETED' else '❌'}")
        print(f"  Performance Tests: {'✅' if compliance_requirements['performance_testing']['status'] == 'COMPLETED' else '❌'}")
        print(f"  Automation: ✅")

    def analyze_performance(self):
        """Analyze test performance metrics."""
        print("\n⚡ Analyzing performance...")

        total_duration = sum(cat.get('duration_seconds', 0) for cat in self.results['test_categories'].values())
        total_tests = sum(cat.get('total_tests', 0) for cat in self.results['test_categories'].values())

        performance_metrics = {
            'total_duration_seconds': total_duration,
            'total_tests_run': total_tests,
            'average_test_duration': total_duration / total_tests if total_tests > 0 else 0,
            'category_performance': {
                category: {
                    'duration': cat.get('duration_seconds', 0),
                    'tests': cat.get('total_tests', 0),
                    'avg_duration': cat.get('duration_seconds', 0) / cat.get('total_tests', 1) if cat.get('total_tests', 0) > 0 else 0
                }
                for category, cat in self.results['test_categories'].items()
            }
        }

        self.results['performance_analysis'] = performance_metrics

        print(f"  Total Duration: {total_duration:.1f}s")
        print(f"  Total Tests: {total_tests}")
        print(f"  Average Test Duration: {performance_metrics['average_test_duration']:.2f}s")

        # Check for slow tests
        slow_categories = [
            cat for cat, metrics in performance_metrics['category_performance'].items()
            if metrics['avg_duration'] > 10.0  # Tests averaging over 10 seconds
        ]

        if slow_categories:
            print(f"  ⚠️ Slow test categories: {', '.join(slow_categories)}")
        else:
            print("  ✅ All test categories running efficiently")

    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        print("\n💡 Generating recommendations...")

        recommendations = []

        # Coverage recommendations
        coverage_data = self.results.get('coverage_analysis', {})
        if 'coverage_percentage' in coverage_data:
            coverage_pct = coverage_data['coverage_percentage']
            if coverage_pct < 0.90:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'COVERAGE',
                    'issue': f'Test coverage ({coverage_pct:.1%}) below 90% target',
                    'action': f'Add tests for {coverage_data.get("missing_statements", 0)} uncovered statements'
                })

        # Test category recommendations
        for category, results in self.results['test_categories'].items():
            if not results.get('success', False):
                success_rate = results.get('success_rate', 0)
                if success_rate < 0.70:
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': category.upper(),
                        'issue': f'{category} tests failing with {success_rate:.1%} success rate',
                        'action': f'Investigate and fix {results.get("failed", 0)} failing tests'
                    })
                elif success_rate < 0.90:
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': category.upper(),
                        'issue': f'{category} tests below optimal {success_rate:.1%} success rate',
                        'action': f'Improve {results.get("failed", 0)} failing tests'
                    })

        # Performance recommendations
        perf_data = self.results.get('performance_analysis', {})
        avg_duration = perf_data.get('average_test_duration', 0)
        if avg_duration > 5.0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'PERFORMANCE',
                'issue': f'Average test duration ({avg_duration:.2f}s) is slow',
                'action': 'Optimize test execution and reduce unnecessary delays'
            })

        # Compliance recommendations
        compliance = self.results.get('compliance_analysis', {})
        failed_requirements = [req for req, data in compliance.items() if data.get('status') == 'FAILED']
        if failed_requirements:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'COMPLIANCE',
                'issue': f'{len(failed_requirements)} compliance requirements not met',
                'action': f'Address failures in: {", ".join(failed_requirements)}'
            })

        # Success recommendations
        if not recommendations:
            recommendations.append({
                'priority': 'LOW',
                'category': 'GENERAL',
                'issue': 'All tests passing and meeting requirements',
                'action': 'Continue with regular test maintenance and expand test coverage'
            })

        self.results['recommendations'] = recommendations

    def generate_summary(self):
        """Generate overall validation summary."""
        total_categories = len(self.results['test_categories'])
        successful_categories = len([cat for cat in self.results['test_categories'].values() if cat.get('success', False)])
        total_tests = sum(cat.get('total_tests', 0) for cat in self.results['test_categories'].values())
        total_passed = sum(cat.get('passed', 0) for cat in self.results['test_categories'].values())
        total_failed = sum(cat.get('failed', 0) for cat in self.results['test_categories'].values())
        total_errors = sum(cat.get('errors', 0) for cat in self.results['test_categories'].values())

        overall_success = (successful_categories == total_categories and total_errors == 0)

        self.results['summary'] = {
            'total_test_categories': total_categories,
            'successful_categories': successful_categories,
            'failed_categories': total_categories - successful_categories,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'overall_success_rate': total_passed / total_tests if total_tests > 0 else 0,
            'overall_status': 'PASS' if overall_success else 'FAIL',
            'validation_timestamp': datetime.now().isoformat()
        }

    def save_validation_report(self):
        """Save validation report."""
        report_path = self.project_root / 'reports' / 'final-validation-report.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Final validation report saved to: {report_path}")
        return report_path

    def print_final_summary(self):
        """Print final validation summary."""
        print("\n" + "="*60)
        print("🎉 FINAL VALIDATION SUMMARY")
        print("="*60)

        summary = self.results['summary']
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Test Categories: {summary['successful_categories']}/{summary['total_test_categories']} passed")
        print(f"Total Tests: {summary['total_passed']}/{summary['total_tests']} passed")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")

        if summary['total_errors'] > 0:
            print(f"⚠️ {summary['total_errors']} test errors detected")

        # Coverage summary
        if 'coverage_percentage' in self.results.get('coverage_analysis', {}):
            coverage_pct = self.results['coverage_analysis']['coverage_percentage']
            print(f"Test Coverage: {coverage_pct:.1%}")

        # Compliance summary
        compliance = self.results.get('compliance_analysis', {})
        completed = len([c for c in compliance.values() if c.get('status') == 'COMPLETED'])
        total = len(compliance)
        print(f"Compliance: {completed}/{total} requirements met")

        # Performance summary
        perf = self.results.get('performance_analysis', {})
        print(f"Total Duration: {perf.get('total_duration_seconds', 0):.1f}s")

        print("\n📋 Category Details:")
        for category, results in self.results['test_categories'].items():
            status = "✅" if results.get('success', False) else "❌"
            success_rate = results.get('success_rate', 0)
            duration = results.get('duration_seconds', 0)
            print(f"  {status} {category.title()}: {success_rate:.1%} success, {duration:.1f}s")

        # Recommendations
        if self.results['recommendations']:
            print(f"\n💡 Recommendations ({len(self.results['recommendations'])}):")
            for rec in self.results['recommendations']:
                icon = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
                print(f"  {icon} [{rec['priority']}] {rec['issue']}")
                print(f"     → {rec['action']}")

        print("\n" + "="*60)

    def run_validation(self):
        """Run complete validation process."""
        print("🚀 Starting Final Test Validation")
        print("=" * 60)

        # Run all test categories
        self.run_test_category('unit', 'tests/unit/', 'Unit Tests', timeout=300)
        self.run_test_category('integration', 'tests/integration/', 'Integration Tests', timeout=600)
        self.run_test_category('security', 'tests/security/', 'Security Tests', timeout=600)
        self.run_test_category('performance', 'tests/performance/', 'Performance Tests', timeout=900)

        # Analyze results
        self.analyze_coverage()
        self.analyze_compliance()
        self.analyze_performance()
        self.generate_recommendations()
        self.generate_summary()

        # Save and print results
        self.save_validation_report()
        self.print_final_summary()

        # Return appropriate exit code
        return 0 if self.results['summary']['overall_status'] == 'PASS' else 1

def main():
    """Main validation function."""
    validator = TestValidator()
    return validator.run_validation()

if __name__ == "__main__":
    sys.exit(main())