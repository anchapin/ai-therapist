"""
Test Documentation Generator
Automatically generates documentation from test files
"""

import os
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Optional

class TestDocumentationGenerator:
    """Generate documentation from test files."""
    
    def __init__(self, test_directory: str = "tests"):
        self.test_directory = Path(test_directory)
        self.docs = {}
    
    def parse_test_file(self, file_path: Path) -> Dict:
        """Parse a test file and extract documentation."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {'error': f'Syntax error in {file_path}'}
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': []
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('test'):
                        method_info = {
                            'name': item.name,
                            'docstring': ast.get_docstring(item),
                            'args': [arg.arg for arg in item.args.args]
                        }
                        class_info['methods'].append(method_info)
                
                classes.append(class_info)
            
            elif isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
                func_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args]
                }
                functions.append(func_info)
        
        return {
            'file': str(file_path),
            'classes': classes,
            'functions': functions
        }
    
    def generate_documentation(self) -> str:
        """Generate comprehensive test documentation."""
        doc_lines = [
            "# AI Therapist - Test Documentation",
            "",
            "This documentation is automatically generated from test files.",
            "",
            "## Test Coverage Overview",
            ""
        ]
        
        # Parse all test files
        test_files = list(self.test_directory.rglob("test_*.py"))
        
        for file_path in sorted(test_files):
            if file_path.name == "__init__.py":
                continue
            
            rel_path = file_path.relative_to(self.test_directory)
            section_name = str(rel_path).replace('_', ' ').replace('/', ' - ').title()
            
            doc_lines.append(f"### {section_name}")
            doc_lines.append("")
            doc_lines.append(f"**File:** `{rel_path}`")
            doc_lines.append("")
            
            file_info = self.parse_test_file(file_path)
            
            if 'error' in file_info:
                doc_lines.append(f"❌ {file_info['error']}")
                doc_lines.append("")
                continue
            
            # Document classes
            for cls in file_info['classes']:
                doc_lines.append(f"#### Class: {cls['name']}")
                
                if cls['docstring']:
                    doc_lines.append(f"*{cls['docstring']}*")
                
                doc_lines.append("**Test Methods:**")
                for method in cls['methods']:
                    method_doc = method['docstring'] or "No description available"
                    doc_lines.append(f"- `{method['name']}`: {method_doc}")
                
                doc_lines.append("")
            
            # Document standalone functions
            if file_info['functions']:
                doc_lines.append("**Standalone Test Functions:**")
                for func in file_info['functions']:
                    func_doc = func['docstring'] or "No description available"
                    doc_lines.append(f"- `{func['name']}`: {func_doc}")
                
                doc_lines.append("")
        
        # Add statistics
        total_files = len(test_files)
        total_classes = sum(len(self.parse_test_file(f).get('classes', [])) for f in test_files)
        total_tests = sum(
            len(cls.get('methods', [])) + len(self.parse_test_file(f).get('functions', []))
            for f in test_files
            for cls in self.parse_test_file(f).get('classes', [])
        )
        
        doc_lines.extend([
            "## Statistics",
            "",
            f"- **Test Files:** {total_files}",
            f"- **Test Classes:** {total_classes}",
            f"- **Test Methods:** {total_tests}",
            "",
            "---",
            "*Generated by TestDocumentationGenerator*"
        ])
        
        return "\n".join(doc_lines)
    
    def save_documentation(self, output_path: str = "TEST_DOCUMENTATION.md"):
        """Save documentation to file."""
        doc_content = self.generate_documentation()
        
        with open(output_path, 'w') as f:
            f.write(doc_content)
        
        print(f"📝 Test documentation saved to {output_path}")
        return output_path

if __name__ == "__main__":
    generator = TestDocumentationGenerator()
    output_file = generator.save_documentation()
    print(f"✅ Documentation generated: {output_file}")