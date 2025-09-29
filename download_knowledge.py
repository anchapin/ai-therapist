#!/usr/bin/env python3
"""
Script to download knowledge files from URLs specified in knowledge_files.txt
"""

import os
import requests
from pathlib import Path
from typing import List, Tuple
import sys

def load_knowledge_files_config() -> List[Tuple[str, str]]:
    """Load knowledge file configuration from knowledge_files.txt"""
    config_path = Path(__file__).parent / "knowledge_files.txt"
    files_to_download = []

    if not config_path.exists():
        print(f"Error: {config_path} not found")
        return files_to_download

    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('|', 1)
            if len(parts) != 2:
                print(f"Warning: Invalid format on line {line_num}: {line}")
                continue

            filename, url = parts
            files_to_download.append((filename.strip(), url.strip()))

    return files_to_download

def download_file(filename: str, url: str, knowledge_dir: Path) -> bool:
    """Download a single file from URL"""
    file_path = knowledge_dir / filename

    try:
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create the file
        with open(file_path, 'wb') as f:
            f.write(response.content)

        print(f"Successfully downloaded {filename} ({len(response.content)} bytes)")
        return True

    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return False
    except IOError as e:
        print(f"Error saving {filename}: {e}")
        return False

def check_missing_files(files_config: List[Tuple[str, str]], knowledge_dir: Path) -> List[Tuple[str, str]]:
    """Check which files are missing from knowledge directory"""
    missing_files = []

    for filename, url in files_config:
        file_path = knowledge_dir / filename
        if not file_path.exists():
            missing_files.append((filename, url))

    return missing_files

def main():
    """Main function to download knowledge files"""
    print("AI Therapist Knowledge File Downloader")
    print("=" * 40)

    # Get project root directory
    project_root = Path(__file__).parent
    knowledge_dir = project_root / "knowledge"

    # Create knowledge directory if it doesn't exist
    knowledge_dir.mkdir(exist_ok=True)
    print(f"Knowledge directory: {knowledge_dir}")

    # Load configuration
    files_config = load_knowledge_files_config()
    if not files_config:
        print("No files configured for download")
        return 1

    print(f"Found {len(files_config)} files in configuration")

    # Check for missing files
    missing_files = check_missing_files(files_config, knowledge_dir)

    if not missing_files:
        print("All knowledge files are already present!")
        return 0

    print(f"Found {len(missing_files)} missing files to download:")
    for filename, url in missing_files:
        print(f"  - {filename}")

    print()

    # Download missing files
    success_count = 0
    failure_count = 0

    for filename, url in missing_files:
        if download_file(filename, url, knowledge_dir):
            success_count += 1
        else:
            failure_count += 1

    print()
    print(f"Download complete: {success_count} successful, {failure_count} failed")

    if failure_count > 0:
        print("Some files failed to download. Please check the URLs and try again.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())