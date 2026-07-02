#!/usr/bin/env python3
"""
Validate that issue template filenames contain only ASCII characters.

This prevents invisible Unicode characters from breaking GitHub's
issue template recognition.
"""

import os
import sys
from pathlib import Path


def validate_template_filenames(template_dir: Path) -> bool:
    """
    Check all files in the template directory for non-ASCII characters.

    Returns True if all filenames are valid, False otherwise.
    """
    valid = True

    if not template_dir.exists():
        print(f"Template directory not found: {template_dir}")
        return True  # Not an error if directory doesn't exist

    for file_path in template_dir.iterdir():
        if file_path.is_file():
            filename = file_path.name
            try:
                filename.encode('ascii')
            except UnicodeEncodeError:
                print(f"ERROR: Non-ASCII characters found in filename: {filename}")
                print(f"  Full path: {file_path}")
                print(f"  Hex dump: {filename.encode('utf-8').hex()}")
                valid = False

    return valid


def main():
    """Main entry point."""
    # Find the template directory relative to the repository root
    # The script is in scripts/, so the repo root is one level up
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    template_dir = repo_root / ".github" / "ISSUE_TEMPLATE"

    print(f"Validating issue template filenames in: {template_dir}")

    if validate_template_filenames(template_dir):
        print("All template filenames are valid (ASCII only)")
        sys.exit(0)
    else:
        print("\nPlease rename the files to remove non-ASCII characters")
        sys.exit(1)


if __name__ == "__main__":
    main()
