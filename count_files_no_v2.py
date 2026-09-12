#!/usr/bin/env python3
"""
Count files in a directory that do NOT have "v2" in their name.
"""

import os
import sys
from pathlib import Path


def count_files_without_v2(directory: str) -> int:
    """
    Count all files in the given directory (non-recursive) that don't have "v2" in their name.
    
    Args:
        directory: Path to the directory to scan
        
    Returns:
        Number of files without "v2" in their name
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)
    
    count = 0
    for item in dir_path.iterdir():
        if item.is_file() and "v2" not in item.name:
            count += 1
    
    return count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_files_no_v2.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    result = count_files_without_v2(directory)
    print(f"Files without 'v2' in name: {result}")