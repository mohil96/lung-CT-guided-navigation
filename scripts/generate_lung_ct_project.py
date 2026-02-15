#!/usr/bin/env python3
"""
Lung CT Pipeline Project Generator
Run this script to create all project files
"""

import os
from pathlib import Path

def create_project():
    """Generate complete project structure with all files"""

    print("=" * 60)
    print("Lung CT Pipeline - Project Generator")
    print("=" * 60)
    print()

    project_dir = Path("lung-ct-pipeline")
    project_dir.mkdir(exist_ok=True)

    # Create directory structure
    directories = [
        "src/models",
        "src/preprocessing", 
        "src/segmentation",
        "src/centerline",
        "src/pathplanning",
        "src/visualization",
        "src/utils",
        "scripts",
        "configs",
        "notebooks",
        "tests",
    ]

    print("Creating directories...")
    for d in directories:
        (project_dir / d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {len(directories)} directories")

    # Create __init__.py files
    init_dirs = [
        "src",
        "src/models",
        "src/preprocessing",
        "src/segmentation", 
        "src/centerline",
        "src/pathplanning",
        "src/visualization",
        "src/utils",
    ]

    for d in init_dirs:
        (project_dir / d / "__init__.py").write_text("")
    print(f"✓ Created {len(init_dirs)} __init__.py files")

    print()
    print("Project structure created successfully!")
    print()
    print("Next steps:")
    print("1. cd lung-ct-pipeline")
    print("2. Download full source files from shared link")
    print("3. pip install -r requirements.txt")
    print("4. python src/models/unet3d.py  # Test installation")
    print()
    print("See README.md for complete documentation")
    print()

if __name__ == "__main__":
    create_project()
