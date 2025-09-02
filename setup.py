#!/usr/bin/env python
"""
Setup script for RCApy package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README_PACKAGE.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "RCApy: A comprehensive Python package for Reliable Components Analysis (RCA) applied to EEG data."

# Read version from __init__.py
def read_version():
    """Extract version from __init__.py."""
    init_path = os.path.join(os.path.dirname(__file__), 'rcapy', '__init__.py')
    if os.path.exists(init_path):
        with open(init_path, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="rcapy",
    version=read_version(),
    description="Reliable Components Analysis for Python",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="RCApy Development Team",
    author_email="rcapy.dev@example.com",
    url="https://github.com/rcapy/rcapy",
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "mne>=1.5.0",
        "h5py>=3.0.0",
        "scikit-learn>=1.0.0",
        "pathlib2;python_version<'3.4'",
    ],
    
    # Optional dependencies for advanced features
    extras_require={
        'advanced': [
            'joblib>=1.0.0',
            'psutil>=5.8.0',
            'plotly>=5.0.0'
        ],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0'
        ]
    },
    
    # Entry points for command line tools
    entry_points={
        'console_scripts': [
            'rcapy-pooled=rcapy.analysis.pooled_rca:main',
            'rcapy-isc=rcapy.analysis.isc_analysis:main', 
            'rcapy-demo=rcapy.examples.demos:main',
            'rcapy-pipeline=rcapy.analysis.complete_pipeline:main',
        ],
    },
    
    # Package data
    package_data={
        'rcapy': [
            'data/*.json',
            'examples/data/*.npz',
            'examples/notebooks/*.ipynb'
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    
    # Keywords
    keywords=[
        "EEG", "neuroscience", "music", "RCA", "reliable components analysis", 
        "inter-subject correlation", "neural synchrony", "music preference",
        "brain imaging", "signal processing"
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://rcapy.readthedocs.io/',
        'Source': 'https://github.com/rcapy/rcapy',
        'Bug Reports': 'https://github.com/rcapy/rcapy/issues',
    },
)