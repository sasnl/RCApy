"""
RCA Examples and Demos
=====================

This module contains example scripts and demonstrations of the RCA package
functionality.

Modules:
--------
- demos: Basic demonstration scripts
- tutorials: Step-by-step tutorials

Usage:
------
>>> from music_rca.examples import run_basic_demo
>>> run_basic_demo()
"""

from .demos import (
    demo_rca_with_music_data,
    demo_synthetic_rca,
    main
)

__all__ = [
    'demo_rca_with_music_data',
    'demo_synthetic_rca',
    'main'
]