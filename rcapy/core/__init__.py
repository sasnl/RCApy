"""
Core RCA Implementation
======================

This module contains the core Reliable Components Analysis implementation
and essential utility functions for music preference EEG analysis.

Classes:
--------
- ReliableComponentsAnalysis: Main RCA algorithm implementation
- Utility functions for data loading and preprocessing

For detailed API documentation, see individual module docstrings.
"""

from .rca import ReliableComponentsAnalysis, demo_rca_analysis
from .rca_utils import (
    load_music_preference_data,
    epochs_to_rca_format,
    epochs_to_rca_format_fixed_length,
    plot_music_rca_topographies,
    compute_rca_reliability_metrics,
    run_rca_on_music_data,
    plot_music_rca_results,
    save_rca_results,
    batch_rca_analysis
)

__all__ = [
    'ReliableComponentsAnalysis',
    'demo_rca_analysis',
    'load_music_preference_data',
    'epochs_to_rca_format',
    'epochs_to_rca_format_fixed_length', 
    'plot_music_rca_topographies',
    'compute_rca_reliability_metrics',
    'run_rca_on_music_data',
    'plot_music_rca_results',
    'save_rca_results',
    'batch_rca_analysis'
]