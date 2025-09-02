"""
RCA Utility Functions
====================

This module contains utility functions for visualization, data validation,
and helper functions for the RCA analysis pipeline.

Functions:
----------
- Visualization utilities
- Data validation functions
- Helper functions for plotting and analysis

For detailed API documentation, see individual module docstrings.
"""

from .plotting import (
    plot_correlation_matrix,
    plot_song_level_analysis
)
from .validation import validate_rca_conversion

__all__ = [
    'plot_correlation_matrix',
    'plot_song_level_analysis', 
    'validate_rca_conversion'
]