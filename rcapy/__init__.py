"""
RCApy - Reliable Components Analysis for Python
==============================================

A comprehensive Python package for Reliable Components Analysis (RCA) 
applied to EEG data, with specialized tools for music preference studies.

This package provides tools for:
- Core RCA algorithm implementation
- Multi-subject pooled RCA analysis
- Inter-subject correlation (ISC) analysis
- Neural-acoustic coupling analysis
- Music preference correlation studies

Main Components:
- core: Core RCA algorithm and utilities
- analysis: High-level analysis pipelines
- utils: Utility functions for data integration
- examples: Example scripts and demos

Quick Start:
-----------
>>> from rcapy import ReliableComponentsAnalysis
>>> from rcapy.analysis import PooledMultiSubjectRCA
>>> 
>>> # Basic RCA analysis
>>> rca = ReliableComponentsAnalysis(n_components=5)
>>> rca.fit(eeg_data, n_trials=50)
>>> 
>>> # Multi-subject pooled analysis
>>> pooled_rca = PooledMultiSubjectRCA()
>>> pooled_rca.run_complete_analysis()

For detailed documentation, see README_PACKAGE.md
"""

# Version information
__version__ = "1.0.0"
__author__ = "RCApy Development Team"
__email__ = "rcapy.dev@example.com"
__description__ = "Reliable Components Analysis for Python"

# Core imports - these are the main classes users will interact with
from .core.rca import ReliableComponentsAnalysis
from .core.rca_utils import (
    load_music_preference_data,
    epochs_to_rca_format,
    epochs_to_rca_format_fixed_length,
    plot_music_rca_topographies,
    compute_rca_reliability_metrics
)

# Analysis pipeline imports
from .analysis.pooled_rca import PooledMultiSubjectRCA
from .analysis.isc_analysis import RC1InterSubjectCorrelation
from .analysis.neural_acoustic_coupling import RC1SpectralFluxCorrelation
from .analysis.neural_preference import RC1PreferenceAnalysis

# Make main classes available at package level
__all__ = [
    # Core classes
    'ReliableComponentsAnalysis',
    'load_music_preference_data',
    'epochs_to_rca_format', 
    'epochs_to_rca_format_fixed_length',
    'plot_music_rca_topographies',
    'compute_rca_reliability_metrics',
    
    # Analysis pipelines
    'PooledMultiSubjectRCA',
    'RC1InterSubjectCorrelation', 
    'RC1SpectralFluxCorrelation',
    'RC1PreferenceAnalysis',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Package-level constants
DEFAULT_N_COMPONENTS = 5
DEFAULT_SUBJECTS = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
DEFAULT_SONGS = [f"{i}-{j}" for i in range(1, 6) for j in range(1, 4)]

# Convenience function for quick analysis
def run_complete_rca_pipeline(base_path=None, subjects=None, songs=None):
    """
    Run the complete RCA analysis pipeline with default parameters.
    
    Parameters:
    -----------
    base_path : str, optional
        Base path to music preference data (default: current directory)
    subjects : list, optional  
        List of subject IDs (default: pilot_1 through pilot_5)
    songs : list, optional
        List of song IDs (default: 1-1 through 5-3)
    
    Returns:
    --------
    dict : Dictionary containing results from all analysis stages
    """
    from .analysis.complete_pipeline import CompletePipeline
    
    pipeline = CompletePipeline(
        base_path=base_path,
        subjects=subjects or DEFAULT_SUBJECTS,
        songs=songs or DEFAULT_SONGS
    )
    
    return pipeline.run_all_analyses()

# Add convenience function to exports
__all__.append('run_complete_rca_pipeline')