"""
RCA Analysis Pipelines
======================

This module contains high-level analysis pipelines for RCA-based music preference studies.

Classes:
--------
- PooledMultiSubjectRCA: Multi-subject pooled RCA analysis
- RC1InterSubjectCorrelation: Inter-subject correlation analysis using RC1
- RC1SpectralFluxCorrelation: Neural-acoustic coupling analysis
- RC1PreferenceAnalysis: Neural-preference relationship analysis

Usage:
------
>>> from music_rca.analysis import PooledMultiSubjectRCA
>>> pooled_rca = PooledMultiSubjectRCA()
>>> results = pooled_rca.run_complete_analysis()
"""

from .pooled_rca import PooledMultiSubjectRCA
from .isc_analysis import RC1InterSubjectCorrelation
from .neural_acoustic_coupling import RC1SpectralFluxCorrelation
from .neural_preference import RC1PreferenceAnalysis

__all__ = [
    'PooledMultiSubjectRCA',
    'RC1InterSubjectCorrelation',
    'RC1SpectralFluxCorrelation', 
    'RC1PreferenceAnalysis'
]