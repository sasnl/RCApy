# RCApy

A comprehensive Python package for **Reliable Components Analysis (RCA)** applied to EEG data, with specialized tools for music studies.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://rcapy.readthedocs.io/)

## Overview

This package implements a complete pipeline for analyzing EEG responses to music using Reliable Components Analysis. It identifies neural components that are reliable across trials and subjects, enabling robust analysis of inter-subject correlation, neural-acoustic coupling, and music preference relationships.

## Key Features

- 🧠 **Core RCA Algorithm**: Robust implementation of Reliable Components Analysis
- 🎵 **Music-Specific Tools**: Specialized functions for music preference EEG studies
- 📊 **Multi-Subject Analysis**: Pooled RCA across multiple participants
- 🔗 **Inter-Subject Correlation**: Measure neural synchrony between participants
- 🎼 **Neural-Acoustic Coupling**: Correlate neural responses with music features
- ❤️ **Preference Analysis**: Investigate neural-preference relationships
- 📈 **Comprehensive Visualization**: Publication-ready plots and topographic maps

## Installation

### From PyPI (when available)
```bash
pip install rcapy
```

### From Source
```bash
git clone https://github.com/rcapy/rcapy.git
cd rcapy
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/sasnl/RCApy.git
cd rcapy
pip install -e .[dev]
```

## Quick Start

### Basic RCA Analysis
```python
from rcapy import ReliableComponentsAnalysis
import numpy as np

# Create synthetic EEG data (channels x timepoints x trials)
eeg_data = np.random.randn(32, 1000, 50)

# Run RCA
rca = ReliableComponentsAnalysis(n_components=5)
rca.fit(eeg_data, n_trials=50)

# Get reliable components
components = rca.components_
eigenvalues = rca.eigenvalues_

print(f"RC1 reliability: {eigenvalues[0]:.4f}")
```

### Complete Analysis Pipeline
```python
from rcapy import run_complete_rca_pipeline

# Run all analysis stages
results = run_complete_rca_pipeline(
    base_path="/path/to/music/data",
    subjects=['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
)

# Results contain all analysis outputs
pooled_rca = results['stage1_pooled_rca']
isc_analysis = results['stage2_isc']
coupling_analysis = results['stage3_coupling']
preference_analysis = results['stage4_preference']
```

### Individual Analysis Components
```python
from rcapy.analysis import (
    PooledMultiSubjectRCA,
    RC1InterSubjectCorrelation,
    RC1SpectralFluxCorrelation,
    RC1PreferenceAnalysis
)

# Stage 1: Pooled RCA
pooled_rca = PooledMultiSubjectRCA()
rca_results = pooled_rca.run_complete_analysis()

# Stage 2: Inter-subject correlations  
isc_analyzer = RC1InterSubjectCorrelation()
isc_results = isc_analyzer.run_complete_analysis()

# Stage 3: Neural-acoustic coupling
coupling_analyzer = RC1SpectralFluxCorrelation()
coupling_results = coupling_analyzer.run_complete_analysis()

# Stage 4: Neural-preference relationships
preference_analyzer = RC1PreferenceAnalysis()
preference_results = preference_analyzer.run_complete_analysis()
```

## Command Line Interface

The package provides command-line tools for common analyses:

```bash
# Run pooled RCA analysis
rcapy-pooled --base-path /path/to/data

# Run ISC analysis
rcapy-isc --base-path /path/to/data

# Run complete pipeline
rcapy-pipeline --base-path /path/to/data

# Run demo analysis
rcapy-demo
```

## Package Structure

```
rcapy/
├── core/                   # Core RCA implementation
│   ├── rca.py             # Main RCA algorithm
│   └── rca_utils.py       # Utility functions
├── analysis/              # Analysis pipelines
│   ├── pooled_rca.py      # Multi-subject pooled RCA
│   ├── isc_analysis.py    # Inter-subject correlation
│   ├── neural_acoustic_coupling.py  # Neural-acoustic coupling
│   ├── neural_preference.py         # Neural-preference analysis
│   └── complete_pipeline.py         # Unified pipeline
├── utils/                 # Utility modules
│   ├── plotting.py        # Visualization utilities
│   └── validation.py      # Data validation
└── examples/              # Example scripts and demos
    └── demos.py          # Basic demonstrations
```

## Analysis Pipeline

The package implements a four-stage analysis pipeline:

### Stage 1: Pooled Multi-Subject RCA
- Pools EEG data from all subjects before running RCA
- Finds components reliable across the entire dataset
- Generates topographic visualizations of spatial patterns

### Stage 2: Inter-Subject Correlation (ISC)
- Applies RC1 spatial filter to extract neural timecourses
- Computes correlations between subjects for each song
- Creates comprehensive correlation heatmaps

### Stage 3: Neural-Acoustic Coupling
- Correlates RC1-filtered responses with spectral flux features
- Analyzes subject-specific and song-specific coupling patterns
- Investigates temporal alignment of neural and acoustic dynamics

### Stage 4: Neural-Preference Relationships
- Examines relationships between neural coupling and behavioral preferences
- Identifies individual differences in neural-preference patterns
- Categorizes songs by neural coupling and preference levels

## Data Requirements

### EEG Data
- **Format**: MNE-Python compatible (.fif files)
- **Preprocessing**: ICA-cleaned, filtered, and epoched
- **Structure**: Standard 10-20 electrode montage (32 channels recommended)
- **Sampling rate**: 1000 Hz recommended

### Music Features
- **Spectral flux**: Pre-computed at 128 Hz sampling rate
- **Format**: NumPy arrays (.npz files)
- **Required features**: `spectral_flux`, `time_s` arrays

### Behavioral Data
- **Preference ratings**: JSON format with subject-song ratings
- **Scale**: 1-9 preference scale
- **Structure**: Hierarchical by subject and song ID

## Scientific Background

**Reliable Components Analysis (RCA)** identifies neural components that show high trial-to-trial reliability by solving a generalized eigenvalue problem that maximizes the ratio of cross-trial covariance to trial-average covariance.

**Reference:**
Dmochowski, J. P., Sajda, P., Dias, J., & Parra, L. C. (2012). Correlated components of ongoing EEG point to emotionally laden attention–a possible marker of engagement?. Frontiers in human neuroscience, 6.

**Mathematical Foundation:**
```
R̂ = argmax (W^T R_xx W) / (W^T R_x̄x̄ W)
```

Where:
- `R_xx`: Cross-trial covariance matrix  
- `R_x̄x̄`: Trial-average covariance matrix
- `W`: Spatial filter weights

**Applications:**
- Inter-subject correlation analysis
- Neural synchrony measurement
- Neural-acoustic coupling studies
- Music preference neural mechanisms

## Key Results

Based on analysis of music preference EEG data:

- **RC1 Properties**: λ = 0.003406, frontocentral (FC2) spatial pattern
- **Inter-Subject Correlation**: Mean ISC = 0.007 ± 0.026 across songs
- **Neural-Acoustic Coupling**: Weak but measurable (r = 0.0024 ± 0.0142)
- **Neural-Preference**: No universal relationship, strong individual differences

## Dependencies

### Core Requirements
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `matplotlib >= 3.5.0`
- `seaborn >= 0.11.0` 
- `pandas >= 1.3.0`
- `mne >= 1.5.0`
- `h5py >= 3.0.0`
- `scikit-learn >= 1.0.0`

### Optional Dependencies
- `joblib >= 1.0.0` (for parallel processing)
- `plotly >= 5.0.0` (for interactive plots)
- `pytest >= 6.0.0` (for testing)

## Examples and Tutorials

### Example 1: Basic RCA with Topographic Visualization
```python
from rcapy import ReliableComponentsAnalysis
from rcapy.core import plot_music_rca_topographies
import mne

# Load your EEG data
epochs = mne.read_epochs('your_data.fif')
data = epochs.get_data()  # Shape: (trials, channels, timepoints)

# Prepare data for RCA (channels x timepoints x trials)
rca_data = data.transpose(1, 2, 0)

# Run RCA
rca = ReliableComponentsAnalysis(n_components=5)
rca.fit(rca_data, n_trials=data.shape[0])

# Plot topographies
fig = plot_music_rca_topographies(
    rca.components_.T, 
    epochs.info,
    title="RCA Components"
)
```

### Example 2: Custom Analysis Pipeline
```python
from rcapy.analysis import CompletePipeline

# Create custom pipeline
pipeline = CompletePipeline(
    base_path="/your/data/path",
    subjects=['subj1', 'subj2', 'subj3'], 
    songs=['song1', 'song2', 'song3']
)

# Run specific stages
results = pipeline.run_all_analyses(stages=[1, 2])  # Only RCA and ISC

# Generate report
report = pipeline.generate_summary_report('analysis_summary.txt')
print(report)
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rcapy

# Run specific test modules
pytest rcapy/tests/test_rca.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rcapy,
    title = {RCApy: Reliable Components Analysis for Python},
    author = {Tong Shan},
    year = {2025},
    url = {https://github.com/rcapy/rcapy},
    version = {1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Based on RCA methodology by Dmochowski et al. (2012)
- MNE-Python community for EEG analysis tools
- Music preference study participants and research team
