# Installation Guide - RCApy Package

## Quick Installation

### From Local Directory (Recommended for Development)
```bash
cd /Users/tongshan/Documents/music_preference/code/analysis/rca_python
pip install -e .
```

### Standard Installation
```bash
cd /Users/tongshan/Documents/music_preference/code/analysis/rca_python
pip install .
```

## Verify Installation

Test that the package installed correctly:

```python
# Test basic imports
from rcapy import ReliableComponentsAnalysis, run_complete_rca_pipeline
from rcapy.analysis import PooledMultiSubjectRCA
from rcapy.core import load_music_preference_data

print("✅ RCApy package installed successfully!")
print(f"Version: {rcapy.__version__}")
```

## Development Installation

For development with additional tools:

```bash
pip install -e .[dev]
```

This includes testing, documentation, and development tools.

## Usage Examples

### Quick Test
```python
import rcapy
import numpy as np

# Create sample data
data = np.random.randn(32, 1000, 50)  # (channels, timepoints, trials)

# Run RCA
rca = rcapy.ReliableComponentsAnalysis(n_components=5)
rca.fit(data, n_trials=50)

print(f"RC1 reliability: {rca.eigenvalues_[0]:.4f}")
print("✅ Basic RCA test passed!")
```

### Complete Pipeline Test
```python
# This requires actual EEG data
from rcapy import run_complete_rca_pipeline

# Uncomment when you have data:
# results = run_complete_rca_pipeline(
#     base_path="/path/to/your/data",
#     subjects=['pilot_1', 'pilot_2'],
#     stages=[1, 2]  # Just run RCA and ISC
# )
```

## Troubleshooting

### Import Errors
If you get import errors, try:
```bash
pip install --upgrade -e .
```

### Missing Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### Development Setup
For full development environment:
```bash
pip install -r requirements-dev.txt
```

## Package Structure Verification

Run this to verify the package structure:
```bash
python -c "
import rcapy
print('Package modules:')
for attr in dir(rcapy):
    if not attr.startswith('_'):
        print(f'  - {attr}')
"
```

Expected output:
```
Package modules:
  - DEFAULT_N_COMPONENTS
  - DEFAULT_SONGS
  - DEFAULT_SUBJECTS
  - PooledMultiSubjectRCA
  - RC1InterSubjectCorrelation
  - RC1PreferenceAnalysis
  - RC1SpectralFluxCorrelation
  - ReliableComponentsAnalysis
  - compute_rca_reliability_metrics
  - epochs_to_rca_format
  - epochs_to_rca_format_fixed_length
  - load_music_preference_data
  - plot_music_rca_topographies
  - run_complete_rca_pipeline
```

## Command Line Tools

After installation, these commands should be available:
```bash
rcapy-demo        # Run demo analysis
rcapy-pooled      # Run pooled RCA
rcapy-isc         # Run ISC analysis
rcapy-pipeline    # Run complete pipeline
```

Test with:
```bash
rcapy-demo --help
```