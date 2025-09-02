#!/usr/bin/env python3
"""
Complete RCA Analysis Pipeline
==============================

This module provides a unified interface to run the complete RCA analysis
pipeline, including all four stages of analysis.

Classes:
--------
- CompletePipeline: Orchestrates all analysis stages

Usage:
------
>>> from music_rca.analysis import CompletePipeline
>>> pipeline = CompletePipeline()
>>> results = pipeline.run_all_analyses()
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .pooled_rca import PooledMultiSubjectRCA
from .isc_analysis import RC1InterSubjectCorrelation
from .neural_acoustic_coupling import RC1SpectralFluxCorrelation
from .neural_preference import RC1PreferenceAnalysis

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePipeline:
    """
    Complete RCA analysis pipeline orchestrator.
    
    This class coordinates all four stages of the RCA analysis:
    1. Pooled multi-subject RCA
    2. Inter-subject correlation analysis
    3. Neural-acoustic coupling analysis
    4. Neural-preference relationship analysis
    """
    
    def __init__(self, base_path: Optional[str] = None, 
                 subjects: Optional[List[str]] = None,
                 songs: Optional[List[str]] = None):
        """
        Initialize the complete pipeline.
        
        Parameters:
        -----------
        base_path : str, optional
            Base path to music preference data
        subjects : list, optional
            List of subject IDs
        songs : list, optional
            List of song IDs
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.subjects = subjects or ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        self.songs = songs or [f"{i}-{j}" for i in range(1, 6) for j in range(1, 4)]
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized complete pipeline with {len(self.subjects)} subjects and {len(self.songs)} songs")
    
    def run_stage1_pooled_rca(self) -> Dict[str, Any]:
        """Run Stage 1: Pooled multi-subject RCA analysis."""
        logger.info("Starting Stage 1: Pooled Multi-Subject RCA")
        
        try:
            pooled_rca = PooledMultiSubjectRCA(base_path=str(self.base_path))
            results = pooled_rca.run_complete_analysis()
            self.results['stage1_pooled_rca'] = results
            
            logger.info("Stage 1 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            raise
    
    def run_stage2_isc_analysis(self) -> Dict[str, Any]:
        """Run Stage 2: Inter-subject correlation analysis."""
        logger.info("Starting Stage 2: Inter-Subject Correlation Analysis")
        
        try:
            isc_analyzer = RC1InterSubjectCorrelation(base_path=str(self.base_path))
            results = isc_analyzer.run_complete_analysis()
            self.results['stage2_isc'] = results
            
            logger.info("Stage 2 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            raise
    
    def run_stage3_neural_acoustic_coupling(self) -> Dict[str, Any]:
        """Run Stage 3: Neural-acoustic coupling analysis."""
        logger.info("Starting Stage 3: Neural-Acoustic Coupling Analysis")
        
        try:
            coupling_analyzer = RC1SpectralFluxCorrelation(base_path=str(self.base_path))
            results = coupling_analyzer.run_complete_analysis()
            self.results['stage3_coupling'] = results
            
            logger.info("Stage 3 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            raise
    
    def run_stage4_neural_preference(self) -> Dict[str, Any]:
        """Run Stage 4: Neural-preference relationship analysis."""
        logger.info("Starting Stage 4: Neural-Preference Analysis")
        
        try:
            preference_analyzer = RC1PreferenceAnalysis(base_path=str(self.base_path))
            results = preference_analyzer.run_complete_analysis()
            self.results['stage4_preference'] = results
            
            logger.info("Stage 4 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            raise
    
    def run_all_analyses(self, stages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run all analysis stages in sequence.
        
        Parameters:
        -----------
        stages : list, optional
            List of stage numbers to run (default: [1, 2, 3, 4])
        
        Returns:
        --------
        dict : Dictionary containing results from all stages
        """
        if stages is None:
            stages = [1, 2, 3, 4]
        
        logger.info(f"Starting complete RCA pipeline (stages: {stages})")
        
        try:
            # Stage 1: Pooled RCA
            if 1 in stages:
                self.run_stage1_pooled_rca()
            
            # Stage 2: ISC Analysis
            if 2 in stages:
                self.run_stage2_isc_analysis()
            
            # Stage 3: Neural-Acoustic Coupling
            if 3 in stages:
                self.run_stage3_neural_acoustic_coupling()
            
            # Stage 4: Neural-Preference Analysis
            if 4 in stages:
                self.run_stage4_neural_preference()
            
            logger.info("Complete RCA pipeline finished successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a summary report of all analysis results.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save summary report
        
        Returns:
        --------
        str : Summary report content
        """
        report_lines = [
            "=" * 60,
            "COMPLETE RCA ANALYSIS PIPELINE SUMMARY",
            "=" * 60,
            f"Base path: {self.base_path}",
            f"Subjects: {len(self.subjects)} ({', '.join(self.subjects)})",
            f"Songs: {len(self.songs)} songs",
            f"Stages completed: {len(self.results)}",
            ""
        ]
        
        # Add results from each stage
        for stage, results in self.results.items():
            report_lines.append(f"📊 {stage.upper()}:")
            report_lines.append("-" * 40)
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"  {key}: {value}")
                    elif isinstance(value, (list, tuple)) and len(value) < 10:
                        report_lines.append(f"  {key}: {value}")
                    else:
                        report_lines.append(f"  {key}: {type(value).__name__}")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 60,
            "PIPELINE COMPLETE",
            "=" * 60
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Summary report saved to: {output_path}")
        
        return report_content

def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete RCA analysis pipeline')
    parser.add_argument('--base-path', type=str, help='Base path to music preference data')
    parser.add_argument('--stages', nargs='+', type=int, choices=[1, 2, 3, 4], 
                       default=[1, 2, 3, 4], help='Stages to run')
    parser.add_argument('--report', type=str, help='Path to save summary report')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CompletePipeline(base_path=args.base_path)
    results = pipeline.run_all_analyses(stages=args.stages)
    
    # Generate report
    report = pipeline.generate_summary_report(output_file=args.report)
    print(report)
    
    return results

if __name__ == "__main__":
    main()