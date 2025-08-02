"""
Wavelet Analysis Architecture
Shows the detailed wavelet processing and pattern extraction flow
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.compute import Rack
from diagrams.aws.analytics import Analytics
from diagrams.programming.flowchart import Action, Decision, Document
from diagrams.generic.storage import Storage

with Diagram("Financial Wavelet Prediction - Wavelet Analysis Architecture", 
             filename="architecture_diagrams/5_wavelet_analysis", 
             show=False,
             direction="LR"):
    
    # Input
    with Cluster("Input Data"):
        raw_timeseries = Document("Raw Time Series\n(OHLCV)")
        
    # Preprocessing
    with Cluster("Preprocessing"):
        normalizer = Action("Normalize Data")
        denoiser = Action("Denoise Signal")
        
    # Wavelet Transform
    with Cluster("Wavelet Transform"):
        cwt = Rack("Continuous\nWavelet Transform")
        dwt = Rack("Discrete\nWavelet Transform")
        wavelet_selector = Decision("Wavelet\nSelection")
        
        with Cluster("Wavelet Types"):
            morlet = Python("Morlet")
            mexican_hat = Python("Mexican Hat")
            daubechies = Python("Daubechies")
            symlet = Python("Symlet")
    
    # Pattern Extraction
    with Cluster("Pattern Extraction"):
        pattern_detector = Python("pattern_detector.py")
        shapelet_extractor = Python("shapelet_extractor.py")
        motif_discovery = Python("motif_discovery.py")
        
        with Cluster("Pattern Features"):
            scale_features = Action("Scale Features")
            frequency_features = Action("Frequency Features")
            amplitude_features = Action("Amplitude Features")
            phase_features = Action("Phase Features")
    
    # Pattern Analysis
    with Cluster("Pattern Analysis"):
        pattern_visualizer = Python("pattern_visualizer.py")
        wavelet_analyzer = Python("wavelet_analyzer.py")
        
        with Cluster("Analysis Types"):
            scalogram = Analytics("Scalogram\nAnalysis")
            ridge_extraction = Analytics("Ridge\nExtraction")
            modulus_maxima = Analytics("Modulus\nMaxima")
            
    # Pattern Classification
    with Cluster("Pattern Classification"):
        pattern_clusterer = Python("Pattern\nClusterer")
        similarity_engine = Python("Similarity\nEngine")
        dtw_calculator = Python("DTW\nCalculator")
        
    # Pattern Storage
    with Cluster("Pattern Management"):
        pattern_db = Storage("Pattern\nDatabase")
        pattern_index = Document("Pattern\nIndex")
        pattern_metadata = Document("Pattern\nMetadata")
        
    # Visualization
    with Cluster("Visualization"):
        dtw_visualizer = Python("DTW\nVisualizer")
        scalogram_viz = Python("Scalogram\nVisualizer")
        pattern_gallery = Python("Pattern\nGallery")
    
    # Flow
    raw_timeseries >> normalizer >> denoiser
    denoiser >> wavelet_selector
    
    wavelet_selector >> Edge(label="Continuous") >> cwt
    wavelet_selector >> Edge(label="Discrete") >> dwt
    
    [morlet, mexican_hat] >> cwt
    [daubechies, symlet] >> dwt
    
    [cwt, dwt] >> pattern_detector
    pattern_detector >> [shapelet_extractor, motif_discovery]
    
    [shapelet_extractor, motif_discovery] >> [scale_features, frequency_features, amplitude_features, phase_features]
    
    [scale_features, frequency_features, amplitude_features, phase_features] >> wavelet_analyzer
    
    wavelet_analyzer >> [scalogram, ridge_extraction, modulus_maxima]
    [scalogram, ridge_extraction, modulus_maxima] >> pattern_visualizer
    
    pattern_visualizer >> pattern_clusterer
    pattern_clusterer >> similarity_engine
    similarity_engine >> dtw_calculator
    
    dtw_calculator >> [pattern_db, pattern_index, pattern_metadata]
    
    [pattern_db, dtw_calculator] >> dtw_visualizer
    [scalogram, pattern_visualizer] >> scalogram_viz
    pattern_db >> pattern_gallery
