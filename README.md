# Spatial Spread of the Local Field Potential and its Laminar Variation in Visual Cortex

The cortical processing of visual information is fundamental to understanding how the brain interprets complex stimuli. This study investigates the spatial spread of the Local Field Potential (LFP) and its laminar variation in the visual cortex, providing insights into the synaptic and neural network mechanisms underlying visual processing.

The small cortical spread of LFP signals, the LFP and multi-unit activity (MUA) recorded simultaneously had similar visual field maps. Therefore, the LFP is a good index of local circuit activity.

## INSTALLATION

### PREREQUISITES
- Python 3.x
- NumPy
- Matplotlib
- SciPy

### SETUP
1. Ensure Python 3 and pip are installed on your system.
2. Install required Python packages:
   ```
   pip install numpy matplotlib scipy
### USAGE
Run the script directly from the command line: ```python retinal_analysis.py```

# Project Overview
### Purpose
The main objective of this project is to analyze retinal outputs, specifically focusing on the response of the retina to vertical bars moving at different velocities. By processing 'ON' and 'OFF' events—where 'ON' events are typically associated with an increase in light intensity detected by retinal cells, and 'OFF' events are associated with a decrease—the project aims to understand how these changes affect retinal activity. Such analysis is crucial for research in visual neuroscience, helping to elucidate how visual information is processed at the retinal level.

## METHODOLOGY
Retinal responses to moving vertical bars were recorded via a multi-electrode array, capturing the neural activity across an 81-unit grid. Event timestamps, reflecting 'ON' (increase in light intensity) and 'OFF' (decrease in light intensity) stimuli responses, were extracted and stored in a structured dataset for analysis.

# CONCLUSION
-    The visual spreads of MUA and LFP were similar , visual spread of MUA being slightly less than LFP
-    Varying MUA spread within 30μm to 100μm range had no notable impact on their estimation of Local Field Potential (LFP) cortical spread.
-   The relatively small value of cortical spread is one important reason why the visual spreads of MUA and LFP are so similar.

