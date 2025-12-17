# HiPo-Gait Hierarchical Pose model decoupling for Gait Recognition

Source code for the manuscript **HiPo-Gait** currently under review.

## Abstract

Gait recognition has traditionally relied on silhouette- or appearance-based representations to analyze walking patterns. While silhouettes capture rich movement information, they are highly dependent on body shape and contours, which are potentially irrelevant to gait analysis. The human pose emerges as a more robust and semantically meaningful alternative. However, pose-based models have typically underperformed compared to silhouette-based approaches. To enhance gait feature extraction from pose representations, this paper proposes HiPo-Gait, a novel hierarchical architecture that analyzes isolated limbs of the body pose and groups them at different levels using a new local attention-based fusion module (LAF) to produce a robust hierarchical descriptor of the walking pattern. Moreover, instead of using point coordinates as input, we use heatmaps, a richer representation of the body pose. 
Our experimental results indicate that *a)* the hierarchical decoupling extracts richer features from every individual limb and *b)* the fusion approach optimally aggregates the limbs, outperforming the classical fusion operations. Finally, our approach achieves a percentage point increase of 3.3% over the top pose-based state-of-the-art model in CASIA-B, 21.9%, and 38.5%, respectively, in CCPG and SUSTech1K, and 1.7% improvement over the top model in the FVG-B dataset.

## Source code will be released soon !!
