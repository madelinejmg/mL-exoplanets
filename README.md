## M-Dwarf Planet Candidates Using Supervised Machine Learning

**Madeline J. Maldonado Gutierrez**  
Barnard College, Columbia University  
ASTR-GR6012: Astro-Skills — Numerical and Statistical Methods  

## Overview

This project investigates the application of **supervised machine learning** to the problem of **exoplanet vetting around M-dwarf stars** using Kepler light curves. Motivated by the increasing volume of exoplanet detections and the limitations of manual vetting, this work evaluates whether the **shape of phase-folded Kepler light curves alone** contains sufficient information to distinguish planetary signals from astrophysical false positives in the **low signal-to-noise regime** characteristic of M-dwarf systems.

Using labeled **Kepler Objects of Interest (KOIs)** as a training set, a **convolutional neural network (CNN)** is trained on uniformly processed light curves to test how much information is contained purely in the *appearance of the transit signal*, without using additional physical, stellar, or diagnostic features.

Rather than optimizing for maximum detection performance, this project quantifies the **limits of shape-based learning** in the low signal-to-noise regime characteristic of M-dwarf planetary systems.

## What “Shape-Based Only” Means?

In this project, **shape-based only** means that the machine-learning model is trained **exclusively on the normalized flux values of phase-folded light curves**. The CNN receives no information beyond how the light curve varies with orbital phase.

### Information the model *can* use

The CNN is allowed to learn from:
- how deep the transit appears in the folded light curve  
- how wide the dip is in phase  
- how sharp or gradual ingress and egress are  
- whether the dip is symmetric around phase 0  
- overall structure of flux as a function of phase  

All of this information is encoded in a single input array: ```flux(phase)``` represented as a fixed-length, median-binned, phase-folded light curve.


## Scientific Motivation

Since the first confirmed detections of extrasolar planets, the **transit method** has become one of the most productive techniques for identifying exoplanets, particularly through NASA’s *Kepler* mission [1]. When a planet transits its host star, it produces a characteristic dip in the stellar light curve defined by ingress, egress, and transit depth.

M-dwarf stars are especially promising targets due to their small radii and high planet occurrence rates [2]. However, transit searches around M dwarfs are challenged by:

- intrinsically faint host stars,  
- strong stellar variability and flaring,  
- shallow transit depths for small planets,  
- low signal-to-noise ratios in long-cadence data.

Traditional vetting of transit candidates relies on manual inspection, which is time-consuming and difficult to scale. While machine-learning approaches have shown success in exoplanet detection [3,4], it remains unclear whether **light-curve morphology alone** is sufficient for robust vetting in the M-dwarf regime.

## Dataset

- **Source:** Kepler Objects of Interest (KOI) catalog [5]  
- **Target population:** M-dwarf host stars  
- **Labels:**
  - `1` — confirmed or candidate exoplanet  
  - `0` — astrophysical false positive (e.g., eclipsing binaries)  
- **Sample size:** 115 unique stellar systems  
  - 37 planets  
  - 78 false positives  

Each system contributes **one example**, ensuring no information leakage between training and validation sets.

## Light-Curve Processing

Long-cadence **Presearch Data Conditioning Simple Aperture Photometry (PDCSAP)** light curves are downloaded using **Lightkurve** [6], which retrieves Kepler data from the Mikulski Archive for Space Telescopes (MAST).

Processing steps:

1. Download all available quarters per target  
2. Stitch quarters and remove NaN values  
3. Flatten long-term trends  
4. Median-normalize flux  
5. Phase-fold using KOI ephemerides  
6. Median-bin to a fixed-length representation (2048 bins)  

The final input to the CNN has shape: `(N_stars, 2048, 1)`

No flare removal, clipping, or additional feature engineering is applied, by design, in order to preserve the raw shape information.

## Model Architecture

A one-dimensional convolutional neural network is used to learn transit-shape features:

- Input layer: `(2048, 1)`  
- Convolutional layers with ReLU activations  
- Max pooling and global average pooling  
- Dropout regularization  
- Sigmoid output for binary classification  

A normalization layer is trained on each training fold only to stabilize optimization while preserving relative signal amplitudes.

## Training and Evaluation

### Cross-Validation

- **5-fold GroupKFold cross-validation**  
- Grouped by `kepid` to prevent stellar leakage  
- Class imbalance handled using class-weighted loss  

### Metrics

- ROC-AUC  
- Precision–Recall AUC (Average Precision)  
- Out-of-fold (OOF) predictions for global evaluation  

## Results

- **Mean CV ROC-AUC:** 0.57 ± 0.10  
- **OOF ROC-AUC:** 0.52  
- **OOF Precision–Recall AUC:** 0.38  
- **Baseline AP (class fraction):** ≈ 0.32  

Predicted probability distributions for planets and false positives show substantial overlap, with most predictions clustered near 0.5.

## Interpretation

These results show that **light-curve shape alone provides limited discriminatory power** for exoplanet vetting around M-dwarf stars. Although the CNN learns weak signal above chance, it does not generalize reliably across different stellar systems when evaluated out-of-fold.

This limitation reflects physical and observational challenges—stellar variability, shallow transits, and low signal-to-noise ratios—rather than shortcomings of the neural network architecture. The findings are consistent with real Kepler vetting pipelines, which rely on multiple sources of information beyond the folded light curve.

## Conclusion

This project demonstrates that CNNs trained only on the shape of phase-folded Kepler light curves are fundamentally limited for detecting and validating low signal-to-noise exoplanets around M-dwarf stars. While folded light curves contain some information about planetary transits, this information alone is insufficient for reliable classification, particularly in the presence of stellar variability and observational noise. By intentionally restricting the model to shape-based inputs, this project provides a controlled measurement of these limitations and motivates the inclusion of additional physically motivated features in future machine-learning-based vetting pipelines.

Beyond its scientific findings, this project represents a significant technical and conceptual expansion of my skill set. The work required exploring computational and statistical methods that were previously unfamiliar to me, including the construction of end-to-end machine-learning pipelines for astronomical time-series data, the design and training of convolutional neural networks, and the use of cross-validation strategies appropriate for grouped astrophysical datasets. I also developed a deeper understanding of performance evaluation using ROC and precision–recall metrics, as well as how to interpret these metrics in a physically meaningful way for exoplanet science.

Overall, this project pushed me into new technical territory by integrating astronomy and machine learning, and it reshaped how I approach exoplanet vetting problems and would most certainly include it in my future research projects.

## References

1. Borucki, W. J., et al. (2010). *Kepler Planet-Detection Mission: Introduction and First Results*. **Science**, 327, 977–980.  
2. Dressing, C. D., & Charbonneau, D. (2015). *The Occurrence Rate of Small Planets around Small Stars*. **ApJ**, 807, 45.  
3. Shallue, C. J., & Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning*. **AJ**, 155, 94.  
4. Ansdell, M., et al. (2018). *Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning*. **ApJ**, 869, 46.  
5. NASA Exoplanet Archive: Kepler Objects of Interest (KOI) Catalog.  
6. Lightkurve Collaboration et al. (2018). *Lightkurve: Kepler and TESS time series analysis in Python*. **Astrophysics Source Code Library**, ascl:1812.013.  
