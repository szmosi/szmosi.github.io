---
layout: page
title: Road Damage Classification
short_title: Road Damage Classification
description: Comparing SVM and convolutional neural networks for image-based infrastructure inspection.
img: assets/img/projects/road_damage/model_comparison.png
importance: 2
category: "Data Science"
github: https://github.com/szmosi/road-damage-classification
---

## Overview

**Can image classification help prioritize road maintenance inspections?** In a collaborative MSc Machine Learning and Deep Learning coursework project (2025), we explored binary classification of road images as **damage** or **no damage** using a linear SVM and two CNN architectures. The motivating application was assisting maintenance teams in reviewing dashcam imagery—not building a production-ready or real-time road defect detector.

The project used the publicly available **Road Damage Detection 2022 (RDD2022)** dataset. The original report describes selecting approximately **34,007 images** before additional quality filtering and training. The models used image-level labels derived from the source annotations; they did not learn bounding boxes or distinguish specific types of defects.

[View the code on GitHub](https://github.com/szmosi/road-damage-classification) · [Original RDD2022 dataset](https://github.com/sekilab/RoadDamageDetector)

## Methodology

We prepared images through annotation parsing, country filtering, perceptual-hash duplicate checks, and image-quality screening for brightness, colorfulness and blur. The original workflow then applied contrast/color transformations and horizontal flipping. Data quality varied between countries, which is an important limitation of cross-country evaluation.

![Original notebook image-quality distributions]({{ '/assets/img/projects/road_damage/image_quality_distributions.png' | relative_url }})
*Image-quality distributions, preserved from the coursework notebook's saved output.*

Three classification approaches were compared:

1. **Linear SVM baseline:** a hinge-loss SGD classifier on flattened, scaled image pixels.
2. **Simple CNN:** three convolution/pooling stages and a binary sigmoid classifier.
3. **Enhanced CNN experiment:** a deeper convolutional architecture and an autoencoder experiment, with added dropout and early stopping.

**Implementation caveat:** while the coursework describes an autoencoder followed by a CNN, the original notebook does not explicitly train the enhanced CNN on encoded features. This prevents us from claiming that the public notebook reproduces an end-to-end autoencoder-based classifier.

## Reported results

![Reported model comparison]({{ '/assets/img/projects/road_damage/model_comparison.png' | relative_url }})
*Scores reported in the submitted coursework report; chart reconstructed from those reported values, not from a new model run.*

| Model | Damage-class recall | Damage-class F1 | Macro F1 |
|:--|--:|--:|--:|
| Linear SVM | 83% | 79% | 66% |
| Simple CNN | 80% | 83% | **77%** |
| Enhanced CNN | **88%** | **85%** | 75% |

The enhanced CNN had the highest reported **recall for damaged roads**, relevant to minimizing missed defects. The simple CNN produced the highest reported **macro F1**, suggesting a more balanced result across the two classes. These are different operational priorities, and neither metric alone establishes readiness for real-world deployment.

## Model training and diagnostics

![Simple CNN training curves]({{ '/assets/img/projects/road_damage/simple_cnn_learning_curves.png' | relative_url }})
*Simple CNN: saved training and validation curves from the original notebook.*

![Enhanced CNN training curves]({{ '/assets/img/projects/road_damage/enhanced_cnn_learning_curves.png' | relative_url }})
*Enhanced CNN: saved training and validation curves from the original notebook.*

![Autoencoder learning curves]({{ '/assets/img/projects/road_damage/autoencoder_learning_curves.png' | relative_url }})
*Autoencoder training diagnostic from the original notebook. The notebook does not establish direct integration of the encoded features into CNN training.*

## Lessons and limitations

The project highlighted the trade-off between detection sensitivity, overall class balance, and training resources. Its limitations matter: country-dependent image distributions, imperfectly documented train/test splits, no external holdout evaluation, and no prospective field testing. The historical report describes a 70/30 split, whereas the preserved notebook specifies 80/20; this discrepancy requires resolution before numerical reproduction.

**This was a collaborative university group assignment.** The GitHub repository preserves a sanitized version of the historical experimental code; the results shown here come from the submitted coursework and have not been independently reproduced. The source dataset is available from [the RDD2022 project](https://github.com/sekilab/RoadDamageDetector); image data is not redistributed.
