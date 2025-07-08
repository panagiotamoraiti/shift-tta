# shift-tta

This repository contains the code developed as part of my [**diploma dissertation thesis**](https://drive.google.com/file/d/1GbQM4dtU-hiPv5iGilRl1_kHJ-4HBvYB/view). The work builds upon the repository provided for the **Test-time Adaptation (TTA) Challenge** at the **Visual Continual Learning (VCL) Workshop**, **ICCV 2023**.

## Overview

In this project, I utilized a pre-trained **YOLOX** object detection model and proposed a novel test-time adaptation method that extends the baseline **Mean Teacher** approach by integrating:
- **Stochastic Restoration**
- **Contrastive Learning**

The combination of these techniques led to consistent performance improvements over the baseline.

## Publications

The proposed method was published in:

- **SETN 2024, AI Conference**  
  [ACM Digital Library Link](https://dl.acm.org/doi/10.1145/3688671.3688755)

## Evaluation

In addition to the original benchmark dataset from the VCL@ICCV2023 challenge, the proposed method was further evaluated on four diverse datasets:
- **KITTI**
- **Cityscapes**
- **CLAD-D**
- **Corrupted-COCO**

These evaluations demonstrated the method's robustness and generalization capability under various distribution shifts.

## Acknowledgements

This repository builds upon the official challenge codebase provided by the organizers of the **VCL@ICCV2023 Test-Time Adaptation Challenge**.
