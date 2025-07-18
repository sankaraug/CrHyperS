





# UC-HSI: UAV-Based Crop Hyperspectral Imaging Datasets and Machine Learning Benchmark Results

**Published in**: IEEE Geoscience and Remote Sensing Letters (GRSL), Vol. 21, 2024  
[IEEE DOI](https://doi.org/10.1109/LGRS.2024.5508005)



## Overview

**UC-HSI** provides a high-resolution hyperspectral dataset collected using a UAV platform, covering ten agriculturally significant crops. It includes crop classification and growth-stage annotation tasks with machine learning and deep learning benchmarks.

This repository offers:
- Dataset access instructions
- Preprocessing pipeline and scripts
- Benchmark model code (SVM, CNNs, ViT, HyperConvFormer)
- Performance results and visualizations

---

## Dataset Summary

| Parameter             | Details                                           |
|----------------------|---------------------------------------------------|
| Sensor               | Resonon Pika-L (VNIR)                             |
| Bands                | 300 (385–1021 nm)                                 |
| Spatial Resolution   | 1.1 cm                                            |
| Platform             | DJI Matrice-600 Pro UAV                           |
| Crops Covered        | 10 (cotton, maize, paddy, sorghum, etc.)         |
| Total Samples        | 69,514                                            |
| Patch Size           | 11 × 11 × 300                                     |
| Growth Stages        | Annotated for sorghum and groundnut              |



## Benchmark Models

Implemented ML/DL models:

-  SVM
-  1D CNN
-  3D CNN
-  Vision Transformer (ViT)
-  SpectralFormer
-  Proposed: **HyperConvFormer**

 Refer to [`benchmark_models/`](./benchmark_models) for model code.



## Results Snapshot

| Model             | Accuracy (%) |
|------------------|--------------|
| SVM              | 93.12        |
| 1D CNN           | 96.88        |
| 3D CNN           | 94.13        |
| ViT              | 98.06        |
| SpectralFormer   | 97.12        |
| **HyperConvFormer** | **98.86** |

 Visual results and confusion matrices are available in [`results/`](./results).



## Preprocessing Pipeline

Steps used to prepare data:

1. **Radiometric Calibration**  
2. **Geometric Rectification**
3. **Quality Filtering using NIQE**
4. **Spectral Denoising** (FastHyDe)
5. **Patch Extraction** (NDVI > 0.5, TCARI > 1500)

Details and scripts are in [`preprocessing/`](./preprocessing).



## Dataset Access (via TiHAN)

Due to file size and access policy, the dataset is hosted on TiHAN servers.

 [Request Dataset Access](https://tihan.iith.ac.in/dataset-access-form) *(placeholder)*

Fill out the form with your research purpose. Access will be granted after approval.



## Citation

Please cite the following paper if you use this dataset or code:

```bibtex
@article{sankararao2024uc-hsi,
  title     = {UC-HSI: UAV-Based Crop Hyperspectral Imaging Datasets and Machine Learning Benchmark Results},
  author    = {Sankararao, Adduru U.G. and Rajalakshmi, P. and Choudhary, Sunita},
  journal   = {IEEE Geoscience and Remote Sensing Letters},
  volume    = {21},
  year      = {2024},
  pages     = {5508005},
  doi       = {10.1109/LGRS.2024.5508005}
}
```



## Contributors

- Dr. Adduru U.G. Sankararao – Principal Investigator  
- P. Rajalakshmi – Research Guide  
- Sunita Choudhary – Crop Domain Expert, ICRISAT  
- Sai Kiran – Code Implementation  
- Maintainer: [Your Name Here]



© 2024 IIT Hyderabad & ICRISAT – All Rights Reserved.

