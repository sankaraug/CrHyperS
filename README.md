
# UC-HSI: UAV-Based Crop Hyperspectral Imaging Datasets and Machine Learning Benchmark Results

**UC-HSI** provides a high-resolution hyperspectral dataset collected using a UAV platform, covering ten agriculturally significant crops. It includes crop classification and growth-stage annotation tasks with machine learning and deep learning benchmarks.


**Published in**: IEEE Geoscience and Remote Sensing Letters (GRSL), Vol. 21, 2024  
[IEEE DOI](https://ieeexplore.ieee.org/document/10605842)







---

## Dataset Summary

| Parameter             | Details                                          |
|----------------------|---------------------------------------------------|
| Sensor               | Resonon Pika-L (VNIR)                             |
| Bands                | 300 (385–1021 nm)                                 |
| Spatial Resolution   | 1.1 cm                                            |
| Platform             | DJI Matrice-600 Pro UAV                           |
| Crops Covered        | 10 (Pearl millet,maize,Groundnut,sorghum,etc.)    |
| Total Samples        | 69,514                                            |
| Patch Size           | 11 × 11 × 300                                     |
| Growth Stages        | sorghum and groundnut               |


## Crop Dataset Class Distribution


| Class | Crop Type     | Train Images | Test Images | Train Samples | Test Samples |
|-------|---------------|--------------|-------------|---------------|--------------|
| 1     | Sorghum       | 51           | 40          | 3,282         | 2,519        |
| 2     | Pearl millet  | 44           | 36          | 5,395         | 5,333        |
| 3     | Maize         | 236          | 127         | 11,559        | 6,464        |
| 4     | Groundnut     | 51           | 42          | 3,622         | 3,403        |
| 5     | Cowpea        | 43           | 41          | 6,774         | 5,638        |
| 6     | Common bean   | 22           | 12          | 1,433         | 763          |
| 7     | Lima bean     | 10           | 4           | 749           | 371          |
| 8     | Mung bean     | 14           | 8           | 2,051         | 1,166        |
| 9     | Chickpea      | 24           | 14          | 1,167         | 884          |
| 10    | Pigeon pea    | 38           | 29          | 3,722         | 3,219        |
| **Total** |           | **533**      | **353**     | **39,754**    | **29,760**   |



## Growth Stage Dataset Distribution
### Sorghum

| Stage               | Train Images | Test Images | Train Samples | Test Samples |
|---------------------|--------------|-------------|----------------|---------------|
| Vegetative          | 51           | 40          | 1,083          | 732           |
| Flowering initiation| 50           | 33          | 2,106          | 1,541         |
| Milk                | 44           | 40          | 2,331          | 2,994         |
| Grain filling       | 36           | 30          | 1,400          | 2,063         |
| Harvesting          | 26           | 22          | 1,769          | 1,672         |
| **Total**           | **207**      | **165**     | **8,689**      | **9,002**     |

### Groundnut

| Stage               | Train Images | Test Images | Train Samples | Test Samples |
|---------------------|--------------|-------------|----------------|---------------|
| Vegetative          | 33           | 31          | 1,927          | 2,290         |
| Flowering initiation| 35           | 31          | 3,097          | 2,968         |
| Peg development     | 46           | 42          | 1,618          | 1,466         |
| Peanut formation    | 22           | 19          | 1,239          | 898           |
| Harvesting          | 23           | 20          | 2,050          | 788           |
| **Total**           | **159**      | **143**     | **9,931**      | **8,410**     |


## Benchmark Models

Implemented ML/DL models:

-  SVM
-  1D CNN
-  3D CNN
-  Vision Transformer (ViT)
-  SpectralFormer
-  Proposed: **HyperConvFormer**

 Refer to [`Codes/`](./Codes) for model code.




**"Architecture of the Proposed HyperConvFormer Model for Crop HSI Classification"**

![HyperConvFormer](https://github.com/sankaraug/CrHyperS/blob/main/HyperConvFormer.png)

---
## Results Snapshot

| Model             | Accuracy (%) |
|------------------|--------------|
| SVM              | 93.12        |
| 1D CNN           | 96.88        |
| 3D CNN           | 94.13        |
| ViT              | 98.06        |
| SpectralFormer   | 97.12        |
| **HyperConvFormer** | **98.86** |

---

![HyperConvFormer](https://github.com/sankaraug/CrHyperS/blob/main/Results/Overall%20Accuracy%20(OA)%20Comparison%20Across%20Models.gif))


 Visual results and confusion matrices are available in [`results/`](./results).
 




## Dataset Access (via TiHAN)

Due to file size and access policy, the dataset is hosted on TiHAN servers.

 [Request Dataset Access](https://tihan.in/tiand-datasets/).
 
Navigate to "Hyperspectral Imaging (Aerial)" section in above link and click on download....
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

- Dr. Adduru U.G. Sankararao - IIT Hyderabad
- Prof. P. Rajalakshmi –  IIT Hyderabad
- Sunita Choudhary – Scientist, ICRISAT  



© 2024 IIT Hyderabad  – All Rights Reserved.

