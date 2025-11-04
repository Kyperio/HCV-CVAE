# HCV-CVAE: Hierarchical Convolutional Variational Transformer for Cloud Removal

This repository contains the implementation of **HCV-CVAE**, an improved conditional variational autoencoder (CVAE)-based model designed for thin-cloud removal in remote sensing imagery.  
The proposed approach builds upon the original CVAE framework and integrates **hierarchical convolutional–Transformer feature modeling**, **cross-dimensional weighted mutual information (CWMI) loss**, and a **conservative linear KL annealing strategy** to enhance both the generative stability and texture reconstruction quality of single-temporal satellite images.

---

## 🌤️ **Datasets**

This project adopts two widely used open-source datasets for training and evaluation:

- **RICE2 Dataset**  
  *[Paper]* [Remote Sensing Image Cloud Removal (RICE)](https://arxiv.org/abs/1901.00600)  
  *[Download link]* [https://github.com/BUPTLdy/RICE_DATASET](https://github.com/BUPTLdy/RICE_DATASET)

- **T-Cloud Dataset**  
  *[Paper]* [Uncertainty-Based Thin Cloud Removal Network via Conditional Variational Autoencoders (CVAE)](https://openaccess.thecvf.com/content/ACCV2022/papers/Ding_Uncertainty-Based_Thin_Cloud_Removal_Network_via_Conditional_Variational_Autoencoders_ACCV_2022_paper.pdf)  
  *[Download link]* [https://pan.baidu.com/s/1LtkcdxMbJQTgEr-JvTM1Ug](https://pan.baidu.com/s/1LtkcdxMbJQTgEr-JvTM1Ug) (Extraction code: `t63d`)

---

## 🧠 **Model Overview**

HCV-CVAE extends the standard CVAE with several structural and training improvements:

- **HCV-ViT Encoder:**  
  A hierarchical convolutional–Transformer encoder for joint local texture enhancement and global semantic modeling.

- **Conservative Linear KL Annealing:**  
  Stabilizes latent space training and prevents posterior collapse during optimization.

- **Cross-Dimensional Weighted Mutual Information (CWMI) Loss:**  
  A lightweight mutual-information-inspired constraint that preserves radiometric consistency and edge structure without explicit wavelet decomposition.

- **Test-Time Augmentation (TTA):**  
  Multi-view inference (horizontal/vertical flips) for robust prediction and improved spatial consistency.

---

## ⚙️ **Usage**

### **Train**
1. Modify the output directory in `config.yml` and select the model (`hcv_cvae`).
2. Update dataset paths in `dataloader/My.py` (RICE or T-Cloud).
3. Run the training command:
   ```bash
   python train.py
