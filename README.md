# Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection (CVPR 2026)

Official PyTorch implementation of the paper: **"Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection"**.

---

## 📢 News
* **[2026.03]** 🚀 Code repository initialized for CVPR 2026.
* **[Coming Soon]** Pre-trained models and full training scripts will be released.

---

## 📊 Dataset Preparation

We evaluate our model on three standard RGB-T SOD benchmarks. Please download and organize them as follows:

### 1. Download Links
| Dataset | Images | Description | Link |
| :--- | :---: | :--- | :---: |
| **VT5000** | 5000 | Largest benchmark for RGB-T SOD | [[Download]](https://github.com/WangXiao2018/VT5000) |
| **VT1000** | 1000 | Diverse scenarios and objects | [[Download]](https://github.com/trash-ai/VT1000) |
| **VT821** | 821 | Classic early-stage benchmark | [[Download]](https://github.com/Zhengzhe-Liu/VT821) |

### 2. Data Organization
```text
data/
├── train/
│   ├── RGB/          # .jpg or .png
│   ├── T/            # Thermal images
│   └── GT/           # Ground Truth (Binary masks)
├── test/
│   ├── VT5000/
│   │   ├── RGB/
│   │   ├── T/
│   │   └── GT/
│   └── ...
