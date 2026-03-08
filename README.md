# Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection (CVPR 2026)

Official PyTorch implementation of the paper: **"Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection"**.

---

## 📢 News
* **[2026.03]** 🎉 Accepted by CVPR 2026.
* **[Coming Soon]** Pre-trained models and full training scripts will be released.

---

## 📊 Dataset Preparation

Our model is evaluated across RGB-T SOD benchmarks (Unaligned & Aligned) and extended to Light Field SOD benchmarks.

### 1. RGB-T SOD Datasets
| Category | Dataset | Images | Link |
| :--- | :--- | :---: | :---: |
| **Unaligned** | **UVT20K** | 40,000 | [[Download]](https://github.com/KunpengWang-96/PCNet) |
| **Unaligned** | **UVT2000** | 2,000 | [[Download]](https://github.com/KunpengWang-96/SACNet) |
| **Weakly Aligned** | **un-VT5000** | 5,000 | [[Download]](https://github.com/ZhengzheTu/DCNet) |
| **Weakly Aligned** | **un-VT1000** | 1,000 | [[Download]](https://github.com/ZhengzheTu/DCNet) |
| **Weakly Aligned** | **un-VT821** | 821 | [[Download]](https://github.com/ZhengzheTu/DCNet) |
| **Aligned** | **VT5000** | 5,000 | [[Download]](https://github.com/WangXiao2018/VT5000) |
| **Aligned** | **VT1000** | 1,000 | [[Download]](https://github.com/trash-ai/VT1000) |
| **Aligned** | **VT821** | 821 | [[Download]](https://github.com/Zhengzhe-Liu/VT821) |

### 2. Light Field SOD Datasets

| Dataset | Training | Testing | Description |
| :--- | :---: | :---: | :--- |
| **DUTLF-V2** | 2,957 | 1,247 | Covers ten representative object categories across real-world scenes. |
| **PKU-LF** | - | - | The largest publicly available light field dataset with 100+ object categories. |

### 3. Data Organization
```text
data/
├── train/
│   ├── RGB/
│   ├── T/
│   └── GT/
├── test/
│   ├── UVT20K/
│   ├── DUTLF-V2/
│   └── ...
