# TTS4CTR - Test-Time Scaling for Click-Through Rate Prediction

Official implementation of **"Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation"**

[![arXiv](https://img.shields.io/badge/arXiv-2512.07650-b31b1b.svg)](https://arxiv.org/abs/2512.07650)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Abstract

Inspired by the success of language models (LM), scaling up deep learning recommendation systems (DLRS) has become a recent trend in the community. All previous methods tend to scale up the model parameters during training time. However, how to efficiently utilize and scale up computational resources during **test time** remains underexplored, which can prove to be a scaling-efficient approach and bring orthogonal improvements in LM domains. 

The key point in applying test-time scaling to DLRS lies in effectively generating diverse yet meaningful outputs for the same instance. We propose two ways:
- **Heterogeneous Architecture Scaling**: Exploring the heterogeneity of different model architectures
- **Homogeneous Initialization Scaling**: Utilizing the randomness of model initialization under a homogeneous architecture

The evaluation is conducted across **8 models** (including both classic and SOTA models) on **3 benchmarks**. Sufficient evidence proves the effectiveness of both solutions. We further prove that under the same inference budget, **test-time scaling can outperform parameter scaling**. Our test-time scaling can also be seamlessly accelerated with the increase in parallel servers when deployed online, without affecting the inference time on the user side.

## 🚀 Quick Start

### Environment Setup

```bash
conda env create -f environment.yml
```

### Usage Example

```bash
bash avazu_train_group.sh
```

## 📂 Repository Structure

```
TTS4CTR/
├── data/
│   └── tfloader.py          # Data loading utilities
├── modules/
│   ├── models.py            # Model definitions (8 models supported)
│   └── layers.py            # Custom layers
├── trainer.py               # Single model training script
├── trainer_group.py         # Group training for test-time scaling
├── avazu_train_group.sh     # Example training script
├── environment.yml          # Conda environment configuration
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 📊 Supported Models

The repository includes implementations for both classic and state-of-the-art recommendation models evaluated in our paper across three benchmark datasets.

## 🎯 Key Features

- ✅ **Test-time scaling** via prediction merging
- ✅ **Heterogeneous architecture** ensemble
- ✅ **Homogeneous initialization** ensemble
- ✅ **Scalable** to parallel deployment
- ✅ **No inference time overhead** on user side
- ✅ **Outperforms parameter scaling** under same inference budget

## 🔬 Experimental Results

Our experiments demonstrate that:
- Test-time scaling provides orthogonal improvements to traditional parameter scaling
- Both heterogeneous and homogeneous approaches are effective
- The method can be seamlessly deployed in production environments with parallel servers

For detailed results, please refer to our [paper](https://arxiv.org/abs/2512.07650).

## 📝 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{lyu2025exploring,
  title={Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation},
  author={Lyu, Fuyuan and Chen, Zhentai and Jiang, Jingyan and Li, Lingjie and Tang, Xing and He, Xiuqiang and Liu, Xue},
  journal={arXiv preprint arXiv:2512.07650},
  year={2025}
}
```

**Paper**: [arXiv:2512.07650 [cs.IR]](https://arxiv.org/abs/2512.07650)  
**DOI**: [10.48550/arXiv.2512.07650](https://doi.org/10.48550/arXiv.2512.07650)  
**Subjects**: Information Retrieval (cs.IR); Machine Learning (cs.LG)

## 👥 Authors

- Fuyuan Lyu
- Zhentai Chen
- Jingyan Jiang
- Lingjie Li
- Xing Tang
- Xiuqiang He
- Xue Liu

## 📧 Contact

For questions or issues, please open an issue in this repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

If this work helps your research or project, please consider giving it a star ⭐️!