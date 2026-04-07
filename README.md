# HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models

Official implementation of our CVPR 2026 paper: 
**[HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](https://arxiv.org/abs/2603.25411)**

[![arXiv](https://img.shields.io/badge/arXiv-2603.25411-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2603.25411) [![Project Page](https://img.shields.io/badge/Project-Homepage-blue?style=flat&logo=github)](https://microsoft.github.io/HiSpatial/) [![Model](https://img.shields.io/badge/Model-HiSpatial3B-yellow?style=flat&logo=huggingface)](https://huggingface.co/lhzzzzzy/HiSpatial-3B)

## Release Progress

- [x] **Inference code & evaluation scripts**
- [x] **Model weights** ([HuggingFace](https://huggingface.co/lhzzzzzy/HiSpatial-3B))
- [ ] **Training data & dataloader** (before May 1, 2026)


## Installation

```bash
# Clone the repository
git clone https://github.com/microsoft/HiSpatial.git
cd HiSpatial

# Install the package (core + evaluation dependencies)
pip install -e ".[eval]"

# Install MoGe depth estimator (required for inference)
pip install -e ".[depth]"
```


## Inference

HiSpatial takes an RGB image and a 3D point cloud (estimated by [MoGe](https://github.com/microsoft/MoGe)) as input, and answers spatial reasoning questions.

```python
from hispatial.inference import MoGeProcessor, HiSpatialPredictor

# Initialize MoGe depth estimator and HiSpatial predictor
moge = MoGeProcessor(device_name="cuda")
predictor = HiSpatialPredictor(model_load_path="lhzzzzzy/HiSpatial-3B")  # auto-downloads from HF Hub

# Load an image (file path, PIL Image, or numpy array)
image = "example.jpg"

# Estimate 3D point cloud from the image
xyz_values = moge.apply_transform(image)

# Ask a spatial question
answer = predictor.query(
    image=image,
    prompt="Which object is closer to the camera, the chair or the table?",
    xyz_values=xyz_values,
)
print(answer)
```

> **Note:** You can also pass a local path to `model_load_path` (e.g. `"/path/to/weights.pt"` or a directory containing `weights.pt` + `config.json`).


## Evaluation

We evaluate HiSpatial on 6 spatial understanding benchmarks. Each eval script can be run independently:

```bash
# CV-Bench (2D Relation + 3D)
python eval/eval_cv_bench.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/cvbench

# 3DSRBench
python eval/eval_3dsrbench.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --tsv_path path/to/3DSRBenchv1.tsv \
    --save_path results/3dsrbench

# EmbSpatial
python eval/eval_emb_spatial.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/embspatial \
    --benchmark_path path/to/embspatial_bench.json

# Q-Spatial (QSpatial+ and QSpatial-ScanNet)
python eval/eval_q_spatial.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/qspatial \
    --scannet_images_dir path/to/scannet/images

# RoboSpatial
python eval/eval_robospatial.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/robospatial

# SpatialRGPT
python eval/eval_spatialrgpt.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/spatialrgpt.jsonl
```

Or run all benchmarks at once (edit paths in the script first):

```bash
bash eval/run_all.sh
```


## Citation

```bibtex
@inproceedings{liang2026hispatial,
  title={HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models},
  author={Liang, Huizhi and Shen, Yichao and Deng, Yu and Xu, Sicheng and Feng, Zhiyuan and Zhang, Tong and Liang, Yaobo and Yang, Jiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
