<p align="center">
  <img src="assets/banner.png" width="45%" alt="RxnCaption Banner" />
</p>

<h1 align="center"><img src="assets/logo.png" width="36" style="vertical-align: middle;" /> RxnCaption</h1>

<p align="center">
  <b>Reformulating Reaction Diagram Parsing as Visual Prompt Guided Captioning</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.02384"><img src="https://img.shields.io/badge/arXiv-2511.02384-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/songjhPKU/RxnCaption-VL"><img src="https://img.shields.io/badge/🤗%20Model-RxnCaption--VL-yellow" alt="Model"></a>
  <a href="https://huggingface.co/datasets/songjhPKU/U-RxnDiagram-15k"><img src="https://img.shields.io/badge/🤗%20Dataset-U--RxnDiagram--15k-blue" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>

---

> **CVPR 2026** — 给定一张从科学论文中提取的化学反应示意图，RxnCaption 能够识别其中所有的分子结构、文本内容和指代符，并将它们组织为结构化的反应图（反应物 → 条件 → 产物）。

<p align="center">
  <img src="assets/pipeline.png" width="100%" alt="RxnCaption Pipeline" />
</p>

## 🔥 最新动态

- 🚀 [02/21/2026] 论文被 **CVPR 2026** 接收！

## ✨ 亮点

- 🏆 **最优性能** — 在 RxnScribe-test 和 U-RxnDiagram-15k 两个基准上均达到 SOTA
- 🔬 **全新 BIVP 策略** — 通过打框索引的视觉提示，将bbox检测任务转化为结构化自然语言生成任务
- 🧪 **1.5 万标注图像** — 最大规模的化学反应图数据集，涵盖 4 种拓扑类型（单行 / 多行 / 树形 / 图状）
- ⚡ **即插即用** — 一条命令运行完整流水线：分子框检测 → 打框 → VL 推理
- 📊 **全面评测** — Hard / Soft / Hybrid 指标 + 可视化报告

## 📊 主要结果

### 在 RxnScribe-test 上的表现

| 方法 | 策略 | Hard F1 | Soft F1 |
|------|------|---------|---------|
| RxnScribe | BROS | 74.0 | 83.8 |
| RxnIm | BROS | 73.2 | 76.9 |
| Gemini-2.5-Pro | BIVP | 49.8 | 76.1 |
| **RxnCaption-VL（本方法）** | **BIVP** | **75.5** | **88.2** |

### 在 U-RxnDiagram-15k-test 上的表现

| 方法 | 策略 | Hard F1 | Soft F1 |
|------|------|---------|---------|
| RxnScribe | BROS | 34.9 | 45.9 |
| RxnIm | BROS | 37.4 | 40.5 |
| Gemini-2.5-Pro | BIVP | 40.4 | 66.6 |
| **RxnCaption-VL（本方法）** | **BIVP** | **55.5** | **67.6** |

---

## 🏗️ 仓库结构

```
RxnCaption/
├── README.md / README_zh.md
├── LICENSE                    # CC-BY-NC-4.0
├── requirements.txt
│
├── molyolo/                   # 模块 1 — 分子结构检测器（YOLOv10）
│   ├── predict.py
│   └── weights/MolYOLO.pt     # （需单独下载）
│
├── rxncaption/                # 模块 2/3 — 核心流水线
│   ├── annotate.py            # BIVP：边界框 + 阅读顺序编号
│   ├── inference.py           # VL 模型推理（prompt 模板）
│   └── evaluate.py            # Hard / Soft / Hybrid 评测
│
├── tools/                     # 数据处理工具
│   ├── generate_mapdict.py
│   ├── transform_yolo_detections.py
│   ├── convert_to_qwen_format.py
│   ├── transform_jsonl_to_json.py
│   └── transform_prediction_to_gtformat.py
│
├── scripts/                   # Shell 流水线
│   ├── run_inference.sh       # 端到端推理
│   ├── run_eval.sh            # 评测
│   └── prepare_data.sh        # 训练数据准备
│
├── demo/                      # 快速演示
│   ├── run_demo.sh
│   └── run_demo_slurm.sh
│
└── docs/
    ├── DATA.md                # 数据集文档
    └── TRAINING.md            # 训练指南
```

---

## ⚡ 快速开始

### 环境安装

```bash
git clone https://github.com/songjhPKU/RxnCaption
cd RxnCaption
pip install -r requirements.txt

# 安装内置的 ultralytics（YOLOv10）
pip install -e molyolo/
```

### 下载权重

```bash
# MolYOLO 检测器权重
mkdir -p molyolo/weights
wget -O molyolo/weights/MolYOLO.pt \
    https://github.com/songjhPKU/MolYOLO/raw/main/weights/MolYOLO.pt

# RxnCaption-VL 模型 — 两种方式：
# 方式 A：自动从 HuggingFace 下载（默认）
#   脚本默认使用 "songjhPKU/RxnCaption-VL"，首次运行时 swift 会自动下载

# 方式 B：使用本地模型（推荐大多数用户使用）
huggingface-cli download songjhPKU/RxnCaption-VL --local-dir /path/to/RxnCaption-VL
#   然后通过 --model 参数传入本地路径：
#   bash scripts/run_inference.sh --model /path/to/RxnCaption-VL ...
```

### 对你的图片运行推理

```bash
bash scripts/run_inference.sh \
    --image_dir  /path/to/reaction_images \
    --output_dir ./outputs \
    --gpu_num    1
```

完整流水线自动执行以下步骤：
1. **MolYOLO** 检测分子结构 → 每张图输出 JSON 格式的边界框
2. **BIVP** 在图像上标注蓝色边界框 + 数字编号
3. **RxnCaption-VL** 读取标注后的图像，预测反应图
4. 后处理将输出转为评测格式

### 快速演示

想快速试一下？使用自带的 demo 脚本：

```bash
# 1. 把几张反应示意图放到 demo/sample_images/
# 2. 运行：
bash demo/run_demo.sh

# 带评测（如果有 ground truth）：
GT_FILE=demo/sample_gt.json bash demo/run_demo.sh

# 使用本地模型：
MODEL=/path/to/RxnCaption-VL bash demo/run_demo.sh
```

详见 [demo/README.md](demo/README.md)。

---

## 🔬 流水线详解

### 第一步 — MolYOLO 分子检测

基于 YOLOv10 微调的模型，检测反应示意图中的所有相关实体（分子、文本、标识符）。

```bash
python molyolo/predict.py \
    --img_dir       /path/to/images \
    --weights       molyolo/weights/MolYOLO.pt \
    --output_dir    outputs/molyolo \
    --output_name   run01 \
    --conf          0.5 \
    --gpu_num       4 \
    --visual_prompt
```

### 第二步 — BIVP 视觉提示标注

**边界框索引视觉提示**（BIVP）模块在每张图像上绘制蓝色边界框和阅读顺序编号，将原始检测结果转化为 VL 模型的视觉提示。

```bash
python rxncaption/annotate.py \
    --image_root_dir    /path/to/images \
    --det_json_root_dir outputs/molyolo/run01/json \
    --middle_root_dir   outputs/annotated \
    --confidence_threshold 0.5
```

### 第三步 — RxnCaption-VL 推理

微调后的 Qwen2.5-VL-7B 模型读取标注后的图像，输出结构化的 JSON 反应列表。

```bash
swift infer \
    --model           songjhPKU/RxnCaption-VL \
    --model_type      qwen2_5_vl \
    --infer_backend   pt \
    --val_dataset     outputs/eval_input.jsonl \
    --result_path     outputs/infer_output.jsonl \
    --max_batch_size  1 \
    --max_new_tokens  16384
```

**输出示例：**
```json
[
  {
    "reactants":  [{"structure": 1}, {"text": "H₂O"}],
    "conditions": [{"text": "Δ, 2h"}],
    "products":   [{"structure": 2}]
  }
]
```

### 第四步 — 评测

三种评测模式反映了不同的匹配严格程度：

| 模式 | 匹配内容 |
|------|---------|
| **Hard** | 所有角色成员（分子 + 文本）需满足 IoU ≥ 0.5 |
| **Soft** | 仅比较分子成员 |
| **Hybrid** | 分子通过 IoU 匹配；文本通过无序集合比较 |

```bash
bash scripts/run_eval.sh \
    --gt_file        data/ground_truth.json \
    --raw_pred_file  outputs/raw_prediction.json \
    --mapdict        data/mapdict_from_yolo_to_gt.json \
    --image_dir      data/images \
    --output_dir     results/ \
    --mode           all
```

---

## 🗃️ 数据集

**U-RxnDiagram-15k** 包含约 15,000 张从科学论文 PDF 中提取的反应示意图，涵盖 4 种拓扑类型，附有完整标注。

```python
from datasets import load_dataset
ds = load_dataset("songjhPKU/U-RxnDiagram-15k")
```

详见 [docs/DATA.md](docs/DATA.md) 了解完整的数据格式和下载说明。

---

## 🏋️ 训练

详见 [docs/TRAINING.md](docs/TRAINING.md) 获取完整训练指南。

简要步骤：

```bash
# 1. 准备数据
bash scripts/prepare_data.sh \
    --raw_gt_json  data/ground_truth_ocr.json \
    --yolo_det_dir data/det_json/ \
    --image_dir    data/annotated_images/ \
    --output_dir   data/processed/

# 2. 训练（单机 8 卡，全参数微调）
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --dataset data/processed/train.jsonl \
    --val_dataset data/processed/val.jsonl \
    --output_dir outputs/train/ \
    # ... 详见 docs/TRAINING.md
```

---

## 🤝 引用

如果你发现本工作对你的研究有帮助，请引用：

```bibtex
@misc{song2026rxncaptionreformulatingreactiondiagram,
      title={RxnCaption: Reformulating Reaction Diagram Parsing as Visual Prompt Guided Captioning}, 
      author={Jiahe Song and Chuang Wang and Bowen Jiang and Yinfan Wang and Hao Zheng and Xingjian Wei and Chengjin Liu and Rui Nie and Junyuan Gao and Jiaxing Sun and Yubin Wang and Lijun Wu and Zhenhua Huang and Jiang Wu and Qian Yu and Conghui He},
      year={2026},
      eprint={2511.02384},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.02384}, 
}
```

---

## 📜 许可证

本项目采用 **CC BY-NC 4.0** 许可协议 — 详见 [LICENSE](LICENSE)。

## 🙏 致谢

本研究由上海人工智能实验室支持。
