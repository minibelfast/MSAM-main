# MambaMIL (GitHub Bundle)

本目录为工程代码的 GitHub 上传整理版：已移除缓存/运行日志等非必要文件，并补充基础使用说明，便于读者复现训练、评估与可解释性分析流程。

## 功能概览
- WSI 生存/预后预测（survival-aware MIL）
- 多种 MIL 模型：Mean/Max/ABMIL/TransMIL/S4/MambaMIL/MambaAttn，以及扩展模型（如 `models/BLCA_MIL.py`）
- 多模态输入：Titan（slide-level 768d）+ UNI（patch-level 1536d）
- 连续学习：冻结参数 / 知识蒸馏（KD）/ EWC（可组合）
- 可解释性：注意力热图、patch-space ERF、t-SNE（2D/3D）、KMeans patch subgroup、SHAP（DeepExplainer）

## 目录结构（核心）
- `main_survival.py`：生存预测训练入口（支持 k-fold 与连续学习相关参数）
- `models/`：模型实现
- `dataset/`：数据加载（包含 multimodal survival dataset）
- `utils/`：训练、评估、损失函数、EWC 等工具
- `dataset_csv/`：示例元数据（case_id、slide_id、censorship、survival_months）
- `splits/`：示例划分文件（splits_*.csv）
- `CLAM/`：WSI 预处理/patching/部分可解释性脚本（用于特征提取与热图等）
- `mamba/`：Mamba/SRMamba 等实现（已 vendor 到仓库中）

## 环境安装（建议）
建议：CUDA 11.8 + Python 3.10。

1) 创建环境
```bash
conda create -n mambamil python=3.10 -y
conda activate mambamil
```

2) 安装 PyTorch（示例：cu118）
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
```

3) 安装 Mamba（本仓库内已包含）
```bash
pip install -e ./mamba
```

4) 其他常用依赖（按需安装）
```bash
pip install scikit-survival pandas h5py lifelines tqdm pyyaml timm scikit-learn matplotlib shap
```

## 数据准备
本工程默认使用“特征级训练”，即不直接输入 WSI 原图，而输入由基础模型预先提取的特征。

### 1) 生存标签 CSV（dataset_csv）
以 `dataset_csv/GBM_processed.csv` 为例，必须包含以下字段：
- `case_id`：患者/病例 ID
- `slide_id`：切片 ID（与特征文件名对应）
- `censorship`：删失标记（以你的定义为准）
- `survival_months`：生存时间（月）

### 2) 多模态特征（Titan + UNI，h5）
multimodal 模式下，数据集会分别读取：
- Titan：`<data_root_dir_titan>/<slide_id>.h5`，h5 内 dataset 名为 `features`（通常为 768 维向量或可堆叠形式）
- UNI：`<data_root_dir_uni>/<slide_id>.h5`，h5 内 dataset 名为 `features`（通常为 N×1536 的 patch 特征矩阵）

目录示例：
```text
data/
  features_titan_h5/
    TCGA-xx-xxxx-....h5
  features_uni_h5/
    TCGA-xx-xxxx-....h5
```

## 训练（Survival）

### 1) 五折/三折训练
训练入口为 `main_survival.py`。你需要至少指定：
- `--task`：用于定位 `dataset_csv/<study>_processed.csv` 的 study 名（代码中会取 task 的第二段作为 `<study>`）
- `--split_dir`：划分文件目录，需包含 `splits_0.csv ... splits_{k-1}.csv`
- `--data_root_dir_titan`、`--data_root_dir_uni`：Titan/UNI 特征目录（h5）

示例（以 GBM 为例，k=5）：
```bash
python main_survival.py \
  --task TCGA_GBM_survival \
  --k 5 \
  --split_dir ./splits/TCGA_GBM_survival_kfold \
  --data_root_dir_titan ./data/features_titan_h5 \
  --data_root_dir_uni ./data/features_uni_h5 \
  --model_type mamba_attn \
  --bag_loss nll_surv \
  --results_dir ./results/TCGA_GBM_mamba_attn
```

### 2) 连续学习（Continual Learning）
本工程在训练参数中提供以下开关（可组合使用）：
- 冻结：`--freeze_layers`
- 知识蒸馏（KD）：`--distillation --teacher_model_paths ... --alpha_kd ... --temperature ...`
- EWC：`--use_ewc --ewc_lambda ...`
- 载入预训练权重：`--pretrained_model <path> --finetune_lr <lr>`

示例：在新队列上基于预训练模型继续训练（冻结 + KD + EWC）
```bash
python main_survival.py \
  --task ZN_GBM_survival \
  --k 3 \
  --split_dir ./splits/ZN_GBM_survival_kfold \
  --data_root_dir_titan ./data/ZN/features_titan_h5 \
  --data_root_dir_uni ./data/ZN/features_uni_h5 \
  --pretrained_model ./results/TCGA_GBM_mamba_attn/s_0_checkpoint.pth \
  --finetune_lr 1e-5 \
  --freeze_layers \
  --distillation \
  --teacher_model_paths ./results/TCGA_GBM_mamba_attn/s_0_checkpoint.pth \
  --alpha_kd 0.5 \
  --temperature 4.0 \
  --use_ewc \
  --ewc_lambda 1.0 \
  --results_dir ./results/ZN_GBM_incremental
```

## 可解释性（Interpretability）
常用输出包括：
- 注意力热图（patch attention → WSI 坐标映射）
- patch-space ERF（对输入 patch 特征求梯度归因，并映射回补丁网格）
- t-SNE（2D/3D）与 KMeans 聚类（patch subgroup discovery）
- SHAP（DeepExplainer）

相关脚本主要位于根目录与 `CLAM/` 目录下。部分脚本为实验性分析工具，可能需要你根据实际数据路径修改参数后运行。

