# ME-BBB：机制增强的血脑屏障通透性预测

[English README](README.md) · [工作流程](docs/workflow.md) · [环境信息](docs/environment.md) · [数据输入](data/README.md) · [第三方许可](THIRD_PARTY_NOTICES.md)

ME-BBB 使用 **ECFP4（Morgan radius 2，2048 bit）** 和五个机制专家（BCRP、MDR1、MRP、PAMPA、Caco-2）的预测概率，建立基于 ExtraTrees 的 BBB 分类模型。机制专家以 RDKit2DNormalized 描述符和 TabPFN 3.5 为基础，使用预定义的 Bemis–Murcko 骨架划分。

## 本次公开内容

这是**代码先行发布版**，包含训练专家模型、传统算法对照、BBB 模型训练、消融实验，以及 Figure 2–6 绘图代码。此次**不包含**原始分子标签、划分 CSV、专家预测分数、拟合模型、TabPFN checkpoint、SHAP 缓存及图件输出。因此，需要自行准备符合许可要求的输入数据后才能运行完整流程；公开代码并不等同于已完整复现论文全部数值。

## 环境信息（2026-11-03 提供）

- Python **3.11.17**；PyTorch **2.7.1+cu118**（其 CUDA 构建版本是 **11.8**）。
- NVIDIA 驱动 **580.95.05**，两张 **Tesla T4（每张约 15 GB）**。
- `nvidia-smi` 显示的 CUDA **13.0** 是驱动支持版本，不能写成 PyTorch 的实际 CUDA 构建版本。
- TabPFN、RDKit 等依赖的原始训练版本尚待补充，详见 [环境文档](docs/environment.md)。

## 基础检查

```bash
conda create -n me-bbb python=3.11 -y
conda activate me-bbb
python -m pip install -r requirements/dev.txt
python -m pytest tests -q
```

GPU 专家训练需另行安装 `requirements/tabpfn.txt`，并从 Prior Labs 官方获取具有合法使用权限的 TabPFN 3.5 checkpoint。`requirements/` 未冒充已验证的原始锁定环境。论文 DOI 和作者引文后续补充。

主脚本位于 `scripts/train_experts/`、`scripts/train_bbb/`、`scripts/analyses/ablation/` 和 `scripts/figures/`。输入约定见 [data/README.md](data/README.md)，模型说明见 [models/README.md](models/README.md)。原始代码按仓库 MIT 许可证发布，但 TabPFN 权重、模型输出和第三方数据不受项目 MIT 许可证覆盖。
