#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import importlib

print("="*60)
print("🔍 Deep Learning Environment Check")
print("="*60)

# 1️⃣ Python & Conda 基本信息
print(f"🐍 Python version: {sys.version.split()[0]}")
print(f"📂 Python executable: {sys.executable}")

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "N/A")
print(f"🧩 Conda environment: {conda_env}")

print("-"*60)

# 2️⃣ 要检查的核心包
packages = [
    "torch", "torchvision", "torchaudio", "torch_geometric", "numpy"
]

for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {pkg:15s} version: {version}")
    except ImportError:
        print(f"❌ {pkg:15s} not found")

print("-"*60)

# 3️⃣ Torch 运行时检查
try:
    import torch

    print(f"🧠 Torch version: {torch.__version__}")
    print(f"📦 Torch CUDA available: {torch.cuda.is_available()}")
    print(f"💻 Torch MPS available (Apple Silicon): {torch.backends.mps.is_available()}")
    print(f"🧮 Torch device: {'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 简单 tensor 测试
    x = torch.randn(2, 3)
    print(f"Tensor test OK, sample: \n{x}")
except Exception as e:
    print(f"⚠️ Torch test failed: {e}")

print("-"*60)

# 4️⃣ 检查 OpenMP 冲突风险
libomp = os.popen("ls $(conda info --base)/envs/*/lib | grep libomp | wc -l").read().strip()
if libomp and int(libomp) > 1:
    print(f"⚠️ Detected multiple libomp libraries ({libomp} copies). May cause OMP Error #15.")
    print("💡 Try setting:")
    print("    export OMP_NUM_THREADS=1")
    print("    export MKL_NUM_THREADS=1")
else:
    print("✅ No multiple libomp libraries detected.")

print("="*60)
print("✅ Environment check complete.")
