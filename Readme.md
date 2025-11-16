# CUDA/Triton 算子开发与性能分析完整教程

本项目提供了一套完整的 CUDA 和 Triton 算子开发、优化和性能分析的教程与实践示例。

## 📚 目录结构

```
profile_cuda/
├── 01_cuda_basics/           # CUDA 算子基础教程
│   ├── tutorial.md           # CUDA 编程基础
│   └── examples/             # 示例代码
├── 02_triton_basics/         # Triton 算子基础教程
│   ├── tutorial.md           # Triton 编程基础
│   └── examples/             # 示例代码
├── 03_performance_optimization/  # 性能优化指南
│   ├── optimization_guide.md     # 优化技巧
│   ├── gpu_architecture.md       # GPU 架构适配
│   └── best_practices.md         # 最佳实践
├── 04_performance_analysis/  # 性能分析工具
│   ├── ncu_guide.md          # NCU 使用指南
│   ├── nsight_systems.md     # Nsight Systems 指南
│   └── profiling_tools.md    # 其他性能分析工具
├── 05_global_optimization/   # 全局性能分析
│   └── global_perspective.md # 全局视角优化
└── examples/                 # 完整项目示例
    ├── vector_add/           # 向量加法
    ├── matrix_multiply/      # 矩阵乘法
    ├── softmax/              # Softmax 算子
    └── flash_attention/      # Flash Attention
```

## 🎯 学习路径

### 1️⃣ CUDA 算子开发
- [CUDA 基础教程](01_cuda_basics/tutorial.md)
- 学习如何编写基础 CUDA kernel
- 理解线程模型、内存层次结构
- 实践向量加法、矩阵乘法等经典算子

### 2️⃣ Triton 算子开发
- [Triton 基础教程](02_triton_basics/tutorial.md)
- 理解 Triton 的编程模型
- 学习如何用 Python 编写高性能算子
- 对比 CUDA 和 Triton 的优缺点

### 3️⃣ 性能优化
- [性能优化指南](03_performance_optimization/optimization_guide.md)
- [GPU 架构适配](03_performance_optimization/gpu_architecture.md)
- 学习内存访问优化、计算优化
- 针对不同 GPU 架构（Ampere、Hopper等）的优化策略

### 4️⃣ 性能分析
- [NCU 完整指南](04_performance_analysis/ncu_guide.md)
- [Nsight Systems 使用](04_performance_analysis/nsight_systems.md)
- 学习如何定位性能瓶颈
- 分析带宽利用率、计算效率

### 5️⃣ 全局优化
- [全局性能分析](05_global_optimization/global_perspective.md)
- 从系统级别优化算子性能
- 多算子融合、内存复用策略

## 🚀 快速开始

### 环境准备

```bash
# 安装必要的工具
pip install torch triton nvidia-pyindex
pip install nvidia-nsight-compute

# 检查 CUDA 环境
nvidia-smi
nvcc --version
```

### 运行第一个示例

```bash
# CUDA 版本
cd examples/vector_add/cuda
python test.py

# Triton 版本
cd examples/vector_add/triton
python test.py
```

## 📊 性能分析示例

```bash
# 使用 NCU 分析 CUDA kernel
ncu --set full -o profile_result python test.py

# 使用 Nsight Systems 分析整体性能
nsys profile -o timeline python test.py
```

## 🎓 核心知识点

### CUDA 关键概念
- 线程层次结构（Thread/Block/Grid）
- 内存层次（Global/Shared/Register）
- 同步机制（__syncthreads）
- 内存访问模式（Coalesced Access）

### Triton 关键概念
- Block-level 编程
- 自动内存管理
- 自动调优（Auto-tuning）
- JIT 编译优化

### 性能优化关键点
- 最大化内存带宽利用率
- 最大化计算吞吐量
- 减少分支分歧
- 优化寄存器使用
- Tile 优化策略

## 📖 推荐阅读顺序

1. 先学习 CUDA 基础（01_cuda_basics）
2. 再学习 Triton 基础（02_triton_basics）
3. 深入性能优化技巧（03_performance_optimization）
4. 掌握性能分析工具（04_performance_analysis）
5. 建立全局优化视角（05_global_optimization）

## 🔗 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

## 📝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License
