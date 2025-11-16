# 示例代码

本目录包含了各种 CUDA 和 Triton 算子的实现示例。

## 目录结构

```
examples/
├── vector_add/           # 向量加法
│   ├── cuda/            # CUDA 实现
│   │   └── vector_add.cu
│   └── triton/          # Triton 实现
│       └── vector_add.py
├── matrix_multiply/      # 矩阵乘法
│   ├── cuda/
│   └── triton/
└── README.md
```

## 运行示例

### CUDA 示例

```bash
# 编译 CUDA 代码
cd vector_add/cuda
nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add

# 运行
./vector_add

# 使用 NCU 分析
ncu --set full -o profile ./vector_add
ncu-ui profile.ncu-rep
```

### Triton 示例

```bash
# 安装依赖
pip install torch triton

# 运行 Triton 代码
cd vector_add/triton
python vector_add.py

# 使用 NCU 分析
ncu python vector_add.py
```

## 示例说明

### 1. Vector Add (向量加法)

最基础的示例，展示：
- CUDA kernel 基础语法
- Triton kernel 基础语法
- 性能测试和验证
- 向量化优化（CUDA）
- 自动调优（Triton）

**学习重点：**
- 线程索引计算
- 内存访问模式
- 性能测试方法

### 2. Matrix Multiply (矩阵乘法)

经典的计算密集型算子，展示：
- Tiling 技术
- Shared Memory 使用
- Tensor Cores 使用
- 性能优化技巧

**学习重点：**
- 数据复用
- 内存层次优化
- 计算吞吐量优化

## 性能测试

每个示例都包含性能测试代码：

1. **正确性验证**：对比 PyTorch/NumPy 结果
2. **性能测试**：测量运行时间和带宽
3. **优化对比**：对比不同优化版本

## 扩展练习

1. 修改 block size，观察性能变化
2. 使用 NCU 分析瓶颈
3. 实现更多优化版本
4. 添加其他数据类型支持（FP16, INT8）

## 参考资料

- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [Triton Examples](https://github.com/openai/triton/tree/main/python/tutorials)
- [NVIDIA GPU Teaching Kit](https://developer.nvidia.com/teaching-kits)
