# 线程模式对比示例

这个示例演示了 CUDA 中不同的线程组织和遍历模式。

## 编译与运行

```bash
# 编译
nvcc -O3 -arch=sm_80 compare_patterns.cu -o compare_patterns

# 运行
./compare_patterns

# 指定数组大小
./compare_patterns 50000000  # 50M elements
```

## 使用 NCU 分析

```bash
# 分析内存访问模式
ncu --section MemoryWorkloadAnalysis ./compare_patterns

# 分析 Warp 效率
ncu --section WarpStateStats ./compare_patterns

# 完整分析
ncu --set full -o profile ./compare_patterns
ncu-ui profile.ncu-rep
```

## 模式说明

### Pattern 1: 标准一对一映射
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    out[idx] = in[idx] * 2.0f + 1.0f;
}
```
- **适用**：大多数情况
- **优点**：简单直观
- **配置**：gridSize = (N + blockSize - 1) / blockSize

### Pattern 2: Grid-Stride Loop
```cpp
int stride = blockDim.x * gridDim.x;
for (int i = tid; i < N; i += stride) {
    out[i] = in[i] * 2.0f + 1.0f;
}
```
- **适用**：超大数据集
- **优点**：更好的数据复用，提高 ILP
- **配置**：固定 gridSize（如 1024）

### Pattern 3: 向量化 (float4)
```cpp
int base = idx * 4;
float4 val = reinterpret_cast<const float4*>(in)[idx];
// process val
```
- **适用**：内存密集型
- **优点**：提高内存带宽利用率
- **提升**：1.5-2x

### Pattern 4: 向量化 + Grid-Stride
- **适用**：生产环境推荐
- **优点**：结合两者优势
- **提升**：2-3x

## 预期输出

```
========================================
CUDA Thread Pattern Comparison
========================================
Array size: 100000000 elements (400.0 MB)

Testing different thread patterns:
Block size: 256

Pattern 1: Standard mapping          :  8.234 ms,  972.11 GB/s  ✓ Correct
Pattern 2: Grid-stride loop           :  7.891 ms, 1014.23 GB/s  ✓ Correct
Pattern 3: Vectorized (float4)        :  5.123 ms, 1562.45 GB/s  ✓ Correct
Pattern 4: Vectorized + Grid-stride   :  4.987 ms, 1604.12 GB/s  ✓ Correct

========================================
Testing different block sizes (Pattern 1):
========================================
Block size:   64                      :  9.123 ms,  877.23 GB/s
Block size:  128                      :  8.456 ms,  946.78 GB/s
Block size:  256                      :  8.234 ms,  972.11 GB/s
Block size:  512                      :  8.345 ms,  959.34 GB/s
Block size: 1024                      :  8.567 ms,  934.12 GB/s

========================================
Recommendations:
========================================
1. Use block size of 256 (8 warps) for most cases
2. Use vectorization (float4) for memory-bound kernels
3. Use Grid-stride for very large data
4. Combine vectorization + Grid-stride for best performance

❌ DO NOT use stride=32 in loops!
✅ Use stride=blockDim.x*gridDim.x for Grid-stride
```

## 关键发现

1. **Block size = 256 通常最优**
   - 64: 太小，启动开销大
   - 1024: 太大，占用率可能降低
   - 256: 平衡点

2. **向量化很重要**
   - 提升 1.5-2x 性能
   - 接近理论带宽

3. **不要用 stride=32**
   - 会导致线程访问重叠
   - 产生竞争条件

4. **组合优化最佳**
   - 向量化 + Grid-stride = 最优性能

## 练习

1. 修改计算操作，观察性能变化
2. 使用 NCU 分析内存访问模式
3. 测试不同的数据大小
4. 尝试 float2 vs float4 向量化
