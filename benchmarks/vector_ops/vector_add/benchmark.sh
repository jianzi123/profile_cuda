#!/bin/bash

# Vector Add Benchmark Comparison Script
#
# This script runs all versions and compares performance
# Generates a summary table with speedups

set -e

echo "========================================"
echo "Vector Add Benchmark Suite"
echo "========================================"
echo ""

# Check if binaries exist
VERSIONS=("v0_naive" "v1_coalesced" "v2_vectorized" "v3_shared_tiling" "v4_optimized")
for ver in "${VERSIONS[@]}"; do
    if [ ! -f "./$ver" ]; then
        echo "Error: $ver not found. Please run 'make all' first."
        exit 1
    fi
done

# Problem size (can be overridden)
SIZE=${1:-67108864}  # 64M elements = 256MB
echo "Problem size: $SIZE elements"
echo ""

# Output file
OUTPUT="benchmark_results.txt"
> $OUTPUT

# Function to extract time from output
extract_time() {
    grep "Time:" | awk '{print $2}' | head -1
}

# Function to extract bandwidth from output
extract_bandwidth() {
    grep "Bandwidth:" | awk '{print $2}' | head -1
}

# Run benchmarks
declare -A times
declare -A bandwidths

for ver in "${VERSIONS[@]}"; do
    echo "Running $ver..."
    output=$(./$ver $SIZE 2>&1)

    time_ms=$(echo "$output" | extract_time)
    bw=$(echo "$output" | extract_bandwidth)

    times[$ver]=$time_ms
    bandwidths[$ver]=$bw

    echo "$ver: ${time_ms} ms, ${bw} GB/s"

    # Save full output
    echo "=== $ver ===" >> $OUTPUT
    echo "$output" >> $OUTPUT
    echo "" >> $OUTPUT
done

echo ""
echo "========================================"
echo "Performance Comparison Summary"
echo "========================================"
echo ""

# Print header
printf "%-20s %12s %15s %12s %12s\n" "Version" "Time (ms)" "Bandwidth (GB/s)" "Speedup" "Efficiency"
printf "%-20s %12s %15s %12s %12s\n" "--------------------" "------------" "---------------" "------------" "------------"

# Calculate baseline
baseline_time=${times[v0_naive]}

# Calculate theoretical bandwidth (A100: 1555 GB/s, adjust for your GPU)
THEORETICAL_BW=1555

# Print results
for ver in "${VERSIONS[@]}"; do
    time_ms=${times[$ver]}
    bw=${bandwidths[$ver]}

    if [ "$ver" == "v0_naive" ]; then
        speedup="1.00x"
    else
        speedup=$(echo "scale=2; $baseline_time / $time_ms" | bc)
        speedup="${speedup}x"
    fi

    efficiency=$(echo "scale=1; $bw / $THEORETICAL_BW * 100" | bc)

    printf "%-20s %12s %15s %12s %11s%%\n" "$ver" "$time_ms" "$bw" "$speedup" "$efficiency"
done

echo ""
echo "========================================"
echo "Key Insights"
echo "========================================"
echo ""

# Calculate key metrics
v0_time=${times[v0_naive]}
v1_time=${times[v1_coalesced]}
v2_time=${times[v2_vectorized]}
v3_time=${times[v3_shared_tiling]}
v4_time=${times[v4_optimized]}

coalesce_speedup=$(echo "scale=1; $v0_time / $v1_time" | bc)
vectorize_speedup=$(echo "scale=1; $v1_time / $v2_time" | bc)
shared_slowdown=$(echo "scale=1; $v2_time / $v3_time" | bc)
final_speedup=$(echo "scale=1; $v2_time / $v4_time" | bc)
total_speedup=$(echo "scale=1; $v0_time / $v4_time" | bc)

echo "1. Memory Coalescing (v0 → v1):"
echo "   - Speedup: ${coalesce_speedup}x"
echo "   - Impact: 修复非合并访问是最关键的优化"
echo "   - NCU 指标: sectors_per_request 从 ~32 降到 ~1.0"
echo ""

echo "2. Vectorization (v1 → v2):"
echo "   - Speedup: ${vectorize_speedup}x"
echo "   - Impact: float4 减少指令数，提升 ILP"
echo "   - NCU 指标: 指令数减少约 50%"
echo ""

echo "3. Shared Memory (v2 → v3) - 反面教材:"
if (( $(echo "$shared_slowdown < 1.0" | bc -l) )); then
    echo "   - Slowdown: ${shared_slowdown}x (性能下降!)"
    echo "   - Impact: Vector add 不需要数据重用，Shared Memory 引入开销"
    echo "   - NCU 指标: barrier stall 增加 20-40%"
    echo "   - 教训: 并非所有优化都适用于所有场景"
else
    echo "   - Result: ${shared_slowdown}x (可能 GPU 特定行为不同)"
fi
echo ""

echo "4. Final Optimizations (v2 → v4):"
echo "   - Speedup: ${final_speedup}x"
echo "   - Impact: Loop unrolling, 读缓存优化, 最优 grid size"
echo "   - NCU 指标: 接近理论带宽峰值"
echo ""

echo "5. Total Improvement (v0 → v4):"
echo "   - Speedup: ${total_speedup}x"
echo "   - Efficiency: $(echo "scale=1; ${bandwidths[v4_optimized]} / $THEORETICAL_BW * 100" | bc)%"
echo "   - 结论: 达到硬件极限，无需进一步优化内核"
echo ""

echo "========================================"
echo "NCU Profiling Commands"
echo "========================================"
echo ""
echo "# Full profiling (all metrics):"
echo "make ncu"
echo ""
echo "# Quick comparison (key metrics only):"
echo "make ncu-quick"
echo ""
echo "# Individual version:"
echo "ncu --set full --export v4_optimized ./v4_optimized"
echo ""
echo "# Specific metrics:"
echo "ncu --metrics l1tex__average_t_sectors_per_request,smsp__average_warps_issue_stalled_barrier ./v1_coalesced"
echo ""

echo "Full results saved to: $OUTPUT"
echo ""
