# Makefile for GEMM Benchmarks

NVCC = nvcc
NVCC_FLAGS = -O3 -use_fast_math -arch=sm_80 -lineinfo
# For A100: sm_80, For V100: sm_70, For RTX 3090: sm_86

TARGETS = gemm_v0_naive gemm_v2_shared_tiling gemm_v3_optimized

.PHONY: all clean run compare help

all: $(TARGETS)

# Build rules
gemm_v0_naive: gemm_v0_naive.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

gemm_v2_shared_tiling: gemm_v2_shared_tiling.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

gemm_v3_optimized: gemm_v3_optimized.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Run all versions
run: $(TARGETS)
	@echo "========================================"
	@echo "Running all GEMM benchmarks"
	@echo "========================================"
	@for target in $(TARGETS); do \
		echo ""; \
		echo ">>> Running $$target (1024×1024×1024)..."; \
		./$$target 1024 1024 1024; \
		echo ""; \
	done

# Compare versions
compare: $(TARGETS)
	@echo "Running comparison..."
	@python3 ../../../tools/compare_versions.py $(TARGETS) --json=gemm_comparison.json

# Profile with NCU
ncu: $(TARGETS)
	@echo "Profiling with NCU..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		ncu --set full --export $$target ./$$target 1024 1024 1024; \
	done

# Clean
clean:
	rm -f $(TARGETS)
	rm -f *.ncu-rep
	rm -f *.nsys-rep
	rm -f *.json

help:
	@echo "GEMM Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all versions"
	@echo "  run      - Run all versions with 1024×1024×1024"
	@echo "  compare  - Compare performance"
	@echo "  ncu      - Profile with NCU"
	@echo "  clean    - Clean artifacts"
	@echo ""
	@echo "Individual versions:"
	@echo "  gemm_v0_naive          - Baseline (no optimization)"
	@echo "  gemm_v2_shared_tiling  - Shared memory tiling"
	@echo "  gemm_v3_optimized      - Bank conflict fix + optimizations"
