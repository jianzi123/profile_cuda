# Makefile for Vector Add Benchmarks
#
# Usage:
#   make all       - Build all versions
#   make run       - Run all versions
#   make ncu       - Profile all versions with NCU
#   make clean     - Clean build artifacts
#   make compare   - Run and compare all versions

NVCC = nvcc
NVCC_FLAGS = -O3 -use_fast_math -arch=sm_80 -lineinfo
# For A100: sm_80, For V100: sm_70, For RTX 3090: sm_86

TARGETS = v0_naive v1_coalesced v2_vectorized v3_shared_tiling v4_optimized
NCU_REPORTS = $(addsuffix .ncu-rep, $(TARGETS))

.PHONY: all clean run ncu compare help

all: $(TARGETS)

# Build rules
v0_naive: v0_naive.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

v1_coalesced: v1_coalesced.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

v2_vectorized: v2_vectorized.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

v3_shared_tiling: v3_shared_tiling.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

v4_optimized: v4_optimized.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Run all versions
run: $(TARGETS)
	@echo "========================================"
	@echo "Running all Vector Add benchmarks"
	@echo "========================================"
	@for target in $(TARGETS); do \
		echo ""; \
		echo ">>> Running $$target..."; \
		./$$target; \
		echo ""; \
	done

# Profile with NCU
ncu: $(TARGETS)
	@echo "========================================"
	@echo "Profiling with NVIDIA Nsight Compute"
	@echo "========================================"
	@for target in $(TARGETS); do \
		echo ""; \
		echo ">>> Profiling $$target..."; \
		ncu --set full --export $$target ./$$target; \
		echo "Generated: $$target.ncu-rep"; \
	done

# Quick metrics
ncu-quick: $(TARGETS)
	@echo "========================================"
	@echo "Quick NCU metrics"
	@echo "========================================"
	@for target in $(TARGETS); do \
		echo ""; \
		echo ">>> $$target:"; \
		ncu --metrics gpu__time_duration.avg,\
l1tex__average_t_sectors_per_request,\
smsp__sass_thread_inst_executed_op_fadd.sum,\
dram__bytes.sum \
		./$$target 2>&1 | grep -A 20 "Metric Name"; \
	done

# Compare all versions
compare: $(TARGETS)
	@./benchmark.sh

# Clean build artifacts
clean:
	rm -f $(TARGETS)
	rm -f *.ncu-rep
	rm -f *.nsys-rep
	rm -f *.csv
	rm -f benchmark_results.txt

# Help
help:
	@echo "Vector Add Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all versions"
	@echo "  run         - Run all versions"
	@echo "  ncu         - Profile with full NCU metrics"
	@echo "  ncu-quick   - Profile with key metrics only"
	@echo "  compare     - Run benchmark comparison script"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Individual versions:"
	@for target in $(TARGETS); do \
		echo "  $$target"; \
	done
	@echo ""
	@echo "Configuration:"
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo "  Architecture: sm_80 (A100)"
	@echo ""
	@echo "Notes:"
	@echo "  - Adjust -arch=sm_XX for your GPU"
	@echo "  - NCU requires sudo on some systems"
	@echo "  - Use 'make compare' for detailed performance analysis"
