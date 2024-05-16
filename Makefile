# Based on sgemm_cuda_siboehm/Makefile

.PHONY: all build clean profile

CMAKE := cmake

BUILD_DIR := build
BENCH_DIR := bench_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release ..
	# XXX @$(MAKE) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

# Run with make profile KERNEL=<integer> (default 1)
KERNEL ?= 1
profile: build
	@ncu --call-stack --set full --export $(BENCH_DIR)/kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)
