# Makefile for building ComfyUI Docker images
# Supports cu130-megapak-pt211 and cu130-slim components with timestamped tags

# Default registry
REGISTRY ?= registry.cn-hangzhou.aliyuncs.com
REPO_NAME ?= eazycloud
IMAGE_NAME ?= $(REPO_NAME)/comfyui-custom

# Build arguments with defaults
MAX_JOBS ?= 1
TORCH_CUDA_ARCH_LIST ?= 8.0;8.6;10.0;12.0;12.0+PTX

# Current date for timestamped tags (format: YYYYMMDD)
DATE := $(shell date +%Y%m%d)

#  Model Path
MODEL_PARENT ?= /media/wwhvw/A63032EE3032C5595/comfyui-docker/storage-models 
OUTPUT_PARENT ?= /media/wwhvw/备份/comfyui

# Test Port
TEST_PORT ?= 8188

# Qwen2511-edit设置--fast会出现花图
# --disable-xformers可以解决xformers无法在cu130-py3.11-pt211 5090架构运行的问题
#CLI_ARGS ?= " --disable-xformers"

# Component definitions
COMPONENTS = cu130-megapak-pt211 
# base-cu130-pt211-cache xpu rocm rocm6 nightly rocm7 xpu-cn cu130-slim-v2 cu130-slim cu128-megapak-pt28 cu128-slim cu128-megapak cu126-slim cu128-megapak-pt29 cu126-megapak cpu base-cu130-pt211 base-cu130-slim-s2 base-cu130-slim-s1 base-cu130-devel base-rocm72-pt211

# Component directories mapping
COMPONENT_DIRS = \
    cu130-megapak-pt211:cu130-megapak-pt211 \
    base-cu130-pt211-cache:base-cu130-pt211-cache \
    xpu:xpu \
    rocm:rocm \
    rocm6:rocm6 \
    nightly:nightly \
    rocm7:rocm7 \
    xpu-cn:xpu-cn \
    cu130-slim-v2:cu130-slim-v2 \
    cu130-slim:cu130-slim \
    cu128-megapak-pt28:cu128-megapak-pt28 \
    cu128-slim:cu128-slim \
    cu128-megapak:cu128-megapak \
    cu126-slim:cu126-slim \
    cu128-megapak-pt29:cu128-megapak-pt29 \
    cu126-megapak:cu126-megapak \
    cpu:cpu \
    base-cu130-pt211:base-cu130-pt211 \
    base-cu130-slim-s2:base-cu130-slim-s2 \
    base-cu130-slim-s1:base-cu130-slim-s1 \
    base-cu130-devel:base-cu130-devel \
    base-rocm72-pt211:base-rocm72-pt211


# Function to get component directory
define get_component_dir
$(strip $(patsubst $(1):%,%,$(filter $(1):%,$(COMPONENT_DIRS))))
endef

# Function to get component tag
define get_component_tag
$(shell basename "$(call get_component_dir,$(1))")
endef

.PHONY: help build-all push-all clean-all test-all $(addprefix build-,$(COMPONENTS)) $(addprefix push-,$(COMPONENTS)) $(addprefix clean-,$(COMPONENTS)) $(addprefix test-,$(COMPONENTS))

help:
	@echo "Makefile for ComfyUI Docker Images"
	@echo ""
	@echo "Usage:"
	@echo "  make build-<component>     - Build specific component"
	@echo "  make push-<component>      - Build and push specific component"
	@echo "  make clean-<component>     - Remove specific component images"
	@echo "  make test-<component>      - Run test for specific component"
	@echo "  make build-all             - Build all components"
	@echo "  make push-all              - Build and push all components"
	@echo "  make clean-all             - Remove all component images"
	@echo "  make test-all              - Run tests for all components"
	@echo "  make build-list LIST=cu130-megapak-pt211,cu130-slim"
	@echo "  IMAGE_NAME=test make build-cu130-megapak-pt211"
	@echo "  IMAGE_NAME=test MODEL_PARENT=/media/wwhvw/A63032EE3032C5591/comfyui-docker/storage-models   make test-cu130-megapak-pt211
	@echo ""
	@echo "Available components: $(COMPONENTS)"

build-all: $(addprefix build-,$(COMPONENTS))
push-all: $(addprefix push-,$(COMPONENTS))
clean-all: $(addprefix clean-,$(COMPONENTS))
test-all: $(addprefix test-,$(COMPONENTS))

# Component template
define COMPONENT_TEMPLATE
build-$(1):
	@echo "------------------------------------------------"
	@echo "Building component: $(1)..."
	@COMPONENT_DIR="$(call get_component_dir,$(1))"; \
	COMPONENT_TAG="$(call get_component_tag,$(1))"; \
	if [ -z "$$$$COMPONENT_DIR" ]; then \
		echo "Error: Component $(1) not found"; exit 1; \
	fi; \
	echo "Building component: $(1)... ->  image: $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG"; \
	PROXY_ARGS=""; \
	if [ ! -z "$$$$http_proxy" ]; then \
		echo "Using http_proxy: $$$$http_proxy"; \
		PROXY_ARGS="$$$$PROXY_ARGS --build-arg HTTP_PROXY=$$$$http_proxy"; \
	elif [ ! -z "$$$$HTTP_PROXY" ]; then \
		echo "Using HTTP_PROXY: $$$$HTTP_PROXY"; \
		PROXY_ARGS="$$$$PROXY_ARGS --build-arg HTTP_PROXY=$$$$HTTP_PROXY"; \
	fi; \
	if [ ! -z "$$$$https_proxy" ]; then \
		echo "Using HTTPS_PROXY: $$$$https_proxy"; \
		PROXY_ARGS="$$$$PROXY_ARGS --build-arg HTTPS_PROXY=$$$$https_proxy"; \
	elif [ ! -z "$$$$HTTPS_PROXY" ]; then \
		echo "Using HTTPS_PROXY: $$$$HTTPS_PROXY"; \
		PROXY_ARGS="$$$$PROXY_ARGS --build-arg HTTPS_PROXY=$$$$HTTPS_PROXY"; \
	fi; \
	if [ ! -z "$$$$no_proxy" ]; then \
		PROXY_ARGS="$$$$PROXY_ARGS --build-arg NOPROXY=$$$$no_proxy"; \
	elif [ ! -z "$$$$NOPROXY" ]; then \
		PROXY_ARGS="$$$$PROXY_ARGS --build-arg NOPROXY=$$$$NO_PROXY"; \
	fi; \
	\
	docker build \
		$$$$PROXY_ARGS \
		--build-arg REGISTRY=$(REGISTRY) \
		--build-arg MAX_JOBS=$(MAX_JOBS) \
		--build-arg TORCH_CUDA_ARCH_LIST='$(TORCH_CUDA_ARCH_LIST)' \
		-t $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG \
		-t $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG-$(DATE) \
		$$$$COMPONENT_DIR

push-$(1): build-$(1)
	@COMPONENT_TAG="$(call get_component_tag,$(1))"; \
	docker push $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG; \
	docker push $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG-$(DATE)

clean-$(1):
	@COMPONENT_TAG="$(call get_component_tag,$(1))"; \
	docker rmi $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG 2>/dev/null || true; \
	docker rmi $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG-$(DATE) 2>/dev/null || true; \
	OLD_IMAGES=$$$$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "$(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG-" | grep -v "-$(DATE)$$$$"); \
	if [ ! -z "$$$$OLD_IMAGES" ]; then \
		for img in $$$$OLD_IMAGES; do docker rmi $$$$img 2>/dev/null || true; done; \
	fi

test-$(1): 
	@COMPONENT_TAG="$(call get_component_tag,$(1))"; \
	TEST_DIR=$$$$(mktemp -d -t comfyui-test-XXXXXX); \
	echo "Using temporary directory: $$$$TEST_DIR"; \
	mkdir -p $$$$TEST_DIR/storage-cache/dot-cache \
	         $$$$TEST_DIR/storage-cache/dot-config \
	         $$$$TEST_DIR/storage-nodes/dot-local \
	         $$$$TEST_DIR/storage-nodes/custom_nodes \
	         $$$$TEST_DIR/storage-models/models \
	         $$$$TEST_DIR/storage-models/hf-hub \
	         $$$$TEST_DIR/storage-models/torch-hub \
	         $$$$TEST_DIR/storage-models/u2net \
	         $$$$TEST_DIR/storage-user/input \
	         $$$$TEST_DIR/storage-user/output \
	         $$$$TEST_DIR/storage-user/user-profile \
	         $$$$TEST_DIR/storage-user/user-scripts; \
	echo "Starting container for component: $(1)..."; \
	HPROXY=""; \
	HSPROXY=""; \
	if [ ! -z "$$$$http_proxy" ]; then \
		echo "Using http_proxy: $$$$http_proxy"; \
		HPROXY="$$$$http_proxy"; \
	elif [ ! -z "$$$$HTTP_PROXY" ]; then \
		echo "Using HTTP_PROXY: $$$$HTTP_PROXY"; \
		HPROXY="$$$$HTTP_PROXY"; \
	fi; \
	if [ ! -z "$$$$https_proxy" ]; then \
		echo "Using HTTPS_PROXY: $$$$https_proxy"; \
		HSPROXY="$$$$https_proxy"; \
	elif [ ! -z "$$$$HTTPS_PROXY" ]; then \
		echo "Using HTTPS_PROXY: $$$$HTTPS_PROXY"; \
		HSPROXY="$$$$HTTPS_PROXY"; \
	fi; \
	docker run -it  \
	  --name comfyui-test-$$$$COMPONENT_TAG \
	  --gpus all \
	  -p $(TEST_PORT):8188 \
	  -v "$$$$TEST_DIR/storage-cache/dot-cache:/root/.cache" \
	  -v "$$$$TEST_DIR/storage-cache/dot-config:/root/.config" \
	  -v "$$$$TEST_DIR/storage-nodes/dot-local:/root/.local" \
	  -v "$$$$TEST_DIR/storage-nodes/custom_nodes:/root/ComfyUI/custom_nodes" \
	  -v "$(MODEL_PARENT)/models:/root/ComfyUI/models" \
	  -v "$(MODEL_PARENT)/u2net:/root/.u2net" \
	  -v "$(MODEL_PARENT)/comfyui_controlnet_aux/ckpts:/root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts" \
	  -v "$(MODEL_PARENT)/pixai:/root/ComfyUI/custom_nodes/booru-helper-mini/models/pixai" \
	  -v "$(MODEL_PARENT)/wd14tagger/models:/root/ComfyUI/custom_nodes/booru-helper-mini/models/ComfyUI-WD14-Tagger/models" \
	  -v "$(MODEL_PARENT)/hf-hub:/root/.cache/huggingface/hub" \
	  -v "$(MODEL_PARENT)/torch-hub:/root/.cache/torch/hub" \
	  -v "$(OUTPUT_PARENT)/assets:/root/ComfyUI/input" \
	  -v "$(OUTPUT_PARENT)/output:/root/ComfyUI/output" \
	  -v  $(OUTPUT_PARENT)/comfyui-ecosystem/workflows:/root/ComfyUI/user/default/workflows \
      -v  $(OUTPUT_PARENT)/comfyui-ecosystem/subgraphs:/root/ComfyUI/user/default/subgraphs \
	  -v "$$$$TEST_DIR/storage-user/user-scripts:/root/user-scripts" \
	  -e CUDA_LAUNCH_BLOCKING=1 \
	  -e HTTP_PROXY=$(HPROXY) \
	  -e HTTPS_PROXY=$(HSPROXY) \
	  -e HF_ENDPOINT="https://hf-mirror.com" \
	  -e CLI_ARGS=$(CLI_ARGS) \
	  $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG; \
	echo "Cleaning up temporary directory: $$$$TEST_DIR"; \
	#rm -rf $$$$TEST_DIR

simpletest-$(1): 
	@COMPONENT_TAG="$(call get_component_tag,$(1))"; \
	docker run -it  \
	  --name comfyui-test-$$$$COMPONENT_TAG \
	  --gpus all \
	  -p $(TEST_PORT):8188 \
	  -v "$(MODEL_PARENT)/models:/root/ComfyUI/models" \
	  -v "$(MODEL_PARENT)/u2net:/root/.u2net" \
	  -v "$(MODEL_PARENT)/hf-hub:/root/.cache/huggingface/hub" \
	  -v "$(MODEL_PARENT)/torch-hub:/root/.cache/torch/hub" \
	  -e CLI_ARGS=$(CLI_ARGS) \
	  $(REGISTRY)/$(IMAGE_NAME):$$$$COMPONENT_TAG; 
endef

$(foreach comp,$(COMPONENTS),$(eval $(call COMPONENT_TEMPLATE,$(comp))))

# --- LIST Rules ---
build-list:
	@if [ -z "$(LIST)" ]; then echo "Usage: make build-list LIST=comp1,comp2"; exit 1; fi
	@for comp in $$(echo $(LIST) | tr ',' ' '); do \
		$(MAKE) build-$$comp; \
	done

push-list:
	@if [ -z "$(LIST)" ]; then echo "Usage: make push-list LIST=comp1,comp2"; exit 1; fi
	@for comp in $$(echo $(LIST) | tr ',' ' '); do \
		$(MAKE) push-$$comp; \
	done

status:
	@echo "Current build status (Date: $(DATE)):"
	@for comp in $(COMPONENTS); do \
		TAG="$(call get_component_tag,$$comp)"; \
		if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "$(IMAGE_NAME):$$TAG$$"; then \
			echo "  $$comp: ✓ Built ($$TAG)"; \
		else \
			echo "  $$comp: ✗ Not built"; \
		fi; \
	done