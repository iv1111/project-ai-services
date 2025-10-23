REGISTRY?=icr.io/ai-services-private
IMAGE=docling
TAG?=2.36.0

CONTAINER_BUILDER?=podman
build:
	$(CONTAINER_BUILDER) build --build-arg DOCLING_VERSION=${TAG} -t $(REGISTRY)/$(IMAGE):$(TAG) .
.PHONY: build

push:
	$(CONTAINER_BUILDER) push $(REGISTRY)/$(IMAGE):$(TAG)
.PHONY: push
