S ?= 01

.PHONY: cluster-up cluster-down scenario-up scenarios-down test lint

cluster-up:
	kind get clusters | grep -qx aiops || kind create cluster --config deploy/kind-config.yaml --wait 120s
	kubectl --context kind-aiops apply -f deploy/rbac.yaml
	scripts/mint-kubeconfigs.sh

cluster-down:
	kind delete cluster --name aiops

scenario-up:
	kubectl --context kind-aiops apply -f scenarios/$(S)-*/manifests.yaml

scenarios-down:
	for f in scenarios/*/manifests.yaml; do kubectl --context kind-aiops delete --ignore-not-found -f $$f; done

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy src
