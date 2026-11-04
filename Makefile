S ?= 01

.PHONY: up cluster-up cluster-down scenario-up scenarios-up scenarios-down fixtures test lint eval

# One command for a reviewer: cluster, RBAC, all scenarios, config file and the agent image.
up: cluster-up scenarios-up .env
	docker compose build agent
	@echo
	@echo "Ready. Faults take ~60s to show up (restarts, probe failures, events). Then:"
	@echo '  docker compose run --rm agent investigate "Why is checkout unhealthy?" -n scenario-01 --backend live'
	@echo "Offline, no API key: docker compose run --rm agent eval --llm scripted"
	@echo "For a real model, set GEMINI_API_KEY or ANTHROPIC_API_KEY and AGENT_LLM_PROVIDER in .env."

# Created once from the example; an existing .env is never overwritten.
.env:
	cp .env.example .env

cluster-up:
	kind get clusters | grep -qx aiops || kind create cluster --config deploy/kind-config.yaml --wait 120s
	kubectl --context kind-aiops apply -f deploy/rbac.yaml
	scripts/mint-kubeconfigs.sh

cluster-down:
	kind delete cluster --name aiops

scenario-up:
	kubectl --context kind-aiops apply -f scenarios/$(S)-*/manifests.yaml

scenarios-up:
	for f in scenarios/*/manifests.yaml; do kubectl --context kind-aiops apply -f $$f; done
	kubectl --context kind-aiops -n scenario-03 rollout status deploy/frontend --timeout=180s

scenarios-down:
	for f in scenarios/*/manifests.yaml; do kubectl --context kind-aiops delete --ignore-not-found -f $$f; done

fixtures:
	uv run python scripts/record_fixtures.py

test:
	uv run pytest

eval:
	uv run aiops eval $(EVAL_ARGS)

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy src
