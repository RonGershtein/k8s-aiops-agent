S ?= 01
Q ?= What is wrong in this namespace?

.PHONY: up run fix cluster-up cluster-down scenario-up scenarios-up scenarios-down fixtures test lint eval

# One command for a reviewer: cluster, RBAC, all scenarios, config file and the agent image.
up: cluster-up scenarios-up .env
	docker compose build agent
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo ""
	@echo "  1. Add a model API key to .env:"
	@echo "       AGENT_LLM_PROVIDER=gemini"
	@echo "       GEMINI_API_KEY=your-key      free at https://aistudio.google.com/api-keys"
	@echo ""
	@echo "  2. Wait about 60 seconds for the faults to appear."
	@echo ""
	@echo "  3. Run the agent:"
	@echo "       make run S=01                investigate a scenario (S=00 to S=05)"
	@echo "       make fix                     review and approve the scenario 03 fix"
	@echo "       make scenario-up S=03        reset scenario 03 after a fix"
	@echo ""
	@echo "  No API key? Run the offline eval:  docker compose run --rm agent eval --llm scripted"

# Investigate one scenario on the live cluster with the model configured in .env.
run:
	docker compose run --rm agent investigate "$(Q)" -n scenario-$(S) --backend live

# Investigate with the approval-gated fix enabled. Only scenario 03's Service selector has a
# supported fix; other scenarios end with a recommendation. Reset with `make scenario-up S=03`.
fix: S = 03
fix:
	docker compose run --rm agent investigate "$(Q)" -n scenario-$(S) --backend live --allow-remediation

# Created once from the example; an existing .env is never overwritten.
.env:
	cp .env.example .env

cluster-up:
	kind get clusters | grep -qx aiops || kind create cluster --config deploy/kind-config.yaml --wait 120s
	kubectl --context kind-aiops apply -f deploy/rbac.yaml
	scripts/mint-kubeconfigs.sh

cluster-down:
	kind delete cluster --name aiops

# apply creates what is missing; replace then restores the exact manifest (also after `make fix`).
scenario-up:
	kubectl --context kind-aiops apply -f scenarios/$(S)-*/manifests.yaml
	kubectl --context kind-aiops replace -f scenarios/$(S)-*/manifests.yaml >/dev/null

scenarios-up:
	for f in scenarios/*/manifests.yaml; do \
		kubectl --context kind-aiops apply -f $$f && kubectl --context kind-aiops replace -f $$f >/dev/null; \
	done
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
