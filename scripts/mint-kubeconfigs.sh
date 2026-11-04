#!/usr/bin/env bash
# Writes one kubeconfig per agent ServiceAccount, so the agent never uses the cluster-admin context.
set -euo pipefail

context="${KIND_CONTEXT:-kind-aiops}"
out_dir="${OUT_DIR:-.kube}"
# shortcut: tokens expire after 24h, rerun this script (or `make cluster-up`) to refresh.
duration="${TOKEN_DURATION:-24h}"

server=$(kubectl config view --raw -o jsonpath="{.clusters[?(@.name=='$context')].cluster.server}")
ca=$(kubectl config view --raw -o jsonpath="{.clusters[?(@.name=='$context')].cluster.certificate-authority-data}")
[[ -n "$server" && -n "$ca" ]] || { echo "context '$context' not found; run 'kind create cluster' first" >&2; exit 1; }

mkdir -p "$out_dir"
for sa in readonly remediator; do
  token=$(kubectl --context "$context" -n aiops-system create token "aiops-agent-$sa" --duration "$duration")
  file="$out_dir/agent-$sa.kubeconfig"
  umask 077
  cat >"$file" <<EOF
apiVersion: v1
kind: Config
clusters:
  - name: aiops
    cluster:
      server: $server
      certificate-authority-data: $ca
users:
  - name: aiops-agent-$sa
    user:
      token: $token
contexts:
  - name: aiops-agent-$sa
    context: {cluster: aiops, user: aiops-agent-$sa}
current-context: aiops-agent-$sa
EOF
  echo "wrote $file"
done
