#!/usr/bin/env bash
# Writes one kubeconfig per agent ServiceAccount, so the agent never uses the cluster-admin context.
# Each also gets an .internal variant for containers on the kind docker network.
set -euo pipefail

context="${KIND_CONTEXT:-kind-aiops}"
out_dir="${OUT_DIR:-.kube}"
# shortcut: tokens expire after 24h, rerun this script (or `make cluster-up`) to refresh.
duration="${TOKEN_DURATION:-24h}"

server=$(kubectl config view --raw -o jsonpath="{.clusters[?(@.name=='$context')].cluster.server}")
ca=$(kubectl config view --raw -o jsonpath="{.clusters[?(@.name=='$context')].cluster.certificate-authority-data}")
[[ -n "$server" && -n "$ca" ]] || { echo "context '$context' not found; run 'kind create cluster' first" >&2; exit 1; }
internal_server="https://${context#kind-}-control-plane:6443"

write_kubeconfig() {
  local file=$1 url=$2 sa=$3 token=$4
  cat >"$file" <<EOF
apiVersion: v1
kind: Config
clusters:
  - name: aiops
    cluster:
      server: $url
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
}

mkdir -p "$out_dir"
umask 077
for sa in readonly remediator; do
  token=$(kubectl --context "$context" -n aiops-system create token "aiops-agent-$sa" --duration "$duration")
  write_kubeconfig "$out_dir/agent-$sa.kubeconfig" "$server" "$sa" "$token"
  write_kubeconfig "$out_dir/agent-$sa.internal.kubeconfig" "$internal_server" "$sa" "$token"
done
