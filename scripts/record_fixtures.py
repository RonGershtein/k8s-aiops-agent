"""Snapshot every scenario namespace from the live cluster into scenarios/<id>/fixture.json.

Run after `make scenario-up` for each scenario and ~60s of settling, so restarts and events exist.
"""

import json
import sys
from pathlib import Path

import yaml

from k8s_aiops_agent.fixture_kube import FIXTURE_FILE, record_namespace
from k8s_aiops_agent.kube import LiveKube

kubeconfig = Path(sys.argv[1] if len(sys.argv) > 1 else ".kube/agent-readonly.kubeconfig")
kube = LiveKube.from_kubeconfig(kubeconfig)

for scenario_file in sorted(Path("scenarios").glob("*/scenario.yaml")):
    namespace = yaml.safe_load(scenario_file.read_text())["namespace"]
    recording = record_namespace(kube, namespace)
    if not recording["pods"]:
        sys.exit(f"{namespace} has no pods; run `make scenario-up S=...` first")
    out = scenario_file.parent / FIXTURE_FILE
    out.write_text(json.dumps(recording, indent=2, sort_keys=True, default=str) + "\n")
    print(f"recorded {namespace} -> {out}")
