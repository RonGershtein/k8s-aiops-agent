# Eval report: fixture-gemini

Pass rate 94%: 16/17 graded trials passed, 1 errored.

## Tuning scenarios

| scenario | pass rate | categories | mean steps | mean tokens | mean latency | judge (c/e/a) |
|---|---|---|---|---|---|---|
| 00-healthy | 2/3 | no_fault | 3.3 | 11,124 | 4.2s | 5.0/5.0/5.0 |
| 01-crashloop-bad-config | 3/3 | bad_configuration | 4.0 | 14,912 | 11.3s | 5.0/5.0/5.0 |
| 02-readiness-wrong-port | 2/2 (+1 errored) | readiness_probe_failure | 4.0 | 14,350 | 4.4s | 5.0/5.0/5.0 |
| 03-service-selector-mismatch | 3/3 | service_selector_mismatch | 3.0 | 7,717 | 7.2s | 5.0/5.0/5.0 |

## Held-out scenarios

| scenario | pass rate | categories | mean steps | mean tokens | mean latency | judge (c/e/a) |
|---|---|---|---|---|---|---|
| 04-image-pull-error | 3/3 | image_pull_error | 3.3 | 10,523 | 4.3s | 5.0/5.0/5.0 |
| 05-missing-configmap | 3/3 | bad_configuration | 4.3 | 13,340 | 18.9s | 5.0/5.0/5.0 |

## Failures

- 00-healthy #3: evidence_has_key_signals | judge: The agent correctly identified that there is no fault in the Kubernetes resources, supported its claim with valid evidence from the cluster state, and recommended a safe and appropriate course of action. (trace f4a80d4b24bf31ca58d47ed1892e2741)

## Errored (model API did not answer; not graded)

- 02-readiness-wrong-port #2: Inconclusive: the investigation stopped (model_error) before a supported conclusion was reached. (trace 8e1316595dd188ab03f9fd2c49f525b1)
