# Eval report: fixture-scripted

## Tuning scenarios

| scenario | pass rate | categories | mean steps | mean tokens | mean latency | judge (c/e/a) |
|---|---|---|---|---|---|---|
| 00-healthy | 1/1 | no_fault | 3.0 | 3,300 | 0.0s | n/a |
| 01-crashloop-bad-config | 1/1 | bad_configuration | 4.0 | 4,400 | 0.0s | n/a |
| 02-readiness-wrong-port | 1/1 | readiness_probe_failure | 3.0 | 3,300 | 0.0s | n/a |
| 03-service-selector-mismatch | 1/1 | service_selector_mismatch | 3.0 | 3,300 | 0.0s | n/a |

## Held-out scenarios

| scenario | pass rate | categories | mean steps | mean tokens | mean latency | judge (c/e/a) |
|---|---|---|---|---|---|---|
| 04-image-pull-error | 1/1 | image_pull_error | 2.0 | 2,200 | 0.0s | n/a |
| 05-missing-configmap | 1/1 | bad_configuration | 3.0 | 3,300 | 0.0s | n/a |

## Failures

None.
