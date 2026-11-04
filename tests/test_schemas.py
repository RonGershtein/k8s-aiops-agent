from typing import Any

import pytest
from pydantic import ValidationError

from k8s_aiops_agent.schemas import Diagnosis, FaultCategory


def diagnosis_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fault_found": True,
        "category": "crash_loop",
        "root_cause": "checkout exits because PAYMENT_TIMEOUT_MS is '30s', not an integer.",
        "evidence": [
            {"claim": "Container exits with FATAL", "tool_call_id": "call_1", "excerpt": "FATAL"},
            {"claim": "ConfigMap holds '30s'", "tool_call_id": "call_2", "excerpt": "30s"},
        ],
        "recommended_action": "Set PAYMENT_TIMEOUT_MS to 30000 in the checkout-config ConfigMap.",
        "action_risk": "low",
        "confidence": 0.9,
    }
    return payload | overrides


def test_diagnosis_with_valid_payload_parses() -> None:
    payload = diagnosis_payload()

    diagnosis = Diagnosis.model_validate(payload)

    assert diagnosis.category is FaultCategory.CRASH_LOOP


def test_diagnosis_inconclusive_without_evidence_or_fault_parses() -> None:
    payload = diagnosis_payload(fault_found=False, category="inconclusive", evidence=[])

    diagnosis = Diagnosis.model_validate(payload)

    assert diagnosis.category is FaultCategory.INCONCLUSIVE


def test_diagnosis_inconclusive_claiming_a_fault_is_rejected() -> None:
    payload = diagnosis_payload(fault_found=True, category="inconclusive")

    with pytest.raises(ValidationError, match="inconclusive"):
        Diagnosis.model_validate(payload)


def test_diagnosis_without_evidence_is_rejected() -> None:
    payload = diagnosis_payload(evidence=[])

    with pytest.raises(ValidationError, match="evidence"):
        Diagnosis.model_validate(payload)


def test_diagnosis_fault_found_with_no_fault_category_is_rejected() -> None:
    payload = diagnosis_payload(fault_found=True, category="no_fault")

    with pytest.raises(ValidationError, match="no_fault"):
        Diagnosis.model_validate(payload)


def test_diagnosis_no_fault_with_fault_category_is_rejected() -> None:
    payload = diagnosis_payload(fault_found=False, category="crash_loop")

    with pytest.raises(ValidationError, match="no_fault"):
        Diagnosis.model_validate(payload)


def test_diagnosis_confidence_above_one_is_rejected() -> None:
    payload = diagnosis_payload(confidence=1.5)

    with pytest.raises(ValidationError, match="confidence"):
        Diagnosis.model_validate(payload)


def test_diagnosis_cited_tool_call_ids_returns_unique_ids() -> None:
    diagnosis = Diagnosis.model_validate(diagnosis_payload())

    cited = diagnosis.cited_tool_call_ids()

    assert cited == {"call_1", "call_2"}
