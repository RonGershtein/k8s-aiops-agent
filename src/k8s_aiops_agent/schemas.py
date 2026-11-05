from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FaultCategory(StrEnum):
    NO_FAULT = "no_fault"
    CRASH_LOOP = "crash_loop"
    BAD_CONFIGURATION = "bad_configuration"
    READINESS_PROBE_FAILURE = "readiness_probe_failure"
    SERVICE_SELECTOR_MISMATCH = "service_selector_mismatch"
    IMAGE_PULL_ERROR = "image_pull_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SCHEDULING_FAILURE = "scheduling_failure"
    OTHER = "other"
    INCONCLUSIVE = "inconclusive"


_NO_FAULT_CLAIMED = {FaultCategory.NO_FAULT, FaultCategory.INCONCLUSIVE}


class Evidence(BaseModel):
    claim: str = Field(description="What this finding shows, in one sentence.")
    tool_call_id: str = Field(description="Id of the tool call whose output supports the claim.")
    excerpt: str = Field(description="Short verbatim quote from that tool output.")


class ServiceSelectorFix(BaseModel):
    service: str = Field(description="Name of the Service whose selector is wrong.")
    proposed_selector: dict[str, str] = Field(
        description="Selector labels that match the intended, healthy pods."
    )


class Diagnosis(BaseModel):
    fault_found: bool
    category: FaultCategory
    root_cause: str = Field(
        description="What is wrong, or an explicit statement that no fault was found."
    )
    affected_resources: list[str] = Field(default_factory=list, description="e.g. 'deployment/web'")
    evidence: list[Evidence] = Field(
        default_factory=list, description="At least one item, unless the result is inconclusive."
    )
    recommended_action: str
    action_risk: Literal["none", "low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_fix: ServiceSelectorFix | None = Field(
        default=None,
        description=(
            "Only for a Service selector mismatch: the selector fix the operator may apply "
            "after approval. Leave empty for every other fault."
        ),
    )

    @model_validator(mode="after")
    def _category_agrees_with_fault_found(self) -> "Diagnosis":
        if self.fault_found == (self.category in _NO_FAULT_CLAIMED):
            raise ValueError(
                "fault_found must be false exactly when category is 'no_fault' or 'inconclusive'"
            )
        if not self.evidence and self.category is not FaultCategory.INCONCLUSIVE:
            raise ValueError("evidence needs at least one item unless the result is inconclusive")
        return self

    def cited_tool_call_ids(self) -> set[str]:
        return {item.tool_call_id for item in self.evidence}


class RemediationProposal(BaseModel):
    action: Literal["patch_service_selector"] = "patch_service_selector"
    namespace: str
    service: str
    current_selector: dict[str, str]
    proposed_selector: dict[str, str]
    matching_ready_pods: list[str]
    expected_effect: str


class RemediationOutcome(BaseModel):
    approved: bool
    applied: bool
    resolved: bool
    verification: list[Evidence] = Field(default_factory=list)
    message: str


StopReason = Literal[
    "diagnosis_submitted",
    "max_steps",
    "token_budget",
    "timeout",
    "repeated_calls",
    "invalid_diagnosis",
    "model_refusal",
    "model_error",
]


class RunStats(BaseModel):
    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    stop_reason: StopReason
    trace_id: str | None = None


class InvestigationResult(BaseModel):
    question: str
    diagnosis: Diagnosis
    stats: RunStats
    remediation: RemediationOutcome | None = None
    ledger: list[dict[str, Any]] = Field(
        default_factory=list, description="Every tool call: id, name, args, output, error, latency."
    )
