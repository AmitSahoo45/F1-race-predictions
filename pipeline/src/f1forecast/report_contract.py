"""Version gate for evaluation evidence consumed by publication and approval."""

EVALUATION_SCHEMA_VERSION = "2"


def validate_report_format(report: dict) -> None:
    if report.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            "evaluation report schema is obsolete; regenerate with current input eligibility rules"
        )
    events = report.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("evaluation report requires per-event evidence")
    for event in events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("required_inputs_available"), bool)
            or not isinstance(event.get("unexpected_actual_entrants"), list)
        ):
            raise ValueError("evaluation report lacks explicit per-event availability diagnostics")  # noqa: TRY004 - invalid JSON evidence, not a Python argument type
