import pytest
from f1forecast.report_contract import validate_report_format


def test_legacy_or_incomplete_report_cannot_be_used_as_current_evidence():
    with pytest.raises(ValueError, match="obsolete"):
        validate_report_format({"events": []})
    with pytest.raises(ValueError, match="diagnostics"):
        validate_report_format(
            {
                "evaluation_schema_version": "2",
                "events": [{"missing_pattern": ["missing-qualifying-result"]}],
            }
        )
    validate_report_format(
        {
            "evaluation_schema_version": "2",
            "events": [{"required_inputs_available": False, "unexpected_actual_entrants": []}],
        }
    )
