"""Export schemas; generated files must never be edited by hand."""

import json
from pathlib import Path

from .contracts import Analysis, Event, EventTelemetry, Forecast, SiteData


def export_schemas(destination: str | Path = "schemas") -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, model in [("site", SiteData), ("forecast", Forecast), ("event", Event),
                        ("analysis", Analysis), ("telemetry", EventTelemetry)]:
        schema = model.model_json_schema(mode="serialization")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        (destination / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8")
