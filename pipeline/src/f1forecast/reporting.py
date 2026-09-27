"""Publish only actual evaluation evidence, with its sample counts and limits."""

import json
from pathlib import Path

from .contracts import Evaluation, SiteData
from .publication import publish_site
from .registry import metric_contract


def publish_evaluation(report_paths: list[str], site_dir: str | Path) -> Evaluation:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    seasons = []
    comparisons = []
    for report in reports:
        for period, year in [("locked_2025", 2025), ("separate_2026", 2026)]:
            # Report each target independently, never imply Q and R are independent races.
            for model_name in ("baseline", "learned"):
                summary = report["periods"][period][model_name]
                if not summary.get("evaluated_events", 0):
                    continue
                comparisons.append(
                    {
                        "name": f"{year} {model_name}",
                        "target": report["target"],
                        "event_count": summary["evaluated_events"],
                        "metrics": metric_contract(summary),
                        "promoted": model_name == "learned"
                        and report.get("promotion_passed", False),
                    }
                )
            chosen = report["periods"][period][
                "learned" if report.get("promotion_passed") else "baseline"
            ]
            if chosen.get("evaluated_events", 0):
                seasons.append(
                    {
                        "season": year,
                        "target": report["target"],
                        "event_count": chosen["evaluated_events"],
                        "kind": "reconstructed",
                        "metrics": metric_contract(chosen),
                        "confidence_intervals": [
                            {
                                "metric": key.removesuffix("_95pct_ci"),
                                "low": interval[0],
                                "high": interval[1],
                            }
                            for key, interval in chosen.items()
                            if key.endswith("_95pct_ci")
                        ],
                        "calibration": [
                            {
                                "outcome": b["outcome"].replace("top_ten", "top10"),
                                "predicted": b["mean_probability"],
                                "observed": b["observed_rate"],
                                "count": b["count"],
                            }
                            for b in chosen.get("calibration_bins", [])
                        ],
                    }
                )
    if not seasons:
        raise ValueError("no evaluated held-out events; cannot publish accuracy claims")
    evidence = Evaluation(
        status="evaluated",
        summary="Chronological historical reconstruction; not prospective performance.",
        seasons=seasons,
        comparison=comparisons,
        limitations=[
            "Archives can include later corrections; these results are historical reconstructions.",
            "2025 is held out; 2026 is reported separately. Counts are evaluated target sessions.",
            "Incomplete result orders, DNS and disqualifications are excluded from full-order metrics; see the report.",
            "Metric confidence intervals resample events; driver rows are not independent races.",
            "A rank-utility model does not separately model retirement risk or strategy.",
        ],
    )
    root = Path(site_dir)
    site = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
    site.evaluation = evidence
    publish_site(site, root)
    # Complete reproducibility evidence accompanies compact presentation metrics.
    (root / "evaluation-reports").mkdir(exist_ok=True)
    for report in reports:
        (root / "evaluation-reports" / f"{report['target']}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return evidence
