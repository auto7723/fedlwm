from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .audit import audit_release
from .config import ExperimentConfig
from .data import FeatureBundle
from .runner import run_experiment


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="fedlwm")
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="validate an anonymized feature bundle")
    validate.add_argument("--config", required=True)
    validate.add_argument("--data-root", required=True)
    train = commands.add_parser("train", help="run the federated protocol")
    train.add_argument("--config", required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--evaluation-suffix", choices=["dev", "test"], default="test")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int)
    summarize = commands.add_parser("summarize", help="summarize completed seed directories")
    summarize.add_argument("--root", required=True)
    summarize.add_argument("--output")
    audit = commands.add_parser("audit-release", help="scan the release tree for identity/path leaks")
    audit.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit-release":
        findings = audit_release(args.root)
        if findings:
            raise SystemExit("\n".join(findings))
        print("release audit passed")
        return
    if args.command == "summarize":
        root = Path(args.root)
        files = sorted(root.glob("**/seed_*/result.json"))
        if not files:
            raise SystemExit("no seed_*/result.json files found")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        summary = {"scenarios": {}}
        for scenario in sorted({result.get("scenario", "unspecified") for result in results}):
            selected = [result for result in results if result.get("scenario", "unspecified") == scenario]
            scenario_report = {
                "seeds": [result["seed"] for result in selected], "views": {},
            }
            for view in ("E-P", "E-G1", "E-G1-prime", "E-G2"):
                scenario_report["views"][view] = {}
                for metric in ("weighted_f1", "macro_f1", "uar", "nll", "class_coverage"):
                    values = [float(result["online"][view][metric]) for result in selected]
                    mean = sum(values) / len(values)
                    scenario_report["views"][view][metric] = {
                        "mean": mean,
                        "std": (
                            sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
                        ) ** 0.5,
                        "std_ddof": 1,
                        "values": values,
                    }
            summary["scenarios"][scenario] = scenario_report
        payload = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        print(payload)
        return
    config = ExperimentConfig.load(args.config)
    if getattr(args, "seed", None) is not None:
        config = dataclasses.replace(
            config, federated=dataclasses.replace(config.federated, seed=args.seed),
        )
    if args.command == "validate-data":
        bundle = FeatureBundle(args.data_root, config.model.modality_dims, config.model.class_count)
        print(json.dumps({
            "samples": len(bundle.records), "clients": len(bundle.clients),
            "data_contract_hash": bundle.digest,
        }, indent=2))
        return
    result = run_experiment(config, args.data_root, args.output, args.evaluation_suffix, args.device)
    print(json.dumps(result["online"], indent=2))


if __name__ == "__main__":
    main()
