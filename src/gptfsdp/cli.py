"""``gptfsdp`` command line: ``train``, ``prepare``, ``heldout``, ``arc-easy``, ``download-evals``,
``hellaswag``, ``bench-layernorm`` and ``report``.

``train`` reads an optional JSON config and applies ``--set key=value`` overrides (typed from the
dataclass defaults), so the same command runs on a laptop and under ``torchrun``::

    torchrun --nproc_per_node 8 -m gptfsdp.cli train --config configs/gpt2_124m.json \\
        --set parallel=fsdp run_name=fsdp_1x8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any


def _coerce(name: str, raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name}: expected a boolean, got {raw!r}")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def build_config(config_path: str | None, overrides: list[str]) -> Any:
    from gptfsdp.train import TrainConfig

    values: dict[str, Any] = {}
    if config_path:
        values.update(json.loads(Path(config_path).read_text()))
    defaults = {f.name: f.default for f in dataclasses.fields(TrainConfig)}
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or key not in defaults:
            raise ValueError(f"--set expects key=value with a TrainConfig field, got {item!r}")
        values[key] = _coerce(key, raw, defaults[key])
    return TrainConfig(**values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gptfsdp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="pretrain (single process, or under torchrun)")
    p_train.add_argument("--config", help="JSON file with TrainConfig fields")
    p_train.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

    p_prep = sub.add_parser("prepare", help="download + tokenize FineWeb-Edu into uint16 shards")
    p_prep.add_argument("--out", required=True)
    p_prep.add_argument("--shard-tokens", type=int, default=100_000_000)
    p_prep.add_argument("--workers", type=int, default=2)
    p_prep.add_argument("--subset", default="sample-10BT")
    p_prep.add_argument("--max-docs", type=int, default=0, help="0 = all (for smoke tests)")

    p_hs = sub.add_parser("hellaswag", help="HellaSwag validation accuracy of a checkpoint")
    p_hs.add_argument("--run", required=True, help="run directory (uses its newest checkpoint)")
    p_hs.add_argument("--data", required=True, help="hellaswag_val.jsonl")
    p_hs.add_argument("--limit", type=int, default=0)

    p_ho = sub.add_parser("heldout", help="loss on the validation shard (same tokens for all)")
    src = p_ho.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory (uses its newest checkpoint)")
    src.add_argument("--gpt2", help="OpenAI GPT-2 124M model.safetensors (file or directory)")
    p_ho.add_argument("--data", required=True, help="shard directory (uses its val shard)")
    p_ho.add_argument("--tokens", type=int, default=10_485_760)
    p_ho.add_argument("--out", default="baselines/openai_gpt2_heldout.json", help="for --gpt2")

    p_arc = sub.add_parser("arc-easy", help="ARC-Easy test accuracy (same harness for all)")
    arc_src = p_arc.add_mutually_exclusive_group(required=True)
    arc_src.add_argument("--run", help="run directory (uses its newest checkpoint)")
    arc_src.add_argument("--gpt2", help="OpenAI GPT-2 124M model.safetensors (file or directory)")
    p_arc.add_argument("--data", required=True, help="ARC-Easy test parquet")
    p_arc.add_argument("--out", default="baselines/openai_gpt2_arc_easy.json", help="for --gpt2")

    p_dl = sub.add_parser("download-evals", help="GPT-2 124M weights + ARC-Easy test (pinned)")
    p_dl.add_argument("--out", required=True, help="directory for the downloads")
    p_dl.add_argument("--manifest", default="baselines/downloads.json")

    p_bench = sub.add_parser("bench-layernorm", help="Triton vs eager vs compile LayerNorm")
    p_bench.add_argument("--out", required=True)
    p_bench.add_argument("--label", default="", help="kernel variant, recorded in the output")

    p_rep = sub.add_parser("report", help="RESULTS.md tables from run logs")
    p_rep.add_argument("--runs", default="runs")
    p_rep.add_argument("--out", default="RESULTS.md")

    args = parser.parse_args(argv)
    if args.command == "train":
        from gptfsdp.train import train

        summary = train(build_config(args.config, args.set))
    elif args.command == "prepare":
        from gptfsdp.prepare import prepare_fineweb

        summary = prepare_fineweb(
            args.out, subset=args.subset, shard_tokens=args.shard_tokens, workers=args.workers,
            max_docs=args.max_docs,
        )  # fmt: skip
    elif args.command == "hellaswag":
        from gptfsdp.hellaswag import evaluate_run

        summary = evaluate_run(args.run, args.data, limit=args.limit)
    elif args.command == "heldout":
        from gptfsdp.evaluate import evaluate_checkpoint, evaluate_openai_gpt2

        if args.run:
            summary = evaluate_checkpoint(args.run, args.data, args.tokens)
        else:
            summary = evaluate_openai_gpt2(args.gpt2, args.data, args.tokens, args.out)
    elif args.command == "arc-easy":
        from gptfsdp.arc import evaluate_run_or_gpt2

        summary = evaluate_run_or_gpt2(args.data, run=args.run, gpt2=args.gpt2, out=args.out)
    elif args.command == "download-evals":
        from gptfsdp import arc
        from gptfsdp.evaluate import download_gpt2

        summary = {
            "gpt2": download_gpt2(Path(args.out) / "gpt2"),
            "arc_easy_test": arc.download(Path(args.out) / "ai2_arc"),
        }
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    elif args.command == "bench-layernorm":
        from gptfsdp.bench import bench_layernorm

        summary = bench_layernorm(args.out, label=args.label)
    else:
        from gptfsdp.report import write_results

        summary = write_results(args.runs, args.out)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
