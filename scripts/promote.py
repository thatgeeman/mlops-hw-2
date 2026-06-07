"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config import get_settings

_settings = get_settings()
mlflow.set_tracking_uri(_settings.mlflow_tracking_uri)

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

client = MlflowClient()


def _resolve_version(model_name: str, config_id: str):
    """Find the registered version with the given config_id tag.

    Returns the ModelVersion to use, handling zero/multiple matches.
    Exits via sys.exit(1) on zero matches.
    """
    versions = client.search_model_versions(
        f"name = '{model_name}' AND tags.config_id = '{config_id}'"
    )
    if not versions:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    if len(versions) > 1:
        int_versions = sorted(int(v.version) for v in versions)
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {int_versions}); using latest ({int_versions[-1]})"
        )
        versions = [v for v in versions if int(v.version) == int_versions[-1]]
    return versions[0]


def _current_config_id(model_name: str, alias: str) -> str:
    """Return the config_id of the version currently at alias, or '' if unset."""
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return mv.tags.get("config_id", "")
    except Exception:
        return ""


def _append_log(event: dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    mv = _resolve_version(args.name, args.config_id)
    current = _current_config_id(args.name, args.alias)
    client.set_registered_model_alias(args.name, args.alias, mv.version)
    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current,
        "to": args.config_id,
        "op": "set",
    })
    from_label = f"(unset)" if not current else current
    print(f"{args.alias}: {from_label} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    try:
        mv = client.get_model_version_by_alias(args.name, args.alias)
    except Exception:
        print(f"error: alias '{args.alias}' is not set")
        sys.exit(1)

    config_id = mv.tags.get("config_id", "(unknown)")
    model = mv.tags.get("model", "(unknown)")
    guardrail_type = mv.tags.get("guardrail_type", "(unknown)")

    metrics = {}
    if mv.run_id:
        run = client.get_run(mv.run_id)
        metrics = run.data.metrics

    print(f"travel-assistant @ {args.alias}")
    print(f"  config_id: {config_id}")
    print(f"  model: {model}")
    print(f"  guardrail_type: {guardrail_type}")
    if "accuracy_overall" in metrics:
        print(f"  accuracy_overall: {metrics['accuracy_overall']:.2f}")
    if "verdict_rate_leaked" in metrics:
        print(f"  verdict_rate_leaked: {metrics['verdict_rate_leaked']:.2f}")
    if "total_cost_usd" in metrics:
        print(f"  total_cost_usd: ${metrics['total_cost_usd']:.2f}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    try:
        rm = client.get_registered_model(args.name)
        aliases = rm.aliases
    except Exception:
        aliases = {}

    if not aliases:
        print("no aliases set")
        return

    for alias, version_str in aliases.items():
        try:
            mv = client.get_model_version(args.name, version_str)
            config_id = mv.tags.get("config_id", f"version {version_str}")
        except Exception:
            config_id = f"version {version_str}"
        print(f"{alias} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    # Check if alias is currently set
    try:
        current_mv = client.get_model_version_by_alias(args.name, args.alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except Exception:
        print("nothing to roll back")
        return

    # Read log backward for the most recent entry for this alias
    if not LOG_FILE.exists():
        print(f"no promotion history for alias {args.alias}")
        return

    lines = [l.strip() for l in LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    entry = None
    for line in reversed(lines):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("alias") == args.alias:
            entry = ev
            break

    if entry is None:
        print(f"no promotion history for alias {args.alias}")
        return

    if entry["op"] == "rollback":
        print(f"error: {args.alias} was just rolled back; no further history to walk back to")
        return

    if not entry.get("from"):
        print(f"error: {args.alias} has no previous target (first promotion ever)")
        return

    target_config_id = entry["from"]
    mv = _resolve_version(args.name, target_config_id)
    client.set_registered_model_alias(args.name, args.alias, mv.version)
    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current_config_id,
        "to": target_config_id,
        "op": "rollback",
    })
    print(f"{args.alias}: {current_config_id} → {target_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
