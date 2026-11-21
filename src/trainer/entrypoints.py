"""Configured entry points for the archived FHCal CPU pipeline."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "src/configs"


def config_override(config, expression):
    key, separator, value = expression.partition("=")
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            raise ValueError(f"Unknown setting: {key}")
        node = node[part]
    if not separator or parts[-1] not in node or isinstance(node[parts[-1]], dict):
        raise ValueError(f"Expected a known setting as key=value: {expression}")
    resolved = yaml.safe_load(value)
    default = node[parts[-1]]
    if default is not None and type(resolved) is not type(default):
        raise ValueError(f"{key} must be a {type(default).__name__}")
    if isinstance(resolved, (dict, list)):
        raise ValueError(f"{key} takes a scalar value")
    node[parts[-1]] = resolved


def run(action, default):
    parser = argparse.ArgumentParser(description=action.__doc__)
    parser.add_argument("-cn", "--config-name", default=default, choices=(default,))
    parser.add_argument("--cfg", choices=("job",), help="Print the resolved configuration")
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    args = parser.parse_args()
    config = yaml.safe_load((CONFIGS / f"{args.config_name}.yaml").read_text(encoding="utf-8"))
    try:
        for override in args.overrides:
            config_override(config, override)
        if args.cfg == "job":
            print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), end="")
        else:
            action(config)
    except (ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


def repository_path(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {field} to a path")
    return (ROOT / Path(value).expanduser()).resolve()


def pipeline(script, *args):
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[variable] = "1"
    subprocess.run([sys.executable, str(ROOT / "src" / script), *map(str, args)],
                   cwd=ROOT, env=environment, check=True)


def train(config):
    """Fit the six-component proton bank and calibrate its packet decoder."""
    source = repository_path(config["data"]["path"], "data.path")
    if not source.is_file():
        raise ValueError(f"Archive not found: {source}")
    output = repository_path(config["run"]["output_dir"], "run.output_dir")
    workers = config["trainer"]["workers"]
    pipeline("conditional_gmm_baseline.py", "train", source, "--output-dir", output, "--workers", workers)
    if config["trainer"]["calibrate"]:
        pipeline("evaluate_gmm_packet_decoder.py", source, output / "bank.npz",
                 "--output", output / "stochastic_decoder.json")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def generate(config):
    """Generate section responses and mixture parameters from a checkpoint bank."""
    bank = repository_path(config["model"]["bank"], "model.bank")
    if not bank.is_file():
        raise ValueError(f"Bank not found: {bank}")
    options = config["inferencer"]
    output = repository_path(options["output"], "inferencer.output")
    if output.suffix != ".npz":
        raise ValueError("inferencer.output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["--decoder", options["decoder"], "--direction", options["direction"],
            "--events", options["events"], "--seed", options["seed"], "--output", output]
    packets = options["packets_per_component"]
    if packets is not None:
        if type(packets) is not int or packets < 1:
            raise ValueError("inferencer.packets_per_component must be a positive integer")
        args.extend(("--packets-per-component", packets))
    pipeline("conditional_gmm_baseline.py", "generate", bank, *args)
    record = {"config": config, "bank_sha256": checksum(bank), "output_sha256": checksum(output)}
    if options["decoder"] == "stochastic" and packets is None:
        record["calibration_sha256"] = checksum(bank.with_name("stochastic_decoder.json"))
    output.with_suffix(".run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
