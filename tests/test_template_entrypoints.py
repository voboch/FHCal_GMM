from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_configured_generation_from_a_different_working_directory(tmp_path):
    subprocess.run([sys.executable, ROOT / "inference.py", "-cn=proton_inference", "inferencer.direction=2",
                    "inferencer.events=12", f"inferencer.output={tmp_path / 'showers.npz'}"],
                   cwd=tmp_path, check=True)
    payload = tmp_path / "showers.npz"
    with np.load(payload, allow_pickle=False) as data:
        assert data["section_energy"].shape == (12, 438)
        assert data["decoder"] == "stochastic"
        assert data["packets_per_component"] == 128
        assert data["direction"] == 2
        np.testing.assert_allclose(data["section_energy"].sum(axis=1), data["energy"])
    provenance = json.loads((tmp_path / "showers.run.json").read_text())
    assert provenance["config"]["inferencer"]["events"] == 12
    assert provenance["output_sha256"] == sha256(payload.read_bytes()).hexdigest()
    assert len(provenance["calibration_sha256"]) == 64


def test_train_can_resolve_config_and_reject_misspelled_settings(tmp_path):
    args = [sys.executable, ROOT / "train.py", "-cn=proton", "--cfg", "job"]
    resolved = subprocess.run([*args, "trainer.workers=2"], cwd=tmp_path, check=True,
                              capture_output=True, text=True)
    assert "workers: 2" in resolved.stdout
    invalid = subprocess.run([*args, "trainer.worker=2"], cwd=tmp_path, capture_output=True, text=True)
    assert invalid.returncode == 2
    assert "trainer.worker" in invalid.stderr
