"""Regression tests: DampedOscillationProblem training data must be reproducible.

Invariant I5 (see evidence INVARIANTS_ASSUMPTIONS.md): fresh-process
construction yields byte-identical x_noisy / train_data.
"""

import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONSTRUCT_SNIPPET = """
import hashlib, pickle, sys
sys.path.insert(0, r"{repo_root}")
from problems.damped_oscillation import DampedOscillationProblem
p = DampedOscillationProblem()
print(hashlib.sha256(pickle.dumps(p.x_noisy)).hexdigest())
"""


def _fresh_process_data_hash():
    proc = subprocess.run(
        [sys.executable, "-c", CONSTRUCT_SNIPPET.format(repo_root=str(REPO_ROOT))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return proc.stdout.strip().splitlines()[-1]


def test_train_data_identical_across_fresh_processes():
    hash_one = _fresh_process_data_hash()
    hash_two = _fresh_process_data_hash()
    assert hash_one == hash_two, (
        "DampedOscillationProblem training data is not reproducible across "
        f"fresh processes: {hash_one} != {hash_two}"
    )


def test_train_data_shape_invariant():
    from problems.damped_oscillation import DampedOscillationProblem

    problem = DampedOscillationProblem()
    assert len(problem.train_data) == 100
    (t_value,), target = problem.train_data[0]
    assert isinstance(t_value, float)
    assert isinstance(target, float)
