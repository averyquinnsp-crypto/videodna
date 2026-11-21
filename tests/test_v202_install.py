"""v2.0.2 installer + doctor regression tests.

Bug 1 (install.sh): the shim lands in ~/.local/bin, which is NOT on PATH on
some machines, and an older /usr/local/bin/videodna shadows it — after a
"successful" v2.0.1 install plain `videodna --version` still said 2.0.0. The
old PATH warning only fired when `command -v videodna` FAILED, which never
happens for upgraders. install.sh must now print a LOUD warning naming the
old binary when one exists on PATH outside ~/.local/bin.

Bug 2 (doctor): `doctor` reported `requests` MISSING although nothing in the
code imports requests (it is only an optional [ai] extra). The check is
dropped; doctor must pass clean after a fresh install.

Sandbox safety: these tests NEVER touch the real /usr/local/bin/videodna or
the real system. They copy the repo to a temp dir, point HOME at a temp dir,
put a fake old `videodna` (a tiny shell script printing a fake old version)
on a temp PATH, and run the copied install.sh there. The sandbox venv's pip
is stubbed to exit 0 (system deps are already installed on this machine and
no network install is under test) and its `videodna` entry point is stubbed
so the installer's final `doctor` step cannot fail on imports.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = sys.executable  # run under: .venv/bin/python -m pytest


def _write_stub(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestInstallShadowWarning(unittest.TestCase):
    def _sandbox_install(self, with_old_binary: bool) -> subprocess.CompletedProcess:
        """Run the copied install.sh in a sandbox; return the completed run."""
        tmp = Path(tempfile.mkdtemp(prefix="videodna-install-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)

        repo_copy = tmp / "repo"
        shutil.copytree(
            REPO_ROOT,
            repo_copy,
            ignore=shutil.ignore_patterns(
                ".venv", ".git", "__pycache__", "*.egg-info", ".pytest_cache",
            ),
        )

        fake_home = tmp / "home"
        fake_home.mkdir()
        fake_bin = tmp / "oldbin"
        fake_bin.mkdir()
        if with_old_binary:
            # Fake OLD videodna, first on PATH — like /usr/local/bin/videodna.
            _write_stub(
                fake_bin / "videodna",
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then echo "videodna 2.0.0"; exit 0; fi\n'
                'echo "fake old videodna 2.0.0"\n',
            )

        # Fresh venv in the sandbox copy (system python3 has ensurepip here).
        venv_dir = repo_copy / ".venv"
        subprocess.run(
            ["python3", "-m", "venv", str(venv_dir)],
            check=True, capture_output=True, text=True, timeout=180,
        )
        # Stub pip: no network install is under test; the venv install step
        # must succeed so the script reaches the shim + PATH warning logic.
        _write_stub(venv_dir / "bin" / "pip", "#!/bin/sh\nexit 0\n")
        # Stub the entry point so the installer's final `doctor` step runs
        # without importing the package under the system python.
        _write_stub(venv_dir / "bin" / "videodna", "#!/bin/sh\nexit 0\n")

        # PATH deliberately EXCLUDES /usr/local/bin (real videodna lives
        # there) and ~/.local/bin, so only the fake old binary can be found.
        path = f"{fake_bin}:/usr/bin:/bin" if with_old_binary else "/usr/bin:/bin"
        env = dict(os.environ, HOME=str(fake_home), PATH=path)
        env.pop("PREFIX", None)  # never take the Termux branch in tests
        return subprocess.run(
            ["bash", str(repo_copy / "install.sh")],
            capture_output=True, text=True, timeout=300,
            env=env, cwd=str(repo_copy),
        )

    def test_old_binary_on_path_warns_and_names_it(self):
        res = self._sandbox_install(with_old_binary=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        out = res.stdout + res.stderr
        self.assertIn("WARNING", out)
        self.assertIn("SHADOW", out)
        self.assertIn("oldbin/videodna", out)  # names the old binary's path
        self.assertIn("2.0.0", out)  # prints the old binary's version
        self.assertIn("~/.local/bin", out)
        self.assertIn("rm", out)  # tells the user how to fix it

    def test_no_old_binary_no_false_warning(self):
        res = self._sandbox_install(with_old_binary=False)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        out = res.stdout + res.stderr
        self.assertNotIn("WARNING", out)
        self.assertNotIn("SHADOW", out)
        # The plain "not on PATH yet" note is still fine.
        self.assertIn("not on this shell's PATH yet", out)


class TestDoctorNoRequests(unittest.TestCase):
    def test_doctor_passes_clean_no_requests_line(self):
        res = subprocess.run(
            [VENV_PYTHON, "-m", "videodna.cli", "doctor"],
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
        )
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertNotIn("requests", res.stdout)
        self.assertNotIn("MISSING", res.stdout)
        self.assertIn("numpy", res.stdout)
        self.assertIn("PIL", res.stdout)


if __name__ == "__main__":
    unittest.main()
