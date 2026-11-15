"""Bundle configuration templates in wheel builds."""

from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPackage(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        for source in (root / "configs").rglob("*.json"):
            if source.name.endswith(".private.json"):
                continue
            target = (
                Path(self.build_lib)
                / "softcas/_configs"
                / source.relative_to(root / "configs")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())


setup(cmdclass={"build_py": BuildPackage})
