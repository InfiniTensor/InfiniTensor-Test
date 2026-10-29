"""Explicitly fetch immutable source revisions; never reset an existing checkout."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from infinitensor_test.models.sources.paddle_source import REPOSITORIES, repository_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, default=Path("data/paddle-repos"))
    p.add_argument("--sources", nargs="+", choices=sorted(REPOSITORIES), default=["paddleocr", "paddleseg"])
    args = p.parse_args(argv)
    root = args.directory.expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        for source in dict.fromkeys(args.sources):
            name, revision = REPOSITORIES[source]
            target = root / name
            if target.exists():
                repository_path(source, {"repository_root": str(root)})
                print(f"Verified {name} {revision}")
                continue
            with tempfile.TemporaryDirectory(prefix=".checkout-", dir=root) as temp:
                checkout = Path(temp) / name
                def git(*args):
                    subprocess.run(["git", "-C", str(checkout), *args], check=True)
                checkout.mkdir()
                git("init")
                git("remote", "add", "origin", f"https://github.com/PaddlePaddle/{name}.git")
                git("config", "remote.origin.promisor", "true")
                git("config", "remote.origin.partialclonefilter", "blob:none")
                git("sparse-checkout", "init", "--cone")
                folders = {"paddleocr": ["ppocr", "configs"], "paddleseg": ["paddleseg", "Matting", "configs"],
                           "paddledetection": ["ppdet", "configs"]}[source]
                git("sparse-checkout", "set", *folders)
                git("fetch", "--filter=blob:none", "--depth=1", "origin", revision)
                git("checkout", "--detach", "FETCH_HEAD")
                checkout.rename(target)
            print(f"Prepared {name} {revision}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Paddle repository setup failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
