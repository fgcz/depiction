"""Downloads the public imzML acquisitions listed in `datasets.py` into a gitignored cache.

    uv run python -m tests.real_data.fetch --list
    uv run python -m tests.real_data.fetch --all           # both, 1.24 GB
    uv run python -m tests.real_data.fetch mouse_kidney    # 59 MB
    uv run python -m tests.real_data.fetch --all --verify  # re-hash what is already there

Stdlib only, so it runs before anything is installed. `DEPICTION_TEST_DATA_DIR` moves the
cache; see `datasets.cache_dir`.

Nothing is ever written under its final name until its SHA-256 matches the manifest: the
download goes to `<filename>.part` and is renamed afterwards. A partial or corrupted file
therefore cannot be mistaken for a complete one, which matters because the tests decide
what is present by file size and would otherwise trust a truncated 1.18 GB `.ibd`.

There is deliberately no record of what was downloaded when. `--verify` re-derives the
answer from the files themselves, which is also correct for a copy someone put in the cache
by hand; a stamp file would only be able to speak for downloads this script performed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from tests.real_data.datasets import DATASETS, DATASETS_BY_NAME, PublicDataset, RemoteFile, cache_dir

#: PRIDE and Zenodo both serve fine without one, but an anonymous urllib UA is the kind of
#: thing that gets rate-limited first when a host tightens up.
_USER_AGENT = "depiction-test-data-fetcher (https://github.com/fgcz/depiction)"

_CHUNK_SIZE = 1024 * 1024


def _human(n_bytes: int) -> str:
    # Decimal, not binary: the sizes in `public-test-data.md` are decimal, and a CLI that
    # calls the same file 1.10 GB where the document calls it 1.18 GB invites a bug report.
    return f"{n_bytes / 1000**2:.1f} MB" if n_bytes < 1000**3 else f"{n_bytes / 1000**3:.2f} GB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _download(remote: RemoteFile, target: Path) -> str:
    """Streams `remote` to `<target>.part`, returning the SHA-256 of what arrived."""
    partial = target.with_name(target.name + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    show_progress = sys.stderr.isatty()

    request = urllib.request.Request(remote.url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != remote.size_bytes:
            # The deposit changed, or a proxy is serving something else. Say so before
            # spending the bandwidth rather than after failing the hash check.
            raise ValueError(
                f"{remote.filename}: the server offers {declared} bytes, the manifest expects "
                f"{remote.size_bytes}. Refusing to download; check the deposit."
            )
        received = 0
        with partial.open("wb") as file:
            while chunk := response.read(_CHUNK_SIZE):
                file.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if show_progress:
                    percent = 100 * received / remote.size_bytes
                    print(f"\r  {remote.filename}: {percent:5.1f}%  {_human(received)}", end="", file=sys.stderr)
    if show_progress:
        print(file=sys.stderr)
    return digest.hexdigest()


def fetch_file(remote: RemoteFile, verify: bool) -> bool:
    """Ensures `remote` is present and correct. Returns False when it is not."""
    target = remote.local_path
    if target.is_file() and target.stat().st_size == remote.size_bytes:
        if not verify:
            print(f"  {remote.filename}: present ({_human(remote.size_bytes)})")
            return True
        actual = _sha256(target)
        if actual == remote.sha256:
            print(f"  {remote.filename}: verified")
            return True
        print(f"  {remote.filename}: SHA-256 MISMATCH\n    expected {remote.sha256}\n    actual   {actual}")
        return False

    print(f"  {remote.filename}: downloading {_human(remote.size_bytes)}")
    partial = target.with_name(target.name + ".part")
    actual = _download(remote, target)
    if actual != remote.sha256:
        # Left in place under `.part`, where nothing will mistake it for the real file, so
        # that a reproducible mismatch can be looked at rather than silently re-downloaded.
        print(f"  {remote.filename}: SHA-256 MISMATCH, left at {partial}")
        print(f"    expected {remote.sha256}\n    actual   {actual}")
        return False
    partial.replace(target)
    print(f"  {remote.filename}: ok")
    return True


def fetch_dataset(dataset: PublicDataset, verify: bool) -> bool:
    print(f"{dataset.name} -- {dataset.source}, {dataset.licence}, {_human(dataset.total_bytes)}")
    # Not short-circuited: a report covering both files is more useful than stopping at the
    # first problem, and the second file may well be fine.
    results = [fetch_file(remote, verify=verify) for remote in dataset.files]
    return all(results)


def list_datasets() -> None:
    print(f"cache: {cache_dir()}\n")
    for dataset in DATASETS:
        status = "present" if dataset.is_available else "missing"
        print(f"{dataset.name:<16} {_human(dataset.total_bytes):>8}  {dataset.licence:<5} {status:<8} {dataset.source}")
        print(f"{'':<16} {dataset.description}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("names", nargs="*", help=f"datasets to fetch: {', '.join(DATASETS_BY_NAME)}")
    parser.add_argument("--all", action="store_true", help="fetch every dataset")
    parser.add_argument("--list", action="store_true", help="show what is defined and what is present")
    parser.add_argument("--verify", action="store_true", help="re-hash files that are already present")
    args = parser.parse_args(argv)

    if args.list:
        list_datasets()
        return 0

    unknown = [name for name in args.names if name not in DATASETS_BY_NAME]
    if unknown:
        parser.error(f"unknown dataset(s) {', '.join(unknown)}; known: {', '.join(DATASETS_BY_NAME)}")

    selected = DATASETS if args.all else tuple(DATASETS_BY_NAME[name] for name in args.names)
    if not selected:
        parser.error("name a dataset, or pass --all or --list")

    print(f"cache: {cache_dir()}")
    ok = [fetch_dataset(dataset, verify=args.verify) for dataset in selected]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
