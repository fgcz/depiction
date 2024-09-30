import hashlib
import shutil
import subprocess
import timeit
from pathlib import Path


def checksum_native(file: Path) -> str:
    binary_path = shutil.which("sha1sum")
    result = subprocess.run(
        [binary_path, str(file)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.split()[0].lower()


def checksum_naive(file: Path) -> str:
    return hashlib.sha1(file.read_bytes()).hexdigest()


def checksum_chunked(file: Path, chunksize=4096) -> str:
    hasher = hashlib.sha1()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(chunksize), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    # create test file of size 200 MiB
    file = Path("../../msi/examples/testfile.png")
    with file.open("wb") as f:
        f.seek(200 * 1024 * 1024 - 1)
        f.write(b"\0")

    # sanity check
    print("Performing sanity check")
    assert checksum_naive(file) == checksum_chunked(file) == checksum_native(file)
    print("Sanity check passed")

    # benchmark
    print("Naive:", timeit.timeit(lambda: checksum_naive(file), number=10))
    print("Chunked 4096:", timeit.timeit(lambda: checksum_chunked(file), number=10))
    print("Chunked 8192:", timeit.timeit(lambda: checksum_chunked(file, chunksize=8192), number=10))
    print("Chunked 16384:", timeit.timeit(lambda: checksum_chunked(file, chunksize=16384), number=10))
    print("Chunked 30000:", timeit.timeit(lambda: checksum_chunked(file, chunksize=30000), number=10))
    print("Chunked 50000:", timeit.timeit(lambda: checksum_chunked(file, chunksize=50000), number=10))
    print("Native:", timeit.timeit(lambda: checksum_native(file), number=10))


if __name__ == "__main__":
    main()
