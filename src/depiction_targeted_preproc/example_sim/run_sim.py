from pathlib import Path


def main() -> None:
    dir_work = Path(__file__).parent / "data-work"
    dir_output = Path(__file__).parent / "data-output"
    dir_work.mkdir(exist_ok=True, parents=True)
    dir_output.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    main()
