import argparse
import tarfile
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with tarfile.open(args.archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            if not member.name.lower().endswith(".png"):
                continue

            relative = Path(member.name)

            # Safety check against malicious/invalid archive paths.
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive path: {member.name}")

            destination = args.output_dir / relative

            # Makes the operation resumable.
            if destination.exists():
                skipped += 1
                continue

            source = tar.extractfile(member)
            if source is None:
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source) as image:
                # Keep files grayscale to minimize disk use.
                # The existing Dataset loader can convert to 3 channels.
                image = image.convert("L")
                image = image.resize(
                    (args.size, args.size),
                    Image.Resampling.BILINEAR,
                )

                image.save(
                    destination,
                    format="PNG",
                    compress_level=6,
                )

            processed += 1

            if processed % 1000 == 0:
                print(f"Processed {processed} images")

    print(
        f"Complete: processed={processed}, "
        f"already_existing={skipped}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
