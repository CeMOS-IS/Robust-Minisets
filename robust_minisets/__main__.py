import warnings

import robust_minisets
from robust_minisets.info import INFO, DEFAULT_ROOT


def available():
    """List all available datasets."""
    print(f"Robust-Minisets v{robust_minisets.__version__} @ {robust_minisets.HOMEPAGE}")

    print("All available datasets:")
    for key in INFO.keys():
        print(
            f"\t{key:<20} | {INFO[key]['python_class']:<20} | Resolution: {INFO[key]['resolution']}."
        )


def download(root=DEFAULT_ROOT):
    """Download all available datasets."""
    for key in INFO.keys():
        print(f"Downloading {key:<15} | {INFO[key]['python_class']:<20}")
        _ = getattr(robust_minisets, INFO[key]["python_class"])(
                split="test", download=True, root=root
                )


def clean(root=DEFAULT_ROOT):
    """Delete all downloaded npz from root."""
    import os
    from glob import glob

    for path in glob(os.path.join(root, "*.npz")):
        os.remove(path)


def info(flag):
    """Print the dataset details given a subset flag."""

    import json

    print(json.dumps(INFO[flag], indent=4))


def save(flag, folder, postfix="png", root=DEFAULT_ROOT, download=False):
    """Save the dataset as standard figures, which could be used for AutoML tools, e.g., Google AutoML Vision."""

    for split in ["train", "val", "test"]:
        print(f"Saving {flag} {split}...")
        try:
            dataset = getattr(robust_minisets, INFO[flag]["python_class"])(
                split=split, download=download, root=root
            )
            dataset.save(folder, postfix)
        except:
            warnings.warn(f"An error occurred while saving {flag} {split}. Could not find {flag} {split}...", Warning)


if __name__ == "__main__":
    import fire

    fire.Fire()
