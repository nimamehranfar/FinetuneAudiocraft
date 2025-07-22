from pathlib import Path

import torch

from audiocraft import __version__


def main(ckpt_path, out_file):
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_file)

    print(f"Loading checkpoint from {ckpt_path}")
    print(f"Will save exported state_dict to {out_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Top-level keys:", ckpt.keys())
    print("loaded ckpt, now saving...")
    new_pkg = {
        "model": {".".join(k.split('.')[1:]): v for k, v in ckpt["state_dict"].items() if k.startswith("model")},
        "xp.cfg": ckpt["hyper_parameters"],
        "version": __version__,
        "exported": True
    }
    out_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_path)

main(f"C:/tmp/020819d8/checkpoint_35.th","NewModel")

