"""Cut the object out of an image with rembg, writing an RGBA PNG.

Runs as a standalone script inside vendor/trellis2mlx/.venv (which has rembg
and onnxruntime) — it must not import `pipeline`. Called by
Trellis2Stage._cut_out_background; prints one JSON line on success.

    python remove_background.py IN.png OUT.png --model birefnet-general

Why this exists: the official TRELLIS.2 pipeline cuts out the object with
BiRefNet, but the vendored MLX port calls rembg with its default model,
u2net (2020). On this project's own concept images the two masks differ on
only 1-2% of the object, but exactly where the 3D shape is decided: u2net
kept white background between Godzilla's dorsal spikes (which becomes
webbing) and clipped toes, fingers and base edges. generate.py skips its own
background removal when the input already has transparency, so handing it
this RGBA cut-out is enough — no vendored code changes.

The first run of a model downloads its weights into ~/.rembg/models
(birefnet-general: 973MB, from rembg's GitHub releases).
"""

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default="birefnet-general")
    args = parser.parse_args()

    from rembg import new_session, remove

    t0 = time.time()
    session = new_session(args.model)
    t1 = time.time()
    cutout = remove(Image.open(args.input).convert("RGB"), session=session)
    t2 = time.time()

    alpha = np.asarray(cutout)[:, :, 3]
    # generate.py crops to pixels with alpha > 0.8; with none it would fall
    # back to the uncut image, so treat that as a failure here instead.
    foreground = float(np.mean(alpha > 0.8 * 255))
    if foreground == 0.0:
        print("no foreground found", file=sys.stderr)
        return 2
    cutout.save(args.output)
    print(
        json.dumps(
            {
                "model": args.model,
                "load_s": round(t1 - t0, 2),
                "run_s": round(t2 - t1, 2),
                "foreground_fraction": round(foreground, 4),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
