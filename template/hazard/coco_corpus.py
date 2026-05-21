"""Load a fixed COCO val2017 subset into :class:`ImageCorpus` (localnet E2E)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import bittensor as bt

from template.hazard.image_corpus import (
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    UnlabeledImage,
)

if TYPE_CHECKING:
    from template.hazard.image_corpus import ImageCorpusConfig


def load_coco_manifest_into_corpus(corpus: ImageCorpus, manifest_path: Path) -> None:
    """Populate ``corpus`` from ``manifest.json`` produced by prepare_coco script."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COCO manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"COCO manifest {path} has no images[] entries.")

    corpus._golden.clear()
    corpus._annotation.clear()
    corpus._benchmark.clear()
    corpus._golden_index.clear()
    corpus._all_image_index.clear()

    for row in images:
        image_id = str(row["image_id"])
        rel = str(row.get("relative_path") or row.get("file_name") or "")
        image_path = (path.parent / rel).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image missing for {image_id}: {image_path}")
        width = int(row["width"])
        height = int(row["height"])
        is_golden = bool(row.get("is_golden"))
        anns = tuple(
            GoldenAnnotation(
                hazard_class=str(a["hazard_class"]),
                bounding_box=tuple(int(v) for v in a["bounding_box"]),
                severity=str(a.get("severity") or "none"),
            )
            for a in row.get("annotations") or []
        )
        corpus._all_image_index[image_id] = image_path
        url = image_path.as_uri()
        if is_golden:
            golden = GoldenImage(
                image_id=image_id,
                image_path=image_path,
                image_url=url,
                width=width,
                height=height,
                annotations=anns,
            )
            corpus._golden.append(golden)
            corpus._golden_index[image_id] = golden
        else:
            corpus._annotation.append(
                UnlabeledImage(
                    image_id=image_id,
                    image_path=image_path,
                    image_url=url,
                    width=width,
                    height=height,
                    source_dataset="coco_val2017",
                )
            )

    bt.logging.info(
        "event=coco_manifest_loaded path=%s golden=%d pool=%d"
        % (path, len(corpus._golden), len(corpus._annotation))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
