#!/usr/bin/env python3
"""
Prepare 500 High-Resolution Forestry Samples from Restor TCD (Tree Crown Delineation)
Replaces previous dataset with real global forest aerial/satellite chips (2048x2048).
Generates:
  - 100 Golden Ground-Truth chips (with polygon segmentations & bounding boxes)
  - 400 Raw Annotation Pool chips
  - golden_labels.json with fine-grained ecological taxonomy & biomes
"""

import io
import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq

def infer_eco_class_from_biome(biome: str, crown_area: float) -> str:
    """Infer fine-grained ecological tree/forest class based on biome and canopy area."""
    b = (biome or "").lower()
    
    # 1. Coastal / Wetland
    if any(k in b for k in ("zanzibar", "coastal", "mangrove", "swamp", "wetland")):
        if crown_area > 1000:
            return "Mangrove (Coastal Dense)"
        return "Mangrove"
        
    # 2. Boreal / Taiga / Conifer
    if any(k in b for k in ("taiga", "boreal", "scandinavian", "russian", "conifer", "pine", "spruce")):
        if crown_area > 2500:
            return "Dense Taiga Canopy"
        elif crown_area > 500:
            return "Scots Pine (Boreal Conifer)"
        return "Boreal Conifer"
        
    # 3. Tropical Rainforest / Humid Lowland
    if any(k in b for k in ("rain forest", "rainforest", "sumatran", "amazon", "tropical moist", "bahia")):
        if crown_area > 3000:
            return "Tropical Emergent Canopy"
        elif crown_area > 800:
            return "Tropical Broadleaf (Dense)"
        return "Tropical Forest Tree"
        
    # 4. Mixed / Deciduous / Temperate
    if any(k in b for k in ("deciduous", "mixed", "puget", "sarmatic", "pannonian", "nihonkai")):
        if crown_area > 1500:
            return "Mature Mixed Hardwood"
        return "Deciduous Broadleaf"
        
    # 5. Steppe / Savanna / Dry
    if any(k in b for k in ("steppe", "dry", "savanna", "scrub", "gangetic")):
        if crown_area < 200:
            return "Understory Shrub"
        return "Dry Forest Tree"

    # Default based on canopy crown size
    if crown_area > 2000:
        return "dense_tree"
    elif crown_area < 150:
        return "plant"
    return "ordinary_tree"


def main():
    test_p = Path("/home/komail/.cache/huggingface/hub/datasets--restor--tcd/snapshots/d97d4da0ebbb6e249ae95ac5e19656babd972eb2/data/test-00000-of-00001.parquet")
    train_p = Path("/home/komail/.cache/huggingface/hub/datasets--restor--tcd/snapshots/d97d4da0ebbb6e249ae95ac5e19656babd972eb2/data/train-00000-of-00007.parquet")

    output_base = Path("/home/komail/DataAnnotation/data/climate_mrv/samples")
    golden_dir = output_base / "golden"
    raw_dir = output_base / "raw"
    golden_labels_file = output_base / "golden_labels.json"

    print("Loading TCD Parquet tables...")
    t_test = pq.read_table(test_p)
    t_train = pq.read_table(train_p)

    candidates = []

    def collect_candidates(table, source_split):
        for i in range(len(table)):
            img_data = table["image"][i].as_py()
            if not img_data or not img_data.get("bytes"):
                continue
            coco_str = table["coco_annotations"][i].as_py()
            if not coco_str or coco_str == "[]":
                continue
            try:
                coco = json.loads(coco_str)
            except Exception:
                continue
            if len(coco) < 3:
                continue

            biome = str(table["biome_name"][i].as_py() or "Mixed Forest").strip()
            w = int(table["width"][i].as_py() or 2048)
            h = int(table["height"][i].as_py() or 2048)
            lat = float(table["lat"][i].as_py() or 0.0)
            lon = float(table["lon"][i].as_py() or 0.0)

            candidates.append({
                "split": source_split,
                "row_idx": i,
                "table": table,
                "biome": biome,
                "width": w,
                "height": h,
                "lat": lat,
                "lon": lon,
                "coco": coco,
            })

    collect_candidates(t_test, "test")
    collect_candidates(t_train, "train")

    print(f"Total valid annotated forestry candidates: {len(candidates)}")
    assert len(candidates) >= 500, f"Need at least 500 candidates, found {len(candidates)}"

    # Set deterministic random seed and sample exactly 500
    rng = random.Random(42)
    rng.shuffle(candidates)
    selected_500 = candidates[:500]

    # Split: 100 Golden, 400 Raw
    golden_selected = selected_500[:100]
    raw_selected = selected_500[100:500]

    print(f"Selected: 100 Golden ground-truth, 400 Raw annotation chips.")

    # Clean previous local directories
    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    golden_labels = {}

    # Process 100 Golden Chips
    print("Exporting 100 Golden Ground-Truth forestry chips...")
    total_golden_crowns = 0
    for idx, item in enumerate(golden_selected):
        table = item["table"]
        row_idx = item["row_idx"]
        raw_bytes = table["image"][row_idx].as_py()["bytes"]
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        chip_name = f"tcd_forestry_golden_{idx:03d}.jpg"
        chip_path = golden_dir / chip_name
        pil_img.save(chip_path, format="JPEG", quality=92)

        biome = item["biome"]
        annotations = []
        for ann in item["coco"]:
            bbox = ann.get("bbox", [0, 0, 10, 10])
            bx, by, bw, bh = [float(v) for v in bbox]
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x1, y1 = max(0.0, bx), max(0.0, by)
            x2, y2 = min(float(item["width"]), bx + bw), min(float(item["height"]), by + bh)
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue

            area = float(ann.get("area", (x2 - x1) * (y2 - y1)))
            eco_class = infer_eco_class_from_biome(biome, area)

            # Extract polygon if available
            seg = ann.get("segmentation")
            poly = None
            if seg and isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                pts = seg[0]
                if len(pts) >= 6:
                    poly = [[round(pts[j], 2), round(pts[j+1], 2)] for j in range(0, len(pts)-1, 2)]

            annotations.append({
                "hazard_class": eco_class,
                "bounding_box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area": round(area, 2),
                "polygon": poly,
            })

        total_golden_crowns += len(annotations)
        primary_class = annotations[0]["hazard_class"] if annotations else "dense_tree"
        golden_labels[chip_name] = {
            "chip_id": chip_name,
            "biome": biome,
            "lat": item["lat"],
            "lon": item["lon"],
            "class": primary_class,
            "annotations": annotations,
            "total_crowns": len(annotations),
        }

    with open(golden_labels_file, "w") as f:
        json.dump(golden_labels, f, indent=2)

    print(f"✓ Saved 100 Golden chips to {golden_dir} ({total_golden_crowns} total tree crowns labeled)")
    print(f"✓ Saved ground-truth metadata to {golden_labels_file}")

    # Process 400 Raw Chips
    print("Exporting 400 Raw Annotation Pool forestry chips...")
    for idx, item in enumerate(raw_selected):
        table = item["table"]
        row_idx = item["row_idx"]
        raw_bytes = table["image"][row_idx].as_py()["bytes"]
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        chip_name = f"tcd_forestry_raw_{idx:03d}.jpg"
        chip_path = raw_dir / chip_name
        pil_img.save(chip_path, format="JPEG", quality=92)

    print(f"✓ Saved 400 Raw chips to {raw_dir}")

    # Clean flywheel image cache so validator and miners immediately regenerate with new dataset
    flywheel_cache = Path("/home/komail/DataAnnotation/data/flywheel/image_cache")
    if flywheel_cache.exists():
        print("Clearing local flywheel image cache...")
        shutil.rmtree(flywheel_cache)
        flywheel_cache.mkdir(parents=True, exist_ok=True)
        print("✓ Local cache cleared.")

    print("\nDataset preparation completed successfully! 500 TCD forestry chips ready.")

if __name__ == "__main__":
    main()
