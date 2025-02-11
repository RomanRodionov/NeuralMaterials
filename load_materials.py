import torchvision.transforms.functional as TF
from datasets import load_dataset
import argparse
from pathlib import Path
import json

resolution = (1024, 1024)
columns = ["basecolor", "diffuse", "displacement", "height", "metallic", "normal", "opacity", "roughness", "specular"]

#https://huggingface.co/datasets/gvecchio/MatSynth

def process_data(data):
    for col in columns:
        data[col] = TF.resize(data[col], resolution)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset.")
    parser.add_argument("--base_dir", required=True, help="Directory to save the downloaded files.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    ds = load_dataset(
        "gvecchio/MatSynth",
        streaming=True,
    )

    ds = ds.filter(lambda x: x["metadata"]["license"] == "CC0")
    ds = ds.filter(lambda x: x["metadata"]["source"] != "deschaintre_2020")
    ds = ds.map(process_data, batched=False)

    for split in ds:
        for item in ds[split]:
            name = item["name"]
            dest_dir = base_dir / split / item["metadata"]["category"] / name
            dest_dir.mkdir(exist_ok=True, parents=True)

            with open(dest_dir / "metadata.json", "w") as f:
                item["metadata"]["physical_size"] = str(
                    item["metadata"]["physical_size"]
                )
                json.dump(item["metadata"], f, indent=4)

            item["basecolor"].save(dest_dir / "basecolor.png")
            item["diffuse"].save(dest_dir / "diffuse.png")
            item["displacement"].save(dest_dir / "displacement.png")
            item["specular"].save(dest_dir / "specular.png")
            item["height"].save(dest_dir / "height.png")
            item["metallic"].save(dest_dir / "metallic.png")
            item["normal"].save(dest_dir / "normal.png")
            item["opacity"].save(dest_dir / "opacity.png")
            item["roughness"].save(dest_dir / "roughness.png")
            if item["blend_mask"] is not None:
                item["blend_mask"].save(dest_dir / "blend_mask.png")
