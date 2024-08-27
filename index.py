from faiss import write_index
from PIL import Image
from tqdm import tqdm

import argparse
import clip
import faiss
import json
import numpy as np
import os
import torch


def index(image_dir_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    images = []
    image_paths = []
    img_dir_path = image_dir_path
    for animal_name in sorted(os.listdir(img_dir_path)):
        print(animal_name)
        if not os.path.isdir(os.path.join(img_dir_path, animal_name)):
            continue
        for img_file in tqdm(os.listdir(os.path.join(img_dir_path, animal_name))):
            if not img_file.endswith(".jpg"):
                continue
            image = Image.open(os.path.join(img_dir_path, animal_name, img_file)).convert("RGB")
            images.append(preprocess(image))
            image_paths.append(os.path.join(img_dir_path, animal_name, img_file))

    batch_size = 32  # Adjust batch size according to your GPU memory
    image_features = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        image_input = torch.tensor(np.stack(batch_images)).to(device)

        with torch.no_grad():
            batch_features = model.encode_image(image_input).float()
        batch_features /= batch_features.norm(dim=-1, keepdim=True)
        image_features.append(batch_features.cpu().numpy())

    image_features = np.concatenate(image_features, axis=0)

    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    write_index(index, "static/index.faiss")

    with open("static/image_paths.json", "w") as f:
        json.dump(image_paths, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_path", type=str, default="static/data/images")
    args = parser.parse_args()
    index(args.image_dir_path)
