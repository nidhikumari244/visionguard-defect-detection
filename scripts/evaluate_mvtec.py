import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm import tqdm


DEFAULT_CATEGORIES = ["bottle", "cable", "carpet", "wood", "screw"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build VisionGuard memory banks and evaluate on MVTec AD."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path containing MVTec category folders, e.g. /content/mvtec_ad",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for memory banks and result JSON files.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="MVTec categories to evaluate.",
    )
    parser.add_argument("--subsample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-methods",
        nargs="+",
        default=["max", "top1", "top5"],
        choices=["max", "top1", "top5", "p95mean"],
    )
    parser.add_argument(
        "--threshold-percentiles",
        nargs="+",
        type=float,
        default=[80, 85, 90, 95],
    )
    return parser.parse_args()


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_backbone(device):
    backbone = timm.create_model(
        "wide_resnet50_2",
        pretrained=True,
        features_only=True,
        out_indices=(1, 2),
    )
    backbone.eval().to(device)
    return backbone


def extract_patch_features(image, backbone, transform, device):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(tensor)

    fine, coarse = features[0], features[1]
    coarse_up = torch.nn.functional.interpolate(
        coarse,
        size=fine.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    combined = torch.cat([fine, coarse_up], dim=1)
    _, channels, height, width = combined.shape
    patches = combined.permute(0, 2, 3, 1).reshape(-1, channels)
    return patches.cpu().numpy().astype(np.float32), (height, width)


def image_files(directory):
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        path for path in Path(directory).iterdir() if path.suffix.lower() in extensions
    )


def score_distances(distances, method):
    flat = distances.reshape(-1)
    if method == "max":
        return float(flat.max())
    if method == "top1":
        k = max(1, int(len(flat) * 0.01))
        return float(np.mean(np.sort(flat)[-k:]))
    if method == "top5":
        k = max(1, int(len(flat) * 0.05))
        return float(np.mean(np.sort(flat)[-k:]))
    if method == "p95mean":
        threshold = np.percentile(flat, 95)
        return float(np.mean(flat[flat >= threshold]))
    raise ValueError(f"Unknown score method: {method}")


def build_memory_bank(category_root, backbone, transform, device, subsample_size, seed):
    train_good_dir = category_root / "train" / "good"
    train_images = image_files(train_good_dir)
    if not train_images:
        raise FileNotFoundError(f"No training images found in {train_good_dir}")

    patches_by_image = []
    for path in tqdm(train_images, desc=f"{category_root.name} train features"):
        image = Image.open(path).convert("RGB")
        patches, _ = extract_patch_features(image, backbone, transform, device)
        patches_by_image.append(patches)

    memory_bank = np.concatenate(patches_by_image, axis=0)
    mean = memory_bank.mean(axis=0)
    std = memory_bank.std(axis=0) + 1e-8

    rng = np.random.default_rng(seed)
    count = min(subsample_size, memory_bank.shape[0])
    indices = rng.choice(memory_bank.shape[0], count, replace=False)
    memory_bank_sub = ((memory_bank[indices] - mean) / std).astype(np.float32)
    return memory_bank_sub, mean, std, len(train_images)


def patch_distances(image, nn_model, mean, std, backbone, transform, device):
    patches, shape = extract_patch_features(image, backbone, transform, device)
    patches_norm = (patches - mean) / std
    distances, _ = nn_model.kneighbors(patches_norm)
    return distances, shape


def load_ground_truth_mask(gt_dir, defect_type, image_name):
    gt_name = f"{Path(image_name).stem}_mask.png"
    gt_path = gt_dir / defect_type / gt_name
    if not gt_path.exists():
        return np.ones((224, 224), dtype=np.uint8)

    mask = Image.open(gt_path).convert("L").resize((224, 224))
    return (np.array(mask) > 127).astype(np.uint8)


def evaluate_category(
    category,
    data_root,
    output_dir,
    backbone,
    transform,
    device,
    args,
):
    category_root = data_root / category
    test_dir = category_root / "test"
    gt_dir = category_root / "ground_truth"
    if not category_root.exists():
        raise FileNotFoundError(f"Missing category folder: {category_root}")

    memory_bank, mean, std, n_train = build_memory_bank(
        category_root,
        backbone,
        transform,
        device,
        args.subsample_size,
        args.seed,
    )

    nn_model = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(
        memory_bank
    )

    method_scores = {method: [] for method in args.score_methods}
    labels = []
    all_pixel_scores = []
    all_pixel_labels = []

    defect_types = sorted(path.name for path in test_dir.iterdir() if path.is_dir())
    for defect_type in defect_types:
        label = 0 if defect_type == "good" else 1
        for image_path in tqdm(
            image_files(test_dir / defect_type),
            desc=f"{category} test/{defect_type}",
        ):
            image = Image.open(image_path).convert("RGB")
            distances, (height, width) = patch_distances(
                image,
                nn_model,
                mean,
                std,
                backbone,
                transform,
                device,
            )

            for method in args.score_methods:
                method_scores[method].append(score_distances(distances, method))
            labels.append(label)

            anomaly_map = cv2.resize(distances.reshape(height, width), (224, 224))
            if defect_type == "good":
                gt_mask = np.zeros((224, 224), dtype=np.uint8)
            else:
                gt_mask = load_ground_truth_mask(gt_dir, defect_type, image_path.name)

            all_pixel_scores.append(anomaly_map.flatten())
            all_pixel_labels.append(gt_mask.flatten())

    labels = np.array(labels)
    pixel_auroc = roc_auc_score(
        np.concatenate(all_pixel_labels),
        np.concatenate(all_pixel_scores),
    )

    train_method_scores = {method: [] for method in args.score_methods}
    for image_path in tqdm(
        image_files(category_root / "train" / "good"),
        desc=f"{category} train scores",
    ):
        image = Image.open(image_path).convert("RGB")
        distances, _ = patch_distances(
            image,
            nn_model,
            mean,
            std,
            backbone,
            transform,
            device,
        )
        for method in args.score_methods:
            train_method_scores[method].append(score_distances(distances, method))

    method_results = {}
    for method in args.score_methods:
        scores = np.array(method_scores[method])
        image_auroc = roc_auc_score(labels, scores)

        candidates = []
        for percentile in args.threshold_percentiles:
            threshold = float(np.percentile(train_method_scores[method], percentile))
            preds = (scores > threshold).astype(int)
            candidates.append(
                {
                    "percentile": percentile,
                    "threshold": threshold,
                    "precision": float(precision_score(labels, preds, zero_division=0)),
                    "recall": float(recall_score(labels, preds, zero_division=0)),
                    "f1": float(f1_score(labels, preds, zero_division=0)),
                }
            )

        best = max(candidates, key=lambda item: item["f1"])
        method_results[method] = {
            "image_auroc": float(image_auroc),
            "best_threshold": best,
            "threshold_candidates": candidates,
        }

    category_results = {
        "category": category,
        "n_train": n_train,
        "n_test": int(len(labels)),
        "pixel_auroc": float(pixel_auroc),
        "methods": method_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{category}_mb_sub_norm.npy", memory_bank)
    np.save(output_dir / f"{category}_mean.npy", mean)
    np.save(output_dir / f"{category}_std.npy", std)
    (output_dir / f"{category}_results.json").write_text(
        json.dumps(category_results, indent=2),
        encoding="utf-8",
    )
    return category_results


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    backbone = load_backbone(device)
    transform = build_transform()

    all_results = {}
    for category in args.categories:
        print(f"\n=== Evaluating {category.upper()} ===")
        all_results[category] = evaluate_category(
            category,
            data_root,
            output_dir,
            backbone,
            transform,
            device,
            args,
        )
        print(json.dumps(all_results[category], indent=2))

    (output_dir / "summary_results.json").write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved summary to {output_dir / 'summary_results.json'}")


if __name__ == "__main__":
    main()
