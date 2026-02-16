"""End-to-end inference pipeline for lung CT segmentation and navigation path planning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import binary_opening, label
from scipy.spatial import cKDTree

from src.data.datasets import load_nifti, resample_volume, window_and_normalize
from src.models.unet3d import UNet3D
from src.models.vnet import VNet
from src.pathplanning.astar import AStarPathPlanner


def generate_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def pad_to_patch(volume: np.ndarray, patch_size: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    pads = []
    for dim, patch in zip(volume.shape, patch_size):
        needed = max(patch - dim, 0)
        pads.append((0, needed))
    padded = np.pad(volume, pads, mode="constant", constant_values=0.0)
    return padded, (volume.shape[0], volume.shape[1], volume.shape[2])


def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    model.to(device)

    patch_size = tuple(int(x) for x in patch_size)
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)

    padded, original_shape = pad_to_patch(volume, patch_size)
    D, H, W = padded.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    d_starts = generate_starts(D, pd, sd)
    h_starts = generate_starts(H, ph, sh)
    w_starts = generate_starts(W, pw, sw)

    with torch.no_grad():
        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    patch = padded[d : d + pd, h : h + ph, w : w + pw]
                    tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                    logits = model(tensor)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                    output[d : d + pd, h : h + ph, w : w + pw] += prob
                    count[d : d + pd, h : h + ph, w : w + pw] += 1

    pred = output / np.maximum(count, 1e-8)
    od, oh, ow = original_shape
    return pred[:od, :oh, :ow]


def extract_centerline(airway_mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize_3d

        return skeletonize_3d(airway_mask > 0)
    except Exception:
        # Fallback when skeletonization is unavailable.
        from scipy.ndimage import binary_erosion

        return binary_erosion(airway_mask > 0, iterations=2)


def build_graph_from_centerline(centerline: np.ndarray, spacing: Sequence[float]) -> nx.Graph:
    coords = np.argwhere(centerline > 0)
    graph = nx.Graph()

    if len(coords) == 0:
        return graph

    coord_to_node: Dict[Tuple[int, int, int], int] = {}
    for idx, coord in enumerate(coords):
        c = tuple(int(x) for x in coord)
        coord_to_node[c] = idx
        world_pos = tuple(float(coord[i] * spacing[i]) for i in range(3))
        graph.add_node(idx, voxel_pos=c, pos=world_pos)

    neighbor_offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    for c, node in coord_to_node.items():
        cz, cy, cx = c
        for dz, dy, dx in neighbor_offsets:
            nbr = (cz + dz, cy + dy, cx + dx)
            nbr_node = coord_to_node.get(nbr)
            if nbr_node is None:
                continue
            if graph.has_edge(node, nbr_node):
                continue
            dist = float(np.linalg.norm(np.array(graph.nodes[node]["pos"]) - np.array(graph.nodes[nbr_node]["pos"])))
            graph.add_edge(node, nbr_node, weight=dist)

    return graph


def detect_nodules_from_prob(
    nodule_prob: np.ndarray,
    threshold: float,
    min_voxels: int,
    spacing: Sequence[float],
) -> List[Dict]:
    mask = binary_opening(nodule_prob > threshold, structure=np.ones((3, 3, 3), dtype=bool))
    labeled, num = label(mask)

    nodules: List[Dict] = []
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])

    for comp_id in range(1, num + 1):
        component = labeled == comp_id
        count = int(component.sum())
        if count < min_voxels:
            continue

        coords = np.argwhere(component)
        probs = nodule_prob[component]
        centroid_voxel = coords.mean(axis=0)
        centroid_mm = [float(centroid_voxel[i] * spacing[i]) for i in range(3)]
        volume_mm3 = count * voxel_volume
        diameter_mm = float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0))

        nodules.append(
            {
                "id": len(nodules) + 1,
                "centroid_mm": centroid_mm,
                "diameter_mm": diameter_mm,
                "volume_mm3": float(volume_mm3),
                "confidence": float(probs.mean()),
                "num_voxels": count,
            }
        )

    nodules.sort(key=lambda x: x["confidence"], reverse=True)
    return nodules


def find_trachea_node(graph: nx.Graph) -> int:
    # Heuristic: superior-most centerline node.
    return min(graph.nodes, key=lambda n: graph.nodes[n]["pos"][0])


def plan_paths(graph: nx.Graph, nodules: List[Dict], start_node: int) -> Dict[str, Dict]:
    if graph.number_of_nodes() == 0:
        return {}

    node_ids = list(graph.nodes)
    node_positions = np.array([graph.nodes[n]["pos"] for n in node_ids], dtype=np.float32)
    tree = cKDTree(node_positions)

    planner = AStarPathPlanner(graph)
    paths: Dict[str, Dict] = {}

    for nodule in nodules:
        _, idx = tree.query(np.array(nodule["centroid_mm"], dtype=np.float32), k=1)
        target_node = node_ids[int(idx)]

        result = planner.find_path(start_node, target_node)
        if not result.get("success", False):
            paths[str(nodule["id"])] = {"success": False}
            continue

        path_nodes = result["path"]
        waypoints_mm = [graph.nodes[n]["pos"] for n in path_nodes]
        tortuosity = planner.compute_path_tortuosity(path_nodes)

        paths[str(nodule["id"])] = {
            "success": True,
            "path_nodes": path_nodes,
            "waypoints_mm": waypoints_mm,
            "length_mm": float(result["length"]),
            "cost": float(result["cost"]),
            "num_waypoints": int(result["num_nodes"]),
            "tortuosity": float(tortuosity),
        }

    return paths


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)


def save_results(
    output_dir: str,
    airway_mask: np.ndarray,
    airway_prob: np.ndarray,
    nodule_prob: np.ndarray,
    nodules: List[Dict],
    paths: Dict[str, Dict],
    metadata: Dict,
    affine: np.ndarray,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(airway_mask.astype(np.uint8), affine), out / "airway_segmentation.nii.gz")
    nib.save(nib.Nifti1Image(airway_prob.astype(np.float32), affine), out / "airway_probability.nii.gz")
    nib.save(nib.Nifti1Image(nodule_prob.astype(np.float32), affine), out / "nodule_probability.nii.gz")

    with open(out / "nodules.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(nodules), "nodules": nodules}, f, indent=2)

    successful = [p for p in paths.values() if p.get("success", False)]
    summary = {
        "total_nodules": len(nodules),
        "successful_paths": len(successful),
        "avg_path_length_mm": float(np.mean([p["length_mm"] for p in successful])) if successful else 0.0,
        "avg_tortuosity": float(np.mean([p["tortuosity"] for p in successful])) if successful else 0.0,
    }

    with open(out / "navigation_paths.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paths": paths}, f, indent=2)

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(args: argparse.Namespace) -> None:
    start = time.time()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    volume, spacing, affine = load_nifti(Path(args.input))
    volume_iso = resample_volume(volume, spacing, args.target_spacing, is_mask=False)
    volume_iso = window_and_normalize(volume_iso, args.intensity_range)

    airway_model = UNet3D(in_channels=1, out_channels=1)
    nodule_model = VNet(in_channels=1, out_channels=1)

    load_model_checkpoint(airway_model, args.airway_model, device)
    load_model_checkpoint(nodule_model, args.nodule_model, device)

    airway_prob = sliding_window_predict(
        airway_model,
        volume_iso,
        patch_size=args.airway_patch_size,
        overlap=args.overlap,
        device=device,
    )
    airway_mask = (airway_prob > args.airway_threshold).astype(np.uint8)

    nodule_prob = sliding_window_predict(
        nodule_model,
        volume_iso,
        patch_size=args.nodule_patch_size,
        overlap=args.overlap,
        device=device,
    )
    nodules = detect_nodules_from_prob(
        nodule_prob,
        threshold=args.nodule_threshold,
        min_voxels=args.nodule_min_voxels,
        spacing=args.target_spacing,
    )

    centerline = extract_centerline(airway_mask)
    graph = build_graph_from_centerline(centerline, spacing=args.target_spacing)

    if graph.number_of_nodes() > 0 and nodules:
        start_node = find_trachea_node(graph)
        paths = plan_paths(graph, nodules, start_node=start_node)
    else:
        paths = {}

    metadata = {
        "input": args.input,
        "original_shape": list(volume.shape),
        "original_spacing": list(spacing),
        "target_spacing": list(args.target_spacing),
        "resampled_shape": list(volume_iso.shape),
        "airway_graph_nodes": int(graph.number_of_nodes()),
        "airway_graph_edges": int(graph.number_of_edges()),
        "runtime_sec": round(time.time() - start, 3),
    }

    save_results(
        output_dir=args.output_dir,
        airway_mask=airway_mask,
        airway_prob=airway_prob,
        nodule_prob=nodule_prob,
        nodules=nodules,
        paths=paths,
        metadata=metadata,
        affine=affine,
    )

    print("Inference complete")
    print(json.dumps({"nodules": len(nodules), "paths": len(paths)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung CT inference + path planning")
    parser.add_argument("--input", type=str, required=True, help="Input CT NIfTI path")
    parser.add_argument("--airway_model", type=str, required=True, help="Path to airway checkpoint")
    parser.add_argument("--nodule_model", type=str, required=True, help="Path to nodule checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--target_spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--intensity_range", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument("--airway_patch_size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--nodule_patch_size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--airway_threshold", type=float, default=0.5)
    parser.add_argument("--nodule_threshold", type=float, default=0.5)
    parser.add_argument("--nodule_min_voxels", type=int, default=20)

    main(parser.parse_args())
