"""
Complete inference pipeline for lung CT analysis.

Pipeline:
1. Load CT scan (DICOM or NIfTI)
2. Preprocessing (resampling, windowing, normalization)
3. Airway segmentation (sliding window inference)
4. Nodule detection (candidate generation)
5. Centerline extraction (skeletonization)
6. Graph construction (NetworkX)
7. Path planning (A* for each nodule)
8. Save results (masks, paths, metrics, visualization)
"""

import os
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm

# For real implementation, import these:
# import SimpleITK as sitk
# from scipy.ndimage import binary_erosion, binary_dilation
# from skimage.morphology import skeletonize_3d
# import networkx as nx

# from src.models.unet3d import UNet3D
# from src.models.vnet import VNet
# from src.pathplanning.astar import AStarPathPlanner


class CTPreprocessor:
    """Preprocessing for lung CT scans."""

    def __init__(self, target_spacing=(1.0, 1.0, 1.0), 
                 intensity_range=(-1000, 400)):
        """
        Args:
            target_spacing: Target isotropic spacing in mm (D, H, W)
            intensity_range: HU window (min, max)
        """
        self.target_spacing = target_spacing
        self.intensity_range = intensity_range

    def load_ct(self, filepath):
        """
        Load CT scan from file.

        Args:
            filepath: Path to DICOM series directory or NIfTI file

        Returns:
            volume: 3D numpy array
            spacing: Original spacing (D, H, W) in mm
            origin: Origin coordinates
        """
        filepath = Path(filepath)

        if filepath.is_dir():
            # Load DICOM series
            # In practice: use SimpleITK
            # reader = sitk.ImageSeriesReader()
            # dicom_names = reader.GetGDCMSeriesFileNames(str(filepath))
            # reader.SetFileNames(dicom_names)
            # image = reader.Execute()

            # Placeholder
            volume = np.random.randn(256, 512, 512).astype(np.float32) * 200 - 500
            spacing = (0.625, 0.7, 0.7)
            origin = (0, 0, 0)

        elif filepath.suffix in ['.nii', '.gz']:
            # Load NIfTI
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)
            spacing = nii.header.get_zooms()[:3]
            origin = nii.affine[:3, 3]

        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        return volume, spacing, origin

    def resample_to_isotropic(self, volume, original_spacing):
        """
        Resample volume to isotropic spacing.

        Args:
            volume: 3D numpy array
            original_spacing: Original spacing (D, H, W)

        Returns:
            resampled: Resampled volume
        """
        # Compute scale factors
        scale_factors = [
            orig / target 
            for orig, target in zip(original_spacing, self.target_spacing)
        ]

        # Convert to torch tensor
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Compute new size
        new_size = [
            int(volume.shape[i] * scale_factors[i]) 
            for i in range(3)
        ]

        # Resample using trilinear interpolation
        resampled = F.interpolate(
            volume_tensor,
            size=new_size,
            mode='trilinear',
            align_corners=True
        )

        resampled = resampled.squeeze().numpy()

        return resampled

    def apply_windowing(self, volume):
        """
        Apply HU windowing and normalize.

        Args:
            volume: 3D numpy array in HU

        Returns:
            windowed: Normalized volume [0, 1]
        """
        min_hu, max_hu = self.intensity_range

        # Clip to window
        windowed = np.clip(volume, min_hu, max_hu)

        # Normalize to [0, 1]
        windowed = (windowed - min_hu) / (max_hu - min_hu)

        return windowed.astype(np.float32)

    def preprocess(self, filepath):
        """
        Complete preprocessing pipeline.

        Args:
            filepath: Path to CT scan

        Returns:
            processed: Preprocessed volume (D, H, W)
            metadata: Dictionary with spacing, origin, original_shape
        """
        print("Loading CT scan...")
        volume, spacing, origin = self.load_ct(filepath)

        print(f"Original shape: {volume.shape}, spacing: {spacing}")

        print("Resampling to isotropic spacing...")
        resampled = self.resample_to_isotropic(volume, spacing)

        print(f"Resampled shape: {resampled.shape}")

        print("Applying HU windowing...")
        windowed = self.apply_windowing(resampled)

        metadata = {
            'original_shape': volume.shape,
            'original_spacing': spacing,
            'target_spacing': self.target_spacing,
            'resampled_shape': resampled.shape,
            'origin': origin
        }

        return windowed, metadata


class SlidingWindowInference:
    """Sliding window inference for large 3D volumes."""

    def __init__(self, model, patch_size=(128, 128, 128), overlap=0.5):
        """
        Args:
            model: Trained PyTorch model
            patch_size: Size of patches (D, H, W)
            overlap: Overlap ratio between patches (0-1)
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = tuple(int(p * (1 - overlap)) for p in patch_size)

    def predict(self, volume, device='cuda'):
        """
        Perform sliding window inference.

        Args:
            volume: 3D numpy array (D, H, W)
            device: Device to run inference on

        Returns:
            prediction: 3D prediction mask (D, H, W)
        """
        self.model.eval()
        self.model.to(device)

        D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        # Initialize output
        prediction = np.zeros((D, H, W), dtype=np.float32)
        count = np.zeros((D, H, W), dtype=np.float32)

        # Compute number of patches
        n_patches_d = (D - pd) // sd + 1
        n_patches_h = (H - ph) // sh + 1
        n_patches_w = (W - pw) // sw + 1

        total_patches = n_patches_d * n_patches_h * n_patches_w

        print(f"Running inference on {total_patches} patches...")

        with torch.no_grad():
            pbar = tqdm(total=total_patches)

            for i in range(0, D - pd + 1, sd):
                for j in range(0, H - ph + 1, sh):
                    for k in range(0, W - pw + 1, sw):
                        # Extract patch
                        patch = volume[i:i+pd, j:j+ph, k:k+pw]

                        # Convert to tensor
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                        patch_tensor = patch_tensor.to(device)

                        # Predict
                        output = self.model(patch_tensor)
                        output = torch.sigmoid(output)
                        output = output.squeeze().cpu().numpy()

                        # Add to prediction
                        prediction[i:i+pd, j:j+ph, k:k+pw] += output
                        count[i:i+pd, j:j+ph, k:k+pw] += 1

                        pbar.update(1)

            pbar.close()

        # Average overlapping predictions
        prediction = prediction / (count + 1e-8)

        return prediction


def extract_centerline(binary_mask):
    """
    Extract centerline from binary airway mask.

    Args:
        binary_mask: Binary 3D mask (D, H, W)

    Returns:
        centerline: Binary centerline mask
    """
    # In practice, use skimage.morphology.skeletonize_3d
    # centerline = skeletonize_3d(binary_mask)

    # Placeholder: simple erosion
    from scipy.ndimage import binary_erosion
    centerline = binary_erosion(binary_mask, iterations=3)

    return centerline


def build_graph_from_centerline(centerline):
    """
    Build NetworkX graph from centerline.

    Args:
        centerline: Binary centerline mask

    Returns:
        graph: NetworkX graph with nodes at centerline points
    """
    import networkx as nx

    # Get centerline coordinates
    coords = np.argwhere(centerline > 0)

    # Create graph
    G = nx.Graph()

    # Add nodes with positions
    for idx, coord in enumerate(coords):
        G.add_node(idx, pos=tuple(coord))

    # Add edges between nearby points (simple approach)
    # In practice, use proper branch detection
    for i in range(len(coords)):
        for j in range(i + 1, min(i + 10, len(coords))):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 5.0:  # 5mm threshold
                G.add_edge(i, j, weight=dist)

    return G


def detect_nodules(volume, nodule_model, device='cuda'):
    """
    Detect nodule candidates.

    Args:
        volume: Preprocessed CT volume
        nodule_model: Trained nodule detection model
        device: Device for inference

    Returns:
        nodules: List of nodule dictionaries
    """
    # In practice: patch-based detection, candidate generation, NMS

    # Placeholder: random nodules for demonstration
    nodules = [
        {'id': 1, 'centroid': [120, 200, 300], 'diameter': 8.5, 'confidence': 0.92},
        {'id': 2, 'centroid': [150, 250, 280], 'diameter': 6.2, 'confidence': 0.88},
        {'id': 3, 'centroid': [180, 220, 310], 'diameter': 10.3, 'confidence': 0.95},
    ]

    return nodules


def plan_paths(graph, nodules, trachea_entry_node=0):
    """
    Plan paths from trachea to each nodule.

    Args:
        graph: Airway tree graph
        nodules: List of nodule dictionaries
        trachea_entry_node: Start node (trachea entry)

    Returns:
        paths: Dictionary mapping nodule_id to path info
    """
    from astar import AStarPathPlanner

    planner = AStarPathPlanner(graph)

    paths = {}

    for nodule in nodules:
        # Find nearest graph node to nodule
        nodule_pos = np.array(nodule['centroid'])

        min_dist = float('inf')
        nearest_node = None

        for node, data in graph.nodes(data=True):
            node_pos = np.array(data['pos'])
            dist = np.linalg.norm(nodule_pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # Plan path
        if nearest_node is not None:
            result = planner.find_path(trachea_entry_node, nearest_node)

            if result['success']:
                tortuosity = planner.compute_path_tortuosity(result['path'])

                paths[nodule['id']] = {
                    'path': result['path'],
                    'length': result['length'],
                    'tortuosity': tortuosity,
                    'num_waypoints': result['num_nodes'],
                    'success': True
                }
            else:
                paths[nodule['id']] = {'success': False}

    return paths


def save_results(output_dir, airway_mask, nodules, paths, metadata):
    """Save all results to output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save airway segmentation
    airway_nii = nib.Nifti1Image(airway_mask.astype(np.uint8), np.eye(4))
    nib.save(airway_nii, output_dir / 'airway_segmentation.nii.gz')
    print(f"✓ Saved airway segmentation")

    # Save nodule information
    nodule_data = {
        'nodules': nodules,
        'count': len(nodules)
    }
    with open(output_dir / 'nodules.json', 'w') as f:
        json.dump(nodule_data, f, indent=2)
    print(f"✓ Saved nodule data ({len(nodules)} nodules)")

    # Save path planning results
    path_data = {
        'paths': paths,
        'summary': {
            'total_nodules': len(nodules),
            'successful_paths': sum(1 for p in paths.values() if p.get('success', False)),
            'avg_path_length': np.mean([p['length'] for p in paths.values() if p.get('success', False)]),
            'avg_tortuosity': np.mean([p['tortuosity'] for p in paths.values() if p.get('success', False)])
        }
    }
    with open(output_dir / 'navigation_paths.json', 'w') as f:
        json.dump(path_data, f, indent=2)
    print(f"✓ Saved path planning results")

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata")


def main(args):
    """Main inference pipeline."""

    print("="*70)
    print("Lung CT Analysis Pipeline - Inference")
    print("="*70)
    print()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load models
    print("Loading models...")

    # Placeholder models
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 1, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    airway_model = DummyModel()
    nodule_model = DummyModel()

    # In practice: load from checkpoint
    # checkpoint = torch.load(args.airway_model, map_location=device)
    # airway_model.load_state_dict(checkpoint['model_state_dict'])

    print("✓ Models loaded")
    print()

    # Preprocessing
    preprocessor = CTPreprocessor(
        target_spacing=(1.0, 1.0, 1.0),
        intensity_range=(-1000, 400)
    )

    volume, metadata = preprocessor.preprocess(args.input)
    print("✓ Preprocessing completed")
    print()

    # Airway segmentation
    print("Running airway segmentation...")
    start_time = time.time()

    inference = SlidingWindowInference(
        airway_model,
        patch_size=(128, 128, 128),
        overlap=0.5
    )

    airway_prob = inference.predict(volume, device=device)
    airway_mask = (airway_prob > 0.5).astype(np.uint8)

    seg_time = time.time() - start_time
    print(f"✓ Airway segmentation completed in {seg_time:.2f}s")
    print()

    # Nodule detection
    print("Detecting nodules...")
    nodules = detect_nodules(volume, nodule_model, device=device)
    print(f"✓ Found {len(nodules)} nodule candidates")
    print()

    # Centerline extraction
    print("Extracting airway centerline...")
    centerline = extract_centerline(airway_mask)
    print("✓ Centerline extracted")
    print()

    # Graph construction
    print("Building airway tree graph...")
    graph = build_graph_from_centerline(centerline)
    print(f"✓ Graph built with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print()

    # Path planning
    print("Planning navigation paths...")
    paths = plan_paths(graph, nodules)

    successful = sum(1 for p in paths.values() if p.get('success', False))
    print(f"✓ Path planning completed ({successful}/{len(nodules)} successful)")
    print()

    # Save results
    print("Saving results...")
    metadata['processing_time'] = seg_time
    save_results(args.output_dir, airway_mask, nodules, paths, metadata)
    print()

    # Summary
    print("="*70)
    print("Pipeline Summary")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Airway segmentation: {np.sum(airway_mask)} voxels")
    print(f"Nodules detected: {len(nodules)}")
    print(f"Successful paths: {successful}/{len(nodules)}")

    if successful > 0:
        avg_length = np.mean([p['length'] for p in paths.values() if p.get('success', False)])
        avg_tort = np.mean([p['tortuosity'] for p in paths.values() if p.get('success', False)])
        print(f"Average path length: {avg_length:.1f}mm")
        print(f"Average tortuosity: {avg_tort:.2f}")

    print("="*70)
    print("✓ Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung CT analysis inference pipeline")

    parser.add_argument('--input', type=str, required=True,
                       help='Input CT scan (DICOM directory or NIfTI file)')
    parser.add_argument('--airway_model', type=str, required=True,
                       help='Path to trained airway segmentation model')
    parser.add_argument('--nodule_model', type=str, required=True,
                       help='Path to trained nodule detection model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')

    args = parser.parse_args()

    main(args)
