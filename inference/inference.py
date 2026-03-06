import argparse
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

import inference_flex_functions as inf
import pre_proc_functions as proc


VIEW_CONFIG = {
    "sag": {"checkpoint_arg": "model_sag_path", "stack_axis": 2},
    "cor": {"checkpoint_arg": "model_cor_path", "stack_axis": 1},
    "ax": {"checkpoint_arg": "model_ax_path", "stack_axis": 0},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PigBET inference on CPU, Metal/MPS, or CUDA."
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default="Example_images-selected",
        help="Path to the input directory containing NIfTI volumes.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="res",
        help="Output directory prefix for inference artifacts.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Optional directory prefix for metrics outputs.",
    )
    parser.add_argument(
        "--truth_folder",
        type=str,
        default=None,
        help="Optional directory containing ground-truth masks for metrics.",
    )

    parser.add_argument(
        "--model_sag_path",
        type=str,
        default="Checkpoints-selected/Unet_efficientnet-b3_sag.pth",
        help="Path to the sagittal checkpoint.",
    )
    parser.add_argument(
        "--model_cor_path",
        type=str,
        default="Checkpoints-selected/Unet_efficientnet-b3_cor.pth",
        help="Path to the coronal checkpoint.",
    )
    parser.add_argument(
        "--model_ax_path",
        type=str,
        default="Checkpoints-selected/Unet_efficientnet-b3_ax.pth",
        help="Path to the axial checkpoint.",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="efficientnet-b3",
        help="Encoder name for the segmentation model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "metal", "mps"],
        help="Inference backend. 'metal' maps to PyTorch MPS on Apple Silicon.",
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default="_mc_restore",
        help="Suffix before .nii/.nii.gz used to match source images. Use '' to accept all NIfTI files.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for converting sigmoid outputs into binary masks.",
    )
    parser.add_argument(
        "--prefer_fsl",
        action="store_true",
        help="Use fslmaths for majority voting when it is available; otherwise a Python fallback is used.",
    )

    # Legacy args kept for CLI compatibility with the original script.
    parser.add_argument("--sag_dim", nargs=2, type=int, default=[288, 288], help=argparse.SUPPRESS)
    parser.add_argument("--cor_dim", nargs=2, type=int, default=[256, 288], help=argparse.SUPPRESS)
    parser.add_argument("--ax_dim", nargs=2, type=int, default=[256, 288], help=argparse.SUPPRESS)

    return parser.parse_args()


def resolve_device(device_name):
    requested = "mps" if device_name == "metal" else device_name

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise RuntimeError("Metal/MPS was requested, but PyTorch MPS is not available.")

    return torch.device(requested)


def clear_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(encoder_type):
    return smp.Unet(
        encoder_name=encoder_type,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_model(checkpoint_path, encoder_type, device):
    checkpoint = torch_load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif checkpoint and all(isinstance(key, str) for key in checkpoint):
            state_dict = checkpoint
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        model = build_model(encoder_type)
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError(f"Unsupported checkpoint type for {checkpoint_path}: {type(checkpoint)}")

    model = model.to(device)
    model.eval()
    return model


def ensure_view_dirs(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    view_dirs = {}
    for view in VIEW_CONFIG:
        view_path = base_dir / view
        view_path.mkdir(parents=True, exist_ok=True)
        view_dirs[view] = view_path
    return view_dirs


def collect_png_files(directory):
    return sorted(str(path) for path in Path(directory).glob("*.png"))


def run_view_inference(view, model, dataset, image_dir, output_mask_dir, output_prob_dir, output_png_dir, threshold, device):
    files = collect_png_files(image_dir)
    if not files:
        raise RuntimeError(f"No PNG slices were generated for the {view} view in {image_dir}.")

    print(f"[INFO] Running {view} inference on {len(files)} slices using {device.type}.")
    with torch.inference_mode():
        for index, filename in enumerate(files, start=1):
            image = inf.get_data_from_filename(filename, dataset)
            batch = torch.from_numpy(image).unsqueeze(0).to(device=device, dtype=torch.float32)
            logits = model(batch).squeeze(0).squeeze(0)
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            mask = (probabilities >= threshold).astype(np.uint8)
            original_height, original_width = inf.get_image_hw(filename)
            probabilities = inf.unpad_array_to_shape(probabilities, original_height, original_width).astype(np.float32)
            mask = inf.unpad_array_to_shape(mask, original_height, original_width).astype(np.uint8)

            base_name = Path(filename).stem
            np.save(output_prob_dir / f"{base_name}.npy", probabilities)
            np.save(output_mask_dir / f"{base_name}.npy", mask)
            inf.display(
                [None, None, mask],
                epoch=base_name,
                save_path=None,
                is_inference=True,
                inference_path=str(output_png_dir),
            )
            if index == 1 or index == len(files) or index % 25 == 0:
                print(f"[INFO] {view}: {index}/{len(files)} -> {base_name}")


def maybe_write_metrics(final_out, volume_out, truth_folder, metrics_out):
    if not metrics_out or not truth_folder:
        return

    metrics_prefix = Path(metrics_out)
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)
    inf.calc3dDice(str(final_out), str(volume_out), truth_folder, f"{metrics_out}_dice.csv")
    inf.calc3dIOU(str(final_out), str(volume_out), truth_folder, f"{metrics_out}_iou.csv")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"images_dir:      {args.images_dir}")
    print(f"study_name:      {args.study_name}")
    print(f"encoder_type:    {args.encoder_type}")
    print(f"device:          {device.type}")
    print(f"image_suffix:    {repr(args.image_suffix)}")
    print(f"mask_threshold:  {args.mask_threshold}")
    print(f"prefer_fsl:      {args.prefer_fsl}")

    images_dir = Path(args.images_dir)
    study_dir = Path(args.study_name)

    source_map = proc.build_source_image_map(str(images_dir), img_fixed=args.image_suffix)
    if not source_map:
        raise RuntimeError(
            f"No input NIfTI files matched in {images_dir} with image_suffix={repr(args.image_suffix)}."
        )

    raw_png_dir = ensure_view_dirs(study_dir / "raw_png")
    raw_png_output_dir = ensure_view_dirs(study_dir / "raw_png_output")
    raw_npy_output_dir = ensure_view_dirs(study_dir / "raw_npy_output")
    volume_out_dir = ensure_view_dirs(study_dir / "volumn_out")
    prob_out_dir = ensure_view_dirs(study_dir / "prob_out")
    final_out_dir = study_dir / "final_out"
    final_out_dir.mkdir(parents=True, exist_ok=True)

    proc.proc_img_masks(
        img_dir=str(images_dir),
        out_dir=str(study_dir / "raw_png"),
        img_fixed=args.image_suffix,
        mask_fixed="-mask",
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_type, "imagenet")
    preprocessing = inf.get_preprocessing(preprocessing_fn)

    for view, config in VIEW_CONFIG.items():
        checkpoint_path = getattr(args, config["checkpoint_arg"])
        model = load_model(checkpoint_path, args.encoder_type, device)
        dataset = inf.Dataset(
            images_dir=str(raw_png_dir[view]),
            preprocessing=preprocessing,
            classes=["brain"],
        )
        run_view_inference(
            view=view,
            model=model,
            dataset=dataset,
            image_dir=raw_png_dir[view],
            output_mask_dir=raw_npy_output_dir[view],
            output_prob_dir=prob_out_dir[view],
            output_png_dir=raw_png_output_dir[view],
            threshold=args.mask_threshold,
            device=device,
        )
        clear_device_cache(device)

    for view, config in VIEW_CONFIG.items():
        inf.stack_slices_and_save_nifti(
            str(raw_npy_output_dir[view]),
            str(volume_out_dir[view]),
            source_map,
            config["stack_axis"],
        )

    inf.run_fslmaths(
        str(volume_out_dir["sag"]),
        str(volume_out_dir["cor"]),
        str(volume_out_dir["ax"]),
        str(final_out_dir),
        prefer_fsl=args.prefer_fsl,
    )

    maybe_write_metrics(final_out_dir, study_dir / "volumn_out", args.truth_folder, args.metrics_out)
    print("[INFO] Inference pipeline completed.")


if __name__ == "__main__":
    main()
