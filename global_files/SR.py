#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import savemat
from skimage.metrics import (
    mean_squared_error as mse,
    normalized_root_mse as nrmse,
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)


def pick_config_value(data: dict[str, Any], key: str, default: Any = None, required: bool = True) -> Any:
    if key in data:
        return data[key]

    for section in ["hyperparameters", "others", "paths", "dataset", "training", "superresolve"]:
        if section in data and isinstance(data[section], dict) and key in data[section]:
            return data[section][key]

    if required and default is None:
        raise KeyError(f"Required key '{key}' was not found in the config file.")

    return default


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config_path does not exist: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_config(data: dict[str, Any], config_path: str | Path) -> defaultdict:
    config = defaultdict()

    config["lr"] = pick_config_value(data, "lr")
    config["batch_size"] = pick_config_value(data, "batch_size")
    config["skip_step"] = pick_config_value(data, "skip_step")

    config["nickname"] = pick_config_value(data, "nickname")
    config["hiresl_h"] = pick_config_value(data, "hiresl_h")
    config["hiresl_w"] = pick_config_value(data, "hiresl_w")

    config["channels"] = pick_config_value(data, "channels")
    config["T_in"] = pick_config_value(data, "T_in")
    config["R"] = pick_config_value(data, "R")
    config["full"] = pick_config_value(data, "full")
    config["patch_size_HR"] = pick_config_value(data, "patch_size_HR")
    config["patch_size_LR"] = pick_config_value(data, "patch_size_LR")
    config["border"] = pick_config_value(data, "border")
    config["spectral_band"] = pick_config_value(data, "spectral_band")

    config["dataset_path"] = pick_config_value(data, "dataset_path")
    config["n_chunks"] = pick_config_value(data, "n_chunks")

    config["RegNet_pretrain_dir"] = pick_config_value(data, "RegNet_pretrain_dir", default="", required=False)
    config["SISRNet_pretrain_dir"] = pick_config_value(data, "SISRNet_pretrain_dir", default="", required=False)

    config["mu"] = pick_config_value(data, "mu")
    config["sigma"] = pick_config_value(data, "sigma")
    config["sigma_rescaled"] = pick_config_value(data, "sigma_rescaled")
    config["metric_crop_border"] = int(
        pick_config_value(data, "metric_crop_border", default=config["border"], required=False)
    )
    config["metric_data_range"] = float(
        pick_config_value(data, "metric_data_range", default=65535.0, required=False)
    )
    config["metric_nrmse_normalization"] = str(
        pick_config_value(data, "metric_nrmse_normalization", default="min-max", required=False)
    )

    tensorboard_dir = pick_config_value(data, "tensorboard_dir", default=None, required=False)
    if tensorboard_dir is None:
        tensorboard_dir = (
            f"{config['nickname'].upper()}_"
            f"{config['spectral_band']}_"
            f"lr_{config['lr']}_"
            f"bsize_{config['batch_size']}"
        )
    config["tensorboard_dir"] = tensorboard_dir
    config["config_path"] = str(Path(config_path).expanduser().resolve())
    return config


def module_dir() -> Path:
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()


def import_module_from_path(module_name: str, module_path: Path):
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module does not exist: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_project_root(data: dict[str, Any], this_module_dir: Path) -> Path:
    value = (
        pick_config_value(data, "project_root", default=None, required=False)
        or pick_config_value(data, "global_files_dir", default=None, required=False)
        or pick_config_value(data, "c_global_files_dir", default=None, required=False)
    )
    return Path(value).expanduser().resolve() if value is not None else this_module_dir.resolve()


def resolve_output_dir(data: dict[str, Any], project_root: Path, config: defaultdict) -> Path:
    value = (
        pick_config_value(data, "superresolve_path", default=None, required=False)
        or pick_config_value(data, "superresolve_output_path", default=None, required=False)
        or pick_config_value(data, "sr_output_path", default=None, required=False)
        or pick_config_value(data, "output_dir", default=None, required=False)
    )
    if value is not None:
        return Path(value).expanduser().resolve()
    return project_root / f"superresolve_{config['nickname']}_{config['spectral_band']}"


def resolve_checkpoint_dir(data: dict[str, Any], project_root: Path, config: defaultdict) -> Path:
    value = (
        pick_config_value(data, "checkpoint_dir", default=None, required=False)
        or pick_config_value(data, "checkpoint_path", default=None, required=False)
    )
    if value is not None:
        return Path(value).expanduser().resolve()
    return (project_root / "checkpoints" / config["tensorboard_dir"]).resolve()


def set_project_environment(project_root: Path) -> None:
    project_root = project_root.resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print("Working directory:", os.getcwd())
    print("Python import path[0]:", sys.path[0])


def check_paths(project_root: Path, udn_path: Path, checkpoint_dir: Path, testset_path: Path) -> None:
    print("project_root:", project_root)
    print("UDN path:", udn_path)
    print("checkpoint dir:", checkpoint_dir)
    print("checkpoint dir exists:", checkpoint_dir.exists())
    print("testset_path:", testset_path)
    print("testset_path exists:", testset_path.exists())
    if not udn_path.exists():
        raise FileNotFoundError(f"UDN.py does not exist: {udn_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            "Expected checkpoint directory was not found:\n"
            f"{checkpoint_dir}\n\n"
            "This can be fixed by adding 'checkpoint_dir' to the config or checking the training process."
        )
    if not testset_path.exists():
        raise FileNotFoundError(f"testset_path does not exist: {testset_path}")


def load_test_arrays(testset_path: Path, band: str, nickname: str) -> tuple[np.ndarray, np.ndarray]:
    lr_path = testset_path / f"dataset_{band}_{nickname}_s_LR_test.npy"
    hr_path = testset_path / f"dataset_{band}_{nickname}_s_HR_test.npy"
    print("LR test path:", lr_path)
    print("HR test path:", hr_path)
    if not lr_path.exists():
        raise FileNotFoundError(f"LR test does not exist: {lr_path}")
    if not hr_path.exists():
        raise FileNotFoundError(f"HR test does not exist: {hr_path}")
    lr = np.load(lr_path, allow_pickle=True)
    hr = np.load(hr_path, allow_pickle=True)
    print("LR test shape:", np.asarray(lr).shape)
    print("HR test shape:", np.asarray(hr).shape)
    return lr, hr


def to_2d(arr: np.ndarray) -> np.ndarray:
    return np.squeeze(arr)


def center_crop_2d(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    h, w = img.shape
    if (2 * border) >= h or (2 * border) >= w:
        raise ValueError(
            f"metric_crop_border={border} is too large for an image of size {(h, w)}"
        )
    return img[border:h - border, border:w - border]


def norm01(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    img = np.asarray(img).astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    return (img - mn) / (mx - mn + eps)


def save_preview_images(sr_images, lr_test, hr_test, image_index: int, out_dir: Path) -> None:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sr = sr_images[image_index, 0, :, :, 0]
    lr = lr_test[image_index][0]
    hr = hr_test[image_index]
    for name, arr in [("SR", sr), ("LR", lr), ("HR", hr)]:
        plt.figure(figsize=[8, 8])
        plt.imshow(arr, cmap="gray", interpolation="none")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(preview_dir / f"image_{image_index:03d}_{name}.png", dpi=150)
        plt.close()
    print("Preview saved in:", preview_dir)


def save_mat_files(sr_images, lr_test, hr_test, out_dir: Path) -> None:
    savemat(out_dir / "super_resolved_images.mat", {"super_resolved_images": sr_images})
    savemat(out_dir / "input_images_LR_test.mat", {"input_images_LR_test": lr_test})
    savemat(out_dir / "input_images_HR_test.mat", {"input_images_HR_test": hr_test})
    print(".mat files saved in:", out_dir)


def compute_single_image_metrics(
    sr_images,
    hr_test,
    image_index: int,
    crop_border: int,
    metric_data_range: float,
    nrmse_normalization: str,
) -> None:
    sr = center_crop_2d(to_2d(sr_images[image_index, 0, :, :, 0]), crop_border)
    hr = center_crop_2d(to_2d(hr_test[image_index]), crop_border)
    print(f"SSIM value for image {image_index}: {ssim(hr, sr, data_range=metric_data_range):.4f}")
    print(f"PSNR value for image {image_index}: {psnr(hr, sr, data_range=metric_data_range):.4f} dB")
    print(f"MSE value for image {image_index}: {mse(hr, sr):.4f}")
    print(
        f"NRMSE value for image {image_index}: "
        f"{nrmse(hr, sr, normalization=nrmse_normalization):.4f}"
    )


def save_all_metrics_and_pngs(
    sr_images,
    lr_test,
    hr_test,
    out_dir: Path,
    excel_name: str,
    crop_border: int,
    metric_data_range: float,
    nrmse_normalization: str,
    save_png: bool = True,
):
    sr_dir = out_dir / "SR"
    lr_dir = out_dir / "LR_test"
    hr_dir = out_dir / "HR_test"
    sr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(sr_images.shape[0]):
        sr = center_crop_2d(to_2d(sr_images[i, 0, :, :, 0]), crop_border)
        lr = to_2d(lr_test[i][0])
        hr = center_crop_2d(to_2d(hr_test[i]), crop_border)
        rows.append({
            "image_index": i,
            "SSIM": float(ssim(hr, sr, data_range=metric_data_range)),
            "PSNR_dB": float(psnr(hr, sr, data_range=metric_data_range)),
            "MSE": float(mse(hr, sr)),
            "NRMSE": float(nrmse(hr, sr, normalization=nrmse_normalization)),
        })
        if save_png:
            plt.imsave(sr_dir / f"img_{i:03d}_SR.png", norm01(sr), cmap="gray")
            plt.imsave(lr_dir / f"img_{i:03d}_LR.png", norm01(lr), cmap="gray")
            plt.imsave(hr_dir / f"img_{i:03d}_HR.png", norm01(hr), cmap="gray")
    df = pd.DataFrame(rows).sort_values("image_index").reset_index(drop=True)
    summary = pd.DataFrame({
        "metric": ["SSIM", "PSNR_dB", "MSE", "NRMSE"],
        "mean": [df["SSIM"].mean(), df["PSNR_dB"].mean(), df["MSE"].mean(), df["NRMSE"].mean()],
        "std": [df["SSIM"].std(ddof=1), df["PSNR_dB"].std(ddof=1), df["MSE"].std(ddof=1), df["NRMSE"].std(ddof=1)],
    })
    excel_path = out_dir / excel_name
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print("Procedure completed")
    print(f"Excel: {excel_path}")
    print(f"SR Images: {sr_dir}")
    print(f"LR Images: {lr_dir}")
    print(f"HR Images: {hr_dir}")
    return df


def run_superresolve(
    config_path: str | Path,
    project_root: str | Path | None = None,
    testset_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    image_index: int | None = None,
    n_slide: int | None = None,
    save_png: bool | None = None,
    save_mat: bool | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    data = load_config(config_path)
    this_dir = module_dir()
    project = Path(project_root).expanduser().resolve() if project_root is not None else resolve_project_root(data, this_dir)
    set_project_environment(project)
    udn_path = project / "UDN.py"
    UDN = import_module_from_path("UDN", udn_path)
    if not hasattr(UDN, "SR_network"):
        raise AttributeError(f"UDN.py does not contain SR_network: {udn_path}")
    print("UDN file:", UDN.__file__)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    config = build_model_config(data, config_path)
    testset = Path(testset_path).expanduser().resolve() if testset_path is not None else Path(pick_config_value(data, "testset_path")).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve() if out_dir is not None else resolve_output_dir(data, project, config)
    checkpoint = resolve_checkpoint_dir(data, project, config)
    resolved_n_slide = n_slide if n_slide is not None else pick_config_value(data, "n_slide", default=0, required=False)
    resolved_save_png = save_png if save_png is not None else bool(pick_config_value(data, "save_png", default=True, required=False))
    resolved_save_mat = save_mat if save_mat is not None else bool(pick_config_value(data, "save_mat", default=True, required=False))
    if image_index is None:
        image_index = pick_config_value(data, "preview_image_index", default=None, required=False)
    excel_name = pick_config_value(data, "metrics_excel_name", default=f"metrics_{config['nickname']}.xlsx", required=False)
    print("Config file:", config_path)
    print("Nickname:", config["nickname"])
    print("Spectral band:", config["spectral_band"])
    print("Tensorboard dir:", config["tensorboard_dir"])
    print("Output dir:", output)
    print("n_slide:", resolved_n_slide)
    print("save_png:", resolved_save_png)
    print("save_mat:", resolved_save_mat)
    print("preview_image_index:", image_index)
    print("metrics_excel_name:", excel_name)
    print("metric_crop_border:", config["metric_crop_border"])
    print("metric_data_range:", config["metric_data_range"])
    print("metric_nrmse_normalization:", config["metric_nrmse_normalization"])
    check_paths(project, udn_path, checkpoint, testset)
    output.mkdir(parents=True, exist_ok=True)
    model = UDN.SR_network(config)
    model.build()
    print("Model built successfully")
    print("Running predict_test...")
    sr_images = model.predict_test(str(testset) + "/", n_slide=resolved_n_slide)
    print("super_resolved_images shape:", sr_images.shape)
    lr_test, hr_test = load_test_arrays(testset, config["spectral_band"], config["nickname"])
    if image_index is not None:
        if 0 <= int(image_index) < sr_images.shape[0]:
            save_preview_images(sr_images, lr_test, hr_test, int(image_index), output)
            compute_single_image_metrics(
                sr_images,
                hr_test,
                int(image_index),
                crop_border=config["metric_crop_border"],
                metric_data_range=config["metric_data_range"],
                nrmse_normalization=config["metric_nrmse_normalization"],
            )
        else:
            print(f"Preview was not saved because image_index={image_index} is out of range.")
    if resolved_save_mat:
        save_mat_files(sr_images, lr_test, hr_test, output)
    metrics_df = save_all_metrics_and_pngs(
        sr_images,
        lr_test,
        hr_test,
        output,
        excel_name,
        crop_border=config["metric_crop_border"],
        metric_data_range=config["metric_data_range"],
        nrmse_normalization=config["metric_nrmse_normalization"],
        save_png=resolved_save_png,
    )
    print("Superresolve ran successfully")
    return {
        "config": config,
        "config_path": config_path,
        "project_root": project,
        "testset_path": testset,
        "checkpoint_dir": checkpoint,
        "out_dir": output,
        "super_resolved_images": sr_images,
        "input_images_LR_test": lr_test,
        "input_images_HR_test": hr_test,
        "metrics": metrics_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs DeepSUM super-resolution from a JSON config.")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--testset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--image-index", type=int, default=None)
    parser.add_argument("--n-slide", type=int, default=None)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-mat", action="store_true")
    return parser.parse_args()


def main(config_path: str | Path | None = None, **kwargs) -> dict[str, Any] | None:
    if config_path is not None:
        return run_superresolve(config_path=config_path, **kwargs)
    args = parse_args()
    return run_superresolve(
        config_path=args.config_path,
        project_root=args.project_root,
        testset_path=args.testset_path,
        out_dir=args.out_dir,
        image_index=args.image_index,
        n_slide=args.n_slide,
        save_png=not args.no_png,
        save_mat=not args.no_mat,
    )


if __name__ == "__main__":
    main()
