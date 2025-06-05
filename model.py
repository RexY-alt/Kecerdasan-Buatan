from __future__ import annotations

import pathlib
import sys
import os # Import os for checking path existence

import numpy as np
import torch
import torch.nn as nn

# Determine the application directory (where model.py is located)
app_dir = pathlib.Path(__file__).parent.resolve()
# Define the path to the CBNetV2 submodule, assuming it's at the same level as model.py
submodule_path_str = str(app_dir / "CBNetV2")

# Add CBNetV2 to sys.path if it exists, so mmdet can find its components
if os.path.isdir(submodule_path_str):
    sys.path.insert(0, submodule_path_str)
else:
    # This is a fallback, but ideally this condition shouldn't be met if app.py also checks.
    print(f"Warning: CBNetV2 directory not found at {submodule_path_str} from model.py. MMDetection might fail to load configs/models correctly.", file=sys.stderr)

from mmdet.apis import inference_detector, init_detector


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = self._load_models()
        # Default model, can be changed by set_model_name
        self.model_name = "Faster R-CNN (DB-ResNet50)" # Changed default to one used in UI

    def _load_models(self) -> dict[str, nn.Module]:
        model_dict = {
            "Faster R-CNN (DB-ResNet50)": {
                # Config paths should be relative to the root of the CBNetV2 submodule
                # or an absolute path if known. Given sys.path modification,
                # mmdet might expect paths from within CBNetV2.
                # Let's assume CBNetV2 is at the root of your project.
                "config": "CBNetV2/configs/cbnet/faster_rcnn_cbv2d1_r50_fpn_1x_coco.py",
                "model": "https://github.com/CBNetwork/storage/releases/download/v1.0.0/faster_rcnn_cbv2d1_r50_fpn_1x_coco.pth.zip",
            },
            "Mask R-CNN (DB-Swin-T)": {
                "config": "CBNetV2/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
                "model": "https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip",
            },
            "Improved HTC (DB-Swin-B)": {
                "config": "CBNetV2/configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py",
                "model": "https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip",
            },
            # Add other models if you plan to use them
        }

        # Weights will be downloaded relative to where model.py is executed,
        # typically the root of your Streamlit app.
        weight_dir = app_dir / "weights"
        weight_dir.mkdir(exist_ok=True)

        def _download_and_extract(model_name_key: str, out_dir: pathlib.Path) -> str:
            import zipfile
            import urllib.parse

            model_url = model_dict[model_name_key]["model"]
            # Derive zip name and final model name from URL
            parsed_url = urllib.parse.urlparse(model_url)
            zip_name_from_url = pathlib.Path(parsed_url.path).name
            
            # The actual .pth filename inside the zip might differ slightly (e.g. no .zip extension)
            # Typically, it's the zip_name_from_url without the .zip
            if zip_name_from_url.endswith('.pth.zip'):
                 final_model_filename = zip_name_from_url[:-4] # remove .zip
            else:
                 # Fallback, adjust if your naming is different
                 final_model_filename = zip_name_from_url.replace('.zip', '')


            checkpoint_path = out_dir / final_model_filename
            zip_dl_path = out_dir / zip_name_from_url

            if not checkpoint_path.exists():
                if not zip_dl_path.exists():
                    print(f"Downloading {model_url} to {zip_dl_path}")
                    torch.hub.download_url_to_file(model_url, str(zip_dl_path), progress=True)
                else:
                    print(f"Zip file {zip_dl_path} already exists.")

                print(f"Extracting {zip_dl_path} to {out_dir}")
                with zipfile.ZipFile(zip_dl_path, 'r') as f:
                    f.extractall(out_dir)
                print(f"Extracted. Checkpoint should be at {checkpoint_path}")
                
                if not checkpoint_path.exists():
                    # Try to find the .pth file if the derived name is wrong
                    pth_files = list(out_dir.glob('*.pth'))
                    if pth_files:
                        actual_checkpoint_path = pth_files[0]
                        if actual_checkpoint_path.name != final_model_filename:
                            print(f"Note: Checkpoint file name '{actual_checkpoint_path.name}' differs from expected '{final_model_filename}'. Using actual.")
                            checkpoint_path = actual_checkpoint_path
                        else:
                             print(f"Confirmed checkpoint path: {checkpoint_path}")
                    else:
                        raise FileNotFoundError(f"Checkpoint file {final_model_filename} not found in {out_dir} after extraction, and no other .pth files found.")
            else:
                print(f"Checkpoint {checkpoint_path} already exists.")
            return str(checkpoint_path)


        loaded_models = {}
        for key, item_dict in model_dict.items():
            if key not in ["Faster R-CNN (DB-ResNet50)", "Mask R-CNN (DB-Swin-T)"]: # Only load models used in UI by default
                if key != self.model_name: # Allow loading default if it's not one of the two main ones.
                    continue
            
            print(f"Initializing model: {key}")
            config_file_path = app_dir / item_dict["config"]
            if not config_file_path.exists():
                 # If CBNetV2 is added to path, mmdet might resolve configs differently.
                 # Try path relative to CBNetV2 root if direct path fails.
                 config_file_path_in_submodule = app_dir / "CBNetV2" / item_dict["config"].replace("CBNetV2/", "", 1)
                 if (app_dir / "CBNetV2").exists() and config_file_path_in_submodule.exists():
                     config_file_path = config_file_path_in_submodule
                 else:
                    # Original config path if CBNetV2 is not at root relative to app_dir
                    # This assumes CBNetV2 is at the project root and item_dict["config"] is like "CBNetV2/configs/..."
                    # And app_dir is the project root.
                    potential_config_path = app_dir / item_dict['config']
                    if not potential_config_path.exists():
                        raise FileNotFoundError(f"Config file {item_dict['config']} not found directly or relative to {app_dir} or {app_dir / 'CBNetV2'}")
                    config_file_path = potential_config_path


            print(f"Using config file: {config_file_path}")
            checkpoint_file_path = _download_and_extract(key, weight_dir)
            print(f"Using checkpoint file: {checkpoint_file_path}")
            
            # init_detector expects string paths
            model_instance = init_detector(str(config_file_path), checkpoint_file_path, device=self.device)
            loaded_models[key] = model_instance
            print(f"Successfully loaded {key}")

        return loaded_models

    def set_model_name(self, name: str) -> None:
        if name in self.models:
            self.model_name = name
        elif name in self._load_models.model_dict: # Access original dict for potential lazy loading
             # Basic lazy loading: if model wasn't loaded initially but is known
            print(f"Lazy loading model: {name}")
            # This part needs to be careful not to reload already loaded models or re-download.
            # For simplicity, the current _load_models loads the two main ones or a specified default.
            # If you need more dynamic loading, _load_models would need refinement.
            # For now, we assume it's loaded if it was one of the initially targeted ones.
            item_dict = self._load_models.model_dict[name]
            weight_dir = pathlib.Path(__file__).parent.resolve() / "weights"
            config_file_path = str(pathlib.Path(__file__).parent.resolve() / item_dict["config"])
            checkpoint_file_path = self._load_models._download_and_extract(name, weight_dir) # Access nested helper
            
            model_instance = init_detector(config_file_path, checkpoint_file_path, device=self.device)
            self.models[name] = model_instance
            self.model_name = name
            print(f"Successfully lazy-loaded {name}")
        else:
            print(f"Warning: Model name {name} not found in available models.", file=sys.stderr)


    def detect_and_visualize(self, image: np.ndarray, score_threshold: float) -> tuple[list[np.ndarray] | tuple, np.ndarray]:
        # inference_detector returns List[np.ndarray] for detection or Tuple[List[np.ndarray], List[np.ndarray]] for segmentation
        detection_results = self.detect(image) 
        vis_image = self.visualize_detection_results(image.copy(), detection_results, score_threshold)
        return detection_results, vis_image

    def detect(self, image: np.ndarray) -> list[np.ndarray] | tuple:
        # Ensure image is BGR for MMDetection
        image_bgr = image[:, :, ::-1] if image.shape[2] == 3 else image 
        model = self.models[self.model_name]
        out = inference_detector(model, image_bgr)
        return out

    def visualize_detection_results(
        self, image: np.ndarray, detection_results: list[np.ndarray] | tuple, score_threshold: float = 0.3
    ) -> np.ndarray:
        # Ensure image is BGR for MMDetection's show_result
        image_bgr = image[:, :, ::-1] if image.shape[2] == 3 else image
        model = self.models[self.model_name]
        
        # Handle if detection_results is a tuple (bbox_results, segm_results)
        # model.show_result can often handle this tuple directly.
        vis = model.show_result(
            image_bgr,
            detection_results, # Pass potentially tuple result
            score_thr=score_threshold,
            bbox_color=None,    # Use MMDetection's default palette
            text_color=(200, 200, 200), # Or your preferred color
            mask_color=None,    # Use MMDetection's default palette
            show=False # Important: do not let it call plt.show()
        )
        return vis[:, :, ::-1]  # Convert BGR output from show_result to RGB for Streamlit
