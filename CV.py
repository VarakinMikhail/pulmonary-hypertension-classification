import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import re
from tqdm import tqdm
import time
import itertools
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from skrebate import ReliefF
from scipy.stats import mode as scipy_mode
import warnings
import traceback

DATA_DIR = 'D://classification_archive'
NUM_CLASSES = 4
BATCH_SIZE = 16
INCEPTION_IMG_SIZE = 299
DEFAULT_IMG_SIZE = 224
NUM_WORKERS = 0
N_SPLITS_CV = 10
RANDOM_STATE = 42
N_FEATURES_TO_SELECT = 250
NCA_MAX_ITER = 300
RELIEFF_N_NEIGHBORS = 10
KNN_NEIGHBORS = 1
KNN_METRIC = 'cityblock'
SVM_KERNEL = 'poly'
SVM_DEGREE = 3
SVM_C = 1.0
IMV_SORTED_MIN_K = 2
N_TOP_CLASSIFIERS_FOR_TRIPLETS = 90
TRIPLET_SIZE = 3
GREEDY_ACC_TOLERANCE = 1e-7
LOG_FILE_PATH = "TopClassifiers.txt"

if torch.backends.mps.is_available(): DEVICE = torch.device("mps")
elif torch.cuda.is_available(): DEVICE = torch.device("cuda")
else: DEVICE = torch.device("cpu")
print(f"Using {DEVICE} device.")

def get_label_from_filename(filename):
    match = re.match(r"(\d+) \((\d+)\)\.png", filename)
    if match:
        label = int(match.group(1)) - 1
        return label if 0 <= label < NUM_CLASSES else None
    return None

class CTScanDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Returning blank image.")
            image = Image.new('RGB', (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), color=0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor

class FeatureExtractor(nn.Module):
    def __init__(self, base_model, extraction_type='fc'):
        super().__init__()
        self.base_model_name = base_model.__class__.__name__
        self.extraction_type = extraction_type
        self.base_model = base_model
        self.hook_handle = None
        self.features_out = None

        if self.base_model_name.startswith('VisionTransformer'):
            if extraction_type != 'vit_pre_head':
                 self.extraction_type = 'vit_pre_head'
            if not hasattr(base_model, '_process_input'): raise AttributeError(f"ViT model {self.base_model_name} lacks '_process_input'")
            if not hasattr(base_model, 'encoder'): raise AttributeError(f"ViT model {self.base_model_name} lacks 'encoder'")
            if not hasattr(base_model, 'class_token'): raise AttributeError(f"ViT model {self.base_model_name} lacks 'class_token'")
            self.class_token = base_model.class_token
            if not hasattr(base_model, 'hidden_dim'):
                 if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                      base_model.hidden_dim = base_model.config.hidden_size
                 else: raise AttributeError(f"ViT model {self.base_model_name} lacks 'hidden_dim' or 'config.hidden_size'")
            self._process_input = base_model._process_input
            self.encoder = base_model.encoder
            self.num_features = base_model.hidden_dim
            self.features = nn.Identity(); self.avgpool = nn.Identity(); self.flatten = nn.Identity(); self.dropout = nn.Identity()

        elif self.base_model_name.startswith('Inception'):
            if not hasattr(base_model, 'fc'): raise AttributeError(f"{self.base_model_name} lacks 'fc'")
            self.fc_layer = base_model.fc
            self.avgpool = nn.Identity(); self.dropout = nn.Identity(); self.flatten = nn.Flatten()
            layer_to_hook = None
            if hasattr(base_model, 'Mixed_7c'): layer_to_hook = base_model.Mixed_7c;
            else: print("Warning: Cannot find standard Mixed_7c for hook target.")
            if hasattr(base_model, 'avgpool') and isinstance(base_model.avgpool, nn.AdaptiveAvgPool2d): self.avgpool = base_model.avgpool;
            else: print(f"Warning: Using default AdaptiveAvgPool2d(1) for {self.base_model_name}."); self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if hasattr(base_model, 'dropout') and isinstance(base_model.dropout, nn.Dropout): self.dropout = base_model.dropout;

            if extraction_type == 'fc':
                if layer_to_hook is None: raise ValueError(f"Could not find a suitable layer to hook for {self.base_model_name} 'fc' extraction.")
                def hook_fn(module, input, output):
                    if isinstance(output, tuple): self.features_out = output[0].detach().clone()
                    elif isinstance(output, torch.Tensor): self.features_out = output.detach().clone()
                    else: print(f"Error: Hook received unexpected output type: {type(output)}"); self.features_out = None
                self.hook_handle = layer_to_hook.register_forward_hook(hook_fn)
                self.num_features = self.fc_layer.in_features
                self.features = nn.Identity()
            elif extraction_type == 'gap':
                 modules = list(base_model.children()); avgpool_idx = -1
                 if self.avgpool is not nn.Identity() and self.avgpool in modules:
                     try: avgpool_idx = modules.index(self.avgpool)
                     except ValueError: pass
                 if avgpool_idx == -1:
                     try: avgpool_idx = modules.index(self.fc_layer); print("Warning: Using fc layer index as boundary for GAP features.")
                     except ValueError: raise ValueError("Cannot determine feature boundary for Inception GAP.")
                 feature_modules = [];
                 for i in range(avgpool_idx):
                     module_classname = modules[i].__class__.__name__
                     if module_classname != 'AuxLogits': feature_modules.append(modules[i])
                     else: print(f"  Excluding AuxLogits found at index {i} for GAP")
                 self.features = nn.Sequential(*feature_modules)
                 self.avgpool = nn.AdaptiveAvgPool2d(1); self.flatten = nn.Flatten(); self.dropout = nn.Identity()
                 self.features.to(DEVICE)
                 dummy_input = torch.zeros(1, 3, INCEPTION_IMG_SIZE, INCEPTION_IMG_SIZE).to(DEVICE)
                 with torch.no_grad(): dummy_output = self.features(dummy_input)
                 if isinstance(dummy_output, tuple): dummy_output = dummy_output[0]
                 if len(dummy_output.shape) == 4: self.num_features = dummy_output.shape[1];
                 else: raise ValueError(f"Unexpected feature map shape for Inception GAP: {dummy_output.shape}")
            else: raise ValueError(f"Unsupported extraction_type '{extraction_type}' for Inception")

        else:
            classifier_module_found = None; self.features = None
            self.avgpool = nn.Identity(); self.dropout = nn.Identity(); self.flatten = nn.Flatten()
            if hasattr(base_model, 'features'): self.features = base_model.features
            elif hasattr(base_model, '_features'): self.features = base_model._features
            else:
                 modules = list(base_model.children()); classifier_attr_names = ['classifier', 'fc', 'head']; final_layer_idx = -1
                 for name in classifier_attr_names:
                     if hasattr(base_model, name):
                         classifier_module = getattr(base_model, name)
                         if modules and classifier_module is modules[-1]: final_layer_idx = len(modules) - 1; classifier_module_found = classifier_module;
                         break
                         try:
                             potential_idx = modules.index(classifier_module)
                             if potential_idx != -1 and len(modules) > 1: final_layer_idx = potential_idx; classifier_module_found = classifier_module;
                             break
                         except ValueError: pass
                 if final_layer_idx != -1: self.features = nn.Sequential(*modules[:final_layer_idx])
                 elif modules and isinstance(modules[-1], (nn.Linear, nn.Sequential)) and len(modules) > 1:
                     self.features = nn.Sequential(*modules[:-1]); classifier_module_found = modules[-1]
                 else: raise ValueError(f"Cannot determine feature block for {self.base_model_name}")

            if extraction_type == 'fc':
                if hasattr(base_model, 'avgpool'): self.avgpool = base_model.avgpool
                elif self.base_model_name.startswith('DenseNet'): self.avgpool = nn.AdaptiveAvgPool2d((1, 1));
                else:
                    self.avgpool = nn.AdaptiveAvgPool2d(1)
                final_classifier = None; classifier_to_inspect = classifier_module_found
                if not classifier_to_inspect:
                     if hasattr(base_model, 'classifier'): classifier_to_inspect = base_model.classifier
                     elif hasattr(base_model, 'fc'): classifier_to_inspect = base_model.fc
                     elif hasattr(base_model, 'head'): classifier_to_inspect = base_model.head
                if isinstance(classifier_to_inspect, nn.Linear): final_classifier = classifier_to_inspect
                elif isinstance(classifier_to_inspect, nn.Sequential):
                     linear_layers = [m for m in classifier_to_inspect.modules() if isinstance(m, nn.Linear)]
                     if linear_layers:
                         final_classifier = linear_layers[-1]
                         try:
                             seq_children = list(classifier_to_inspect.children()); final_linear_idx_in_seq = -1
                             for i in range(len(seq_children) - 1, -1, -1):
                                 if seq_children[i] is final_classifier: final_linear_idx_in_seq = i; break
                             if final_linear_idx_in_seq > 0 and isinstance(seq_children[final_linear_idx_in_seq - 1], nn.Dropout): self.dropout = seq_children[final_linear_idx_in_seq - 1];
                         except Exception as e:
                             pass
                if final_classifier:
                    self.num_features = final_classifier.in_features;
                    if self.dropout is nn.Identity() and hasattr(base_model, 'dropout') and isinstance(base_model.dropout, nn.Dropout): self.dropout = base_model.dropout;
                else:
                    self.features.to(DEVICE); self.avgpool.to(DEVICE); self.flatten.to(DEVICE)
                    img_size = INCEPTION_IMG_SIZE if self.base_model_name.startswith('Inception') else DEFAULT_IMG_SIZE
                    dummy_input = torch.zeros(1, 3, img_size, img_size).to(DEVICE)
                    with torch.no_grad():
                         dummy_features_out = self.features(dummy_input)
                         if isinstance(dummy_features_out, tuple): dummy_features_out = dummy_features_out[0]
                         dummy_pooled = self.avgpool(dummy_features_out)
                         dummy_output = self.flatten(dummy_pooled)
                    self.num_features = dummy_output.shape[1];
                    self.dropout = nn.Identity()

            elif extraction_type == 'gap':
                self.avgpool = nn.AdaptiveAvgPool2d(1); self.flatten = nn.Flatten(); self.dropout = nn.Identity()
                self.features.to(DEVICE)
                img_size = INCEPTION_IMG_SIZE if self.base_model_name.startswith('Inception') else DEFAULT_IMG_SIZE
                dummy_input = torch.zeros(1, 3, img_size, img_size).to(DEVICE)
                with torch.no_grad():
                    dummy_output = self.features(dummy_input)
                    if isinstance(dummy_output, tuple): dummy_output = dummy_output[0]
                if len(dummy_output.shape) == 4: self.num_features = dummy_output.shape[1];
                elif len(dummy_output.shape) == 3:
                    self.num_features = dummy_output.shape[-1]; self.avgpool = nn.Identity(); self.flatten = nn.Identity();
                elif len(dummy_output.shape) == 2:
                    self.num_features = dummy_output.shape[-1]; self.avgpool = nn.Identity(); self.flatten = nn.Identity();
                else: raise ValueError(f"Cannot infer GAP features from shape: {dummy_output.shape}")
            else: raise ValueError(f"Unknown extraction_type '{extraction_type}' for CNN model")

    def forward(self, x):
        try: model_device = next(self.parameters()).device
        except StopIteration: model_device = DEVICE
        x = x.to(model_device)

        if self.extraction_type == 'vit_pre_head':
            processed_x = self._process_input(x); n = processed_x.shape[0]
            batch_class_token = self.class_token.expand(n, -1, -1).to(processed_x.device)
            concatenated_x = torch.cat([batch_class_token, processed_x], dim=1)
            encoder_output = self.encoder(concatenated_x)
            if len(encoder_output.shape) == 3: out = encoder_output[:, 0]
            else: raise ValueError(f"Unexpected ViT encoder output shape: {encoder_output.shape}")

        elif self.base_model_name.startswith('Inception') and self.extraction_type == 'fc':
            if self.hook_handle is None: raise RuntimeError("Hook not registered for InceptionV3 'fc'")
            self.features_out = None; _ = self.base_model(x)
            if self.features_out is None: raise RuntimeError("Hook did not capture features for InceptionV3.")
            features_captured = self.features_out.to(model_device)
            out = self.avgpool(features_captured); out = self.flatten(out); out = self.dropout(out)

        else:
            features_out = self.features(x)
            if isinstance(features_out, tuple): features_out = features_out[0]
            if self.base_model_name.startswith('DenseNet') and self.extraction_type == 'fc':
                 out = nn.functional.relu(features_out, inplace=False)
                 out = self.avgpool(out); out = self.flatten(out)
            elif self.extraction_type == 'fc' or self.extraction_type == 'gap':
                 out = self.avgpool(features_out); out = self.flatten(out)
                 if self.extraction_type == 'fc': out = self.dropout(out)
            else: raise ValueError(f"Forward logic error: unknown/unhandled extraction_type '{self.extraction_type}'")
        return out

    def remove_hook(self):
        if self.hook_handle: self.hook_handle.remove(); self.hook_handle = None
    def __del__(self):
        try: self.remove_hook()
        except Exception: pass


print(f"Scanning directory: {DATA_DIR}")
if not os.path.isdir(DATA_DIR): raise ValueError(f"Директория '{DATA_DIR}' не найдена.")
all_image_files = []
all_labels = []
for filename in sorted(os.listdir(DATA_DIR)):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(DATA_DIR, filename)
        label = get_label_from_filename(filename)
        if label is not None: all_image_files.append(file_path); all_labels.append(label)
print(f"Found {len(all_image_files)} valid image files with labels.")
if not all_image_files: raise ValueError(f"В папке '{DATA_DIR}' не найдено валидных файлов.")
all_labels_np = np.array(all_labels)
all_image_files_np = np.array(all_image_files)

def create_custom_preprocess(target_size):
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

print("\nLoading models and defining BOTH preprocessing pipelines...")
models_dict = {}
transforms_dict_standard = {}
transforms_dict_custom = {}
model_input_sizes = {}

try:
    print("Loading EfficientNet B0...")
    effnet_weights = models.EfficientNet_B0_Weights.DEFAULT
    effnet_base = models.efficientnet_b0(weights=effnet_weights)
    effnet_preprocess_standard = effnet_weights.transforms()
    effnet_size = effnet_preprocess_standard.crop_size[0]
    effnet_preprocess_custom = create_custom_preprocess(effnet_size)
    models_dict['effnet'] = effnet_base
    transforms_dict_standard['effnet'] = effnet_preprocess_standard
    transforms_dict_custom['effnet'] = effnet_preprocess_custom
    model_input_sizes['effnet'] = effnet_size

    print("Loading DenseNet 201...")
    densenet_weights = models.DenseNet201_Weights.DEFAULT
    densenet_base = models.densenet201(weights=densenet_weights)
    densenet_preprocess_standard = densenet_weights.transforms()
    densenet_size = densenet_preprocess_standard.crop_size[0]
    densenet_preprocess_custom = create_custom_preprocess(densenet_size)
    models_dict['densenet'] = densenet_base
    transforms_dict_standard['densenet'] = densenet_preprocess_standard
    transforms_dict_custom['densenet'] = densenet_preprocess_custom
    model_input_sizes['densenet'] = densenet_size

    print("Loading Inception V3...")
    inception_weights = models.Inception_V3_Weights.IMAGENET1K_V1
    inception_base = models.inception_v3(weights=inception_weights)
    if hasattr(inception_base, 'aux_logits'): inception_base.aux_logits = False
    else: print("  Model does not have 'aux_logits' attribute (normal).")
    inception_preprocess_standard = inception_weights.transforms()
    inception_size = inception_preprocess_standard.crop_size[0]
    if inception_size != INCEPTION_IMG_SIZE:
        print(f"Warning: Inception standard preprocess size {inception_size} != configured {INCEPTION_IMG_SIZE}. Using {inception_size}.")
    inception_preprocess_custom = create_custom_preprocess(inception_size)
    models_dict['inception'] = inception_base
    transforms_dict_standard['inception'] = inception_preprocess_standard
    transforms_dict_custom['inception'] = inception_preprocess_custom
    model_input_sizes['inception'] = inception_size

    print("Loading ViT B 16 (SWAG Linear)...")
    vit_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    vit_base = models.vit_b_16(weights=vit_weights)
    vit_preprocess_standard = vit_weights.transforms()
    vit_size = vit_preprocess_standard.crop_size[0]
    vit_preprocess_custom = create_custom_preprocess(vit_size)
    models_dict['vit'] = vit_base
    transforms_dict_standard['vit'] = vit_preprocess_standard
    transforms_dict_custom['vit'] = vit_preprocess_custom
    model_input_sizes['vit'] = vit_size

except AttributeError as e:
    print(f"\nError: Failed to load models or weights ({e}). Check torchvision version.")
    traceback.print_exc(); raise RuntimeError("Failed to load models.") from e
except Exception as e:
    print(f"\nError during model loading: {e}")
    traceback.print_exc(); raise RuntimeError("Failed to load models.") from e

print("\nModels and both preprocessing pipelines loaded.")

print("\nInstantiating Feature Extractors...")
extractors = {}
feature_dims = {}
try:
    extractor_definitions = [
        ('densenet_fc', models_dict['densenet'], 'fc'),
        ('densenet_gap', models_dict['densenet'], 'gap'),
        ('effnet_fc', models_dict['effnet'], 'fc'),
        ('effnet_gap', models_dict['effnet'], 'gap'),
        ('inception_fc', models_dict['inception'], 'fc'),
        ('vit_prehead', models_dict['vit'], 'vit_pre_head'),
    ]

    f_key_map = {}
    f_desc_map_for_print = {}

    for i, (name, model, ext_type) in enumerate(extractor_definitions):
        f_key = f"f{i+1}"
        extractor_instance = FeatureExtractor(model, extraction_type=ext_type).to(DEVICE).eval()
        extractors[name] = extractor_instance
        feature_dims[name] = extractor_instance.num_features
        f_key_map[name] = f_key
        f_desc_map_for_print[f_key] = f"{name} (Features: {feature_dims[name]})"

except Exception as e:
    print(f"\nError during FeatureExtractor instantiation: {e}")
    traceback.print_exc(); raise RuntimeError("Failed to instantiate feature extractors.") from e

print("\n--- Feature Key Mapping (f-keys) ---")
for f_key, desc in sorted(f_desc_map_for_print.items(), key=lambda item: int(item[0][1:])):
    print(f"  {f_key}: {desc}")
print("------------------------------------")


def custom_collate_pil(batch):
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], 0)
    return images, labels

def extract_features_for_fold(
    image_files, labels, extractors_dict, feature_dims_dict,
    transforms_to_use, model_keys_map, device, batch_size, num_workers,
    pin_memory_flag, desc_prefix=""
):
    print(f"\n{desc_prefix} Starting feature extraction...")
    dataset = CTScanDataset(image_files, labels)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory_flag,
        collate_fn=custom_collate_pil
    )

    all_features_batches = {name: [] for name in extractors_dict.keys()}
    for extractor in extractors_dict.values(): extractor.eval()

    try:
        with torch.no_grad():
            pbar_extract = tqdm(dataloader, desc=f"{desc_prefix} Extracting", leave=False)
            batch_num = 0
            for pil_images_list, labels_batch in pbar_extract:
                batch_num += 1
                if not pil_images_list: continue

                for name, extractor in extractors_dict.items():
                    pbar_extract.set_postfix({'Extractor': name})
                    model_key = model_keys_map[name]
                    transform = transforms_to_use[model_key]

                    try:
                        inputs_tensor_list = [transform(img) for img in pil_images_list]
                        inputs_tensor = torch.stack(inputs_tensor_list)
                    except Exception as e:
                        print(f"\n!!! {desc_prefix} ERROR applying transform '{model_key}' batch {batch_num} for '{name}': {e}")
                        traceback.print_exc()
                        print(f"    Skipping batch {batch_num} for extractor {name} due to transform error.")
                        continue

                    expected_channels = 3
                    if inputs_tensor.shape[1] != expected_channels:
                         print(f"\n!!! {desc_prefix} CRITICAL ERROR: Tensor after transform '{model_key}' for '{name}' has {inputs_tensor.shape[1]} channels. Expected {expected_channels}. Skipping batch {batch_num}.")
                         continue

                    use_non_blocking = pin_memory_flag and device.type == 'cuda'
                    inputs_tensor = inputs_tensor.to(device, non_blocking=use_non_blocking)

                    try:
                        outputs = extractor(inputs_tensor)
                        features_np = outputs.cpu().numpy()

                        expected_dim = feature_dims_dict.get(name)
                        if expected_dim is None:
                            print(f"\n!!! {desc_prefix} WARNING: Cannot find expected dimension for '{name}'. Skipping validation.")
                        elif len(features_np.shape) != 2 or features_np.shape[1] != expected_dim:
                             print(f"\n!!! {desc_prefix} WARNING: Unexpected output shape for {name} batch {batch_num}. Got {features_np.shape}, Expected (batch_size, {expected_dim}).")
                             if len(features_np.shape) == 4 and features_np.shape[2:] == (1, 1) and features_np.shape[1] == expected_dim:
                                 features_np = features_np.reshape(features_np.shape[0], features_np.shape[1])
                                 print(f"    Reshaped to {features_np.shape}.")
                             else:
                                 print(f"    Could not fix shape mismatch. Skipping batch {batch_num} for extractor {name}.")
                                 continue

                        if features_np.shape[0] != len(pil_images_list):
                            print(f"\n!!! {desc_prefix} WARNING: Batch size mismatch for {name} batch {batch_num}. Input: {len(pil_images_list)}, Output: {features_np.shape[0]}. Skipping batch.")
                            continue

                        all_features_batches[name].append(features_np)

                    except Exception as e:
                        print(f"\n!!! {desc_prefix} ERROR during forward pass {name} batch {batch_num}: {e}")
                        print(f"   Input shape: {inputs_tensor.shape}, Device: {inputs_tensor.device}")
                        traceback.print_exc()
                        print(f"    Skipping batch {batch_num} for extractor {name} due to forward pass error.")
                        continue
        pbar_extract.close()

    except Exception as e:
         print(f"!!! {desc_prefix} Feature extraction failed catastrophically: {e}")
         traceback.print_exc()
         return {}
    finally:
        if 'inputs_tensor' in locals(): del inputs_tensor
        if 'outputs' in locals(): del outputs
        if 'features_np' in locals(): del features_np
        if 'inputs_tensor_list' in locals(): del inputs_tensor_list
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()


    print(f"{desc_prefix} Combining extracted features...")
    base_features_np = {}
    total_samples = len(labels)
    if not any(all_features_batches.values()):
        print(f"Warning: {desc_prefix} No features collected for any extractor.")
        return {}

    for name, feat_list in all_features_batches.items():
        if not feat_list:
             print(f"Warning: {desc_prefix} No features collected for extractor {name}. Skipping.")
             continue

        valid_feat_list = [arr for arr in feat_list if isinstance(arr, np.ndarray) and len(arr.shape) == 2]
        if not valid_feat_list:
            print(f"Warning: {desc_prefix} No valid 2D arrays for {name} after extraction. Skipping.");
            continue

        num_samples_in_batches = sum(arr.shape[0] for arr in valid_feat_list)
        if num_samples_in_batches != total_samples:
             print(f"Error combining {desc_prefix} features for {name}: Sample count mismatch. Expected {total_samples}, got {num_samples_in_batches} from batches. Skipping {name}.")
             continue

        try:
            combined_arr = np.concatenate(valid_feat_list, axis=0)
            expected_dim = feature_dims_dict.get(name)
            if combined_arr.shape[0] != total_samples: raise ValueError(f"Final sample count mismatch. Expected {total_samples}, got {combined_arr.shape[0]}.")
            if len(combined_arr.shape) != 2: raise ValueError(f"Final array is not 2D. Shape: {combined_arr.shape}")
            if expected_dim is not None and combined_arr.shape[1] != expected_dim: raise ValueError(f"Final dimension mismatch. Expected {expected_dim}, got {combined_arr.shape[1]}.")

            base_features_np[name] = combined_arr
        except Exception as e:
            print(f"Error during final concatenation for {desc_prefix} features for {name}: {e}");
            continue

    if not base_features_np:
        print(f"Warning: {desc_prefix} Combining features failed for all models.")
        return {}

    print(f"{desc_prefix} Successfully extracted and combined {len(base_features_np)} base feature sets.")
    return base_features_np

def combine_features_for_experiments(base_features_dict, f_key_map):
    if not base_features_dict: return {}, {}

    valid_feature_keys_original = sorted(
        [key for key in base_features_dict.keys() if key in f_key_map],
        key=lambda k: int(f_key_map[k][1:])
    )

    if not valid_feature_keys_original:
        print("Warning: No valid features found matching the f_key_map.")
        return {}, {}

    f_map_renamed = {f_key_map[key]: base_features_dict[key] for key in valid_feature_keys_original}
    f_indices_renamed = list(f_map_renamed.keys())
    original_to_fkey_map = {key: f_key_map[key] for key in valid_feature_keys_original}
    fkey_to_original_map = {v: k for k, v in original_to_fkey_map.items()}

    combined_features = {}
    combined_features_fkey_desc = {}

    for i, f_key_renamed in enumerate(f_indices_renamed):
        combo_id = i + 1
        combined_features[combo_id] = f_map_renamed[f_key_renamed]
        original_name = fkey_to_original_map[f_key_renamed]
        combined_features_fkey_desc[combo_id] = f_key_renamed

    comb_idx = len(f_indices_renamed) + 1
    min_combination_size = 2
    max_combination_size = len(f_indices_renamed)
    for k in range(min_combination_size, max_combination_size + 1):
        for combo_fkeys_renamed_tuple in itertools.combinations(f_indices_renamed, k):
            combo_fkeys_renamed = list(combo_fkeys_renamed_tuple)
            features_to_combine = [f_map_renamed[key] for key in combo_fkeys_renamed]

            try:
                first_dim = features_to_combine[0].shape[0]
                if not all(f.shape[0] == first_dim for f in features_to_combine): raise ValueError("Inconsistent sample count")
                if not all(len(f.shape) == 2 for f in features_to_combine): raise ValueError("Non-2D arrays")

                combined_features[comb_idx] = np.concatenate(features_to_combine, axis=1)
                combined_features_fkey_desc[comb_idx] = " + ".join(combo_fkeys_renamed)
                comb_idx += 1
            except Exception as e:
                 original_names_combined_for_error = sorted([fkey_to_original_map[key] for key in combo_fkeys_renamed])
                 print(f"Warning: Failed to combine {original_names_combined_for_error}: {e}. Skipping this combination.")


    return combined_features, combined_features_fkey_desc


def run_classification_pipeline(
    combined_features, combined_features_fkey_desc,
    train_index, val_index, y_train, y_val,
    fold_num, config, desc_prefix=""
):
    if not combined_features:
        print(f"{desc_prefix} No combined features to process.")
        return 0.0, None, {}

    print(f"{desc_prefix} Running classification pipeline...")
    num_train_samples = len(train_index)
    num_val_samples = len(val_index)

    fold_predictions_map = {}
    fold_results_details = {}
    config_counter = 0

    pbar_combo = tqdm(combined_features.items(), total=len(combined_features), desc=f"{desc_prefix} Fold {fold_num+1} Combos", leave=False)
    for combo_id, X_combo_full in pbar_combo:
        current_combo_fkey_desc = combined_features_fkey_desc[combo_id]
        pbar_combo.set_postfix({'Combo': f"{combo_id} ({current_combo_fkey_desc[:20]}..)"})

        if X_combo_full.shape[0] != (num_train_samples + num_val_samples):
             print(f" [Err c{combo_id}: Shape mismatch {X_combo_full.shape[0]} vs expected {num_train_samples + num_val_samples}]", end='|')
             continue

        X_train_combo, X_val_combo = X_combo_full[train_index], X_combo_full[val_index]

        selected_features_train = {}; selected_features_val = {}
        feature_selectors = {}; scalers = {}
        current_num_features = X_train_combo.shape[1]
        if current_num_features == 0:
            print(f" [Skip c{combo_id}: 0 features]", end='|')
            continue

        fs_name_nca = 'nca'
        try:
            scaler_nca = StandardScaler(); X_train_scaled_nca = scaler_nca.fit_transform(X_train_combo); X_val_scaled_nca = scaler_nca.transform(X_val_combo); scalers[fs_name_nca] = scaler_nca
            nca_k = min(config['N_FEATURES_TO_SELECT'], current_num_features, num_train_samples - 1)
            if nca_k < 1: raise ValueError(f"NCA k<1 ({nca_k}) for combo {combo_id} (num_feat={current_num_features}, train_samples={num_train_samples})")
            nca = NeighborhoodComponentsAnalysis(n_components=nca_k, random_state=config['RANDOM_STATE'], max_iter=config['NCA_MAX_ITER'], tol=1e-4, verbose=0)
            selected_features_train[fs_name_nca] = nca.fit_transform(X_train_scaled_nca, y_train); selected_features_val[fs_name_nca] = nca.transform(X_val_scaled_nca); feature_selectors[fs_name_nca] = nca
        except Exception as e: print(f" [NCA Err c{combo_id}: {e}]", end='|'); selected_features_train[fs_name_nca] = None; selected_features_val[fs_name_nca] = None

        fs_name_relief = 'relief'
        try:
            scaler_relief = StandardScaler(); X_train_scaled_relief = scaler_relief.fit_transform(X_train_combo); X_val_scaled_relief = scaler_relief.transform(X_val_combo); scalers[fs_name_relief] = scaler_relief
            relief_n_neighbors = min(config['RELIEFF_N_NEIGHBORS'], num_train_samples - 1)
            if relief_n_neighbors < 1: raise ValueError(f"ReliefF nn<1 ({relief_n_neighbors}) for combo {combo_id}")
            relief_k = min(config['N_FEATURES_TO_SELECT'], current_num_features)
            if relief_k < 1: raise ValueError(f"ReliefF k<1 ({relief_k}) for combo {combo_id}")
            relief = ReliefF(n_features_to_select=relief_k, n_neighbors=relief_n_neighbors, n_jobs=-1)
            relief.fit(X_train_scaled_relief, y_train)
            top_indices_relief = relief.top_features_[:relief_k]
            if len(top_indices_relief) == 0 : raise ValueError(f"ReliefF selected 0 feats (k={relief_k}) for combo {combo_id}.")
            selected_features_train[fs_name_relief] = X_train_scaled_relief[:, top_indices_relief]; selected_features_val[fs_name_relief] = X_val_scaled_relief[:, top_indices_relief]; feature_selectors[fs_name_relief] = top_indices_relief
        except Exception as e: print(f" [Relief Err c{combo_id}: {e}]", end='|'); selected_features_train[fs_name_relief] = None; selected_features_val[fs_name_relief] = None

        fs_name_chi2 = 'chi2'
        try:
            min_val_train = X_train_combo.min()
            if min_val_train < 0:
                scaler_chi2 = MinMaxScaler(feature_range=(1e-6, 1));
                X_train_scaled_chi2 = scaler_chi2.fit_transform(X_train_combo)
                X_val_scaled_chi2 = scaler_chi2.transform(X_val_combo)
                scalers[fs_name_chi2] = scaler_chi2
                if X_train_scaled_chi2.min() < 0:
                    raise ValueError(f"Negative values detected in Chi2 scaled training data for combo {combo_id} despite MinMaxScaler.")
            else:
                X_train_scaled_chi2 = X_train_combo.copy(); X_val_scaled_chi2 = X_val_combo.copy(); scalers[fs_name_chi2] = None

            chi2_k = min(config['N_FEATURES_TO_SELECT'], current_num_features)
            if chi2_k < 1: raise ValueError(f"Chi2 k<1 ({chi2_k}) for combo {combo_id}")
            selector_chi2 = SelectKBest(chi2, k=chi2_k)
            selected_features_train[fs_name_chi2] = selector_chi2.fit_transform(X_train_scaled_chi2, y_train); selected_features_val[fs_name_chi2] = selector_chi2.transform(X_val_scaled_chi2); feature_selectors[fs_name_chi2] = selector_chi2
        except ValueError as e:
            if "Input X must be non-negative" in str(e):
                 print(f" [Chi2 Skip c{combo_id}: Input contains negative values even after attempt to scale]", end='|');
            else:
                print(f" [Chi2 VErr c{combo_id}: {e}]", end='|');
            selected_features_train[fs_name_chi2] = None; selected_features_val[fs_name_chi2] = None
        except Exception as e: print(f" [Chi2 Err c{combo_id}: {e}]", end='|'); selected_features_train[fs_name_chi2] = None; selected_features_val[fs_name_chi2] = None

        for fs_name, X_fs_train in selected_features_train.items():
            if X_fs_train is None or selected_features_val[fs_name] is None: continue
            if X_fs_train.shape[0] != num_train_samples or X_fs_train.shape[1] == 0:
                print(f" [Skip Clf c{combo_id}, fs={fs_name}: invalid shape {X_fs_train.shape}]", end='|');
                continue
            if selected_features_val[fs_name].shape[0] != num_val_samples:
                 print(f" [Skip Clf c{combo_id}, fs={fs_name}: val shape mismatch {selected_features_val[fs_name].shape}]", end='|');
                 continue

            X_fs_val = selected_features_val[fs_name]; num_selected_features = X_fs_train.shape[1]

            clf_name_knn = 'knn'
            try:
                actual_knn_neighbors = min(config['KNN_NEIGHBORS'], num_train_samples)
                if actual_knn_neighbors < 1: raise ValueError(f"kNN k<1 ({actual_knn_neighbors})")
                knn = KNeighborsClassifier(n_neighbors=actual_knn_neighbors, metric=config['KNN_METRIC'], n_jobs=-1)
                knn.fit(X_fs_train, y_train); y_pred_knn = knn.predict(X_fs_val)

                if len(y_pred_knn) != num_val_samples:
                    raise ValueError(f"kNN prediction length mismatch: expected {num_val_samples}, got {len(y_pred_knn)}")

                config_counter += 1; config_id = f"config_{config_counter}"; acc_knn = accuracy_score(y_val, y_pred_knn)
                fold_predictions_map[config_id] = y_pred_knn
                fold_results_details[config_id] = {'fold': fold_num + 1, 'combo_id': combo_id, 'desc': current_combo_fkey_desc, 'fs': fs_name, 'clf': clf_name_knn, 'acc': acc_knn, 'num_features': num_selected_features, 'clf_params': {'k': actual_knn_neighbors, 'metric': config['KNN_METRIC']}}
            except Exception as e: print(f" [KNN Err c{combo_id} fs={fs_name}: {e}]", end='|')

            clf_name_svm = 'svm'
            try:
                scaler_svm = StandardScaler(); X_train_svm = scaler_svm.fit_transform(X_fs_train); X_val_svm = scaler_svm.transform(X_fs_val)
                svm = SVC(kernel=config['SVM_KERNEL'], degree=config['SVM_DEGREE'], C=config['SVM_C'], random_state=config['RANDOM_STATE'], probability=False, cache_size=500)
                svm.fit(X_train_svm, y_train); y_pred_svm = svm.predict(X_val_svm)

                if len(y_pred_svm) != num_val_samples:
                    raise ValueError(f"SVM prediction length mismatch: expected {num_val_samples}, got {len(y_pred_svm)}")

                config_counter += 1; config_id = f"config_{config_counter}"; acc_svm = accuracy_score(y_val, y_pred_svm)
                fold_predictions_map[config_id] = y_pred_svm
                fold_results_details[config_id] = {'fold': fold_num + 1, 'combo_id': combo_id, 'desc': current_combo_fkey_desc, 'fs': fs_name, 'clf': clf_name_svm, 'acc': acc_svm, 'num_features': num_selected_features, 'clf_params': {'kernel': config['SVM_KERNEL'], 'degree': config['SVM_DEGREE'], 'C': config['SVM_C']}}
            except Exception as e: print(f" [SVM Err c{combo_id} fs={fs_name}: {e}]", end='|')
    pbar_combo.close(); print()

    print(f"{desc_prefix} Fold {fold_num+1}: Evaluating {len(fold_results_details)} individual classifier results...")
    best_fold_acc = 0.0
    best_fold_config_id = None
    best_fold_details = {}
    best_fold_method = "None"

    if not fold_results_details:
        print(f"  {desc_prefix} Fold {fold_num+1}: No individual classifiers produced results.")
        return best_fold_acc, best_fold_config_id, fold_results_details

    classifiers_for_sorting = []
    for config_id, details in fold_results_details.items():
        preds = fold_predictions_map.get(config_id)
        is_valid_acc = isinstance(details.get('acc'), (int, float))
        is_valid_preds = isinstance(preds, np.ndarray) and preds.shape == (num_val_samples,)

        if is_valid_acc and is_valid_preds:
             classifiers_for_sorting.append({'id': config_id, 'acc': details['acc'], 'details': details})
        else:
            print(f"    Warning: {desc_prefix} Skipping invalid base classifier ID {config_id} (Acc: {details.get('acc')}, Pred shape: {preds.shape if isinstance(preds, np.ndarray) else 'N/A'}). Expected shape: {(num_val_samples,)}")

    if not classifiers_for_sorting:
        print(f"  {desc_prefix} Fold {fold_num+1}: No valid base classifier results found for any ensemble method.")
        return best_fold_acc, best_fold_config_id, fold_results_details

    sorted_classifiers = sorted(classifiers_for_sorting, key=lambda x: x['acc'], reverse=True)
    num_valid_classifiers = len(sorted_classifiers)
    print(f"  {desc_prefix} Found {num_valid_classifiers} valid base classifiers. Adding ranks...")

    for rank_idx, classifier_info in enumerate(sorted_classifiers):
        config_id_to_rank = classifier_info['id']
        rank = rank_idx + 1
        if config_id_to_rank in fold_results_details:
            fold_results_details[config_id_to_rank]['rank_in_fold'] = rank
        else:
             print(f"    Warning: Could not find config_id {config_id_to_rank} in fold_results_details to assign rank.")


    best_single_classifier = sorted_classifiers[0]
    best_fold_acc = best_single_classifier['acc']
    best_fold_config_id = best_single_classifier['id']
    best_fold_details = best_single_classifier['details']
    best_fold_method = "Single"
    print(f"  {desc_prefix} Initial Best (Single): Acc={best_fold_acc:.4f} (ID: {best_fold_config_id}, Desc: {best_single_classifier['details']['desc'][:20]}..., FS: {best_single_classifier['details']['fs']}, Clf: {best_single_classifier['details']['clf']})")

    best_imv_sorted_acc = -1.0
    best_imv_sorted_k = -1
    best_imv_sorted_config_id = None
    best_imv_sorted_details = {}

    if num_valid_classifiers >= config['IMV_SORTED_MIN_K']:
        print(f"  {desc_prefix} Running IMV (Sorted k) for k={config['IMV_SORTED_MIN_K']} to {num_valid_classifiers}...")
        sorted_pred_ids = [c['id'] for c in sorted_classifiers]
        try:
            preds_to_stack = []
            valid_ids_for_stacking = []
            for pid in sorted_pred_ids:
                if pid in fold_predictions_map:
                    preds = fold_predictions_map[pid]
                    if isinstance(preds, np.ndarray) and preds.shape == (num_val_samples,):
                        preds_to_stack.append(preds)
                        valid_ids_for_stacking.append(pid)
                    else:
                         print(f"    Warning: Skipping ID {pid} for IMV Sorted k stacking due to invalid preds shape {preds.shape if isinstance(preds, np.ndarray) else 'N/A'}.")
                else:
                    print(f"    Warning: Skipping ID {pid} for IMV Sorted k stacking as it's missing in predictions map.")

            if len(preds_to_stack) >= config['IMV_SORTED_MIN_K']:
                pred_matrix_sorted = np.array(preds_to_stack).T
                if pred_matrix_sorted.shape[0] != num_val_samples:
                    print(f"    !!! {desc_prefix} IMV Sorted k shape mismatch! Expected {num_val_samples} rows, got {pred_matrix_sorted.shape[0]}. Skipping.")
                else:
                    max_k_to_try = pred_matrix_sorted.shape[1]
                    for k_imv in range(config['IMV_SORTED_MIN_K'], max_k_to_try + 1):
                        current_preds = pred_matrix_sorted[:, :k_imv]
                        try:
                            mode_result, _ = scipy_mode(current_preds, axis=1)
                            y_pred_voted = mode_result.flatten()
                            if len(y_pred_voted) != num_val_samples:
                                print(f"    Warning: scipy.mode output length mismatch for k={k_imv}. Skipping.")
                                continue
                            acc_voted = accuracy_score(y_val, y_pred_voted)
                        except Exception as mode_exc:
                            print(f"    Warning: scipy.mode failed for k={k_imv}. Skipping this k. Error: {mode_exc}")
                            continue

                        imv_config_id = f"imv_sorted_k{k_imv}"
                        involved_classifier_ids = valid_ids_for_stacking[:k_imv]
                        imv_details = {'fold': fold_num + 1, 'combo_id': 'imv_sorted', 'desc': f'IMV Sorted Top {k_imv}', 'fs': 'imv_ensemble', 'clf': 'imv_majority_vote', 'acc': acc_voted, 'k': k_imv, 'involved_ids': involved_classifier_ids, 'num_features': 'N/A'}
                        fold_results_details[imv_config_id] = imv_details

                        if acc_voted > best_imv_sorted_acc:
                            best_imv_sorted_acc = acc_voted
                            best_imv_sorted_k = k_imv
                            best_imv_sorted_config_id = imv_config_id
                            best_imv_sorted_details = imv_details

                    if best_imv_sorted_config_id and best_imv_sorted_acc > best_fold_acc:
                        print(f"    {desc_prefix} IMV Sorted k Update: k={best_imv_sorted_k} -> New Overall Best Acc: {best_imv_sorted_acc:.4f} (beats {best_fold_acc:.4f})")
                        best_fold_acc = best_imv_sorted_acc
                        best_fold_config_id = best_imv_sorted_config_id
                        best_fold_details = best_imv_sorted_details
                        best_fold_method = "IMV Sorted k"
                    elif best_imv_sorted_config_id:
                        print(f"    {desc_prefix} Best IMV Sorted k result: Acc={best_imv_sorted_acc:.4f} (k={best_imv_sorted_k}), but didn't beat overall best ({best_fold_acc:.4f})")

            else:
                 print(f"  {desc_prefix} Skipping IMV Sorted k: Only {len(preds_to_stack)} valid classifiers available for stacking (Min required: {config['IMV_SORTED_MIN_K']}).")

        except Exception as e:
            print(f"    !!! {desc_prefix} Error during IMV Sorted k calculation: {e}"); traceback.print_exc()
    elif num_valid_classifiers > 0:
        print(f"  {desc_prefix} Skipping IMV Sorted k: Only {num_valid_classifiers} valid classifiers found (Min required: {config['IMV_SORTED_MIN_K']}).")


    best_greedy_acc = -1.0
    best_greedy_ensemble_ids = []
    best_greedy_config_id = None
    best_greedy_details = {}

    min_classifiers_needed_greedy = config['TRIPLET_SIZE']
    if num_valid_classifiers >= min_classifiers_needed_greedy:
        print(f"\n  {desc_prefix} Starting IMV (Semi-Greedy)...")

        n_top = min(config['N_TOP_CLASSIFIERS_FOR_TRIPLETS'], num_valid_classifiers)
        print(f"    Searching for the best triplet among top {n_top} classifiers...")
        top_n_classifier_ids = [c['id'] for c in sorted_classifiers[:n_top]]

        best_triplet_acc = -1.0
        best_triplet_ids = None

        if len(top_n_classifier_ids) >= config['TRIPLET_SIZE']:
            triplet_search_start_time = time.time()
            num_triplets_evaluated = 0
            triplet_iterator = itertools.combinations(top_n_classifier_ids, config['TRIPLET_SIZE'])
            total_triplets = sum(1 for _ in itertools.combinations(top_n_classifier_ids, config['TRIPLET_SIZE']))

            pbar_triplets = tqdm(triplet_iterator, total=total_triplets, desc=f"{desc_prefix} Triplet Search", leave=False)
            for triplet_ids_tuple in pbar_triplets:
                triplet_ids = list(triplet_ids_tuple)
                try:
                    preds_triplet = []
                    valid_triplet = True
                    for tid in triplet_ids:
                        preds = fold_predictions_map.get(tid)
                        if preds is None or preds.shape != (num_val_samples,):
                            valid_triplet = False
                            break
                        preds_triplet.append(preds)

                    if not valid_triplet: continue

                    matrix_triplet = np.array(preds_triplet).T
                    if matrix_triplet.shape[0] != num_val_samples: continue

                    mode_result, _ = scipy_mode(matrix_triplet, axis=1)
                    y_pred_triplet = mode_result.flatten()
                    if len(y_pred_triplet) != num_val_samples: continue

                    acc_triplet = accuracy_score(y_val, y_pred_triplet)
                    num_triplets_evaluated += 1

                    if acc_triplet > best_triplet_acc:
                        best_triplet_acc = acc_triplet
                        best_triplet_ids = triplet_ids
                        pbar_triplets.set_postfix({'BestAcc': f'{best_triplet_acc:.4f}'})


                except KeyError as ke: pass
                except Exception as triplet_exc: pass
            pbar_triplets.close()
            triplet_search_duration = time.time() - triplet_search_start_time
            print(f"    Triplet search finished in {triplet_search_duration:.2f}s. Successfully evaluated {num_triplets_evaluated} triplets.")

            if best_triplet_ids:
                print(f"    Best initial triplet found: IDs={best_triplet_ids}, Acc={best_triplet_acc:.4f}")

                print(f"    Starting semi-greedy expansion from the best triplet...")
                current_ensemble_ids = set(best_triplet_ids)
                current_best_greedy_acc = best_triplet_acc

                all_valid_classifier_ids = {c['id'] for c in sorted_classifiers}
                candidate_ids = list(all_valid_classifier_ids - current_ensemble_ids)

                greedy_start_time = time.time()
                iteration_num = 0
                while True:
                    iteration_num += 1
                    best_improvement = -float('inf')
                    best_candidate_to_add = None
                    candidate_loop_start_time = time.time()
                    num_candidates_checked = 0

                    if current_best_greedy_acc >= 1.0 - config['GREEDY_ACC_TOLERANCE']:
                         print(f"      Semi-Greedy Step {iteration_num}: Stopping expansion as accuracy ~1.0 ({current_best_greedy_acc:.4f}) achieved.")
                         break

                    if not candidate_ids:
                         print(f"      Semi-Greedy Step {iteration_num}: No more candidates to add.")
                         break

                    print(f"      Semi-Greedy Step {iteration_num}: Checking {len(candidate_ids)} candidates...")
                    for candidate_id in candidate_ids:
                        temp_ensemble_ids = list(current_ensemble_ids | {candidate_id})
                        try:
                            preds_temp = []
                            valid_temp_ensemble = True
                            for tid in temp_ensemble_ids:
                                preds = fold_predictions_map.get(tid)
                                if preds is None or preds.shape != (num_val_samples,):
                                    valid_temp_ensemble = False
                                    break
                                preds_temp.append(preds)

                            if not valid_temp_ensemble: continue

                            matrix_temp = np.array(preds_temp).T
                            if matrix_temp.shape[0] != num_val_samples: continue

                            mode_result, _ = scipy_mode(matrix_temp, axis=1)
                            y_pred_temp = mode_result.flatten()
                            if len(y_pred_temp) != num_val_samples: continue

                            acc_temp = accuracy_score(y_val, y_pred_temp)
                            num_candidates_checked +=1

                            improvement = acc_temp - current_best_greedy_acc

                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_candidate_to_add = candidate_id

                        except KeyError: pass
                        except Exception as greedy_exc: pass

                    if best_candidate_to_add is not None and best_improvement > config['GREEDY_ACC_TOLERANCE']:
                        new_acc = current_best_greedy_acc + best_improvement
                        print(f"      Semi-Greedy Step {iteration_num}: Adding classifier ID: {best_candidate_to_add}. Improvement: {best_improvement:+.4f}. New Acc: {new_acc:.4f} (Prev: {current_best_greedy_acc:.4f}).")
                        current_ensemble_ids.add(best_candidate_to_add)
                        candidate_ids.remove(best_candidate_to_add)
                        current_best_greedy_acc = new_acc

                        if current_best_greedy_acc >= 1.0 - config['GREEDY_ACC_TOLERANCE']:
                            print(f"      Semi-Greedy Step {iteration_num}: Stopping expansion as accuracy ~1.0 ({current_best_greedy_acc:.4f}) achieved after adding candidate.")
                            break

                    else:
                        print(f"      Semi-Greedy Step {iteration_num}: Semi-greedy expansion stopped. No candidate provided sufficient improvement (Best improvement found: {best_improvement:.4f}, Required > {config['GREEDY_ACC_TOLERANCE']}).")
                        break

                greedy_duration = time.time() - greedy_start_time
                print(f"    Semi-Greedy expansion finished in {greedy_duration:.2f}s after {iteration_num-1} additions.")
                best_greedy_acc = current_best_greedy_acc
                best_greedy_ensemble_ids = list(current_ensemble_ids)
                best_greedy_config_id = f"imv_semigreedy_n{len(best_greedy_ensemble_ids)}"
                best_greedy_details = {
                    'fold': fold_num + 1, 'combo_id': 'imv_semigreedy',
                    'desc': f'IMV Semi-Greedy (n={len(best_greedy_ensemble_ids)})',
                    'fs': 'imv_ensemble', 'clf': 'imv_majority_vote',
                    'acc': best_greedy_acc, 'k': len(best_greedy_ensemble_ids),
                    'involved_ids': sorted(best_greedy_ensemble_ids),
                    'num_features': 'N/A'
                }
                fold_results_details[best_greedy_config_id] = best_greedy_details

                if best_greedy_config_id and best_greedy_acc > best_fold_acc:
                    print(f"    {desc_prefix} IMV Semi-Greedy Update: Ensemble size {len(best_greedy_ensemble_ids)} -> New Overall Best Acc: {best_greedy_acc:.4f} (beats {best_fold_acc:.4f})")
                    best_fold_acc = best_greedy_acc
                    best_fold_config_id = best_greedy_config_id
                    best_fold_details = best_greedy_details
                    best_fold_method = "IMV Semi-Greedy"
                elif best_greedy_config_id:
                     print(f"    {desc_prefix} Best IMV Semi-Greedy result: Acc={best_greedy_acc:.4f} (n={len(best_greedy_ensemble_ids)}), but didn't beat overall best ({best_fold_acc:.4f})")

            else:
                print(f"    Skipping semi-greedy expansion: No valid initial triplet found or triplet search failed.")
        else:
             print(f"    Skipping triplet search: Not enough classifiers in top {n_top} (need {config['TRIPLET_SIZE']}).")
    elif num_valid_classifiers > 0:
        print(f"  {desc_prefix} Skipping IMV Semi-Greedy: Only {num_valid_classifiers} valid classifiers found (Min required: {min_classifiers_needed_greedy}).")


    print(f"\n{desc_prefix} Fold {fold_num+1} Final Result:")
    print(f"  Best Overall Accuracy: {best_fold_acc:.4f}")
    print(f"  Achieved by Method:  {best_fold_method}")
    if best_fold_config_id and best_fold_details:
         print(f"  Winning Config ID:   {best_fold_config_id}")
         print(f"  Winning Config Desc: {best_fold_details.get('desc', 'N/A')}")
         if 'k' in best_fold_details: print(f"  Winning Ensemble Size (k): {best_fold_details['k']}")
         if 'clf' in best_fold_details and best_fold_method=="Single": print(f"  Winning Classifier: {best_fold_details.get('clf','N/A')} (FS: {best_fold_details.get('fs','N/A')})")
         if 'involved_ids' in best_fold_details and best_fold_method != "Single":
             print(f"  Winning Ensemble Components ({len(best_fold_details['involved_ids'])}):")
             for component_id in best_fold_details['involved_ids']:
                 comp_details = fold_results_details.get(component_id)
                 if comp_details:
                     comp_fkey_desc = comp_details.get('desc', 'N/A')
                     comp_rank = comp_details.get('rank_in_fold', 'N/A')

                     print(f"    - ID: {component_id}")
                     print(f"      Rank in Fold:        {comp_rank}")
                     print(f"      Features:            {comp_fkey_desc}")
                     print(f"      FS Method:           {comp_details.get('fs', 'N/A')}")
                     print(f"      Classifier:          {comp_details.get('clf', 'N/A')}")
                     print(f"      Individual Acc:      {comp_details.get('acc', -1):.4f}")
                 else:
                     print(f"    - ID: {component_id} (Details not found)")
    else:
        print("  No successful configuration identified as the best for this fold.")

    print(f"{desc_prefix} Classification pipeline finished.")
    return best_fold_acc, best_fold_config_id, fold_results_details


def log_ranked_classifiers_to_file(filepath, fold_num, preproc_desc, results_details):
    if not results_details:
        return

    ranked_items_list = []
    for config_id, details in results_details.items():
        if 'rank_in_fold' in details and config_id.startswith("config_"):
             ranked_items_list.append({'id': config_id, 'details': details})

    if not ranked_items_list:
        print(f"    Log Info: No ranked individual classifiers found for {preproc_desc} Fold {fold_num + 1} to log.")
        return

    sorted_classifiers = sorted(
        ranked_items_list,
        key=lambda x: x['details']['rank_in_fold']
    )

    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'='*15} Fold {fold_num + 1} - {preproc_desc} - Top Classifiers {'='*15}\n")
            f.write(f"(Found {len(sorted_classifiers)} ranked individual classifiers)\n")

            for classifier_info in sorted_classifiers:
                details = classifier_info['details']
                config_id = classifier_info['id']
                rank = details.get('rank_in_fold', 'N/A')
                acc = details.get('acc', -1.0)
                combo_desc = details.get('desc', 'N/A')
                fs = details.get('fs', 'N/A')
                clf = details.get('clf', 'N/A')
                params = details.get('clf_params', {})
                num_feat = details.get('num_features', 'N/A')

                f.write(f"\n--- Rank {rank} ---\n")
                f.write(f"  Config ID:                    {config_id}\n")
                f.write(f"  Accuracy:                     {acc:.4f}\n")
                f.write(f"  Feature Combination (f-keys): {combo_desc}\n")
                f.write(f"  Feature Selection:            {fs}\n")
                f.write(f"  Classifier:                   {clf}\n")
                f.write(f"  Classifier Params:            {params}\n")
                f.write(f"  Num Features Used:            {num_feat}\n")
            f.write(f"\n{'='*50}\n")
    except Exception as e:
        print(f"\n!!! ERROR: Failed to write ranked classifiers to log file '{filepath}' for Fold {fold_num+1} ({preproc_desc}). Error: {e}")
        traceback.print_exc()


if os.path.exists(LOG_FILE_PATH):
    try:
        os.remove(LOG_FILE_PATH)
        print(f"Cleared previous log file: {LOG_FILE_PATH}")
    except Exception as e:
        print(f"Warning: Could not clear previous log file {LOG_FILE_PATH}. Error: {e}")


print(f"\nStarting {N_SPLITS_CV}-Fold Cross-Validation with Preprocessing Comparison...")
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
fold_best_accuracies = []
all_fold_winning_results = {}

config = {
    'N_FEATURES_TO_SELECT': N_FEATURES_TO_SELECT, 'NCA_MAX_ITER': NCA_MAX_ITER,
    'RELIEFF_N_NEIGHBORS': RELIEFF_N_NEIGHBORS, 'KNN_NEIGHBORS': KNN_NEIGHBORS,
    'KNN_METRIC': KNN_METRIC, 'SVM_KERNEL': SVM_KERNEL, 'SVM_DEGREE': SVM_DEGREE,
    'SVM_C': SVM_C, 'IMV_SORTED_MIN_K': IMV_SORTED_MIN_K,
    'RANDOM_STATE': RANDOM_STATE,
    'N_TOP_CLASSIFIERS_FOR_TRIPLETS': N_TOP_CLASSIFIERS_FOR_TRIPLETS,
    'TRIPLET_SIZE': TRIPLET_SIZE,
    'GREEDY_ACC_TOLERANCE': GREEDY_ACC_TOLERANCE
}

extractor_to_model_key = {
    'densenet_fc': 'densenet', 'densenet_gap': 'densenet',
    'effnet_fc': 'effnet', 'effnet_gap': 'effnet',
    'inception_fc': 'inception',
    'vit_prehead': 'vit'
}


start_cv_time = time.time()
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

pin_memory_flag = True if DEVICE.type == 'cuda' else False

for fold, (train_index, val_index) in enumerate(skf.split(all_image_files_np, all_labels_np)):
    fold_start_time = time.time()
    print(f"\n{'='*15} Fold {fold + 1}/{N_SPLITS_CV} {'='*15}")
    y_train, y_val = all_labels_np[train_index], all_labels_np[val_index]
    X_files_train, X_files_val = all_image_files_np[train_index], all_image_files_np[val_index]
    print(f"Train samples: {len(train_index)}, Validation samples: {len(val_index)}")


    prefix_std = "[Standard Preproc]"
    print(f"\n{prefix_std} --- Running Feature Extraction ---")
    base_features_std = extract_features_for_fold(
        X_files_train, y_train, extractors, feature_dims,
        transforms_dict_standard, extractor_to_model_key,
        DEVICE, BATCH_SIZE, NUM_WORKERS, pin_memory_flag,
        desc_prefix=f"{prefix_std} Train"
    )
    base_features_val_std = extract_features_for_fold(
        X_files_val, y_val, extractors, feature_dims,
        transforms_dict_standard, extractor_to_model_key,
        DEVICE, BATCH_SIZE, NUM_WORKERS, pin_memory_flag,
        desc_prefix=f"{prefix_std} Val"
    )

    combined_features_full_std = {}
    combined_features_std, combined_features_fkey_desc_std = {}, {}
    if base_features_std and base_features_val_std:
        print(f"{prefix_std} Combining train/val features...")
        temp_full_features = {}
        all_keys = set(base_features_std.keys()) & set(base_features_val_std.keys())
        print(f"  Found {len(all_keys)} common feature keys for Standard Preproc.")
        if not all_keys:
             print(f"  Warning: No common feature keys found between train and val for Standard Preproc. Skipping fold.")
        else:
            for key in all_keys:
                 if base_features_std[key].shape[1] != base_features_val_std[key].shape[1]:
                     print(f"  Warning: Feature dimension mismatch for key '{key}' in Standard Preproc (Train: {base_features_std[key].shape[1]}, Val: {base_features_val_std[key].shape[1]}). Skipping key.")
                     continue
                 if base_features_std[key].shape[0] != len(train_index) or base_features_val_std[key].shape[0] != len(val_index):
                     print(f"  Warning: Sample count mismatch for key '{key}' in Standard Preproc (Train: {base_features_std[key].shape[0]} vs {len(train_index)}, Val: {base_features_val_std[key].shape[0]} vs {len(val_index)}). Skipping key.")
                     continue

                 num_total_samples = len(all_labels_np)
                 num_features = base_features_std[key].shape[1]
                 dtype_to_use = base_features_std[key].dtype if base_features_std[key].dtype == np.float64 else np.float32
                 full_arr = np.zeros((num_total_samples, num_features), dtype=dtype_to_use)
                 try:
                     full_arr[train_index] = base_features_std[key]
                     full_arr[val_index] = base_features_val_std[key]
                     temp_full_features[key] = full_arr
                 except IndexError as ie:
                      print(f"  Error assigning features for key '{key}': {ie}. Check train/val indices. Skipping key.")

            combined_features_std, combined_features_fkey_desc_std = combine_features_for_experiments(temp_full_features, f_key_map)
            del temp_full_features; gc.collect()
    else:
        print(f"{prefix_std} Feature extraction failed for train or validation set, skipping classification.")

    acc_std, config_id_std, results_details_std = run_classification_pipeline(
        combined_features_std, combined_features_fkey_desc_std,
        train_index, val_index, y_train, y_val, fold, config,
        desc_prefix=prefix_std
    )

    log_ranked_classifiers_to_file(LOG_FILE_PATH, fold, "Standard Preprocessing", results_details_std)

    del base_features_std, base_features_val_std, combined_features_std, combined_features_fkey_desc_std
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    print(f"{prefix_std} Finished. Best Accuracy: {acc_std:.4f}")


    acc_custom, config_id_custom, results_details_custom = -1.0, None, {}
    skip_custom = False
    if acc_std >= 1.0 - config['GREEDY_ACC_TOLERANCE']:
        print(f"\n[Fold {fold+1}] Standard Preprocessing achieved near-perfect accuracy ({acc_std:.4f}). Skipping Custom Preprocessing.")
        skip_custom = True

    if not skip_custom:
        prefix_custom = "[Custom Preproc]"
        print(f"\n{prefix_custom} --- Running Feature Extraction ---")
        base_features_custom = extract_features_for_fold(
            X_files_train, y_train, extractors, feature_dims,
            transforms_dict_custom, extractor_to_model_key,
            DEVICE, BATCH_SIZE, NUM_WORKERS, pin_memory_flag,
            desc_prefix=f"{prefix_custom} Train"
        )
        base_features_val_custom = extract_features_for_fold(
            X_files_val, y_val, extractors, feature_dims,
            transforms_dict_custom, extractor_to_model_key,
            DEVICE, BATCH_SIZE, NUM_WORKERS, pin_memory_flag,
            desc_prefix=f"{prefix_custom} Val"
        )

        combined_features_full_custom = {}
        combined_features_custom, combined_features_fkey_desc_custom = {}, {}
        if base_features_custom and base_features_val_custom:
            print(f"{prefix_custom} Combining train/val features...")
            temp_full_features = {}
            all_keys = set(base_features_custom.keys()) & set(base_features_val_custom.keys())
            print(f"  Found {len(all_keys)} common feature keys for Custom Preproc.")
            if not all_keys:
                 print(f"  Warning: No common feature keys found between train and val for Custom Preproc. Skipping fold.")
            else:
                for key in all_keys:
                    if base_features_custom[key].shape[1] != base_features_val_custom[key].shape[1]:
                        print(f"  Warning: Feature dimension mismatch for key '{key}' in Custom Preproc (Train: {base_features_custom[key].shape[1]}, Val: {base_features_val_custom[key].shape[1]}). Skipping key.")
                        continue
                    if base_features_custom[key].shape[0] != len(train_index) or base_features_val_custom[key].shape[0] != len(val_index):
                         print(f"  Warning: Sample count mismatch for key '{key}' in Custom Preproc (Train: {base_features_custom[key].shape[0]} vs {len(train_index)}, Val: {base_features_val_custom[key].shape[0]} vs {len(val_index)}). Skipping key.")
                         continue

                    num_total_samples = len(all_labels_np)
                    num_features = base_features_custom[key].shape[1]
                    dtype_to_use = base_features_custom[key].dtype if base_features_custom[key].dtype == np.float64 else np.float32
                    full_arr = np.zeros((num_total_samples, num_features), dtype=dtype_to_use)
                    try:
                        full_arr[train_index] = base_features_custom[key]
                        full_arr[val_index] = base_features_val_custom[key]
                        temp_full_features[key] = full_arr
                    except IndexError as ie:
                        print(f"  Error assigning features for key '{key}': {ie}. Check train/val indices. Skipping key.")

            combined_features_custom, combined_features_fkey_desc_custom = combine_features_for_experiments(temp_full_features, f_key_map)
            del temp_full_features; gc.collect()
        else:
            print(f"{prefix_custom} Feature extraction failed for train or validation set, skipping classification.")


        acc_custom, config_id_custom, results_details_custom = run_classification_pipeline(
            combined_features_custom, combined_features_fkey_desc_custom,
            train_index, val_index, y_train, y_val, fold, config,
            desc_prefix=prefix_custom
        )

        log_ranked_classifiers_to_file(LOG_FILE_PATH, fold, "Custom Preprocessing", results_details_custom)

        del base_features_custom, base_features_val_custom, combined_features_custom, combined_features_fkey_desc_custom
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        print(f"{prefix_custom} Finished. Best Accuracy: {acc_custom:.4f}")


    fold_end_time = time.time()
    print(f"\n--- Fold {fold + 1} Comparison & Selection ---")
    print(f"  Standard Preprocessing Best Accuracy: {acc_std:.4f} (Config: {config_id_std})")
    if not skip_custom:
        print(f"  Custom Preprocessing Best Accuracy:   {acc_custom:.4f} (Config: {config_id_custom})")
    else:
        print(f"  Custom Preprocessing Skipped.")


    winning_acc = 0.0
    winning_method_type = "None"
    winning_config_id = None
    winning_details_all = {}

    if acc_std >= acc_custom and acc_std > 0:
        winning_acc = acc_std
        winning_method_type = "Standard"
        winning_config_id = config_id_std
        winning_details_all = results_details_std if results_details_std else {}
        print(f"  Winner: Standard Preprocessing (Acc: {winning_acc:.4f})")
    elif acc_custom > acc_std and acc_custom > 0:
        winning_acc = acc_custom
        winning_method_type = "Custom"
        winning_config_id = config_id_custom
        winning_details_all = results_details_custom if results_details_custom else {}
        print(f"  Winner: Custom Preprocessing (Acc: {winning_acc:.4f})")
    else:
        print("  Winner: None (Both preprocessing methods failed or resulted in 0 accuracy)")
        if winning_method_type == "None" and results_details_std is not None:
             winning_details_all = results_details_std

    fold_best_accuracies.append(winning_acc)

    winning_config_details_specific = winning_details_all.get(winning_config_id) if winning_config_id else None

    all_fold_winning_results[f'fold_{fold+1}'] = {
        'winning_preprocessing': winning_method_type,
        'best_accuracy': winning_acc,
        'winning_config_id': winning_config_id,
        'winning_config_details': winning_config_details_specific,
        'all_run_results': winning_details_all.copy()
    }

    print(f"  Fold {fold+1} processing time: {fold_end_time - fold_start_time:.2f} sec")

    if 'results_details_std' in locals(): del results_details_std
    if 'results_details_custom' in locals(): del results_details_custom
    if 'winning_details_all' in locals(): del winning_details_all
    if 'winning_config_details_specific' in locals(): del winning_config_details_specific
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()


end_cv_time = time.time(); total_cv_time = end_cv_time - start_cv_time
print("\n{'='*15} Cross-Validation Final Results {'='*15}")
print(f"Total CV time: {total_cv_time:.2f} sec ({total_cv_time/60:.2f} min)")

valid_accuracies = [acc for acc in fold_best_accuracies if isinstance(acc, (int, float)) and acc > 0]

if valid_accuracies:
    mean_accuracy = np.mean(valid_accuracies)
    std_accuracy = np.std(valid_accuracies)
    print(f"\nMean Best Overall Accuracy across {len(valid_accuracies)} successful folds: {mean_accuracy:.4f}")
    print(f"Std Dev of Best Accuracy: {std_accuracy:.4f}")
    print("\n--- Detailed Results per Fold ---")
    for i, acc in enumerate(fold_best_accuracies):
        fold_key = f'fold_{i+1}'
        winner_info = all_fold_winning_results.get(fold_key)
        fold_status = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
        print(f"\n=== Fold {i+1} ===")
        print(f"  Best Accuracy: {fold_status}")

        if winner_info and winner_info['winning_config_details']:
            winning_prep = winner_info['winning_preprocessing']
            winning_details = winner_info['winning_config_details']
            winning_config_id = winner_info['winning_config_id']
            all_run_results_for_fold = winner_info.get('all_run_results', {})

            winning_method = "Unknown"
            config_id = winning_config_id or ""
            desc = winning_details.get('desc', '')
            if config_id.startswith("config_"): winning_method = "Single"
            elif config_id.startswith("imv_sorted_k"): winning_method = "IMV Sorted k"
            elif config_id.startswith("imv_semigreedy_n"): winning_method = "IMV Semi-Greedy"
            elif 'IMV Sorted' in desc: winning_method = "IMV Sorted k"
            elif 'IMV Semi-Greedy' in desc: winning_method = "IMV Semi-Greedy"
            elif winning_details.get('clf') and winning_details.get('fs'): winning_method = "Single"

            print(f"  Winning Preprocessing: {winning_prep}")
            print(f"  Winning Method: {winning_method}")
            print(f"  Winning Config ID: {winning_config_id}")
            print(f"  Winning Config Description (f-keys): {winning_details.get('desc', 'N/A')}")


            if winning_method != "Single" and 'involved_ids' in winning_details:
                involved_ids = winning_details['involved_ids']
                print(f"  Winning Ensemble Size: {len(involved_ids)}")
                print(f"  Winning Ensemble Components:")
                sorted_component_ids = sorted(
                    involved_ids,
                    key=lambda cid: all_run_results_for_fold.get(cid, {}).get('rank_in_fold', float('inf'))
                )

                for component_id in sorted_component_ids:
                    comp_details = all_run_results_for_fold.get(component_id)
                    if comp_details:
                        comp_fkey_desc = comp_details.get('desc', 'N/A')
                        comp_rank = comp_details.get('rank_in_fold', 'N/A')

                        print(f"    ---------------------------------")
                        print(f"    Component ID: {component_id}")
                        print(f"      Rank in Fold:                 {comp_rank}")
                        print(f"      Feature Combination (f-keys): {comp_fkey_desc}")
                        print(f"      Feature Selection:            {comp_details.get('fs', 'N/A')}")
                        print(f"      Classifier:                   {comp_details.get('clf', 'N/A')}")
                        print(f"      Classifier Params:            {comp_details.get('clf_params', {})}")
                        print(f"      Individual Accuracy:          {comp_details.get('acc', -1):.4f}")
                        print(f"      Num Features Used:            {comp_details.get('num_features', 'N/A')}")
                    else:
                        print(f"    - Component ID: {component_id} (Details not found in run results)")
                print(f"    ---------------------------------")

            elif winning_method == "Single":
                 print(f"  Winning Feature Combination (f-keys): {winning_details.get('desc', 'N/A')}")
                 print(f"  Winning Feature Selection:   {winning_details.get('fs', 'N/A')}")
                 print(f"  Winning Classifier:          {winning_details.get('clf', 'N/A')}")
                 print(f"  Winning Classifier Params:   {winning_details.get('clf_params', {})}")
                 print(f"  Num Features Used:           {winning_details.get('num_features', 'N/A')}")

        else:
             winning_prep = winner_info.get('winning_preprocessing', 'N/A') if winner_info else 'N/A'
             print(f"  Winning Preprocessing: {winning_prep}")
             print(f"  Winning Method: N/A (No valid configuration found for this fold)")
else:
    print("\nNo successful configurations found across any fold.")

print("\nCleaning up models and extractors from memory...")
hooks_removed_count = 0
if 'extractors' in locals():
     for name, extractor_obj in extractors.items():
         if hasattr(extractor_obj, 'remove_hook'):
             try: extractor_obj.remove_hook(); hooks_removed_count += 1
             except Exception: pass
     del extractors
     print(f"Final hook cleanup finished. Removed {hooks_removed_count} hooks.")
else: print("Extractors dictionary not found for cleanup.")

if 'models_dict' in locals(): del models_dict; print("Models dictionary deleted.")
if 'effnet_base' in locals(): del effnet_base;
if 'densenet_base' in locals(): del densenet_base;
if 'inception_base' in locals(): del inception_base;
if 'vit_base' in locals(): del vit_base;
if DEVICE.type == 'cuda': torch.cuda.empty_cache(); print("CUDA cache cleared.")
gc.collect()
print("Cleanup complete.")

print("\nScript finished.")