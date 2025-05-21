import cv2
import numpy as np
import onnxruntime
import os
import requests
from tqdm import tqdm
import math
import onnx
import sys
import subprocess
import tempfile
import shutil
import concurrent.futures
import threading
import gc # For garbage collection
from typing import Any, Dict, Generator, List, Optional, Tuple, Literal

# Configuration & Model Paths
MODEL_DIR = ".assets/models"
os.makedirs(MODEL_DIR, exist_ok=True)

SwapperModelType = Literal["inswapper", "simswap"]

MODELS_TO_DOWNLOAD = {
    "face_detector_retinaface": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx",
        "path": os.path.join(MODEL_DIR, "retinaface_10g.onnx")
    },
    "face_landmarker_2dfan4": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx",
        "path": os.path.join(MODEL_DIR, "2dfan4.onnx")
    },
    "face_recognizer_arcface": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx",
        "path": os.path.join(MODEL_DIR, "arcface_w600k_r50.onnx")
    },
    "face_swapper_inswapper_128": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
        "path": os.path.join(MODEL_DIR, "inswapper_128.onnx")
    },
    "face_swapper_simswap_256": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx",
        "path": os.path.join(MODEL_DIR, "simswap_256.onnx")
    },
    "arcface_converter_simswap": { # Converter for SimSwap
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_converter_simswap.onnx",
        "path": os.path.join(MODEL_DIR, "arcface_converter_simswap.onnx")
    },
    "face_enhancer_gfpgan_1.4": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
        "path": os.path.join(MODEL_DIR, "gfpgan_1.4.onnx")
    },
    "face_occluder_xseg1": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx",
        "path": os.path.join(MODEL_DIR, "xseg_1.onnx")
    },
    "face_parser_bisenet_resnet34": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx",
        "path": os.path.join(MODEL_DIR, "bisenet_resnet_34.onnx")
    }
}

FACE_MASK_REGION_MAP = {
    'skin': 1, 'left-eyebrow': 2, 'right-eyebrow': 3,
    'left-eye': 4, 'right-eye': 5, 'glasses': 6,
    'nose': 10, 'mouth': 11, 'upper-lip': 12, 'lower-lip': 13
}
# Base regions for swapping, eye protection will modify this list
BASE_SWAP_MASK_DESIRED_REGIONS = ['skin', 'nose', 'mouth', 'upper-lip', 'lower-lip', 'left-eyebrow', 'right-eyebrow']


# Model Download Utility
def verify_model_file(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        # Check if file is empty or too small
        file_size = os.path.getsize(model_path)
        if file_size < 1024:  # Less than 1KB is definitely wrong
            print(f"Model file is too small ({file_size} bytes): {model_path}")
            return False
            
        # Try to load the model to verify it's valid
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            return True
        except Exception as e:
            print(f"Failed to validate ONNX model {model_path}: {e}")
            return False
            
    except Exception as e:
        print(f"Error verifying model file {model_path}: {e}")
        return False

def download_model(model_info):
    url = model_info["url"]
    path = model_info["path"]
    name = os.path.basename(path)
    
    # If file exists, verify it first
    if os.path.exists(path):
        if verify_model_file(path):
            print(f"Model {name} already exists and is valid.")
            return True
        else:
            print(f"Existing model {name} is invalid. Re-downloading...")
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to remove invalid model file: {e}")
                return False
    
    print(f"Downloading {name} from {url}...")
    temp_path = path + ".temp"
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(temp_path, 'wb') as f, tqdm(
            desc=name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
                
        if total_size != 0 and bar.n != total_size:
            print(f"ERROR: Download incomplete for {name}. Expected {total_size}, got {bar.n}")
            if os.path.exists(temp_path): os.remove(temp_path)
            return False
            
        # Verify the downloaded file
        if verify_model_file(temp_path):
            # If verification successful, move to final location
            if os.path.exists(path): os.remove(path)
            os.rename(temp_path, path)
            print(f"Downloaded and verified {name} successfully.")
            return True
        else:
            print(f"Downloaded file failed verification: {name}")
            if os.path.exists(temp_path): os.remove(temp_path)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download {name}: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error while downloading {name}: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return False

def download_all_models():
    all_successful = True
    for model_name, model_info in MODELS_TO_DOWNLOAD.items():
        if not download_model(model_info):
            print(f"Failed to download {model_name}. Subsequent operations might fail.")
            all_successful = False
    return all_successful

# FaceFusion Helper Functions (Adapted and Simplified)
WARP_TEMPLATE_SET = {
    'arcface_112_v1': np.array([
        [0.35473214, 0.45658929], [0.64526786, 0.45658929], [0.50000000, 0.61154464],
        [0.37913393, 0.77687500], [0.62086607, 0.77687500]]),
    'arcface_112_v2': np.array([
        [0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
        [0.37097589, 0.82469196], [0.63151696, 0.82325089]]),
    'arcface_128': np.array([
        [0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
        [0.38710391, 0.72160547], [0.61507734, 0.72034453]]),
    'ffhq_512': np.array([
        [0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
        [0.39308822, 0.72541100], [0.61150205, 0.72490465]])
}

def estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template_str, crop_size_tuple):
    normed_warp_template = WARP_TEMPLATE_SET[warp_template_str] * np.array(crop_size_tuple)
    affine_matrix, _ = cv2.estimateAffinePartial2D(face_landmark_5, normed_warp_template, method=cv2.RANSAC, ransacReprojThreshold=100)
    if affine_matrix is None:
         affine_matrix, _ = cv2.estimateAffinePartial2D(face_landmark_5, normed_warp_template, method=cv2.LMEDS)
    return affine_matrix

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, warp_template_str, crop_size_tuple):
    affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template_str, crop_size_tuple)
    if affine_matrix is None:
        print("Warning: Affine transform estimation failed in warp_face_by_face_landmark_5. Using fallback crop.")
        target_width, target_height = crop_size_tuple
        x_min, y_min = np.min(face_landmark_5, axis=0).astype(int)
        x_max, y_max = np.max(face_landmark_5, axis=0).astype(int)
        padding_x = int((x_max - x_min) * 0.1); padding_y = int((y_max - y_min) * 0.1)
        crop_x1 = max(0, x_min - padding_x); crop_y1 = max(0, y_min - padding_y)
        crop_x2 = min(temp_vision_frame.shape[1], x_max + padding_x)
        crop_y2 = min(temp_vision_frame.shape[0], y_max + padding_y)
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            print("Error: Fallback crop resulted in invalid dimensions.")
            return np.zeros((target_height, target_width, 3), dtype=np.uint8), None
        cropped_face = temp_vision_frame[crop_y1:crop_y2, crop_x1:crop_x2]
        return cv2.resize(cropped_face, (target_width, target_height), interpolation=cv2.INTER_AREA), None
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size_tuple, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop_vision_frame, affine_matrix

def paste_back(target_image, warped_content, mask_for_pasting, affine_matrix_used_for_warp):
    if affine_matrix_used_for_warp is None:
        print("Warning: Affine matrix is None in paste_back. Cannot perform paste.")
        return target_image
    inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix_used_for_warp)
    target_size_wh = (target_image.shape[1], target_image.shape[0])
    inv_warped_mask = cv2.warpAffine(mask_for_pasting, inverse_affine_matrix, target_size_wh).clip(0, 1)
    inv_warped_content = cv2.warpAffine(warped_content, inverse_affine_matrix, target_size_wh, borderMode=cv2.BORDER_REPLICATE)
    if inv_warped_mask.ndim == 2:
        inv_warped_mask_bgr = cv2.cvtColor(inv_warped_mask, cv2.COLOR_GRAY2BGR)
    else:
        inv_warped_mask_bgr = inv_warped_mask

    pasted_image = target_image.astype(np.float32) # Promote target to float for precision
    pasted_image = inv_warped_mask_bgr * inv_warped_content.astype(np.float32) + \
                   (1 - inv_warped_mask_bgr) * pasted_image
    return pasted_image.astype(np.uint8)

def create_static_box_mask(crop_size_hw_tuple, face_mask_blur_ratio, face_mask_padding_ltrb_percent):
    height, width = crop_size_hw_tuple
    blur_amount = int(width * 0.5 * face_mask_blur_ratio)
    blur_area = max(blur_amount // 2, 1) # Ensure blur_area is at least 1 if blur_amount > 0
    box_mask = np.ones(crop_size_hw_tuple, dtype=np.float32)
    pad_top_p, pad_right_p, pad_bottom_p, pad_left_p = face_mask_padding_ltrb_percent

    abs_pad_top    = max(blur_area if blur_amount > 0 else 0, int(height * pad_top_p / 100.0))
    abs_pad_right  = max(blur_area if blur_amount > 0 else 0, int(width  * pad_right_p / 100.0))
    abs_pad_bottom = max(blur_area if blur_amount > 0 else 0, int(height * pad_bottom_p / 100.0))
    abs_pad_left   = max(blur_area if blur_amount > 0 else 0, int(width  * pad_left_p / 100.0))

    if abs_pad_top > 0 and abs_pad_top < height : box_mask[:abs_pad_top, :] = 0
    if abs_pad_bottom > 0 and abs_pad_bottom < height: box_mask[height - abs_pad_bottom:, :] = 0
    if abs_pad_left > 0 and abs_pad_left < width: box_mask[:, :abs_pad_left] = 0
    if abs_pad_right > 0 and abs_pad_right < width: box_mask[:, width - abs_pad_right:] = 0

    if blur_amount > 0:
        k_size = blur_amount if blur_amount % 2 != 0 else blur_amount + 1
        box_mask = cv2.GaussianBlur(box_mask, (k_size, k_size), 0)
    return box_mask

def restrict_frame(vision_frame, resolution_wh_tuple):
    original_height, original_width = vision_frame.shape[:2]
    target_width, target_height = resolution_wh_tuple
    if original_height > target_height or original_width > target_width:
        scale = min(target_height / original_height, target_width / original_width)
        new_width = int(original_width * scale); new_height = int(original_height * scale)
        return cv2.resize(vision_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return vision_frame

def transform_points(points, matrix):
    points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.transform(points_reshaped, matrix)
    if transformed_points is None:
        print("Warning: cv2.transform returned None in transform_points.")
        return points
    return transformed_points.reshape(-1, 2)

def warp_face_by_translation(temp_vision_frame, translation_xy, scale, crop_size_wh_tuple):
    affine_matrix = np.array([[scale, 0, translation_xy[0]], [0, scale, translation_xy[1]]], dtype=np.float32)
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size_wh_tuple, borderMode=cv2.BORDER_REPLICATE)
    return crop_vision_frame, affine_matrix

# ONNX Session Initialization
providers = ['CUDAExecutionProvider']
detector_session, landmarker_session, recognizer_session = None, None, None
inswapper_session, simswap_session, simswap_arcface_converter_session = None, None, None
enhancer_session, occluder_session, parser_session = None, None, None
inswapper_matrix_global = None

def initialize_sessions_and_globals():
    global detector_session, landmarker_session, recognizer_session
    global inswapper_session, simswap_session, simswap_arcface_converter_session
    global enhancer_session, occluder_session, parser_session
    global inswapper_matrix_global
    print("Initializing ONNX Runtime sessions and globals...")
    
    # First verify all model files
    for model_name, model_info in MODELS_TO_DOWNLOAD.items():
        if not verify_model_file(model_info["path"]):
            print(f"Model file verification failed for {model_name}")
            if not download_model(model_info):
                raise RuntimeError(f"Failed to download or verify {model_name}")
    
    try:
        detector_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_detector_retinaface"]["path"], providers=providers)
        landmarker_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_landmarker_2dfan4"]["path"], providers=providers)
        recognizer_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_recognizer_arcface"]["path"], providers=providers)
        inswapper_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_swapper_inswapper_128"]["path"], providers=providers)
        simswap_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_swapper_simswap_256"]["path"], providers=providers)
        simswap_arcface_converter_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["arcface_converter_simswap"]["path"], providers=providers)
        enhancer_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_enhancer_gfpgan_1.4"]["path"], providers=providers)
        occluder_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_occluder_xseg1"]["path"], providers=providers)
        parser_session = onnxruntime.InferenceSession(MODELS_TO_DOWNLOAD["face_parser_bisenet_resnet34"]["path"], providers=providers)

        inswapper_model_path = MODELS_TO_DOWNLOAD["face_swapper_inswapper_128"]["path"]
        if os.path.exists(inswapper_model_path):
            onnx_model = onnx.load(inswapper_model_path)
            inswapper_matrix_global = onnx.numpy_helper.to_array(onnx_model.graph.initializer[-1])
            print("InSwapper matrix loaded.")
        else:
            print(f"WARNING: InSwapper model for matrix not found at {inswapper_model_path}. Matrix not loaded.")

        print("All ONNX sessions and globals initialized successfully.")
    except Exception as e:
        print(f"Error initializing ONNX sessions or globals: {e}")
        raise

def unload_models_and_clear_memory():
    global detector_session, landmarker_session, recognizer_session
    global inswapper_session, simswap_session, simswap_arcface_converter_session
    global enhancer_session, occluder_session, parser_session
    global inswapper_matrix_global

    print("Unloading ONNX Runtime sessions and clearing globals...")

    sessions_to_clear = {
        'detector_session': detector_session, 'landmarker_session': landmarker_session,
        'recognizer_session': recognizer_session, 'inswapper_session': inswapper_session,
        'simswap_session': simswap_session, 'simswap_arcface_converter_session': simswap_arcface_converter_session,
        'enhancer_session': enhancer_session, 'occluder_session': occluder_session,
        'parser_session': parser_session
    }

    for session_name, session_obj in sessions_to_clear.items():
        if session_obj is not None:
            # For ONNX Runtime, explicitly deleting the session object or setting to None is how it's typically handled.
            # There isn't a .close() or .release() method on InferenceSession.
            globals()[session_name] = None
            print(f"{session_name} unloaded.")
    
    del detector_session, landmarker_session, recognizer_session
    del inswapper_session, simswap_session, simswap_arcface_converter_session
    del enhancer_session, occluder_session, parser_session


    if inswapper_matrix_global is not None:
        inswapper_matrix_global = None
        print("inswapper_matrix_global cleared.")
    
    del inswapper_matrix_global

    # Attempt to clear CUDA cache if PyTorch was involved (not directly here, but good practice if it were)
    # For ONNXRuntime with CUDA, this is usually handled by the provider.
    # Example for PyTorch (if it were used):
    # try:
    #     import torch
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #         print("CUDA cache cleared (PyTorch).")
    # except ImportError:
    #     pass

    gc.collect() # Force garbage collection
    print("Memory cleanup attempted.")

# RetinaFace Detection (same as before)
def _create_static_anchors_retinaface(feature_stride, anchor_total_per_location, stride_height_map, stride_width_map):
    y_coords_grid, x_coords_grid = np.mgrid[:stride_height_map, :stride_width_map]
    anchors_yx = np.stack((y_coords_grid, x_coords_grid), axis=-1)
    anchors_xy = anchors_yx[:,:,::-1] # YX to XY
    anchors = (anchors_xy * feature_stride).reshape((-1, 2))
    anchors = np.repeat(anchors, anchor_total_per_location, axis=0)
    return anchors

def _distance_to_bounding_box_retinaface(anchor_points_xy, deltas_ltrb):
    x1 = anchor_points_xy[:, 0] - deltas_ltrb[:, 0]
    y1 = anchor_points_xy[:, 1] - deltas_ltrb[:, 1]
    x2 = anchor_points_xy[:, 0] + deltas_ltrb[:, 2]
    y2 = anchor_points_xy[:, 1] + deltas_ltrb[:, 3]
    return np.column_stack([x1, y1, x2, y2])

def _distance_to_face_landmark_5_retinaface(anchor_points_xy, landmark_deltas_xy_pairs):
    num_landmarks = landmark_deltas_xy_pairs.shape[1] // 2
    decoded_landmarks = np.zeros_like(landmark_deltas_xy_pairs) # Re-use shape for output
    for i in range(num_landmarks):
        decoded_landmarks[:, i*2]     = anchor_points_xy[:, 0] + landmark_deltas_xy_pairs[:, i*2]
        decoded_landmarks[:, i*2 + 1] = anchor_points_xy[:, 1] + landmark_deltas_xy_pairs[:, i*2 + 1]
    return decoded_landmarks.reshape(-1, num_landmarks, 2)

def detect_faces_retinaface(image_bgr, face_detector_size_str="640x640", score_threshold=0.5, nms_iou_threshold=0.4):
    if detector_session is None: raise ValueError("Detector session not initialized.")
    original_height, original_width = image_bgr.shape[:2]
    detector_width, detector_height = map(int, face_detector_size_str.split('x'))
    
    temp_vision_frame = restrict_frame(image_bgr, (detector_width, detector_height))
    current_height, current_width = temp_vision_frame.shape[:2]
    
    actual_scale_w = current_width / original_width
    actual_scale_h = current_height / original_height
    
    detect_input_frame = np.zeros((detector_height, detector_width, 3), dtype=np.uint8)
    pad_x = (detector_width - current_width) // 2
    pad_y = (detector_height - current_height) // 2
    detect_input_frame[pad_y:pad_y+current_height, pad_x:pad_x+current_width, :] = temp_vision_frame
    
    detect_input_tensor = (detect_input_frame.astype(np.float32) - 127.5) / 128.0
    detect_input_tensor = np.expand_dims(detect_input_tensor.transpose(2, 0, 1), axis=0)
    
    input_name = detector_session.get_inputs()[0].name
    detection_outputs = detector_session.run(None, {input_name: detect_input_tensor})
    
    all_bboxes, all_scores, all_landmarks_raw = [], [], []
    feature_strides = [8, 16, 32]; num_strides = len(feature_strides); anchor_total_per_location = 2
    
    for idx, stride in enumerate(feature_strides):
        scores_raw = detection_outputs[idx]
        bboxes_raw = detection_outputs[idx + num_strides]
        landmarks_raw = detection_outputs[idx + 2 * num_strides]
        
        current_scores = scores_raw[:, 0]
            
        keep_indices = np.where(current_scores >= score_threshold)[0]
        if len(keep_indices) == 0: continue
        
        stride_height_map = math.ceil(detector_height / stride)
        stride_width_map = math.ceil(detector_width / stride)
        anchors_for_stride = _create_static_anchors_retinaface(stride, anchor_total_per_location, stride_height_map, stride_width_map)
        anchors_for_stride = anchors_for_stride[:scores_raw.shape[0]]
        
        selected_anchors = anchors_for_stride[keep_indices]
        selected_bboxes_deltas = bboxes_raw[keep_indices]
        selected_landmarks_deltas = landmarks_raw[keep_indices]
        
        decoded_bboxes = _distance_to_bounding_box_retinaface(selected_anchors, selected_bboxes_deltas * stride)
        decoded_landmarks_pts = _distance_to_face_landmark_5_retinaface(selected_anchors, selected_landmarks_deltas * stride)
        
        for bbox in decoded_bboxes:
            x1 = (bbox[0] - pad_x) / actual_scale_w; y1 = (bbox[1] - pad_y) / actual_scale_h
            x2 = (bbox[2] - pad_x) / actual_scale_w; y2 = (bbox[3] - pad_y) / actual_scale_h
            all_bboxes.append(np.array([max(0, x1), max(0, y1), min(original_width, x2), min(original_height, y2)]))

        for landmark_set in decoded_landmarks_pts:
            landmarks_orig = landmark_set.copy()
            landmarks_orig[:, 0] = (landmarks_orig[:, 0] - pad_x) / actual_scale_w
            landmarks_orig[:, 1] = (landmarks_orig[:, 1] - pad_y) / actual_scale_h
            landmarks_orig[:, 0] = np.clip(landmarks_orig[:, 0], 0, original_width - 1)
            landmarks_orig[:, 1] = np.clip(landmarks_orig[:, 1], 0, original_height - 1)
            all_landmarks_raw.append(landmarks_orig)
            
        all_scores.extend(current_scores[keep_indices])
        
    if not all_bboxes: return [], [], []
    
    final_bboxes_xywh = [[bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]] for bb in all_bboxes]
    all_scores_np = np.array(all_scores, dtype=np.float32)

    nms_indices = cv2.dnn.NMSBoxes(final_bboxes_xywh, all_scores_np, score_threshold, nms_iou_threshold)
    
    if isinstance(nms_indices, tuple):
        nms_indices = np.array([], dtype=int) if not nms_indices else nms_indices[0]
    if nms_indices is None: 
        nms_indices = np.array([], dtype=int)
    if nms_indices.ndim > 1:
        nms_indices = nms_indices.flatten()

    final_bboxes_out = [all_bboxes[i] for i in nms_indices]
    final_scores_out = [all_scores[i] for i in nms_indices]
    final_landmarks_out = [all_landmarks_raw[i] for i in nms_indices]
    
    return final_bboxes_out, final_scores_out, final_landmarks_out

# 2DFAN Landmarker (same as before)
def detect_landmarks_2dfan_precise(temp_vision_frame, bbox_xyxy, face_angle=0): # face_angle not used here
    if landmarker_session is None: raise ValueError("2DFAN Landmarker session not initialized.")
    model_size_tuple_wh = (256, 256)
    bbox_tl_br = np.array(bbox_xyxy); bbox_wh = bbox_tl_br[2:] - bbox_tl_br[:2]
    
    if bbox_wh.min() <= 0:
        print("Warning: Bounding box for landmark detection has non-positive width or height.")
        return np.zeros((68, 2), dtype=np.float32) 

    scale = 195.0 / bbox_wh.max() 
    translation_xy = (np.array(model_size_tuple_wh) - (bbox_tl_br[2:] + bbox_tl_br[:2]) * scale) * 0.5
    
    crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation_xy, scale, model_size_tuple_wh)
    
    input_tensor = crop_vision_frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    ort_inputs = {landmarker_session.get_inputs()[0].name: input_tensor}
    prediction_outputs = landmarker_session.run(None, ort_inputs)
    
    face_landmark_68_heatmap_coords = prediction_outputs[0] 
    face_landmark_68_in_crop = face_landmark_68_heatmap_coords[0, :, :, :2].squeeze() 
    face_landmark_68_in_crop = (face_landmark_68_in_crop / 64.0) * 256.0 
    
    face_landmark_68_original = transform_points(face_landmark_68_in_crop, cv2.invertAffineTransform(affine_matrix))
    return face_landmark_68_original.astype(np.float32)


# XSeg Occlusion Mask (same as before)
def create_occlusion_mask_xseg(face_crop_bgr_256x256):
    if occluder_session is None: raise ValueError("XSeg Occluder session not initialized.")
    input_name = occluder_session.get_inputs()[0].name
    
    img_normalized = face_crop_bgr_256x256.astype(np.float32) / 255.0
    input_tensor = img_normalized[np.newaxis, ...]
    
    ort_inputs = {input_name: input_tensor}
    occlusion_mask_raw = occluder_session.run(None, ort_inputs)[0]
    
    occlusion_mask = occlusion_mask_raw.squeeze().clip(0, 1)
    occlusion_mask_blurred = cv2.GaussianBlur(occlusion_mask, (0,0), sigmaX=5)
    occlusion_mask_final = (occlusion_mask_blurred.clip(0.5, 1.0) - 0.5) * 2.0 
    return occlusion_mask_final.clip(0, 1.0)

# BiSeNet Region Mask (same as before)
def create_region_mask_bisenet(face_crop_bgr_512x512, desired_regions_names: list):
    if parser_session is None: raise ValueError("BiSeNet Parser session not initialized.")
    
    img_rgb = cv2.cvtColor(face_crop_bgr_512x512, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_standardized = (img_normalized - mean) / std
    
    input_tensor = np.transpose(img_standardized, (2, 0, 1))[np.newaxis, ...]
    ort_inputs = {parser_session.get_inputs()[0].name: input_tensor}
    region_logits = parser_session.run(None, ort_inputs)[0]
    
    segmentation_map = np.argmax(region_logits.squeeze(), axis=0)
    target_class_indices = [FACE_MASK_REGION_MAP[region_name] for region_name in desired_regions_names if region_name in FACE_MASK_REGION_MAP]
    
    region_mask = np.isin(segmentation_map, target_class_indices).astype(np.float32)
    region_mask_blurred = cv2.GaussianBlur(region_mask, (0,0), sigmaX=5)
    region_mask_final = (region_mask_blurred.clip(0.5, 1.0) - 0.5) * 2.0
    return region_mask_final.clip(0, 1.0)


# ArcFace Embedding (same as before)
def get_arcface_embedding(aligned_face_112x112_bgr):
    if recognizer_session is None: raise ValueError("ArcFace Recognizer session not initialized.")
    
    img_normalized = (aligned_face_112x112_bgr.astype(np.float32) / 127.5) - 1.0
    img_rgb_normalized = img_normalized[:, :, ::-1] 
    input_tensor = img_rgb_normalized.transpose(2, 0, 1)[np.newaxis, ...] 
    
    ort_inputs = {recognizer_session.get_inputs()[0].name: input_tensor}
    embedding = recognizer_session.run(None, ort_inputs)[0].flatten()
    
    norm = np.linalg.norm(embedding)
    return (embedding / norm if norm != 0 else embedding).astype(np.float32)


# InSwapper Face Swapping (same as before)
def swap_face_inswapper(target_crop_128x128_bgr, source_embedding_512d_arcface):
    if inswapper_session is None: raise ValueError("InSwapper session not initialized.")
    if inswapper_matrix_global is None: raise ValueError("InSwapper matrix not initialized.")

    img_target_rgb = target_crop_128x128_bgr[:,:,::-1]
    img_target_normalized = (img_target_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    
    source_embedding_reshaped = source_embedding_512d_arcface.reshape(1, -1)
    transformed_source_embedding = np.dot(source_embedding_reshaped, inswapper_matrix_global)
    norm = np.linalg.norm(transformed_source_embedding)
    final_source_embedding = (transformed_source_embedding / norm if norm != 0 else transformed_source_embedding).astype(np.float32)
    
    ort_inputs = {
        inswapper_session.get_inputs()[0].name: img_target_normalized,
        inswapper_session.get_inputs()[1].name: final_source_embedding
    }
    swapped_face_normalized_rgb = inswapper_session.run(None, ort_inputs)[0][0].transpose(1,2,0)
    
    swapped_face_rgb = (swapped_face_normalized_rgb.clip(0, 1) * 255.0).astype(np.uint8)
    swapped_face_bgr = swapped_face_rgb[:,:,::-1]
    return swapped_face_bgr

# SimSwap Face Swapping
def get_simswap_source_embedding(source_embedding_512d_arcface):
    if simswap_arcface_converter_session is None: raise ValueError("SimSwap ArcFace Converter session not initialized.")
    embedding_reshaped = source_embedding_512d_arcface.reshape(1, -1) # Already float32 from get_arcface_embedding
    ort_inputs = {simswap_arcface_converter_session.get_inputs()[0].name: embedding_reshaped}
    converted_embedding = simswap_arcface_converter_session.run(None, ort_inputs)[0]
    return converted_embedding.astype(np.float32)

def swap_face_simswap(target_crop_256x256_bgr, source_embedding_simswap_converted):
    if simswap_session is None: raise ValueError("SimSwap session not initialized.")

    img_target_rgb = target_crop_256x256_bgr[:, :, ::-1] # BGR to RGB
    # Normalize with SimSwap's specific mean/std for target
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_target_normalized = (img_target_rgb.astype(np.float32) / 255.0 - mean) / std
    img_target_tensor = img_target_normalized.transpose(2, 0, 1)[np.newaxis, ...]

    ort_inputs = {
        simswap_session.get_inputs()[0].name: source_embedding_simswap_converted, # This is 'latent_id'
        simswap_session.get_inputs()[1].name: img_target_tensor              # This is 'target_image'
    }
    swapped_face_normalized_rgb = simswap_session.run(None, ort_inputs)[0][0].transpose(1, 2, 0) # NCHW to HWC

    # Denormalize (SimSwap outputs in [-1, 1] range generally, clip and scale)
    # As per FaceFusion, SimSwap denormalization is just clip to [0,1] and scale
    swapped_face_rgb = (swapped_face_normalized_rgb.clip(0, 1) * 255.0).astype(np.uint8)
    swapped_face_bgr = swapped_face_rgb[:, :, ::-1] # RGB to BGR
    return swapped_face_bgr


# GFPGAN Face Enhancement (same as before)
def enhance_face_gfpgan(face_crop_bgr_512x512):
    if enhancer_session is None: raise ValueError("GFPGAN Enhancer session not initialized.")
    
    img_rgb = face_crop_bgr_512x512[:, :, ::-1] 
    img_normalized = ((img_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2,0,1)[np.newaxis,...]
    
    ort_inputs = {enhancer_session.get_inputs()[0].name: img_normalized}
    enhanced_face_normalized_rgb = enhancer_session.run(None, ort_inputs)[0][0].transpose(1,2,0)
    
    enhanced_face_rgb = ((enhanced_face_normalized_rgb.clip(-1, 1) * 0.5 + 0.5) * 255.0).astype(np.uint8)
    enhanced_face_bgr_512 = enhanced_face_rgb[:,:,::-1]
    return enhanced_face_bgr_512

# Source Image Processing (same as before)
def get_source_face_embedding(source_img_bgr):
    print("Processing source image...")
    source_bboxes, _, source_all_landmarks_5pt = detect_faces_retinaface(source_img_bgr)
    if not source_bboxes: 
        print("No face detected in source image.")
        return None # Return None if no face detected
    
    source_landmarks_5pt_for_arcface = source_all_landmarks_5pt[0] 
    aligned_source_face_112, _ = warp_face_by_face_landmark_5(
        source_img_bgr, source_landmarks_5pt_for_arcface, "arcface_112_v2", (112, 112)
    )

    source_arcface_embedding = get_arcface_embedding(aligned_source_face_112)
    print("Source face ArcFace embedding extracted.")
    return source_arcface_embedding

# Main Swap Function with Model Choice and Eye Mask
def perform_face_swap(
    source_arcface_embedding, 
    target_img_bgr, 
    swapper_model_name: SwapperModelType = "inswapper",
    use_eye_mask: bool = False, # New argument
    cache_file_prefix=None
):
    if source_arcface_embedding is None:
        print("Error: Source embedding is None. Cannot perform face swap.")
        return target_img_bgr.copy()
        
    print(f"Processing target image using {swapper_model_name}...")
    
    # --- Cache Handling ---
    cached_data = {}
    cache_exists = False
    full_cache_file_path = None

    if cache_file_prefix is not None:
        full_cache_file_path = f"{cache_file_prefix}_{swapper_model_name}_eyes{int(use_eye_mask)}.npz"
        if os.path.exists(full_cache_file_path):
            print(f"Loading cached face data from {full_cache_file_path}...")
            try:
                loaded_data = np.load(full_cache_file_path, allow_pickle=True) # allow_pickle for SWAPPER_CROP_SIZE_WH
                if "empty_cache" in loaded_data:
                    print("Loaded empty cache - no valid faces in this image for current settings.")
                    return target_img_bgr.copy()
                
                # Validate essential keys
                required_keys = ["target_landmarks_5pt", "aligned_target_face_for_swap_bgr", 
                                 "affine_matrix_target_to_swapsize", "combined_mask_feathered_for_swap",
                                 "affine_matrix_enhancer_to_512", "final_mask_for_enhancer_paste_feathered_512",
                                 "SWAPPER_CROP_SIZE_WH_cached"]
                if not all(key in loaded_data for key in required_keys):
                    print("Cache file is missing required keys. Recomputing.")
                else:
                    cached_data = dict(loaded_data) # Convert NpzFile to dict for easier access
                    cache_exists = True
            except Exception as e:
                print(f"Error loading cache file {full_cache_file_path}: {e}. Recomputing.")
                cache_exists = False # Force recompute

    output_image = target_img_bgr.copy()

    if not cache_exists:
        target_bboxes, target_scores, target_all_landmarks_5pt = detect_faces_retinaface(target_img_bgr)
        if not target_bboxes: 
            print("No faces detected in target image.")
            if full_cache_file_path: np.savez(full_cache_file_path, empty_cache=np.array([True]))
            return target_img_bgr

        valid_faces = []
        image_center = np.array([target_img_bgr.shape[1] / 2, target_img_bgr.shape[0] / 2])
        
        for i, (bbox, score, landmarks) in enumerate(zip(target_bboxes, target_scores, target_all_landmarks_5pt)):
            face_width = bbox[2] - bbox[0]; face_height = bbox[3] - bbox[1]
            face_size = face_width * face_height
            face_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            distance_to_center = np.linalg.norm(face_center - image_center)
            
            if score > 0.7 and face_size > 20000:
                valid_faces.append({'index': i, 'bbox': bbox, 'landmarks': landmarks, 'distance_to_center': distance_to_center})
        
        if not valid_faces:
            print("No faces meet the criteria.")
            if full_cache_file_path: np.savez(full_cache_file_path, empty_cache=np.array([True]))
            return target_img_bgr
        
        most_centered_face = min(valid_faces, key=lambda x: x['distance_to_center'])
        target_landmarks_5pt = most_centered_face['landmarks']
        
        # --- Swapper Specific Parameters ---
        if swapper_model_name == "inswapper":
            SWAPPER_CROP_SIZE_WH = (128, 128)
            SWAPPER_ALIGN_TEMPLATE = "arcface_128"
        elif swapper_model_name == "simswap":
            SWAPPER_CROP_SIZE_WH = (256, 256)
            SWAPPER_ALIGN_TEMPLATE = "arcface_112_v1"
        else:
            raise ValueError(f"Unknown swapper model: {swapper_model_name}")

        aligned_target_face_for_swap_bgr, affine_matrix_target_to_swapsize = warp_face_by_face_landmark_5(
            target_img_bgr, target_landmarks_5pt, SWAPPER_ALIGN_TEMPLATE, SWAPPER_CROP_SIZE_WH
        )
        if affine_matrix_target_to_swapsize is None:
            print(f"Skipping face: Failed to align for swapper.")
            if full_cache_file_path: np.savez(full_cache_file_path, empty_cache=np.array([True]))
            return target_img_bgr

        # --- Masking for Swapper (at SWAPPER_CROP_SIZE_WH) ---
        # Occlusion Mask
        occlusion_input_crop_256 = cv2.resize(aligned_target_face_for_swap_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        occlusion_mask_float_256 = create_occlusion_mask_xseg(occlusion_input_crop_256)
        occlusion_mask_for_swap = cv2.resize(occlusion_mask_float_256, SWAPPER_CROP_SIZE_WH, interpolation=cv2.INTER_AREA)

        # Region Mask (adjusting for eye mask)
        current_swap_mask_regions = BASE_SWAP_MASK_DESIRED_REGIONS.copy()
        if use_eye_mask: # Protect eyes by removing them from the swap mask regions
            if 'left-eye' in current_swap_mask_regions: current_swap_mask_regions.remove('left-eye')
            if 'right-eye' in current_swap_mask_regions: current_swap_mask_regions.remove('right-eye')
        
        region_input_crop_512 = cv2.resize(aligned_target_face_for_swap_bgr, (512, 512), interpolation=cv2.INTER_AREA)
        region_mask_float_512 = create_region_mask_bisenet(region_input_crop_512, current_swap_mask_regions)
        region_mask_for_swap = cv2.resize(region_mask_float_512, SWAPPER_CROP_SIZE_WH, interpolation=cv2.INTER_AREA)
        
        # Box Mask
        box_mask_blur_ratio_swap = 0.15
        box_mask_padding_swap = (10, 10, 10, 10) 
        box_mask_for_swap = create_static_box_mask(SWAPPER_CROP_SIZE_WH[::-1], box_mask_blur_ratio_swap, box_mask_padding_swap) # HW format for create_static_box_mask

        combined_mask_for_swap = np.minimum.reduce([
            box_mask_for_swap, occlusion_mask_for_swap, region_mask_for_swap
        ]).clip(0,1)
        
        mask_feather_ksize_swap = int(SWAPPER_CROP_SIZE_WH[0] * 0.15) 
        if mask_feather_ksize_swap % 2 == 0: mask_feather_ksize_swap += 1
        if mask_feather_ksize_swap > 1:
            combined_mask_feathered_for_swap = cv2.GaussianBlur(combined_mask_for_swap, (mask_feather_ksize_swap, mask_feather_ksize_swap), 0)
        else:
            combined_mask_feathered_for_swap = combined_mask_for_swap

        # --- Prepare data for Enhancer step (even if it's a dummy swap initially) ---
        dummy_swap_crop = aligned_target_face_for_swap_bgr.copy()
        dummy_frame_with_raw_swap = paste_back(
            output_image.copy(), dummy_swap_crop, combined_mask_feathered_for_swap, affine_matrix_target_to_swapsize
        )
        
        _, affine_matrix_enhancer_to_512 = warp_face_by_face_landmark_5(
            dummy_frame_with_raw_swap, target_landmarks_5pt, "ffhq_512", (512, 512)
        )
        if affine_matrix_enhancer_to_512 is None:
             final_mask_for_enhancer_paste_feathered_512 = np.ones((512,512), dtype=np.float32) # Fallback
        else:
            enhancer_box_mask_512 = create_static_box_mask((512,512), 0.05, (0,0,0,0))
            occlusion_mask_for_enhancer_512 = cv2.resize(combined_mask_feathered_for_swap, (512,512), interpolation=cv2.INTER_AREA)
            final_mask_for_enhancer_paste_512 = np.minimum(enhancer_box_mask_512, occlusion_mask_for_enhancer_512).clip(0,1)
            
            enhancer_mask_feather_ksize_abs = int(512 * 0.05) 
            if enhancer_mask_feather_ksize_abs % 2 == 0: enhancer_mask_feather_ksize_abs += 1
            final_mask_for_enhancer_paste_feathered_512 = cv2.GaussianBlur(final_mask_for_enhancer_paste_512, (enhancer_mask_feather_ksize_abs, enhancer_mask_feather_ksize_abs), 0) if enhancer_mask_feather_ksize_abs > 1 else final_mask_for_enhancer_paste_512
        
        # Save computed data to cache
        if full_cache_file_path:
            print(f"Saving face data to cache: {full_cache_file_path}")
            np.savez(
                full_cache_file_path,
                target_landmarks_5pt=target_landmarks_5pt,
                aligned_target_face_for_swap_bgr=aligned_target_face_for_swap_bgr, # This is at SWAPPER_CROP_SIZE_WH
                affine_matrix_target_to_swapsize=affine_matrix_target_to_swapsize,
                combined_mask_feathered_for_swap=combined_mask_feathered_for_swap,
                affine_matrix_enhancer_to_512=affine_matrix_enhancer_to_512,
                final_mask_for_enhancer_paste_feathered_512=final_mask_for_enhancer_paste_feathered_512,
                SWAPPER_CROP_SIZE_WH_cached=np.array(SWAPPER_CROP_SIZE_WH) # Store to validate cache later
            )
    else: # Cache exists and was loaded
        target_landmarks_5pt = cached_data["target_landmarks_5pt"]
        aligned_target_face_for_swap_bgr = cached_data["aligned_target_face_for_swap_bgr"]
        affine_matrix_target_to_swapsize = cached_data["affine_matrix_target_to_swapsize"]
        combined_mask_feathered_for_swap = cached_data["combined_mask_feathered_for_swap"]
        affine_matrix_enhancer_to_512 = cached_data["affine_matrix_enhancer_to_512"]
        final_mask_for_enhancer_paste_feathered_512 = cached_data["final_mask_for_enhancer_paste_feathered_512"]
        SWAPPER_CROP_SIZE_WH_cached = tuple(cached_data["SWAPPER_CROP_SIZE_WH_cached"])

        # Validate cached crop size matches current model expectation
        current_swapper_crop_size = (128,128) if swapper_model_name == "inswapper" else (256,256)
        if SWAPPER_CROP_SIZE_WH_cached != current_swapper_crop_size:
            print("Cache mismatch for swapper crop size. Recomputing.")
            # This means we need to re-run the non-cached block for this frame.
            # For simplicity, we'll just return the original image and print a warning.
            # A more robust solution would re-run the 'if not cache_exists' block.
            # Or, simply delete the cache file and recall perform_face_swap.
            if full_cache_file_path and os.path.exists(full_cache_file_path):
                os.remove(full_cache_file_path)
            return perform_face_swap(source_arcface_embedding, target_img_bgr, swapper_model_name, use_eye_mask, cache_file_prefix) # Recurse once

    # --- Perform Actual Swap ---
    if swapper_model_name == "inswapper":
        swapped_face_crop_bgr = swap_face_inswapper(aligned_target_face_for_swap_bgr, source_arcface_embedding)
    elif swapper_model_name == "simswap":
        source_embedding_for_simswap = get_simswap_source_embedding(source_arcface_embedding)
        swapped_face_crop_bgr = swap_face_simswap(aligned_target_face_for_swap_bgr, source_embedding_for_simswap)
    else: # Should not happen due to earlier check
        raise ValueError(f"Unsupported swapper model in swap stage: {swapper_model_name}")

    frame_with_raw_swap = paste_back(
        output_image.copy(), swapped_face_crop_bgr, combined_mask_feathered_for_swap, affine_matrix_target_to_swapsize
    )
    print("Raw swap pasted.")

    # --- Enhancer ---
    if affine_matrix_enhancer_to_512 is None:
        print("Skipping enhancement: Failed to align for enhancer.")
        return frame_with_raw_swap
    
    face_crop_for_enhancer_512, _ = warp_face_by_face_landmark_5(
        frame_with_raw_swap, target_landmarks_5pt, "ffhq_512", (512, 512)
    )
    enhanced_face_gfpgan_512 = enhance_face_gfpgan(face_crop_for_enhancer_512)
    print("Face enhanced by GFPGAN.")
    
    pasted_enhanced_image = paste_back(
        frame_with_raw_swap.copy(), enhanced_face_gfpgan_512, final_mask_for_enhancer_paste_feathered_512, affine_matrix_enhancer_to_512
    )
    
    face_enhancer_blend_ratio = 0.8
    output_image = cv2.addWeighted(
        frame_with_raw_swap, 1.0 - face_enhancer_blend_ratio,
        pasted_enhanced_image, face_enhancer_blend_ratio, 0
    )
    print("Face enhanced and blended.")

    return output_image

# Video processing functions remain largely the same, but pass swapper_model_name and use_eye_mask
def perform_face_swap_video(source_arcface_embedding, target_video_path, output_path, 
                            swapper_model_name: SwapperModelType = "inswapper", use_eye_mask: bool = False,
                            temp_dir=None, cache_dir=None):
    print(f"Processing video: {target_video_path} with {swapper_model_name}, eye mask: {use_eye_mask}")
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    video = cv2.VideoCapture(target_video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {target_video_path}"); return False
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    actual_temp_dir = tempfile.mkdtemp() if temp_dir is None else temp_dir
    os.makedirs(actual_temp_dir, exist_ok=True)
    print(f"Using temporary directory: {actual_temp_dir}")
    
    try:
        frame_number = 0
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = video.read()
                if not ret: break
                
                cache_file = os.path.join(cache_dir, f"frame_{frame_number:06d}") if cache_dir else None
                processed_frame = perform_face_swap(source_arcface_embedding, frame, swapper_model_name, use_eye_mask, cache_file)
                
                frame_path = os.path.join(actual_temp_dir, f"frame_{frame_number:06d}.jpg")
                cv2.imwrite(frame_path, processed_frame if processed_frame is not None else frame)
                
                frame_number += 1
                pbar.update(1)
        
        video.release()
        print("Creating output video with ffmpeg...")
        input_pattern = os.path.join(actual_temp_dir, "frame_%06d.jpg")
        codec = 'libx264'; pix_fmt = 'yuv420p'

        ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', input_pattern,
                      '-c:v', codec, '-pix_fmt', pix_fmt, '-r', str(fps), output_path]
        
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video processing complete! Output saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error during video processing: {e}")
        return False
    finally:
        if os.path.exists(actual_temp_dir) and temp_dir is None: # Only remove if we created it
             print(f"Cleaning up temporary directory: {actual_temp_dir}")
             shutil.rmtree(actual_temp_dir)


def _process_frame_for_video_worker(frame_number, frame_data, output_frame_path, 
                                    source_arcface_embedding, swapper_model_name, use_eye_mask, 
                                    cache_dir_for_frames):
    try:
        cache_file_prefix = os.path.join(cache_dir_for_frames, f"frame_{frame_number:06d}") if cache_dir_for_frames else None
        processed_frame = perform_face_swap(source_arcface_embedding, frame_data, swapper_model_name, use_eye_mask, cache_file_prefix)
        cv2.imwrite(output_frame_path, processed_frame if processed_frame is not None else frame_data)
        return output_frame_path
    except Exception as e:
        print(f"Error processing frame {frame_number} in thread {threading.get_ident()}: {e}")
        try:
            if not os.path.exists(output_frame_path): 
                 cv2.imwrite(output_frame_path, frame_data)
        except Exception as e_save:
            print(f"Critical Error: Failed to save original frame {frame_number}: {e_save}")
        return None

def perform_face_swap_video_threaded(source_arcface_embedding, target_video_path, output_path, 
                                     swapper_model_name: SwapperModelType = "inswapper", use_eye_mask: bool = False,
                                     temp_dir_base=None, cache_dir=None, num_threads=10, 
                                     progress_callback=None):
    if source_arcface_embedding is None:
        print("Error: Source embedding is None."); return False
        
    print(f"Processing video: {target_video_path} using {swapper_model_name}, eye_mask={use_eye_mask}, threads={num_threads}.")

    if cache_dir: os.makedirs(cache_dir, exist_ok=True)
    
    video = cv2.VideoCapture(target_video_path)
    if not video.isOpened(): print(f"Error: Could not open video {target_video_path}"); return False

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {width}x{height}, {fps:.2f} fps")

    actual_temp_dir = tempfile.mkdtemp() if temp_dir_base is None else os.path.join(temp_dir_base, f"{os.path.splitext(os.path.basename(target_video_path))[0]}_temp_{os.getpid()}")
    os.makedirs(actual_temp_dir, exist_ok=True)
    print(f"Using temporary directory for frames: {actual_temp_dir}")

    futures = []; frame_number_counter = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            print("Reading frames and submitting to thread pool...")
            while True:
                ret, frame = video.read()
                if not ret: break
                
                output_frame_path = os.path.join(actual_temp_dir, f"frame_{frame_number_counter:06d}.jpg")
                futures.append(executor.submit(_process_frame_for_video_worker, 
                                         frame_number_counter, frame.copy(), output_frame_path, 
                                         source_arcface_embedding, swapper_model_name, use_eye_mask, cache_dir))
                frame_number_counter += 1
            
            if not futures: print("No frames submitted."); return False
            print(f"All {len(futures)} frames submitted. Waiting for completion...")
            
            total_frames_actual = len(futures); processed_count = 0
            if progress_callback: progress_callback(0.0, f"Starting processing of {total_frames_actual} frames...")
            
            with tqdm(total=total_frames_actual, desc="Processing frames (threaded)") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"A thread task raised an exception: {e}")
                    processed_count += 1; pbar.update(1)
                    if progress_callback: progress_callback(0.8 * (processed_count / total_frames_actual), f"Processing: {processed_count}/{total_frames_actual}")
            
            if processed_count != total_frames_actual: print(f"Warning: {total_frames_actual} tasks, but {processed_count} completed.")

        if progress_callback: progress_callback(0.8, "Assembling final video...")
        print("Assembling output video...")
        input_pattern = os.path.join(actual_temp_dir, "frame_%06d.jpg")
        
        saved_frames_check = [f for f in os.listdir(actual_temp_dir) if f.startswith('frame_') and f.endswith('.jpg')]
        if not saved_frames_check: 
            print(f"Error: No frames in {actual_temp_dir}."); 
        if progress_callback: 
            progress_callback(0.8, "Error: No frames."); return False

        codec = 'libx264'; pix_fmt = 'yuv420p'
        ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', str(fps if fps > 0 else 25), '-i', input_pattern,
                      '-c:v', codec, '-pix_fmt', pix_fmt, '-r', str(fps if fps > 0 else 25), output_path]
        
        try:
            if progress_callback: progress_callback(0.9, "Encoding final video...")
            process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            if progress_callback: progress_callback(1.0, "Video processing complete!")
            print(f"Video processing complete! Output: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFMPEG Error: {e.cmd}\n{e.returncode}\n{e.stdout}\n{e.stderr}")
            if progress_callback: progress_callback(0.9, f"FFMPEG Error: {e.returncode}")
            return False
    finally:
        if video.isOpened(): video.release()
        if actual_temp_dir and os.path.exists(actual_temp_dir) and temp_dir_base is None: # only remove if we created it fully
            print(f"Cleaning up temporary directory: {actual_temp_dir}")
            shutil.rmtree(actual_temp_dir)

# Main Execution
if __name__ == "__main__":
    print("Starting ADVANCED face swap script...")
    if not download_all_models():
        print("One or more models failed to download. Exiting.")
        sys.exit(1)
    
    try:
        initialize_sessions_and_globals()
    except Exception as e:
        print(f"Exiting due to ONNX session/globals initialization error: {e}")
        sys.exit(1)

    source_path = "source.png" # A clear, frontal face image
    target_image_path = "target.jpg"
    output_image_path = "output_swapped.jpg"
    
    target_video_path = "target_vid.mp4" # A short video for testing
    output_video_path = "output_swapped_video.mp4"

    swapper_choice: SwapperModelType = "inswapper" # or "simswap"
    protect_eyes = False # True to protect eyes, False to swap them as part of the face mask

    source_img_bgr = cv2.imread(source_path)
    if source_img_bgr is None:
        print(f"Error: Could not load source image from {source_path}"); sys.exit(1)

    source_arcface_embedding = get_source_face_embedding(source_img_bgr)
    if source_arcface_embedding is None:
        print("Could not get source face embedding. Exiting."); sys.exit(1)

    # Image Swap
    print(f"\n--- Swapping to Image ({swapper_choice}, Eye Mask: {protect_eyes}) ---")
    target_img_bgr = cv2.imread(target_image_path)
    if target_img_bgr is None:
        print(f"Error: Could not load target image from {target_image_path}"); sys.exit(1)
    
    output_image = perform_face_swap(source_arcface_embedding, target_img_bgr, 
                                     swapper_model_name=swapper_choice, 
                                     use_eye_mask=protect_eyes, 
                                     cache_file_prefix="image_cache") # One cache file per image+settings
    cv2.imwrite(output_image_path, output_image)
    print(f"Output image saved to {output_image_path}")

    # Video Swap
    print(f"\n--- Swapping to Video ({swapper_choice}, Eye Mask: {protect_eyes}) ---")
    if not os.path.exists(target_video_path):
        print(f"Target video {target_video_path} not found. Skipping video swap.")
    else:
        perform_face_swap_video_threaded(
            source_arcface_embedding, 
            target_video_path, 
            output_video_path,
            swapper_model_name=swapper_choice,
            use_eye_mask=protect_eyes,
            temp_dir_base="video_temp", # Optional: specify base for temp frame folders
            cache_dir="video_frame_cache", # Optional: specify base for frame data caches
            num_threads=4 # Adjust based on your CPU cores
        )

    # Example of unloading models
    print("\n--- Unloading models ---")
    unload_models_and_clear_memory()
    print("Models unloaded. Script finished.")