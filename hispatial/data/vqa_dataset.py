from __future__ import annotations

import io
import json
import os
import random
import re
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from webdataset import handlers
from webdataset.filters import pipelinefilter, reraise_exception

import utils3d
from typing import IO, Tuple, Union



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def read_depth(path: Union[str, IO]) -> Tuple[np.ndarray, float]:
    """Read a 16-bit PNG depth image and return (depth_float32, ...)."""
    if isinstance(path, str):
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = path.read()

    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get("near"))
    far = float(pil_image.info.get("far"))
    depth = np.array(pil_image)
    mask_nan = depth == 0
    mask_inf = depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth,


def _decode_pt(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


def _decode_json(b: bytes):
    return json.loads(b.decode("utf-8"))


def _decode_img_jpg(b: bytes):
    return Image.open(io.BytesIO(b)).convert("RGB")


def _decode_depth_png(b: bytes):
    return read_depth(io.BytesIO(b))


def _decode_depth_npy(b: bytes):
    return np.load(io.BytesIO(b))


def _not_none(x):
    return x is not None


def add_bbox_perturbation(xyxy, scale_range=(1.0, 1.2)):
    """Randomly scale a bbox around its center."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    sx = random.uniform(*scale_range)
    sy = random.uniform(*scale_range)
    new_w, new_h = w * sx, h * sy
    return [cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2]


# ---------------------------------------------------------------------------
# map_keep_none: like wds.map but keeps None results in the stream
# ---------------------------------------------------------------------------

def _map_keep_none(data, f, handler=reraise_exception):
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result


map_keep_none = pipelinefilter(_map_keep_none)


# ---------------------------------------------------------------------------
# safe_processor_wrapper
# ---------------------------------------------------------------------------

class safe_processor_wrapper:
    """Wrap any callable so it never raises; returns None on error."""

    def __init__(self, processor, name: str = "processor"):
        self.processor = processor
        self.name = name

    def __call__(self, sample):
        try:
            return self.processor(sample)
        except Exception as e:
            warnings.warn(f"[SafeProcessor:{self.name}] Error processing sample: {e}")
            return None


# ---------------------------------------------------------------------------
# Unprojection helpers (for CA1M with K matrix)
# ---------------------------------------------------------------------------

def get_camera_coords(depth: torch.Tensor):
    height, width = depth.shape
    device = depth.device
    camera_coords = torch.stack(
        torch.meshgrid(
            torch.arange(0, width, device=device),
            torch.arange(0, height, device=device),
            indexing="xy",
        ),
        dim=-1,
    )
    return camera_coords


def unproject(depth: torch.Tensor, K: torch.Tensor, RT: torch.Tensor, max_depth: float = 10.0):
    camera_coords = get_camera_coords(depth) * depth[..., None]

    intrinsics_4x4 = torch.eye(4, device=depth.device)
    intrinsics_4x4[:3, :3] = K

    valid = depth > 0
    if max_depth is not None:
        valid &= depth < max_depth

    depth_unsq = depth[..., None]
    uvd = torch.cat((camera_coords, depth_unsq, torch.ones_like(depth_unsq)), dim=-1)

    camera_xyz = torch.linalg.inv(intrinsics_4x4) @ uvd.view(-1, 4).T
    world_xyz = RT @ camera_xyz

    return world_xyz.T[..., :-1].reshape(uvd.shape[0], uvd.shape[1], 3), valid


# ---------------------------------------------------------------------------
# Unified sample converters
# ---------------------------------------------------------------------------

def to_unified_sample(s: dict) -> dict:
    """Convert a webdataset sample dict (o365 / coyo) to unified format."""
    file_id = os.path.basename(s.get("__key__", ""))
    return {
        "id": file_id,
        "img_pil": s.get("img_pil"),
        "depth": s.get("depth"),
        "intrinsics": s.get("intrinsics"),
        "meta": None,
        "qa": s.get("qa"),
        "__url__": s.get("__url__"),
        "__key__": s.get("__key__"),
    }


def to_unified_sample_ca1m(s: dict) -> dict:
    file_id = os.path.basename(s.get("__key__", ""))
    return {
        "id": file_id,
        "img_pil": s.get("img_pil"),
        "depth": s.get("depth"),
        "intrinsics": s.get("intrinsics"),
        "meta": None,
        "qa": s.get("qa"),
        "__url__": s.get("__url__"),
        "__key__": s.get("__key__"),
    }


def to_general_sample(s: dict) -> dict:
    file_id = os.path.basename(s.get("__key__", ""))
    return {
        "id": file_id,
        "img_pil": s.get("img_pil"),
        "moge_tensor": s.get("moge_tensor"),
        "meta": s.get("meta"),
        "shard_path": s.get("__url__", ""),
        "__url__": s.get("__url__"),
        "__key__": s.get("__key__"),
    }


# ---------------------------------------------------------------------------
# CA1M helpers: group-by-keys for multi-file samples
# ---------------------------------------------------------------------------

_CA1M_PART_PAT = re.compile(r"^(?P<base>.+?)_(?P<part>image|depth|intrinsics|qa|meta)$")


def _annotate_and_rekey(sample: dict) -> dict:
    key = sample.get("__key__", "")
    m = _CA1M_PART_PAT.match(key)
    if m:
        base, part = m.group("base"), m.group("part")
    else:
        base, part = key, None

    sample["_part"] = part
    sample["__key__"] = base

    if part == "image" and "jpg" in sample:
        sample["img_jpg"] = sample.pop("jpg")
    if part == "depth" and "npy" in sample:
        sample["depth_npy"] = sample.pop("npy")
    if part == "intrinsics" and "pt" in sample:
        sample["intrinsics_pt"] = sample.pop("pt")
    if part == "qa" and "pt" in sample:
        sample["qa_pt"] = sample.pop("pt")
    return sample


def group_by_keys(data_iter, maxsize=10000, lcase=True):
    """Merge consecutive samples with the same key into one dict."""
    curkey = None
    merged: Dict[str, Any] = {}
    for sample in data_iter:
        key = sample.get("__key__")
        if key is None:
            continue
        if lcase:
            key = key.lower()
        if curkey is None:
            curkey = key
        if key != curkey:
            yield merged
            merged = {}
            curkey = key
        merged.update(sample)
    if merged:
        yield merged


def _decode_by_keys(sample: dict) -> dict:
    if "img_jpg" in sample:
        sample["img_pil"] = _decode_img_jpg(sample.pop("img_jpg"))
    if "depth_npy" in sample:
        sample["depth"] = _decode_depth_npy(sample.pop("depth_npy"))
    if "intrinsics_pt" in sample:
        sample["intrinsics"] = _decode_pt(sample.pop("intrinsics_pt"))
    if "qa_pt" in sample:
        sample["qa"] = _decode_pt(sample.pop("qa_pt"))
    return sample


# =====================================================================
# Transform classes (called per-sample in the wds pipeline)
# =====================================================================


class WildSampleTransform:
    """
    Transform for *in-the-wild* data (o365, coyo).

    Adapted from ``vqa_v3_processor`` in the original RGBD codebase.
    Each sample has depth from monocular estimation + intrinsics.
    Depth is stored as a 16-bit PNG; XYZ is computed via ``utils3d.numpy.depth_to_points``.

    QA structure (v3 format):
        qa is a dict of level lists, e.g. ``level1_orientation_list``,
        ``level2_distance_list``, etc.  Each item has ``qa_type``, ``object``,
        and either ``qa_pairs`` or ``question``/``answer``/``prefix``/``suffix``.
    """

    # QA type classes and their sampling weights
    QA_TYPE_CLASSES = [
        'vqa_distance_single', 'vqa_relative_position_new_perspective_no_orient',
        'vqa_orientation_comparison_two_objects', 'vqa_counting_objects',
        'vqa_distance_each_axis_multi', 'vqa_distance_multi',
        'vqa_object_relation_multi_world', 'vqa_orientation',
        'vqa_closest_distance_multi', 'vqa_object_direction_multi',
        'vqa_relative_position_new_perspective', 'vqa_distance_three_objects',
        'vqa_distance_three_objects_comparision', 'vqa_object_relation_multi',
        'vqa_distance_each_axis_multi_world', 'vqa_most_objects_in_class',
        'vqa_query_aabb', 'vqa_object_size_single',
        'vqa_comparison_multiple_objects', 'level0',
        'vqa_spatial_mcq_multiangle_world', 'vqa_spatial_mcq_multiangle',
    ]

    DEFAULT_QA_TYPE_WEIGHTS = {
        'vqa_distance_single': 180,
        'vqa_relative_position_new_perspective_no_orient': 600,
        'vqa_orientation_comparison_two_objects': 500,
        'vqa_counting_objects': 700,
        'vqa_distance_each_axis_multi': 350,
        'vqa_distance_multi': 600,
        'vqa_object_relation_multi_world': 700,
        'vqa_orientation': 500,
        'vqa_closest_distance_multi': 400,
        'vqa_object_direction_multi': 300,
        'vqa_relative_position_new_perspective': 600,
        'vqa_distance_three_objects': 350,
        'vqa_distance_three_objects_comparision': 350,
        'vqa_object_relation_multi': 700, 
        'vqa_distance_each_axis_multi_world': 350,
        'vqa_most_objects_in_class': 600,
        'vqa_query_aabb': 300,
        'vqa_object_size_single': 300,
        'vqa_comparison_multiple_objects': 650,
        'level0': 350,
        'vqa_spatial_mcq_multiangle_world': 625,
        'vqa_spatial_mcq_multiangle': 625,
    }

    def __init__(self, processor, img_size: int = 448, qa_type_weights: Optional[Dict[str, float]] = None):
        self.processor = processor
        self.img_size = img_size
        self.num = 0
        self.qa_type_weights = qa_type_weights if qa_type_weights is not None else self.DEFAULT_QA_TYPE_WEIGHTS

    # -----------------------------------------------------------------
    # Model input
    # -----------------------------------------------------------------

    def process_model_input(self, result, xyz_values, I2, qa_type, I=None):
        result["prefix"] = result["prefix"].strip().replace("<image>", "")
        inputs = self.processor(
            text="<image>" + result["prefix"],
            suffix=result["suffix"],
            images=I2,
            return_tensors="pt",
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "xyz_values": xyz_values,
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
            "token_type_ids": inputs["token_type_ids"].squeeze(0),
            "inputs": inputs,
            "new_caption": result["prefix"],
            "qa_type": qa_type,
        }

    # -----------------------------------------------------------------
    # Bbox helpers
    # -----------------------------------------------------------------

    def convert_bbox_to_tokens(self, image_width, image_height, pixel_bbox):
        x1, y1, x2, y2 = pixel_bbox
        assert 0 <= x1 <= image_width and 0 <= y1 <= image_height
        assert 0 <= x2 <= image_width and 0 <= y2 <= image_height
        assert x1 <= x2 and y1 <= y2
        norm_x1 = round((x1 / image_width) * 1024)
        norm_y1 = round((y1 / image_height) * 1024)
        norm_x2 = round((x2 / image_width) * 1024)
        norm_y2 = round((y2 / image_height) * 1024)
        return (
            f"<loc{str(norm_y1).zfill(4)}><loc{str(norm_x1).zfill(4)}>"
            f"<loc{str(norm_y2).zfill(4)}><loc{str(norm_x2).zfill(4)}>"
        )

    def resize_xyxy(self, xyxy, h, w):
        x1 = int(xyxy[0] * self.img_size / w)
        y1 = int(xyxy[1] * self.img_size / h)
        x2 = int(xyxy[2] * self.img_size / w)
        y2 = int(xyxy[3] * self.img_size / h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.img_size - 1, x2), min(self.img_size - 1, y2)
        return (x1, y1, x2, y2)

    def query_pixel_xyz(self, I2, mask, xyz):
        indices = torch.nonzero(mask == 1, as_tuple=False)
        random_index = indices[torch.randint(0, len(indices), (1,))]
        y1, x1 = random_index[0][0].item(), random_index[0][1].item()
        xyz_value = xyz[y1, x1]
        x = round(xyz_value[0].item(), 2)
        y = round(xyz_value[1].item(), 2)
        z = round(xyz_value[2].item(), 2)
        bbox_tokens = self.convert_bbox_to_tokens(I2.shape[1], I2.shape[0], [x1, y1, x1, y1])
        return {
            "prefix": f"what is the xyz of the pixel in {bbox_tokens}",
            "suffix": f"{x:.2f}m {y:.2f}m {z:.2f}m",
            "qa_type": "query_pixel_xyz",
            "bbox": [x1, y1, x1, y1],
        }

    # -----------------------------------------------------------------
    # MCQ augmentation
    # -----------------------------------------------------------------

    def mcq_choice_augment(self, options, answer):
        alphabet = random.choice([
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            list("abcdefghijklmnopqrstuvwxyz"),
            [str(i) for i in range(1, 11)],
            ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
            ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        ])
        random.shuffle(options)
        option_format = random.choice(["with_()", "without_()"])
        options_str = ""
        answer_str = ""
        for i, option in enumerate(options):
            if option_format == "with_()":
                options_str += f"({alphabet[i]}). {option}\n"
            else:
                options_str += f"{alphabet[i]}. {option}\n"
            if option == answer:
                answer_str = f"({alphabet[i]})" if option_format == "with_()" else alphabet[i]
        return options_str, answer_str

    # -----------------------------------------------------------------
    # Draw bounding boxes on image
    # -----------------------------------------------------------------

    def draw_bbox(self, I2, I, objects):
        if objects is None:
            return I2, I
        referring_keys = [
            ("referring", "raw"), ("referring_i", "raw_i"),
            ("referring_j", "raw_j"), ("referring_k", "raw_k"),
            ("referring_l", "raw_l"), ("referring_m", "raw_m"),
            ("referring_n", "raw_n"),
        ]
        color_map = {
            "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
            "yellow": (255, 255, 0), "purple": (255, 0, 255),
        }
        for ref_key, raw_key in referring_keys:
            if ref_key not in objects or objects[ref_key] is None or objects.get(raw_key) is None:
                continue
            referring_text = objects[ref_key]
            xyxy = self.resize_xyxy(objects[raw_key]["raw_bbox"], I.shape[0], I.shape[1])
            xyxy = add_bbox_perturbation(xyxy, scale_range=(1.0, 1.2))
            xyxy_raw = objects[raw_key]["raw_bbox"]
            # Clip to image bounds
            xyxy = (max(0, xyxy[0]), max(0, xyxy[1]),
                    min(self.img_size - 1, xyxy[2]), min(self.img_size - 1, xyxy[3]))
            for color_name, color_val in color_map.items():
                if f"(highlighted by the {color_name} box)" in referring_text or \
                   f"(highlighted by a {color_name} box)" in referring_text:
                    I2 = cv2.rectangle(I2, (int(xyxy[0]), int(xyxy[1])),
                                       (int(xyxy[2]), int(xyxy[3])), color_val, 2)
                    I = cv2.rectangle(I, (int(xyxy_raw[0]), int(xyxy_raw[1])),
                                      (int(xyxy_raw[2]), int(xyxy_raw[3])), color_val, 2)
        return I2, I

    # -----------------------------------------------------------------
    # get_question_answer — convert qa_info → (question, answer) strings
    # -----------------------------------------------------------------

    def get_question_answer(self, qa_info):
        """
        Parse a single QA item into (question, answer) strings.

        Handles three formats:
        - ``qa_pairs``: list of {question, answer, options, format, qa_type} dicts
        - ``question`` / ``answer`` keys directly
        - ``prefix`` / ``suffix`` keys directly
        """
        if "qa_pairs" in qa_info:
            qa_pairs = qa_info["qa_pairs"]
            if not qa_pairs:
                return None, None

            # Remove known-buggy qa types
            qa_pairs = [
                qp for qp in qa_pairs
                if qp.get("qa_type", "") not in [
                    "x_distance_w_lf", "y_distance_w_lf", "z_distance_w_lf", "distance_description",
                ]
            ]

            # Split by format: text / mcq / true_false
            text_list, mcq_list, tf_list = [], [], []
            z_overlap_exist = y_overlap_exist = x_overlap_exist = False

            for qp in qa_pairs:
                qa_format = qp.get("format")
                if qa_format is None:
                    qa_format = qp.get("type")
                    if qa_format is None:
                        continue
                    qa_format = qa_format.replace("open_ended", "text")
                    qp["format"] = qa_format

                if "text" in qa_format:
                    text_list.append(qp)
                elif qa_format == "mcq":
                    if "relation_3choice" in qp.get("qa_type", ""):
                        continue
                    mcq_list.append(qp)
                elif qa_format == "true_false":
                    ans_lower = qp.get("answer", "").lower()
                    qt = qp.get("qa_type", "")
                    if "z_overlap_tf" in qt and ans_lower in ("true", "yes"):
                        z_overlap_exist = True
                    if "y_overlap_tf" in qt and ans_lower in ("true", "yes"):
                        y_overlap_exist = True
                    if "x_overlap_tf" in qt and ans_lower in ("true", "yes"):
                        x_overlap_exist = True
                    tf_list.append(qp)

            # Filter tf_list: remove biased overlap / directional QAs
            lr_type = ["x_left_tf", "x_right_tf"]
            ab_type = ["y_above_tf", "y_below_tf"]
            fb_type = ["z_front_tf", "z_behind_tf"]

            new_tf_list = []
            for tf_qa in tf_list:
                qt = tf_qa.get("qa_type", "")
                if z_overlap_exist and ("front" in qt or "behind" in qt):
                    continue
                if y_overlap_exist and ("above_tf" in qt or "below_tf" in qt):
                    continue
                if x_overlap_exist and ("left_tf" in qt or "right_tf" in qt):
                    continue
                if "overlap" in qt:
                    continue  # delete all overlap qa
                new_tf_list.append(tf_qa)

            # Check balance of true/false per axis direction
            lr_neg_cnt = lr_overall_cnt = 0
            ab_neg_cnt = ab_overall_cnt = 0
            fb_neg_cnt = fb_overall_cnt = 0
            for tf_qa in new_tf_list:
                qt = tf_qa.get("qa_type", "")
                is_neg = tf_qa.get("answer", "").lower() in ("no", "false")
                if qt in lr_type:
                    lr_overall_cnt += 1
                    if is_neg:
                        lr_neg_cnt += 1
                elif qt in ab_type:
                    ab_overall_cnt += 1
                    if is_neg:
                        ab_neg_cnt += 1
                elif qt in fb_type:
                    fb_overall_cnt += 1
                    if is_neg:
                        fb_neg_cnt += 1

            if lr_neg_cnt * 2 != lr_overall_cnt:
                new_tf_list = [q for q in new_tf_list if q.get("qa_type", "") not in lr_type]
            if ab_neg_cnt * 2 != ab_overall_cnt:
                new_tf_list = [q for q in new_tf_list if q.get("qa_type", "") not in ab_type]
            if fb_neg_cnt * 2 != fb_overall_cnt:
                new_tf_list = [q for q in new_tf_list if q.get("qa_type", "") not in fb_type]

            tf_list = new_tf_list

            # Weighted selection among text / mcq / true_false
            qa_pair_weights = [0.3, 0.4, 0.3]
            if not text_list:
                qa_pair_weights[0] = 0.0
            if not mcq_list:
                qa_pair_weights[1] = 0.0
            if not tf_list:
                qa_pair_weights[2] = 0.0
            if sum(qa_pair_weights) == 0:
                return None, None

            qa_pair_class = random.choices(
                population=["text", "mcq", "true_false"],
                weights=qa_pair_weights,
            )[0]
            if qa_pair_class == "text":
                qa_pair = random.choice(text_list)
            elif qa_pair_class == "mcq":
                qa_pair = random.choice(mcq_list)
            else:
                qa_pair = random.choice(tf_list)

        elif "question" in qa_info and "answer" in qa_info:
            qa_pair = {
                "question": qa_info["question"],
                "answer": qa_info["answer"],
                "options": qa_info.get("options"),
            }
        elif "prefix" in qa_info and "suffix" in qa_info:
            qa_pair = {
                "question": qa_info["prefix"],
                "answer": qa_info["suffix"],
                "options": qa_info.get("options"),
            }
        else:
            raise ValueError("qa_info must contain either 'prefix'/'suffix', 'question'/'answer', or 'qa_pairs'.")

        # Resolve question (may be a list of rephrasings)
        question = qa_pair["question"]
        if isinstance(question, list):
            question = random.choice(question)

        # Fix known bug: distance_farther answer should be the non-answer option
        if qa_info.get("qa_type") == "distance_farther":
            for option in qa_pair.get("options", []):
                if option != qa_pair["answer"]:
                    qa_pair["answer"] = option
                    break

        # MCQ formatting
        if qa_pair.get("options") is not None and qa_pair.get("format") == "mcq":
            options = qa_pair["options"]
            answer = qa_pair["answer"]
            # Handle list-of-lists options (pick one variant per option)
            if isinstance(options[0], list) and isinstance(answer, list):
                for i in range(len(options)):
                    options[i] = random.choice(options[i])
            elif isinstance(options[0], list):
                for i in range(len(options)):
                    options[i] = random.choice(options[i])

            for option in options:
                if option in qa_pair["answer"]:
                    qa_pair["answer"] = option
                    break

            options_str, answer_str = self.mcq_choice_augment(options, qa_pair["answer"])
            question = question + "\n" + options_str
            answer = answer_str
        else:
            answer = qa_pair["answer"]
            if isinstance(answer, list):
                answer = random.choice(answer)

        # True/False → Yes/No conversion
        if "True or False" in question:
            question = question.replace("True or False", "Yes or No")
        if answer.lower() in ("true", "false"):
            if "Yes or No" not in question and random.random() < 0.5:
                question = question + " Answer with Yes or No."
            answer = "Yes" if answer.lower() == "true" else "No"

        return question, answer

    # -----------------------------------------------------------------
    # __call__
    # -----------------------------------------------------------------

    def __call__(self, sample):
        img = sample["img_pil"]
        depth_np = sample["depth"][0]
        intrinsics = sample["intrinsics"]
        qa = sample["qa"]

        intrinsics_np = intrinsics.cpu().numpy()
        mask_nan, mask_inf = np.isnan(depth_np), np.isinf(depth_np)
        mask = (~(mask_nan | mask_inf)).astype(np.int8)
        depth_clean = depth_np.copy()
        depth_clean[mask_nan] = 0.0
        depth_clean[mask_inf] = 0.0
        xyz = utils3d.numpy.depth_to_points(depth_clean, intrinsics=intrinsics_np)

        I = np.array(img)
        depth = depth_np.copy()

        I2 = cv2.resize(I, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        xyz = cv2.resize(xyz, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(mask)
        xyz = torch.from_numpy(xyz)
        depth = torch.from_numpy(depth)

        # Clip
        xyz[xyz > 250] = 250
        xyz[torch.isinf(xyz)] = 250
        xyz[torch.isnan(xyz)] = 250
        depth[depth > 250] = 250
        depth[torch.isinf(depth)] = 250
        depth[torch.isnan(depth)] = 250

        # xyz_values: [4, H, W]
        xyz_values = xyz.permute(2, 0, 1)
        xyz_values = torch.cat((xyz_values, mask.unsqueeze(0)), dim=0)

        # ---- Weighted QA type sampling (v3 style) ----
        qa_type_weights = dict(self.DEFAULT_QA_TYPE_WEIGHTS)

        # First pass: select qa_type class (including level0 → pixel_xyz)
        qa_class_weights = [qa_type_weights[qt] for qt in self.QA_TYPE_CLASSES]
        current_qa_class = random.choices(
            population=self.QA_TYPE_CLASSES,
            weights=qa_class_weights,
        )[0]

        if current_qa_class == "level0":
            result = self.query_pixel_xyz(I2, mask, xyz)
            return self.process_model_input(result, xyz_values, I2, "pixel_xyz", I)

        # Flatten all QA items from the level-based dict
        qa_item_list = []
        qa_type_set = set()
        for level_key in qa:
            if "metadata" in level_key:
                continue
            for qa_item in qa[level_key]:
                if qa_item is None:
                    continue
                qa_item_list.append(qa_item)
                qa_type_set.add(qa_item["qa_type"])

        # Zero out weights for qa_types not present in this sample
        for qt in list(qa_type_weights.keys()):
            if qt not in qa_type_set:
                qa_type_weights[qt] = 0

        # Re-sample with updated weights
        if sum(qa_type_weights.values()) == 0:
            result = self.query_pixel_xyz(I2, mask, xyz)
            return self.process_model_input(result, xyz_values, I2, "pixel_xyz", I)

        selected_qa_type = random.choices(
            population=list(qa_type_weights.keys()),
            weights=list(qa_type_weights.values()),
        )[0]

        selected_qa_list = [q for q in qa_item_list if q["qa_type"] == selected_qa_type]

        if not selected_qa_list:
            result = self.query_pixel_xyz(I2, mask, xyz)
            return self.process_model_input(result, xyz_values, I2, "pixel_xyz", I)

        qa_info = random.choice(selected_qa_list)

        question, answer = self.get_question_answer(qa_info)

        if question is None or answer is None:
            result = self.query_pixel_xyz(I2, mask, xyz)
            return self.process_model_input(result, xyz_values, I2, "pixel_xyz", I)

        result = {
            "prefix": question,
            "suffix": answer,
            "object": qa_info.get("object"),
        }

        I2, I = self.draw_bbox(I2, I, result["object"])

        return self.process_model_input(result, xyz_values, I2, qa_info.get("qa_type"), I)


# =====================================================================
# GeneralSampleTransform
# =====================================================================


class GeneralSampleTransform:
    """Transform for general VQA data (conversation-format JSON with bboxes)."""

    _PAT_BBOX = re.compile(
        r"""\[\s*
            ([+-]?\d+(?:\.\d+)?)\s*,
            \s*([+-]?\d+(?:\.\d+)?)\s*,
            \s*([+-]?\d+(?:\.\d+)?)\s*,
            \s*([+-]?\d+(?:\.\d+)?)\s*
        \]""",
        re.VERBOSE,
    )

    def __init__(self, processor, img_size: int = 448, vlm_type: str = "paligemma2"):
        self.processor = processor
        self.img_size = img_size
        self.vlm_type = vlm_type

    def convert_bbox_to_tokens(self, image_width, image_height, pixel_bbox):
        x1, y1, x2, y2 = pixel_bbox
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        norm_x1 = round((x1 / image_width) * 1024)
        norm_y1 = round((y1 / image_height) * 1024)
        norm_x2 = round((x2 / image_width) * 1024)
        norm_y2 = round((y2 / image_height) * 1024)
        return (
            f"<loc{str(norm_y1).zfill(4)}><loc{str(norm_x1).zfill(4)}>"
            f"<loc{str(norm_y2).zfill(4)}><loc{str(norm_x2).zfill(4)}>"
        )

    def replace_bbox(self, text: str, image_width, image_height) -> str:
        def _repl(match: re.Match) -> str:
            x1, y1, x2, y2 = map(float, match.groups())
            return self.convert_bbox_to_tokens(
                image_width,
                image_height,
                (x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height),
            )
        return self._PAT_BBOX.sub(_repl, text)

    def __call__(self, sample):
        img = sample["img_pil"]
        moge = sample["moge_tensor"]
        conversations = sample["meta"]

        I = np.array(img)
        mask = moge["mask"].cpu().numpy().astype(np.int8)
        xyz = moge["points"].cpu().numpy()
        depth = moge["depth"].cpu().numpy()

        I2 = cv2.resize(I, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        xyz = cv2.resize(xyz, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(mask)
        xyz = torch.from_numpy(xyz)
        depth = torch.from_numpy(depth)

        xyz[xyz > 250] = 250
        xyz[torch.isinf(xyz)] = 250
        xyz[torch.isnan(xyz)] = 250
        depth[depth > 250] = 250
        depth[torch.isinf(depth)] = 250
        depth[torch.isnan(depth)] = 250

        xyz_values = xyz.permute(2, 0, 1)
        xyz_values = torch.cat((xyz_values, mask.unsqueeze(0)), dim=0)

        # Pick random conversation pair
        i = random.randint(0, len(conversations) // 2 - 1) * 2
        conversation = conversations[i : i + 2]
        assert conversation[0]["from"] == "human" and conversation[1]["from"] == "gpt"

        prefix = conversation[0]["value"].replace("\n", " ")
        suffix = conversation[1]["value"].replace("\n", " ")

        prefix = self.replace_bbox(prefix, img.size[0], img.size[1])
        suffix = self.replace_bbox(suffix, img.size[0], img.size[1])

        prefix = prefix.strip().replace("<image>", "")
        inputs = self.processor(
            text="<image>" + prefix,
            suffix=suffix,
            images=I2,
            return_tensors="pt",
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "xyz_values": xyz_values,
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
            "token_type_ids": inputs["token_type_ids"].squeeze(0),
            "inputs": inputs,
            "new_caption": prefix,
            "qa_type": "general",
        }


# =====================================================================
# CA1MSampleTransform
# =====================================================================


class CA1MSampleTransform:
    """
    Transform for CA1M indoor data (uses K matrix for unprojection).

    Adapted from ``vqa_processor_ca1m`` in the original RGBD codebase.
    Depth is stored as ``.npy``; XYZ is computed via ``unproject()`` with K matrix.

    Supports per-QA-type sampling via ``qa_type_weights``.
    """

    # CA1M QA type categories
    QA_TYPE_CLASSES = [
        "orientation",
        "perspective_taking",
        "spatial_relation",
        "distance",
        "object_depth_comparison",
        "object_width",
        "object_height",
        "problem_solving",
    ]

    DEFAULT_QA_TYPE_WEIGHTS = {
        "orientation": 0.8,
        "perspective_taking": 1.0,
        "spatial_relation": 1.0,
        "distance": 0.8,
        "object_depth_comparison": 1.0,
        "object_width": 1.0,
        "object_height": 1.0,
        "problem_solving": 1.0,
    }

    def __init__(
        self,
        processor,
        img_size: int = 448,
        qa_type_weights: Optional[Dict[str, float]] = None,
    ):
        self.processor = processor
        self.img_size = img_size
        self.num = 0

        if qa_type_weights is None:
            self.qa_type_weights = dict(self.DEFAULT_QA_TYPE_WEIGHTS)
        else:
            self.qa_type_weights = qa_type_weights

    # -----------------------------------------------------------------
    # Model input
    # -----------------------------------------------------------------

    def process_model_input(self, result, xyz_values, I2, qa_type, I=None):
        result["prefix"] = result["prefix"].strip().replace("<image>", "")
        inputs = self.processor(
            text="<image>" + result["prefix"],
            suffix=result["suffix"],
            images=I2,
            return_tensors="pt",
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "xyz_values": xyz_values,
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
            "token_type_ids": inputs["token_type_ids"].squeeze(0),
            "inputs": inputs,
            "new_caption": result["prefix"],
            "qa_type": qa_type,
        }


    # -----------------------------------------------------------------
    # Bbox helpers
    # -----------------------------------------------------------------

    def convert_bbox_to_tokens(self, image_width, image_height, pixel_bbox):
        x1, y1, x2, y2 = pixel_bbox
        assert 0 <= x1 <= image_width and 0 <= y1 <= image_height
        assert 0 <= x2 <= image_width and 0 <= y2 <= image_height
        assert x1 <= x2 and y1 <= y2
        norm_x1 = round((x1 / image_width) * 1024)
        norm_y1 = round((y1 / image_height) * 1024)
        norm_x2 = round((x2 / image_width) * 1024)
        norm_y2 = round((y2 / image_height) * 1024)
        return (
            f"<loc{str(norm_y1).zfill(4)}><loc{str(norm_x1).zfill(4)}>"
            f"<loc{str(norm_y2).zfill(4)}><loc{str(norm_x2).zfill(4)}>"
        )

    def convert_bbox_to_pixelbbox(self, raw_image_width, raw_image_height, bbox2d):
        x1, y1, x2, y2 = bbox2d
        x1 = int((x1 / raw_image_width) * self.img_size)
        y1 = int((y1 / raw_image_height) * self.img_size)
        x2 = int((x2 / raw_image_width) * self.img_size)
        y2 = int((y2 / raw_image_height) * self.img_size)
        return (x1, y1, x2, y2)

    def resize_xyxy(self, xyxy, h, w):
        x1 = int(xyxy[0] * self.img_size / w)
        y1 = int(xyxy[1] * self.img_size / h)
        x2 = int(xyxy[2] * self.img_size / w)
        y2 = int(xyxy[3] * self.img_size / h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.img_size - 1, x2), min(self.img_size - 1, y2)
        return (x1, y1, x2, y2)

    def query_pixel_xyz(self, I2, mask, xyz):
        indices = torch.nonzero(mask == 1, as_tuple=False)
        if len(indices) == 0:
            return {
                "prefix": "describe this image",
                "suffix": "an indoor scene",
                "qa_type": "query_pixel_xyz",
                "bbox": [0, 0, 0, 0],
            }
        random_index = indices[torch.randint(0, len(indices), (1,))]
        y1, x1 = random_index[0][0].item(), random_index[0][1].item()
        xyz_value = xyz[y1, x1]
        x = round(xyz_value[0].item(), 2)
        y = round(xyz_value[1].item(), 2)
        z = round(xyz_value[2].item(), 2)
        bbox_tokens = self.convert_bbox_to_tokens(I2.shape[1], I2.shape[0], [x1, y1, x1, y1])
        return {
            "prefix": f"what is the xyz of the pixel in {bbox_tokens}",
            "suffix": f"{x:.2f}m {y:.2f}m {z:.2f}m",
            "qa_type": "query_pixel_xyz",
            "bbox": [x1, y1, x1, y1],
        }

    # -----------------------------------------------------------------
    # Draw bounding boxes on image (same style as WildSampleTransform)
    # -----------------------------------------------------------------

    def draw_bbox(self, I2, I, objects):
        if objects is None:
            return I2, I
        referring_keys = [
            ("referring", "raw"), ("referring_i", "raw_i"),
            ("referring_j", "raw_j"), ("referring_k", "raw_k"),
            ("referring_l", "raw_l"), ("referring_m", "raw_m"),
            ("referring_n", "raw_n"),
        ]
        color_map = {
            "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
            "yellow": (255, 255, 0), "purple": (255, 0, 255),
        }
        for ref_key, raw_key in referring_keys:
            if ref_key not in objects or objects[ref_key] is None or objects.get(raw_key) is None:
                continue
            referring_text = objects[ref_key]
            xyxy = self.resize_xyxy(objects[raw_key]["raw_bbox"], I.shape[0], I.shape[1])
            xyxy = add_bbox_perturbation(xyxy, scale_range=(1.0, 1.2))
            xyxy_raw = objects[raw_key]["raw_bbox"]
            xyxy = (max(0, xyxy[0]), max(0, xyxy[1]),
                    min(self.img_size - 1, xyxy[2]), min(self.img_size - 1, xyxy[3]))
            for color_name, color_val in color_map.items():
                if f"(highlighted by the {color_name} box)" in referring_text or \
                   f"(highlighted by a {color_name} box)" in referring_text:
                    I2 = cv2.rectangle(I2, (int(xyxy[0]), int(xyxy[1])),
                                       (int(xyxy[2]), int(xyxy[3])), color_val, 2)
                    I = cv2.rectangle(I, (int(xyxy_raw[0]), int(xyxy_raw[1])),
                                      (int(xyxy_raw[2]), int(xyxy_raw[3])), color_val, 2)
        return I2, I

    # -----------------------------------------------------------------
    # QA type sampling
    # -----------------------------------------------------------------

    def _sample_qa_by_type(self, qa_list: list) -> dict:
        """
        Sample a QA pair from *qa_list* respecting per-type weights.

        Groups QAs by their ``qa_type`` field (matched against ``QA_TYPE_CLASSES``),
        then does a weighted category draw followed by a uniform draw within
        that category.  Falls back to uniform if no type matches.
        """
        if not qa_list:
            raise ValueError("qa_list is empty")

        # Bucket by qa_type
        buckets: Dict[str, List[dict]] = {}
        unmatched: List[dict] = []
        for q in qa_list:
            qt = q.get("qa_type", "") or ""
            matched = False
            for canonical in self.QA_TYPE_CLASSES:
                if canonical in qt:
                    buckets.setdefault(canonical, []).append(q)
                    matched = True
                    break
            if not matched:
                unmatched.append(q)

        # Build weighted draw over available buckets
        available_types = [t for t in self.QA_TYPE_CLASSES if t in buckets]
        if not available_types:
            return random.choice(qa_list)

        weights = [self.qa_type_weights.get(t, 1.0) for t in available_types]
        if unmatched:
            available_types.append("__other__")
            weights.append(0.05)
            buckets["__other__"] = unmatched

        chosen_type = random.choices(available_types, weights=weights)[0]
        return random.choice(buckets[chosen_type])

    # -----------------------------------------------------------------
    # __call__
    # -----------------------------------------------------------------

    def __call__(self, sample):
        img = sample["img_pil"]
        depth_np = sample["depth"]
        qa = sample["qa"]

        # Unproject depth → XYZ using K matrix
        depth_clean = depth_np.copy()
        depth_tensor = torch.from_numpy(depth_clean)
        K = sample["intrinsics"]["K"]
        RT = torch.eye(4)
        xyz, valid = unproject(depth_tensor, K, RT, max_depth=30.0)
        xyz = xyz.cpu().numpy()
        mask = valid.cpu().numpy().astype(np.int8)

        I = np.array(img)
        depth = depth_np.copy()

        I2 = cv2.resize(I, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        xyz = cv2.resize(xyz, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # ------------- convert to tensor -------------
        mask = torch.from_numpy(mask)
        xyz = torch.from_numpy(xyz)
        depth = torch.from_numpy(depth)

        # ------------- clip image -------------
        xyz[xyz > 250] = 250
        xyz[torch.isinf(xyz)] = 250
        xyz[torch.isnan(xyz)] = 250
        depth[depth > 250] = 250
        depth[torch.isinf(depth)] = 250
        depth[torch.isnan(depth)] = 250

        # ------------- xyz_values -------------
        xyz_values = xyz.permute(2, 0, 1)
        xyz_values = torch.cat((xyz_values, mask.unsqueeze(0)), dim=0)

        # ---- Empty QA fallback ----
        if not qa or len(qa) == 0:
            result = self.query_pixel_xyz(I2, mask, xyz)
            return self.process_model_input(result, xyz_values, I2, "query_pixel_xyz", I)

        # ---- Sample QA by type ----
        result = self._sample_qa_by_type(qa)

        # Handle problem_solving CoT
        plus_cot = [
            " Please think step by step",
            " Answer the question with cot",
            " Explain your reasoning step by step",
            " Please provide a detailed explanation with cot",
        ]
        if result.get("qa_type") == "problem_solving":
            result["prefix"] = result["prefix"] + random.choice(plus_cot)
            original_suffix = result["suffix"]
            try:
                student_cot_str = result["meta"]["student_cot"]
                student_cot_json = json.loads(student_cot_str)
                cot_simple = student_cot_json["cot_simple"]
                if isinstance(cot_simple, list):
                    result["suffix"] = "\n".join(cot_simple) + "\n"
                elif isinstance(cot_simple, dict):
                    result["suffix"] = "\n".join(cot_simple.values()) + "\n"
                else:
                    result["suffix"] = str(cot_simple)
            except Exception:
                result["suffix"] = original_suffix

        # ---- Draw bbox ----
        # CA1M uses "highlighted by a red box" with meta.2dbox
        if "(highlighted by a red box)" in result.get("prefix", ""):
            xyxy = result.get("meta", {}).get("2dbox")
            if xyxy is not None:
                x1, y1, x2, y2 = self.convert_bbox_to_pixelbbox(I.shape[1], I.shape[0], xyxy)
                I2 = cv2.rectangle(I2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                I = cv2.rectangle(I, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)

        # Also handle generic object-level bboxes (same as WildSampleTransform)
        if result.get("object") is not None:
            I2, I = self.draw_bbox(I2, I, result["object"])
        
        return self.process_model_input(result, xyz_values, I2, result.get("qa_type"), I)


# =====================================================================
# VQACollateFn
# =====================================================================


class VQACollateFn:
    """Collate batches: pad input_ids/labels/token_type_ids, stack pixel & xyz values."""

    def __init__(self, pad_token_id: int = 0, max_length: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        xyz_values = torch.stack([x["xyz_values"] for x in batch])

        input_ids_list = []
        label_list = []
        token_type_ids_list = []

        for x in batch:
            ids = x["input_ids"]
            lab = x["labels"]
            ttids = x["token_type_ids"]
            if self.max_length is not None:
                ids = ids[: self.max_length]
                lab = lab[: self.max_length]
                ttids = ttids[: self.max_length]
            input_ids_list.append(ids)
            label_list.append(lab)
            token_type_ids_list.append(ttids)

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(label_list, batch_first=True, padding_value=-100)
        token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)
        attention_mask = input_ids.ne(self.pad_token_id)

        return {
            "pixel_values": pixel_values,
            "xyz_values": xyz_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_type_ids": token_type_ids,
        }


# =====================================================================
# WebDataset pipeline builders
# =====================================================================


def build_vqa_wds_pipeline(
    shards: Union[str, List[str]],
    *,
    tar_shuffle: bool = False,
    sample_shuffle: bool = False,
    sample_shuffle_buffer: int = 320,
    resampled: bool = False,
    map_processor=None,
):
    """Pipeline for o365 / coyo data (jpg + png depth + intrinsics.pt + qa.pt)."""
    pipe = []
    pipe.append(wds.ResampledShards(shards) if resampled else wds.SimpleShardList(shards))
    if tar_shuffle:
        pipe.append(wds.shuffle(1000))
    pipe.append(wds.split_by_worker)
    pipe.append(wds.tarfile_to_samples(handler=handlers.ignore_and_continue))
    if sample_shuffle:
        pipe.append(wds.shuffle(sample_shuffle_buffer))
    pipe.append(wds.rename(
        img_pil="jpg",
        depth="png",
        intrinsics="intrinsics.pt",
        qa="pt",
    ))
    pipe.append(wds.map_dict(
        img_pil=_decode_img_jpg,
        depth=_decode_depth_png,
        intrinsics=_decode_pt,
        qa=_decode_pt,
    ))
    pipe.append(wds.map(to_unified_sample))
    if map_processor is not None:
        pipe.append(map_keep_none(map_processor))
    return wds.DataPipeline(*pipe)


def build_general_vqa_wds_pipeline(
    shards: Union[str, List[str]],
    *,
    tar_shuffle: bool = False,
    sample_shuffle: bool = False,
    sample_shuffle_buffer: int = 320,
    resampled: bool = False,
    map_processor=None,
):
    """Pipeline for general VQA data (jpg + moge .pt + meta .json)."""
    pipe = []
    pipe.append(wds.ResampledShards(shards) if resampled else wds.SimpleShardList(shards))
    if tar_shuffle:
        pipe.append(wds.shuffle(1000))
    pipe.append(wds.split_by_worker)
    pipe.append(wds.tarfile_to_samples(handler=handlers.ignore_and_continue))
    if sample_shuffle:
        pipe.append(wds.shuffle(sample_shuffle_buffer))
    pipe.append(wds.rename(
        img_pil="jpg",
        moge_tensor="pt",
        meta="json",
    ))
    pipe.append(wds.map_dict(
        img_pil=_decode_img_jpg,
        moge_tensor=_decode_pt,
        meta=_decode_json,
    ))
    pipe.append(wds.map(to_general_sample))
    if map_processor is not None:
        pipe.append(wds.map(map_processor))
    return wds.DataPipeline(*pipe)


def build_vqa_wds_pipeline_ca1m(
    shards: Union[str, List[str]],
    *,
    tar_shuffle: bool = False,
    sample_shuffle: bool = False,
    sample_shuffle_buffer: int = 320,
    resampled: bool = False,
    map_processor=None,
):
    """Pipeline for CA1M data (multi-file samples: image.jpg, depth.npy, intrinsics.pt, qa.pt, meta.json)."""
    pipe = []
    pipe.append(wds.ResampledShards(shards) if resampled else wds.SimpleShardList(shards))
    if tar_shuffle:
        pipe.append(wds.shuffle(1000))
    pipe.append(wds.split_by_worker)
    pipe.append(wds.tarfile_to_samples(handler=wds.handlers.ignore_and_continue))
    pipe.append(wds.map(_annotate_and_rekey))
    pipe.append(group_by_keys)
    if sample_shuffle:
        pipe.append(wds.shuffle(sample_shuffle_buffer))
    pipe.append(wds.map(_decode_by_keys))
    pipe.append(wds.map(to_unified_sample_ca1m))
    if map_processor is not None:
        pipe.append(map_keep_none(map_processor))
    return wds.DataPipeline(*pipe)


# =====================================================================
# build_vqa_dataloader  — the main entry point
# =====================================================================


def build_vqa_dataloader(
    shards_coyo: List[str],
    shards_o365: List[str],
    shards_general: List[str],
    shards_ca1m: List[str],
    *,
    sample_transform_coyo,
    sample_transform_o365,
    sample_transform_general,
    sample_transform_ca1m,
    batch_size: int = 4,
    num_workers: int = 4,
    tar_shuffle: bool = False,
    pin_memory: bool = True,
    collate_fn=None,
    sample_shuffle: bool = False,
    sample_shuffle_buffer: int = 320,
    resampled: bool = False,
    mix_weights: Tuple[float, float, float, float] = (0.35, 0.33, 0.12, 0.20),
    **dl_kwargs,
):
    """
    Build a mixed WebDataset dataloader with four streams:
    coyo, o365 (both wild), general, and ca1m.

    Parameters
    ----------
    mix_weights : tuple of 4 floats
        (coyo, o365, general, ca1m) sampling probabilities.
    """
    w_coyo, w_o365, w_general, w_ca1m = mix_weights

    # Build streams, skipping any with empty shard lists
    datasets = []
    probs = []

    stream_configs = [
        ("coyo",    shards_coyo,    sample_transform_coyo,    w_coyo,    build_vqa_wds_pipeline),
        ("o365",    shards_o365,    sample_transform_o365,    w_o365,    build_vqa_wds_pipeline),
        ("general", shards_general, sample_transform_general, w_general, build_general_vqa_wds_pipeline),
        ("ca1m",    shards_ca1m,    sample_transform_ca1m,    w_ca1m,    build_vqa_wds_pipeline_ca1m),
    ]

    for name, shards, transform, weight, builder in stream_configs:
        if not shards:
            print(f"[build_vqa_dataloader] WARNING: no shards for '{name}', skipping stream.")
            continue
        proc = safe_processor_wrapper(transform, name=name)
        ds = builder(
            shards,
            tar_shuffle=tar_shuffle,
            sample_shuffle=sample_shuffle,
            sample_shuffle_buffer=sample_shuffle_buffer,
            resampled=resampled,
            map_processor=proc,
        )
        datasets.append(ds)
        probs.append(weight)

    if not datasets:
        raise ValueError("No data streams have shards. Cannot build dataloader.")

    mix = wds.RandomMix(datasets, probs=probs)

    dataset = wds.DataPipeline(
        mix,
        wds.select(_not_none),
        wds.batched(batch_size, partial=False, collation_fn=collate_fn),
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None,
        **dl_kwargs,
    )
    return loader



def processor_debug():
    """
    Full mixed dataloader debug (mirrors the original processor_debug).
    Uses num_workers=0 so breakpoints work inside transforms.
    """
    import random
    from pathlib import Path
    from transformers import PaliGemmaProcessor

    variant = {
        "batch_size": 4,
        "image_size": 448,
        "train_dataset": {
            "o365_dir": "/home/v-huizliang/huiz_eai/rgbd_data/dataset_v3/objects365/o365_data/",
            "ca1m_data_dir": "/home/v-huizliang/huiz_eai/rgbd_data/dataset_v2/ca1m_data/3d_annotated/",
        },
        "mix_weights": {
            "coyo_vqa": 0.35,
            "o365_vqa": 0.33,
            "general": 0.12,
            "ca1m": 0.20,
        },
    }

    processor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-mix-448")
    img_size = variant.get("image_size", 448)

    wild_transform = WildSampleTransform(processor, img_size=img_size)
    ca1m_transform = CA1MSampleTransform(processor, img_size=img_size)
    general_transform = GeneralSampleTransform(processor, img_size=img_size)
    collate_fn = VQACollateFn(processor.tokenizer.pad_token_id, max_length=1500)

    # --- Gather tar shards ---
    o365_dir = Path(variant["train_dataset"]["o365_dir"])
    o365_tar_list = [str(p) for p in o365_dir.glob("*/*.tar")]

    ca1m_data_dir = Path(variant["train_dataset"]["ca1m_data_dir"])
    ca1m_tar_list = [str(p) for p in ca1m_data_dir.glob("*.tar")]

    # For debug: reuse o365 shards as coyo placeholder; general left empty
    coyo_tar_list = list(o365_tar_list)
    general_tar_list = []  # no general data for quick test

    print(f"Found {len(o365_tar_list)} shards in {o365_dir}")
    print(f"Found {len(ca1m_tar_list)} shards in {ca1m_data_dir}")
    print(f"Using {len(coyo_tar_list)} shards as coyo placeholder")
    print(f"General shards: {len(general_tar_list)} (skipped)")

    random.shuffle(coyo_tar_list)
    random.shuffle(o365_tar_list)
    random.shuffle(ca1m_tar_list)

    # If general is empty, redistribute its weight
    weights = variant["mix_weights"]
    if not general_tar_list:
        mix_w = (weights["coyo_vqa"], weights["o365_vqa"], 0.0, weights["ca1m"])
        # Use o365 as a dummy for general to avoid empty pipeline error
        general_tar_list = o365_tar_list[:1]
    else:
        mix_w = (weights["coyo_vqa"], weights["o365_vqa"], weights["general"], weights["ca1m"])

    dataloader = build_vqa_dataloader(
        shards_coyo=coyo_tar_list,
        shards_o365=o365_tar_list,
        shards_general=general_tar_list,
        shards_ca1m=ca1m_tar_list,
        sample_transform_coyo=wild_transform,
        sample_transform_o365=wild_transform,
        sample_transform_general=general_transform,
        sample_transform_ca1m=ca1m_transform,
        batch_size=variant["batch_size"],
        num_workers=0,
        tar_shuffle=True,
        sample_shuffle=True,
        sample_shuffle_buffer=400,
        resampled=True,
        collate_fn=collate_fn,
        mix_weights=mix_w,
    )

    for i, batch in enumerate(dataloader):
        # ---- BREAKPOINT: inspect batch here ----
        # batch keys: pixel_values [B,C,H,W], xyz_values [B,4,H,W],
        #             input_ids [B,L], labels [B,L],
        #             attention_mask [B,L], token_type_ids [B,L]
        #
        # Decode first sample:
        #   ids = batch["input_ids"][0]
        #   ids = ids[ids != processor.tokenizer.pad_token_id]
        #   print(processor.tokenizer.decode(ids, skip_special_tokens=False))

        print(f"==== batch {i} ====")
        print(f"  pixel_values:   {batch['pixel_values'].shape}")
        print(f"  xyz_values:     {batch['xyz_values'].shape}")
        print(f"  input_ids:      {batch['input_ids'].shape}")
        print(f"  labels:         {batch['labels'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")

        # Decode first sample's text
        ids = batch["input_ids"][0]
        ids_clean = ids[ids != processor.tokenizer.pad_token_id]
        decoded = processor.tokenizer.decode(ids_clean, skip_special_tokens=False)
        print(f"  decoded[0]: {decoded[:200]}")
        print()


if __name__ == "__main__":
    processor_debug()
