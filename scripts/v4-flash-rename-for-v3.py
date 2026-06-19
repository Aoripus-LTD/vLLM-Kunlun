#!/usr/bin/env python3
"""
v4-flash-rename-for-v3.py — mmap-based V4->V3 weight renamer.

Uses mmap to avoid per-tensor disk seeks and full-shard RAM usage.
OS pages in data on-demand. torch.frombuffer creates zero-copy tensor views.
Peak memory: just the output tensors (~same size as input).
"""

import argparse
import json
import mmap
import os
import re
import shutil
import time
from glob import glob

import torch
from safetensors.torch import save_file


SAFE_DTYPE = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E8M0": torch.uint8, "F8_E4M3": torch.uint8, "F8_E5M2": torch.uint8, "F4_E2M1": torch.uint8,
}


def log(msg):
    print(msg, flush=True)


def is_v4_specific(key):
    if any(s in key for s in ["hc_attn_", "hc_ffn_", "hc_head_"]):
        return True
    if key.endswith("attn_sink") or key.endswith("wkv.weight") or key.endswith("attn_norm.weight"):
        return True
    if ".indexer." in key or ".tid2eid" in key:
        return True
    return False


def mmap_rename_shard(src_path, dst_path, stats):
    """mmap-based rename: zero-copy tensor views, no per-tensor seeks."""
    file_size = os.path.getsize(src_path)

    with open(src_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), file_size, prot=mmap.PROT_READ)
        try:
            header_len = struct.unpack("<Q", mm[:8])[0]
            header = json.loads(mm[8:8+header_len].decode("utf-8"))
            data_section_offset = 8 + header_len

            output_tensors = {}
            expert_buffer = {}  # (prefix, e_idx) -> {"w1": t, "w3": t}
            shared_buffer = {}  # prefix -> {"sw1": t, "sw3": t, "wo_a": t, "wo_b": t}
            kv_b_buffer = {}    # prefix -> {"wk_b": t, "wv_b": t}

            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                if is_v4_specific(key):
                    stats["stripped"] += 1
                    continue

                dtype_str = meta["dtype"]
                shape = tuple(meta["shape"])
                dstart, dend = meta["data_offsets"]
                abs_start = data_section_offset + dstart
                abs_end = data_section_offset + dend
                # Zero-copy view into mmap
                tensor = torch.frombuffer(mm[abs_start:abs_end], dtype=SAFE_DTYPE.get(dtype_str, torch.uint8)).reshape(shape)

                # Top-level
                if key == "embed.weight":
                    output_tensors["model.embed_tokens.weight"] = tensor
                    stats["renamed"] += 1; continue
                if key == "norm.weight":
                    output_tensors["model.norm.weight"] = tensor
                    stats["renamed"] += 1; continue
                if key == "lm_head.weight":
                    output_tensors["lm_head.weight"] = tensor
                    stats["renamed"] += 1; continue

                m = re.match(r"^layers\.(\d+)\.(.+)$", key)
                if not m:
                    stats["unmapped"] += 1; continue
                layer_idx, rest = m.group(1), m.group(2)
                prefix = f"model.layers.{layer_idx}"

                # Simple renames
                simple_map = {
                    "attn.wq_a.weight": f"{prefix}.self_attn.q_a_proj.weight",
                    "attn.wq_b.weight": f"{prefix}.self_attn.q_b_proj.weight",
                    "attn.q_norm.weight": f"{prefix}.self_attn.q_a_layernorm.weight",
                    "attn.kv_norm.weight": f"{prefix}.self_attn.kv_a_layernorm.weight",
                    "attn.wo.weight": f"{prefix}.self_attn.o_proj.weight",
                    "mlp.gate.weight": f"{prefix}.mlp.gate.weight",
                    "mlp.gate.e_score_correction_bias": f"{prefix}.mlp.gate.e_score_correction_bias",
                    "input_layernorm.weight": f"{prefix}.input_layernorm.weight",
                    "post_attention_layernorm.weight": f"{prefix}.post_attention_layernorm.weight",
                    "ffn.shared_experts.w2.weight": f"{prefix}.mlp.shared_experts.down_proj.weight",
                }
                if rest in simple_map:
                    output_tensors[simple_map[rest]] = tensor
                    stats["renamed"] += 1; continue

                # wk_a -> kv_a_proj_with_mqa (best effort: use wk_a as-is)
                if rest == "attn.wk_a.weight":
                    output_tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = tensor
                    stats["renamed"] += 1; continue
                if rest == "attn.wv_a.weight":
                    stats["stripped"] += 1; continue

                # kv_b fusion
                if rest == "attn.wk_b.weight":
                    kv_b_buffer.setdefault(prefix, {})["wk_b"] = tensor
                    if "wv_b" in kv_b_buffer[prefix]:
                        output_tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = torch.cat([kv_b_buffer[prefix]["wk_b"], kv_b_buffer[prefix]["wv_b"]], dim=0)
                        stats["fused"] += 1; del kv_b_buffer[prefix]
                    continue
                if rest == "attn.wv_b.weight":
                    kv_b_buffer.setdefault(prefix, {})["wv_b"] = tensor
                    if "wk_b" in kv_b_buffer[prefix]:
                        output_tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = torch.cat([kv_b_buffer[prefix]["wk_b"], kv_b_buffer[prefix]["wv_b"]], dim=0)
                        stats["fused"] += 1; del kv_b_buffer[prefix]
                    continue

                # O LoRA fusion
                if rest == "attn.wo_a.weight":
                    shared_buffer.setdefault(prefix, {})["wo_a"] = tensor
                    if "wo_b" in shared_buffer[prefix]:
                        output_tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.matmul(shared_buffer[prefix]["wo_b"], shared_buffer[prefix]["wo_a"])
                        stats["fused"] += 1; del shared_buffer[prefix]
                    continue
                if rest == "attn.wo_b.weight":
                    shared_buffer.setdefault(prefix, {})["wo_b"] = tensor
                    if "wo_a" in shared_buffer[prefix]:
                        output_tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.matmul(shared_buffer[prefix]["wo_b"], shared_buffer[prefix]["wo_a"])
                        stats["fused"] += 1; del shared_buffer[prefix]
                    continue

                # Experts
                em = re.match(r"^ffn\.experts\.(\d+)\.w([123])\.weight$", rest)
                if em:
                    e_idx, w_num = em.group(1), em.group(2)
                    if w_num == "2":
                        output_tensors[f"{prefix}.mlp.experts.{e_idx}.w2_weight"] = tensor
                        stats["renamed"] += 1; continue
                    buf_key = (prefix, e_idx)
                    expert_buffer.setdefault(buf_key, {})[f"w{w_num}"] = tensor
                    if "w1" in expert_buffer[buf_key] and "w3" in expert_buffer[buf_key]:
                        output_tensors[f"{prefix}.mlp.experts.{e_idx}.w13_weight"] = torch.cat([expert_buffer[buf_key]["w1"], expert_buffer[buf_key]["w3"]], dim=0)
                        stats["fused"] += 1; del expert_buffer[buf_key]
                    continue

                # Shared experts
                if rest == "ffn.shared_experts.w1.weight":
                    shared_buffer.setdefault(prefix, {})["sw1"] = tensor; continue
                if rest == "ffn.shared_experts.w3.weight":
                    shared_buffer.setdefault(prefix, {})["sw3"] = tensor
                    if "sw1" in shared_buffer[prefix]:
                        output_tensors[f"{prefix}.mlp.shared_experts.gate_up_proj.weight"] = torch.cat([shared_buffer[prefix]["sw1"], shared_buffer[prefix]["sw3"]], dim=0)
                        stats["fused"] += 1; del shared_buffer[prefix]
                    continue

                stats["unmapped"] += 1

            # Flush incomplete buffers
            for buf_key in list(expert_buffer.keys()):
                log(f"  WARN: incomplete expert {buf_key}")
            for prefix in list(kv_b_buffer.keys()):
                log(f"  WARN: incomplete kv_b {prefix}")
            for prefix in list(shared_buffer.keys()):
                log(f"  WARN: incomplete shared {prefix}")

            save_file(output_tensors, dst_path)
        finally:
            mm.close()


import struct  # for unpack

def main(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    log("Copying auxiliary files...")
    for file_path in glob(os.path.join(input_path, "*")):
        fname = os.path.basename(file_path)
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        dst = os.path.join(output_path, fname)
        if os.path.isfile(file_path):
            shutil.copy2(file_path, dst)

    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    new_weight_map = {}
    total_stats = {"renamed": 0, "fused": 0, "stripped": 0, "unmapped": 0}

    t_start = time.time()
    for i, safetensor_file in enumerate(safetensor_files, 1):
        file_name = os.path.basename(safetensor_file)
        dst_file = os.path.join(output_path, file_name)
        t0 = time.time()

        shard_stats = {"renamed": 0, "fused": 0, "stripped": 0, "unmapped": 0}
        mmap_rename_shard(safetensor_file, dst_file, shard_stats)

        # Rebuild weight_map from saved file
        with open(dst_file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            dst_header = json.loads(f.read(header_len).decode("utf-8"))
            for new_key in dst_header.keys():
                if new_key != "__metadata__":
                    new_weight_map[new_key] = file_name

        t_total = time.time() - t0
        for k, v in shard_stats.items():
            total_stats[k] += v
        log(f"[{i}/{len(safetensor_files)}] {file_name}: {t_total:.1f}s "
            f"(renamed={shard_stats['renamed']}, fused={shard_stats['fused']}, "
            f"stripped={shard_stats['stripped']}, unmapped={shard_stats['unmapped']})")

    new_index = {"metadata": model_index.get("metadata", {}), "weight_map": new_weight_map}
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    elapsed = time.time() - t_start
    log(f"\n=== Done in {elapsed:.1f}s ===")
    log(f"Input: {len(weight_map)}, Output: {len(new_weight_map)}, Stats: {total_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bf16-path", required=True)
    parser.add_argument("--output-renamed-path", required=True)
    args = parser.parse_args()
    main(args.input_bf16_path, args.output_renamed_path)
