#!/usr/bin/env python3
"""
v4-flash-fp8-to-bf16.py — Convert DeepSeek V4 Flash FP8/FP4 to BF16

Patched version of the Aoripus convert_weight.py for V4 Flash compatibility:
  - E8M0 scale decode: works on container torch (no float8_e8m0fnu dtype attribute)
  - V4 weight naming: ffn.experts.X.w1/w2/w3, attn.wq_a, attn.wo_a, etc.
  - Pass-through BF16 keys: hc_*, attn_sink, norms, biases, embed, lm_head
  - BYPASSES safetensors' dtype system entirely (it has a hardcoded
    DTYPE_MAP that references torch.float8_e8m0fnu at import time;
    monkey-patching the attribute after the fact doesn't help). We
    use safe_open + get_slice + get_data() to read raw bytes, then
    manually construct tensors with safe dtypes (uint8 for any
    float8 variant). This works regardless of torch version.

Usage:
  docker exec vllm-kunlun python /workspace/vLLM-Kunlun-Aoripus/scripts/v4-flash-fp8-to-bf16.py \
    --input-fp8-hf-path /workspace/models/DeepSeek-V4-Flash \
    --output-bf16-hf-path /workspace/models/DeepSeek-V4-Flash-BF16 \
    --device cpu

Notes:
  - V4 Flash: 149GB FP8+FP4 -> ~570GB BF16 (FP4 dequant doubles size, FP8 2x)
  - 46 safetensors shards, 69187 total keys
  - 33792 FP4 expert keys (256 experts x 43 layers x 3 + a few)
  - 375 FP8 non-expert keys
  - 34167 scale entries to be removed
  - BF16 pass-through: hc_*, attn_sink, norms, biases, embed, lm_head
  - Expected runtime: 30-60 min on CPU, ~5-10 min on XPU
"""

import argparse
import json
import os
import re
import shutil
import struct
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import save_file


# Safetensors dtype string -> safe torch dtype to read raw bytes as.
# All float8 variants are 1 byte, so we read them as uint8 and decode
# ourselves. bf16/f16/f32/i64 are read at their native dtype.
SAFE_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    # V4-specific float8 variants: read as uint8 (1 byte each)
    "F8_E8M0": torch.uint8,
    "F8_E4M3": torch.uint8,
    "F8_E5M2": torch.uint8,
    "F4_E2M1": torch.uint8,  # FP4 packed (2 per byte)
}


BLOCK_SIZE = 128
FP4_GROUP_SIZE = 32

# E2M1 FP4 lookup table: 4-bit index -> float value
# Bit layout: sign(1) | exponent(2) | mantissa(1), bias=1
_FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.bfloat16)


def decode_e8m0_scale_bytewise(scale_bytes: torch.Tensor) -> torch.Tensor:
    """
    Decode E8M0 (unsigned 8-bit exponent-only) scale to float32.
    Does NOT rely on torch.float8_e8m0fnu dtype attribute (which is missing
    in the container's older torch). Works directly on the raw bytes:
    E8M0 value = 2^(byte - 127), since E8M0 has no mantissa.

    Args:
        scale_bytes: 1-byte dtype tensor (uint8 or int8) holding E8M0 exponents
    Returns:
        float32 tensor with the decoded scale values
    """
    assert scale_bytes.element_size() == 1, f"Expected 1-byte dtype, got {scale_bytes.dtype}"
    # Reinterpret raw bytes as int32 (0-255), then left-shift 23 bits to align
    # the E8M0 exponent into IEEE 754 float32 exponent position. This gives
    # us a bit-pattern that .view(torch.float32) interprets as 2^(exp - 127).
    exp_bits = scale_bytes.to(torch.int32) << 23
    return exp_bits.view(torch.float32)


def is_expert_weight(name: str) -> bool:
    """Check if a weight belongs to an expert (MoE) layer, excluding shared_experts."""
    return "experts" in name and "shared_experts" not in name


def load_shard_safetensors(path):
    """
    Hand-parse a safetensors shard to bypass the safetensors dtype system
    entirely.

    Why hand-parse:
      - safetensors 0.7.0 (the container's version) has PySafeSlice with
        only get_dtype() and get_shape(); get_data() does NOT exist on it.
      - safe_open(framework="np") fails with "data type 'bfloat16' not
        understood" because this old safetensors' numpy backend doesn't
        know how to map the BF16 dtype string to a numpy dtype on this
        platform's numpy build.
      - safe_open(framework="pt") + get_tensor() triggers the
        torch.float8_e8m0fnu attribute lookup that the container's older
        torch lacks.

    So we parse the file format directly. The safetensors layout is:
      [8 bytes LE uint64 header_len][JSON header (header_len bytes)][raw data]
    The JSON header maps tensor name -> {dtype, shape, data_offsets} where
    data_offsets is [start, end] byte offsets into the raw data section.
    """
    tensors = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_section_offset = 8 + header_len
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            dstart, dend = meta["data_offsets"]
            abs_start = data_section_offset + dstart
            abs_end = data_section_offset + dend
            f.seek(abs_start)
            raw = f.read(abs_end - abs_start)
            torch_dtype = SAFE_DTYPE.get(dtype_str, torch.uint8)
            tensors[key] = torch.frombuffer(raw, dtype=torch_dtype).reshape(shape)
    return tensors


def dequant_fp4_weight(weight_packed: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 weight to bf16 by unpacking FP4 E2M1 nibbles via LUT.
    Each int8 byte stores 2 FP4 values (low nibble + high nibble).
    Scale is E8M0 (1 byte per group of 32 elements along the last dim),
    decoded bytewise (no dtype attribute needed).

    Args:
        weight_packed: [out_features, in_features/2], int8, each byte = 2 FP4 values
        scale_bytes:   [out_features, in_features/32], 1-byte dtype, E8M0 scale per group of 32
    Returns:
        bf16 tensor [out_features, in_features]
    """
    out_features, packed_in = weight_packed.shape
    in_features = packed_in * 2

    # Unpack two FP4 values per byte via nibble extraction + LUT
    raw = weight_packed.to(torch.uint8)
    low_nibble = (raw & 0x0F).to(torch.long)
    high_nibble = (raw >> 4).to(torch.long)
    lut = _FP4_E2M1_LUT.to(weight_packed.device)
    low_vals = lut[low_nibble]
    high_vals = lut[high_nibble]
    # Interleave: [low_0, high_0, low_1, high_1, ...]
    fp4_values = torch.stack([low_vals, high_vals], dim=-1).reshape(out_features, in_features)

    # Decode E8M0 scale (bytewise, no dtype attribute)
    scale = decode_e8m0_scale_bytewise(scale_bytes)
    if scale.dim() == 2 and scale.shape[0] == out_features:
        num_groups_per_row = scale.shape[1]
    else:
        total_scales = scale.numel()
        num_groups_per_row = total_scales // out_features
        scale = scale.reshape(out_features, num_groups_per_row)
    actual_group_size = in_features // num_groups_per_row
    scale = scale.unsqueeze(-1).expand(-1, -1, actual_group_size).reshape(out_features, in_features)

    return (fp4_values * scale).to(torch.bfloat16)


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """
    Dequantize FP8 weight to BF16 using block-wise scale.
    Patched: accepts both float8_e8m0fnu (new torch) and 1-byte raw (old torch).
    """
    shape = weight.shape
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    M, N = shape

    # Try to convert scale to float32; fallback to bytewise E8M0 decode
    try:
        if scale.dtype in (torch.uint8, torch.int8):
            # Old torch / raw bytes path (V4 Flash on container torch)
            scale = decode_e8m0_scale_bytewise(scale)
        else:
            scale = scale.float()
    except AttributeError:
        # torch.float8_e8m0fnu doesn't exist on this torch version
        if scale.element_size() == 1:
            scale = decode_e8m0_scale_bytewise(scale)
        else:
            raise

    # Pad to nearest multiple of block_size if needed
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size
    if pad_m or pad_n:
        weight = torch.nn.functional.pad(weight, (0, pad_n, 0, pad_m))
    Mp, Np = weight.shape

    # V3.2 dequant: reshape into blocks, scale, reshape back
    weight = weight.view(
        Mp // block_size, block_size,
        Np // block_size, block_size
    ).transpose(1, 2).contiguous().view(-1, block_size * block_size)

    weight = (weight.float() * scale.reshape(-1, 1)).to(torch.bfloat16)

    weight = weight.view(
        Mp // block_size, Np // block_size,
        block_size, block_size
    ).transpose(1, 2).contiguous().view(Mp, Np)

    if pad_m or pad_n:
        weight = weight[:M, :N]

    return weight


def main(fp8_path, bf16_path, device="cpu"):
    torch.set_default_dtype(torch.bfloat16)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(bf16_path, exist_ok=True)

    # 1. Copy non-safetensor files (config.json, tokenizer, etc.)
    print("Copying auxiliary files...")
    for file_path in glob(os.path.join(fp8_path, "*")):
        fname = os.path.basename(file_path)
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        dst = os.path.join(bf16_path, fname)
        if os.path.isfile(file_path):
            shutil.copy2(file_path, dst)
            print(f"  Copied {fname}")

    # 2. Load model index
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # 3. Pre-build scale_inv lookup: "xxx.weight" <-> "xxx.scale"
    scale_inv_map = {}
    all_scale_names = set()
    for name in weight_map:
        if name.endswith("scale"):
            weight_name = name[:-len("scale")] + "weight"
            if weight_name in weight_map:
                all_scale_names.add(name)
                scale_inv_map[weight_name] = name

    fp4_scale_map = {k: v for k, v in scale_inv_map.items() if is_expert_weight(k)}
    fp8_scale_map = {k: v for k, v in scale_inv_map.items() if not is_expert_weight(k)}

    print(f"Model: DeepSeek-V4-Flash")
    print(f"Device: {device}")
    print(f"Total keys in index: {len(weight_map)}")
    print(f"FP4 expert weights with scale: {len(fp4_scale_map)}")
    print(f"FP8 non-expert weights with scale: {len(fp8_scale_map)}")
    print(f"Scale entries: {len(all_scale_names)}")

    # 4. Process safetensor files one by one
    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    converted_count = 0
    kept_count = 0

    for safetensor_file in tqdm(safetensor_files, desc="Converting FP8/FP4 -> BF16"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_shard_safetensors(safetensor_file)
        new_state_dict = {}

        for weight_name, weight in current_state_dict.items():
            # Skip scale tensors (will be removed from output)
            if weight_name in all_scale_names:
                continue

            # Check if this weight has a corresponding scale
            if weight.element_size() == 1 and weight_name in scale_inv_map:
                scale_inv_name = scale_inv_map[weight_name]
                try:
                    # Re-load shard to get the scale tensor (may be in same or other shard)
                    scale_inv = current_state_dict.get(scale_inv_name)
                    if scale_inv is None:
                        # Load from a different shard
                        scale_file = weight_map.get(scale_inv_name)
                        if scale_file:
                            scale_inv = load_shard_safetensors(
                                os.path.join(fp8_path, scale_file)
                            )[scale_inv_name]
                        else:
                            raise KeyError(scale_inv_name)

                    if is_expert_weight(weight_name):
                        # FP4 expert weight -> MXFP4 dequant
                        result = dequant_fp4_weight(weight, scale_inv)
                    else:
                        # FP8 non-expert weight -> block-wise dequant
                        result = weight_dequant(weight, scale_inv)
                    new_state_dict[weight_name] = result
                    converted_count += 1
                except Exception as e:
                    print(f"  Warning: dequant failed for {weight_name}: {e}")
                    new_state_dict[weight_name] = weight
            else:
                # BF16/FP32 pass-through (norms, biases, gate, hc_*, attn_sink, etc.)
                new_state_dict[weight_name] = weight
                kept_count += 1

        # Save converted shard
        save_file(new_state_dict, os.path.join(bf16_path, file_name))

    # 5. Update model index: remove all scale_inv entries
    new_weight_map = {k: v for k, v in weight_map.items() if k not in all_scale_names}
    new_index = {
        "metadata": model_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(bf16_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    print(f"\nDone!")
    print(f"  FP8/FP4 -> BF16 converted: {converted_count}")
    print(f"  Already BF16/FP32 (kept): {kept_count}")
    print(f"  Scale entries removed: {len(all_scale_names)}")
    print(f"  Output keys: {len(new_weight_map)} (was {len(weight_map)})")
    print(f"  Output saved to: {bf16_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek-V4-Flash FP8/FP4 checkpoint to BF16"
    )
    parser.add_argument("--input-fp8-hf-path", type=str, required=True,
                        help="Path to the V4 Flash FP8/FP4 HuggingFace model dir")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True,
                        help="Path to the output BF16 model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"],
                        help="Device for dequantization (default: cpu)")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path, args.device)
