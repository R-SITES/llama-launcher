"""Convert Qwen3.5MoE GGUF → Orthrus-MoE GGUF with diff tensor copies (v4)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llama.cpp', 'gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

if len(sys.argv) < 3:
    print("Usage: convert-orthrus-moe-gguf.py <input.gguf> <output.gguf>")
    sys.exit(1)
MODEL_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

print("Reading source...")
reader = GGUFReader(MODEL_PATH)
fields = reader.fields
tensor_lookup = {t.name: t for t in reader.tensors}
n_layer = int(fields['qwen35moe.block_count'].parts[-1].tolist()[0])
attn_layers = [i for i in range(n_layer) if f'blk.{i}.attn_output.weight' in tensor_lookup]
print(f"Attention layers: {attn_layers}")

SKIP_KEYS = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}

def get_gtype(f):
    """Determine GGUF value type from field parts"""
    parts = f.parts
    if len(parts) >= 3:
        ti = parts[1]
        tval = ti.tolist()[0] if hasattr(ti, 'tolist') else int(ti)
        try:
            return GGUFValueType(tval)
        except ValueError:
            pass
    # Fallback: check last part dtype
    last = parts[-1]
    if hasattr(last, 'dtype'):
        if last.dtype == np.uint8 and len(parts) >= 5:
            return GGUFValueType.STRING
        elif last.dtype in (np.float32, np.float64):
            return GGUFValueType.FLOAT32
        elif last.dtype == np.int32:
            return GGUFValueType.INT32
    return GGUFValueType.UINT32  # default

def get_val(f):
    """Extract value from field"""
    parts = f.parts
    last = parts[-1]
    raw = last.tolist() if hasattr(last, 'tolist') else last
    gtype = get_gtype(f)
    
    if gtype == GGUFValueType.STRING:
        return bytes(raw).decode('utf-8', errors='replace')
    elif gtype == GGUFValueType.ARRAY:
        return [int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for v in raw]
    elif gtype in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return float(raw[0]) if isinstance(raw, (list, np.ndarray)) else float(raw)
    elif gtype == GGUFValueType.INT32:
        return int(raw[0]) if isinstance(raw, (list, np.ndarray)) else int(raw)
    elif gtype == GGUFValueType.BOOL:
        return bool(raw[0]) if isinstance(raw, (list, np.ndarray)) else bool(raw)
    else:
        # UINT32, UINT64, etc.
        return int(raw[0]) if isinstance(raw, (list, np.ndarray)) else int(raw)

# Build KV entries
kv_entries = []
for key in reader.fields:
    if key in SKIP_KEYS or key.startswith('GGUF.'):
        continue
    f = fields[key]
    gtype = get_gtype(f)
    val = get_val(f)
    
    if key == 'general.architecture':
        continue  # handled by writer constructor
    elif key.startswith('qwen35moe.'):
        key = 'orthrus_moe.' + key[len('qwen35moe.'):]
    
    # Skip tokenizer keys (handled separately via native API)
    if key.startswith('tokenizer.') or key == 'general.chat_template':
        continue
    
    kv_entries.append((key, val, gtype))

print(f"  {len(kv_entries)} metadata keys")

# Write
writer = GGUFWriter(OUT_PATH, "orthrus_moe", use_temp_file=True)

for key, val, gtype in kv_entries:
    if gtype == GGUFValueType.STRING:
        writer.add_string(key, val)
    elif gtype == GGUFValueType.ARRAY:
        writer.add_array(key, val)
    elif gtype in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        writer.add_float32(key, float(val))
    elif gtype == GGUFValueType.INT32:
        writer.add_int32(key, int(val))
    elif gtype == GGUFValueType.BOOL:
        writer.add_bool(key, bool(val))
    else:
        writer.add_uint32(key, int(val))

writer.add_uint32('orthrus_moe.block_size', 32)
writer.add_uint32('orthrus_moe.mask_token_id', 151669)

# Add tokenizer using field data indices
for key in fields:
    if key.startswith('tokenizer.') or key == 'general.chat_template':
        f = fields[key]
        gtype = get_gtype(f)
        
        if key == 'tokenizer.ggml.model':
            raw = bytes(f.parts[-1].tolist()).decode('utf-8', errors='replace')
            writer.add_tokenizer_model(raw)
        elif key == 'tokenizer.ggml.tokens':
            # Extract token strings from data indices
            tokens = []
            for idx in f.data:
                token_bytes = bytes(f.parts[idx].tolist())
                tokens.append(token_bytes.decode('utf-8', errors='replace'))
            # Pad to model vocab size for mask_token_id support (151669 needs padded list)
            # Vocab size from token_embd shape
            tok_embd_t = tensor_lookup.get('token_embd.weight')
            n_vocab_val = int(tok_embd_t.shape[0]) if tok_embd_t else 151936
            while len(tokens) < n_vocab_val:
                tokens.append(f"[PAD{len(tokens)}]")
            writer.add_token_list(tokens)
        elif key == 'tokenizer.ggml.merges':
            merges = []
            for idx in f.data:
                merge_bytes = bytes(f.parts[idx].tolist())
                merges.append(merge_bytes.decode('utf-8', errors='replace'))
            writer.add_token_merges(merges)
        elif key == 'tokenizer.ggml.bos_token_id':
            val = get_val(f)
            writer.add_bos_token_id(val)
        elif key == 'tokenizer.ggml.eos_token_id':
            val = get_val(f)
            writer.add_eos_token_id(val)
        elif key == 'tokenizer.ggml.padding_token_id':
            val = get_val(f)
            writer.add_pad_token_id(val)
        elif key == 'tokenizer.ggml.pre':
            raw = bytes(f.parts[-1].tolist()).decode('utf-8', errors='replace')
            writer.add_string(key, raw)  # pre-tokenizer type string, not the boolean add_space_prefix
        elif key == 'tokenizer.ggml.token_type':
            types = []
            for idx in f.data:
                type_bytes = bytes(f.parts[idx].tolist())
                types.append(int.from_bytes(type_bytes, 'little'))
            writer.add_token_types(types)
        elif key == 'general.chat_template':
            raw = bytes(f.parts[-1].tolist()).decode('utf-8', errors='replace')
            writer.add_chat_template(raw)

DIFF = {
    'attn_q': 'attn_q_diff', 'attn_k': 'attn_k_diff', 'attn_v': 'attn_v_diff',
    'attn_output': 'attn_output_diff', 'attn_q_norm': 'attn_q_norm_diff', 'attn_k_norm': 'attn_k_norm_diff'
}

def add_t(t, name=None):
    nm = name or t.name
    if t.tensor_type == GGMLQuantizationType.F32:
        writer.add_tensor(nm, t.data)
    else:
        writer.add_tensor(nm, t.data, raw_shape=list(t.data.shape), raw_dtype=t.tensor_type)

print(f"  Adding {len(reader.tensors)} source tensors...")
for i, t in enumerate(reader.tensors):
    add_t(t)
    if (i+1) % 100 == 0: print(f"    {i+1}/{len(reader.tensors)}")

diff_count = 0
for il in attn_layers:
    for src, dst in DIFF.items():
        sn = f'blk.{il}.{src}.weight'
        if sn in tensor_lookup:
            add_t(tensor_lookup[sn], name=f'blk.{il}.{dst}.weight')
            diff_count += 1
print(f"  {diff_count} diff tensors (total: {len(reader.tensors) + diff_count})")

print("  Writing header/KV/tensors...")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()

print(f"\nDone! {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e9:.2f} GB)")
