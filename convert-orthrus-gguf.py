#!/usr/bin/env python3
"""Convert Orthrus safetensors to GGUF using official gguf library."""
import os, sys, json, torch, numpy as np
from pathlib import Path
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType

def convert(model_dir, output_path):
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    n_layer = config['num_hidden_layers']
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    n_head_kv = config['num_key_value_heads']
    n_ff = config['intermediate_size']
    n_embd_head = config.get('head_dim', 128)
    n_vocab = config['vocab_size']
    n_gqa = n_embd_head * n_head_kv
    
    print(f"Orthrus: {n_layer} layers, {n_embd} embd, {n_head} heads, {n_head_kv} KV heads")
    
    gguf_writer = GGUFWriter(output_path, "orthrus")
    
    # Architecture metadata
    gguf_writer.add_block_count(n_layer)
    gguf_writer.add_embedding_length(n_embd)
    gguf_writer.add_feed_forward_length(n_ff)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_head_count_kv(n_head_kv)
    gguf_writer.add_context_length(config.get('max_position_embeddings', 40960))
    gguf_writer.add_layer_norm_rms_eps(float(config.get('rms_norm_eps', 1e-6)))
    gguf_writer.add_rope_dimension_count(n_embd_head)
    gguf_writer.add_rope_freq_base(float(config.get('rope_parameters', {}).get('rope_theta', 1000000.0)))
    gguf_writer.add_file_type(0)  # F32
    gguf_writer.add_uint32("orthrus.block_size", config.get('block_size', 32))
    gguf_writer.add_uint32("orthrus.mask_token_id", config.get('mask_token_id', 151669))
    
    # Tokenizer
    gguf_writer.add_tokenizer_model("gpt2")
    gguf_writer.add_bos_token_id(config.get('bos_token_id', 151643))
    gguf_writer.add_eos_token_id(config.get('eos_token_id', 151645))
    gguf_writer.add_add_bos_token(True)
    
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path) as f:
            tok_data = json.load(f)
        vocab = tok_data.get('model', {}).get('vocab', {})
        if vocab:
            max_id = max(vocab.values())
            tokens = [""] * (max_id + 1)
            for token_str, token_id in vocab.items():
                tokens[token_id] = token_str
            # Pad to model vocab size so mask_token is in valid range
            while len(tokens) < n_vocab:
                tokens.append(f"[PAD{len(tokens)}]")
            gguf_writer.add_token_list(tokens)
        merges = tok_data.get('model', {}).get('merges', [])
        if merges:
            # Qwen3 merges are lists [token1, token2], convert to "token1 token2" format
            merge_strings = [f"{m[0]} {m[1]}" if isinstance(m, list) else m for m in merges]
            gguf_writer.add_token_merges(merge_strings)
        print(f"Tokenizer: {len(tokens)} tokens, {len(merges)} merges")

    # Define tensor name mapping
    TENSORS = {}
    def add(safe_name, gguf_name, shape):
        TENSORS[safe_name] = (gguf_name, shape)
    add("model.embed_tokens.weight", "token_embd.weight", [n_vocab, n_embd])
    add("model.norm.weight", "output_norm.weight", [n_embd])
    add("lm_head.weight", "output.weight", [n_vocab, n_embd])
    for i in range(n_layer):
        b = f"blk.{i}"
        add(f"model.layers.{i}.input_layernorm.weight", f"{b}.attn_norm.weight", [n_embd])
        add(f"model.layers.{i}.self_attn.q_proj.weight", f"{b}.attn_q.weight", [n_embd_head * n_head, n_embd])
        add(f"model.layers.{i}.self_attn.k_proj.weight", f"{b}.attn_k.weight", [n_gqa, n_embd])
        add(f"model.layers.{i}.self_attn.v_proj.weight", f"{b}.attn_v.weight", [n_gqa, n_embd])
        add(f"model.layers.{i}.self_attn.o_proj.weight", f"{b}.attn_output.weight", [n_embd, n_embd_head * n_head])
        add(f"model.layers.{i}.self_attn.q_norm.weight", f"{b}.attn_q_norm.weight", [n_embd_head])
        add(f"model.layers.{i}.self_attn.k_norm.weight", f"{b}.attn_k_norm.weight", [n_embd_head])
        add(f"model.layers.{i}.post_attention_layernorm.weight", f"{b}.ffn_norm.weight", [n_embd])
        add(f"model.layers.{i}.mlp.gate_proj.weight", f"{b}.ffn_gate.weight", [n_ff, n_embd])
        add(f"model.layers.{i}.mlp.down_proj.weight", f"{b}.ffn_down.weight", [n_embd, n_ff])
        add(f"model.layers.{i}.mlp.up_proj.weight", f"{b}.ffn_up.weight", [n_ff, n_embd])
        add(f"model.layers.{i}.self_attn.q_proj_diff.weight", f"{b}.attn_q_diff.weight", [n_embd_head * n_head, n_embd])
        add(f"model.layers.{i}.self_attn.k_proj_diff.weight", f"{b}.attn_k_diff.weight", [n_gqa, n_embd])
        add(f"model.layers.{i}.self_attn.v_proj_diff.weight", f"{b}.attn_v_diff.weight", [n_gqa, n_embd])
        add(f"model.layers.{i}.self_attn.o_proj_diff.weight", f"{b}.attn_output_diff.weight", [n_embd, n_embd_head * n_head])
        add(f"model.layers.{i}.self_attn.q_norm_diff.weight", f"{b}.attn_q_norm_diff.weight", [n_embd_head])
        add(f"model.layers.{i}.self_attn.k_norm_diff.weight", f"{b}.attn_k_norm_diff.weight", [n_embd_head])
    
    # Write all tensors before header (gguf library handles ordering internally)
    st_file = list(model_dir.glob("*.safetensors"))[0]
    with safe_open(st_file, framework="pt", device="cpu") as st:
        for idx, (safe_name, (gguf_name, shape)) in enumerate(TENSORS.items()):
            if safe_name not in st.keys():
                print(f"  MISSING[{idx}]: {safe_name}")
                continue
            tensor = st.get_tensor(safe_name)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()  # bf16 → f32
            arr = tensor.numpy()
            gguf_writer.add_tensor(gguf_name, arr)
            if idx % 50 == 0:
                print(f"  [{idx}] {gguf_name}: {arr.shape} ({arr.nbytes/1e6:.1f}MB)", flush=True)
    
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    file_size = os.path.getsize(output_path)
    print(f"\nGGUF written: {output_path}")
    print(f"File size: {file_size/1e9:.2f} GB")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert-orthrus-gguf.py <model_dir> [output_path]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(model_dir, "orthrus-qwen3-1.7b.gguf")
    success = convert(model_dir, output_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
