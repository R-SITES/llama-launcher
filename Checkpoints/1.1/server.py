#!/usr/bin/env python3
"""
llama.cpp Launch Command Builder
Serves a web UI at localhost:9876 that lists your GGUF models,
recommends launch flags based on model size/architecture & hardware,
and generates copy-paste ready commands.
"""

import http.server
import json
import os
import re
import socketserver
import subprocess


PORT = 9876
MODELS_DIR = os.path.expanduser("~/models")


# ── System hardware detection ────────────────────────────────────────
def get_system_info():
    info = {}

    # CPU
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.split("\n"):
            if "Model name:" in line:
                info["cpu"] = line.split(":", 1)[1].strip()
            elif "Core(s) per socket:" in line:
                info["cpu_cores"] = line.split(":", 1)[1].strip()
            elif "Thread(s) per core:" in line:
                info["cpu_threads"] = line.split(":", 1)[1].strip()
    except Exception:
        info["cpu"] = "Unknown CPU"
        info["cpu_cores"] = "?"
        info["cpu_threads"] = "?"

    # RAM from /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = parts[1].strip()
        kb = lambda s: int(s.replace(" kB", "").strip())
        info["ram_total_kb"] = kb(meminfo.get("MemTotal", "0 kB"))
        info["ram_total_gb"] = round(info["ram_total_kb"] / 1024 / 1024, 1)
        info["ram_available_kb"] = kb(meminfo.get("MemAvailable", "0 kB"))
        info["ram_available_gb"] = round(info["ram_available_kb"] / 1024 / 1024, 1)
    except Exception:
        info["ram_total_kb"] = 0
        info["ram_total_gb"] = 0
        info["ram_available_kb"] = 0
        info["ram_available_gb"] = 0

    # GPU VRAM from lspci -v (prefetchable memory on VGA/3D controllers)
    try:
        out = subprocess.run(["lspci", "-v"], capture_output=True, text=True, timeout=5)
        blocks = out.stdout.split("\n\n")
        vram_bytes = 0
        gpu_name = ""
        for block in blocks:
            lines = block.split("\n")
            first = lines[0] if lines else ""
            if ("VGA" in first or "3D" in first) and "Intel" in first:
                # Try to get GPU name from subsystem or description
                for l in lines:
                    if "Subsystem:" in l:
                        gpu_name = l.split(":", 1)[1].strip()
                        break
                if not gpu_name:
                    # Extract from first line
                    gpu_name = first.split(":", 1)[-1].strip()
                for l in lines:
                    m = re.search(r"prefetchable\)\s+\[size=(\d+[MGT]?)\]", l)
                    if m:
                        val = m.group(1)
                        if val.endswith("G"):
                            vram_bytes += int(val[:-1]) * 1024 * 1024 * 1024
                        elif val.endswith("M"):
                            vram_bytes += int(val[:-1]) * 1024 * 1024
                        elif val.endswith("T"):
                            vram_bytes += int(val[:-1]) * 1024 * 1024 * 1024 * 1024
        info["gpu_name"] = gpu_name or "Intel Arc"
        info["vram_gb"] = round(vram_bytes / (1024 ** 3), 1) if vram_bytes else 0
        info["vram_bytes"] = vram_bytes
    except Exception:
        info["gpu_name"] = "Unknown GPU"
        info["vram_gb"] = 0
        info["vram_bytes"] = 0

    return info


SYSTEM_INFO = get_system_info()


# ── Recommendation engine ────────────────────────────────────────────
def recommend_flags(filename, filesize_bytes):
    size_gb = filesize_bytes / (1024 ** 3)
    vram_gb = SYSTEM_INFO.get("vram_gb", 0)
    ram_gb = SYSTEM_INFO.get("ram_total_gb", 0)
    name_lower = filename.lower()

    is_reasoning = any(kw in name_lower for kw in ["reason", "r1", "o1", "opus", "sonnet", "claude"])
    is_uncensored = "uncensored" in name_lower or "hc" in name_lower or "hauhau" in name_lower
    quant_type = ""
    for kw in ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0", "q3_k_m", "q3_k_s", "q2_k", "q2_x", "q2_k_p", "iq4_xs", "iq4_nl", "iq3_xxs", "iq3_s", "iq3_m", "iq2_s", "iq2_m", "iq2_xxs", "tq3_4s", "tq4_0"]:
        if kw in name_lower:
            quant_type = kw
            break

    # Context window — base on size, but cap by VRAM
    if size_gb >= 20:
        ctx = 8192
    elif size_gb >= 10:
        ctx = 4096
    elif size_gb >= 5:
        ctx = 4096
    else:
        ctx = 2048
    if is_reasoning and ctx < 4096:
        ctx = 4096

    # Batch size — scale by VRAM and model size
    if vram_gb >= 24:
        batch = 2048
    elif vram_gb >= 12:
        batch = 1024
    elif vram_gb >= 8:
        batch = 512
    else:
        batch = 256

    # For very large models, reduce batch slightly to avoid OOM
    if size_gb >= 20 and batch >= 2048:
        batch = 1024

    # Thread count — use half of physical cores for best performance
    try:
        cores = int(SYSTEM_INFO.get("cpu_cores", 0))
        threads = cores * int(SYSTEM_INFO.get("cpu_threads", 1))
        threads = max(threads // 2, 4)  # at least 4 threads
    except Exception:
        threads = 8

    flags = []
    flags.append(f"-m {MODELS_DIR}/{filename}")
    flags.append(f"-c {ctx}")
    flags.append("-ngl 99")
    flags.append(f"-b {batch}")
    flags.append(f"-t {threads}")
    flags.append("--port 8080")

    if is_uncensored:
        flags.append("--temp 0.7")
        flags.append("--repeat_penalty 1.1")
    else:
        flags.append("--temp 0.5")
        flags.append("--repeat_penalty 1.05")

    # Large models: memory lock & NUMA
    if size_gb >= 15 or vram_gb >= 16:
        flags.append("--mlock")
        flags.append("--numa")

    # MMDOT / multimodal if mmproj present
    mmproj_path = os.path.join(MODELS_DIR, "mmproj-BF16.gguf")
    if os.path.exists(mmproj_path):
        flags.append(f"--mmproj {mmproj_path}")

    # Chat template hint
    if "qwen" in name_lower:
        flags.append("--chat-template qwen2.5")
    elif "gemma" in name_lower:
        flags.append("--chat-template gemma")

    return {
        "flags": flags,
        "size_gb": round(size_gb, 2),
        "arch_hints": {
            "reasoning": is_reasoning,
            "uncensored": is_uncensored,
            "quant": quant_type or "unknown",
        },
        "ctx": ctx,
        "batch": batch,
        "threads": threads,
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/models":
            self._serve_models()
        elif self.path == "/api/system":
            self._serve_system()
        elif self.path == "/api/flags":
            self._serve_flags()
        elif self.path == "/":
            self._serve_index()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/recommend":
            self._serve_recommend()
        elif self.path == "/api/optimize":
            self._serve_optimize()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_models(self):
        models = []
        if os.path.isdir(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                fp = os.path.join(MODELS_DIR, f)
                if os.path.isfile(fp) and f.endswith(".gguf"):
                    size = os.path.getsize(fp)
                    rec = recommend_flags(f, size)
                    models.append({
                        "name": f,
                        "size_gb": rec["size_gb"],
                        "flags": rec["flags"],
                        "arch_hints": rec["arch_hints"],
                        "ctx": rec["ctx"],
                        "batch": rec["batch"],
                        "threads": rec["threads"],
                    })
        self._json({"models": models})

    def _serve_system(self):
        self._json(SYSTEM_INFO)

    def _serve_flags(self):
        flags_path = os.path.join(os.path.dirname(__file__), "flags_db.json")
        if os.path.exists(flags_path):
            with open(flags_path, "r") as f:
                flags_db = json.load(f)
            self._json(flags_db)
        else:
            self._json({"error": "flags_db.json not found"}, 404)

    def _serve_recommend(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        fname = body.get("name", "")
        size = body.get("size", 0)
        rec = recommend_flags(fname, size)
        self._json(rec)

    def _serve_optimize(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        model_info = body.get("model", {})
        sys_info = body.get("system", {})
        current_flags = body.get("currentFlags", {})
        flag_reference = body.get("flagReference", [])
        timestamp = body.get("timestamp", "")

        # Build a prompt for Hermes to analyze and recommend optimal flags
        model_name = model_info.get("name", "unknown")
        model_size = model_info.get("size_gb", 0)
        model_ctx = model_info.get("ctx", 0)
        model_batch = model_info.get("batch", 0)
        model_threads = model_info.get("threads", 0)
        model_arch = model_info.get("arch_hints", {})
        model_flags = model_info.get("flags", [])

        cpu = sys_info.get("cpu", "Unknown")
        cpu_cores = sys_info.get("cpu_cores", "?")
        cpu_threads = sys_info.get("cpu_threads", "?")
        ram_total = sys_info.get("ram_total_gb", 0)
        ram_available = sys_info.get("ram_available_gb", 0)
        gpu_name = sys_info.get("gpu_name", "Unknown")
        vram_gb = sys_info.get("vram_gb", 0)

        is_reasoning = model_arch.get("reasoning", False)
        is_uncensored = model_arch.get("uncensored", False)
        quant = model_arch.get("quant", "unknown")

        current_flag_descriptions = []
        for fid, fdata in current_flags.items():
            if fdata.get("active"):
                flag_name = fdata.get("flag", "?")
                flag_val = fdata.get("value", "")
                desc = f"  - {flag_name}"
                if flag_val:
                    desc += f" = {flag_val}"
                current_flag_descriptions.append(desc)

        // Build flag reference for Hermes — compact format
        flag_ref_text = ""
        if flag_reference:
            flag_ref_text = "AVAILABLE FLAG IDS (use these as keys in recommended_flags):\n"
            for fr in flag_reference:
                flag_ref_text += "  - %s: %s" % (fr['id'], fr['name'])
                if fr.get('full'):
                    flag_ref_text += " (%s)" % fr['full']
                flag_ref_text += " [type=%s, default=%s]\n" % (fr.get('type','bool'), fr.get('default',''))

        prompt = f"""Analyze this LLM model and system to recommend optimal llama.cpp server launch flags.

MODEL:
- Name: {model_name}
- Size: {model_size} GB
- Quantization: {quant}
- Context: {model_ctx} tokens
- Architecture hints: Reasoning={is_reasoning}, Uncensored={is_uncensored}
- Current recommended flags: {', '.join(model_flags[:5])}{'...' if len(model_flags) > 5 else ''}

SYSTEM:
- CPU: {cpu} ({cpu_cores} cores / {cpu_threads} threads)
- RAM: {ram_total} GB total, {ram_available} GB available
- GPU: {gpu_name} ({vram_gb} GB VRAM)
- Framework: Intel oneAPI SYCL

CURRENT ACTIVE FLAGS:
{chr(10).join(current_flag_descriptions) if current_flag_descriptions else 'None set'}

{flag_ref_text}

Please provide optimized flag recommendations. Consider:
1. Best context size (-c / --ctx-size) for this model size and GPU VRAM. For reasoning models, use at least 4096.
2. Optimal batch size (-b / --batch-size) for VRAM capacity.
3. GPU offload strategy (--gpu-layers / -ngl). Use 99 to offload all layers to GPU.
4. Best sampling parameters for this model type (reasoning vs conversational vs uncensored).
5. Special flags for Intel Arc SYCL backend (--mlock, --numa for large models).
6. Performance tuning: --threads (-t), --threads-batch (-T), --verbose.
7. Memory and NUMA flags for models >= 15GB or VRAM >= 16GB.

Return your response as JSON with this EXACT structure:
{{
  "recommended_flags": {{
    "flag_id": {{
      "active": true,
      "value": "recommended_value"
    }}
  }},
  "message": "Human-readable explanation of your recommendations"
}}

CRITICAL: 
- Use ONLY the exact flag IDs listed above (e.g., '--ctx-size', '--gpu-layers', '--threads').
- Only include flags that differ from defaults or are specifically recommended.
- Focus on performance-critical settings for Intel Arc B70 with {vram_gb}GB VRAM.
- Set "active": true for flags to enable, false to disable.
- Keep the message concise but informative."""

        # Spawn Hermes to analyze
        hermes_prompt = f"[OPTIMIZE TASK]\n{prompt}"

        try:
            result = subprocess.run(
                ["hermes", "chat", "-q", hermes_prompt, "--quiet", "--toolsets", "web,terminal,search"],
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ, "HERMES_HOME": os.path.expanduser("~/.hermes")}
            )

            # Parse Hermes output - look for JSON in the response
            output = result.stdout or result.stderr

            # Try to extract JSON from the output
            json_start = output.find("{")
            json_end = output.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                try:
                    response_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try a more lenient parse
                    response_data = json.loads(json_str, strict=False)
            else:
                # No JSON found, return raw output as message
                response_data = {
                    "message": f"Hermes response:\n{output[:500]}",
                    "recommended_flags": {}
                }

            self._json(response_data)

        except subprocess.TimeoutExpired:
            self._json({
                "message": "Hermes timed out after 90 seconds. Try again or check if Hermes is running.",
                "recommended_flags": {}
            }, 200)
        except Exception as e:
            self._json({
                "message": f"Error running Hermes: {str(e)}",
                "recommended_flags": {}
            }, 200)

    def _serve_index(self):
        with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
            html = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 llama.cpp Launch Command Builder")
        print(f"   Open http://localhost:{PORT} in your browser")
        print(f"   Models dir: {MODELS_DIR}")
        print(f"   CPU: {SYSTEM_INFO.get('cpu', 'Unknown')}")
        print(f"   RAM: {SYSTEM_INFO.get('ram_total_gb', '?')}GB | VRAM: {SYSTEM_INFO.get('vram_gb', '?')}GB")
        print(f"   Press Ctrl+C to stop")
        httpd.serve_forever()
