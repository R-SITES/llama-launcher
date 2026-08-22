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
import threading
import time
import urllib.request
import urllib.parse
import urllib.error
import concurrent.futures
from pathlib import Path


PORT = 9876
MODELS_DIR = os.path.expanduser("~/models")

# ── Bench history store ──────────────────────────────────────────────
BENCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_history.json")
BENCH_MAX = 500  # keep at most this many entries

def load_bench_history():
    """Load bench history entries from disk (list of dicts, newest last)."""
    try:
        with open(BENCH_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def save_bench_history(entries):
    """Persist bench history, capped at BENCH_MAX entries."""
    try:
        entries = entries[-BENCH_MAX:]
        with open(BENCH_FILE, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass

def append_bench_entry(entry):
    """Append one entry (dict) to the bench history file."""
    entries = load_bench_history()
    entries.append(entry)
    save_bench_history(entries)
    return entries

def parse_acceptance_from_log(log_path):
    """Extract the last 'draft acceptance = X' value from a llama-server log."""
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines):
            m = re.search(r"draft acceptance = ([\d.]+)", line)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None

def parse_tg_from_log(log_path):
    """Extract the last 'tg = X t/s' value from a llama-server log."""
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines):
            m = re.search(r"tg = ([\d.]+) t/s", line)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None

def current_llama_server_state():
    """Return (running, model_name, build_name) of the running llama-server if any."""
    try:
        result = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True, timeout=5)
        if not result.stdout.strip():
            return False, None, None
        pid = result.stdout.strip().split("\n")[0]
        proc_result = subprocess.run(["ps", "-p", pid, "-o", "args="], capture_output=True, text=True, timeout=5)
        args = proc_result.stdout.strip() if proc_result.stdout.strip() else ""
        model = None
        m = re.search(r"-m\s+(\S+)", args)
        if m:
            model = os.path.basename(m.group(1).rstrip("'\""))
        build = None
        m = re.search(r"(build-[^\s/]+)/bin/llama-server", args)
        if m:
            build = m.group(1)
        return True, model, build
    except Exception:
        return False, None, None


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


# ── HuggingFace Hub (GGUF models) ────────────────────────────────────
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def _get_architecture(config, tags):
    """Extract architecture from HF model config."""
    if not config:
        # Check tags
        for t in (tags or []):
            if t.startswith("architecture:"):
                return t.replace("architecture:", "", 1)
        return ""
    arch = config.get("model_type", "") or config.get("architectures", [""])[0] if isinstance(config.get("architectures"), list) else ""
    return arch

def search_hf_gguf(query="", page=1, page_size=100):
    """Search HuggingFace for GGUF models — fetches up to 5 pages."""
    all_results = []
    total_raw = 0
    seen_ids = set()

    for p in range(5):
        params = {
            "search": query or "llama",
            "sort": "downloads",
            "direction": -1,
            "limit": page_size,
            "full": "true",
            "config": "true",
        }
        if p > 0:
            params["offset"] = p * page_size

        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llama-launcher/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
        except Exception:
            break

        if not data:
            break

        for m in data:
            model_id = m.get("id", "")
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)

            siblings = m.get("siblings", [])
            rfilenames = [s.get("rfilename", "") for s in siblings]

            # GGUF filter — only show models with .gguf files
            gguf_files = [fn for fn in rfilenames if fn.endswith(".gguf")]
            if not gguf_files:
                continue

            # Collect ALL GGUF files with their quants
            gguf_file_list = []
            for fn in gguf_files:
                n = fn.lower()
                fsize = 0
                for s in siblings:
                    if s.get("rfilename") == fn:
                        fsize = s.get("size", 0)
                        break
                # Extract quantization from filename
                quant = ""
                for kw in ["q8_0","q6_k","q5_k_m","q5_k_s","q5_0","q4_k_m","q4_k_s","q4_0",
                           "q3_k_m","q3_k_s","q2_k","q2_k_p","q2_x","iq4_xs","iq4_nl",
                           "iq3_xxs","iq3_s","iq3_m","iq2_s","iq2_m","iq2_xxs","iq1_s","iq1_m",
                           "tq3_4s","tq4_0","bf16","f16","f32","fp16","fp32",
                           "nvfp4","apex","mxfp4","q4_k_xl"]:
                    if kw in n:
                        quant = kw
                        break
                if not quant:
                    quant = fn.rsplit(".", 2)[-2] if fn.count(".") >= 2 else ""
                gguf_file_list.append({
                    "filename": fn,
                    "size": fsize,
                    "quantization": quant,
                })

            # Pick representative info for legacy fields
            gguf_file_list.sort(key=lambda x: x["size"], reverse=True)
            gguf_filename = gguf_file_list[0]["filename"] if gguf_file_list else ""
            gguf_size = gguf_file_list[0]["size"] if gguf_file_list else 0
            quantization = gguf_file_list[0]["quantization"] if gguf_file_list else ""

            cache_dir_name = f"models--{model_id.replace('/', '--')}"
            is_cached = os.path.isdir(os.path.join(HF_CACHE_DIR, cache_dir_name))

            all_results.append({
                "model_id": model_id,
                "name": model_id,
                "gguf_file": gguf_filename,
                "gguf_size": gguf_size,
                "gguf_files": gguf_file_list,
                "size_gb": round(gguf_size / (1024**3), 2) if gguf_size else 0,
                "size_bytes": gguf_size,
                "quantization": quantization,
                "architecture": _get_architecture(m.get("config", {}), m.get("tags", [])),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "is_cached": is_cached,
                "source": "hf_hub",
                "last_modified": m.get("lastModified", ""),
                "description": (m.get("description", "") or "")[:200],
            })

        total_raw += len(data)
        if len(data) < page_size:
            break

    return {"results": all_results, "count": len(all_results), "total": total_raw}


# ── Background download with progress ──
_download_tasks = {}

def get_cache_size_for_model(model_id):
    cache_dir_name = f"models--{model_id.replace('/', '--')}"
    snapshots_dir = os.path.join(HF_CACHE_DIR, cache_dir_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return 0
    total = 0
    for root, dirs, files in os.walk(snapshots_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except:
                pass
    return total

def download_progress_poller(model_id, total_size):
    while _download_tasks.get(model_id, {}).get("status") == "downloading":
        dled = get_cache_size_for_model(model_id)
        _download_tasks[model_id]["downloaded"] = dled
        if total_size > 0:
            _download_tasks[model_id]["progress"] = min(dled / total_size, 1.0)
        time.sleep(2)

def download_worker(model_id, filename):
    _download_tasks[model_id] = {
        "status": "downloading",
        "progress": 0.0,
        "downloaded": 0,
        "total": 0,
        "error": ""
    }
    try:
        import huggingface_hub
        # Download a single GGUF file directly to ~/models/
        local_path = huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        # Verify download completed
        if os.path.isfile(local_path):
            _download_tasks[model_id]["status"] = "done"
            _download_tasks[model_id]["progress"] = 1.0
            _download_tasks[model_id]["downloaded"] = os.path.getsize(local_path)
            _download_tasks[model_id]["total"] = _download_tasks[model_id]["downloaded"]
        else:
            _download_tasks[model_id]["status"] = "error"
            _download_tasks[model_id]["error"] = "Download completed but file not found"
    except Exception as e:
        _download_tasks[model_id]["status"] = "error"
        _download_tasks[model_id]["error"] = str(e)[:500]

def start_download(model_id, filename):
    existing = _download_tasks.get(model_id, {})
    if existing.get("status") in ("downloading", "done"):
        return {"status": existing["status"], "progress": existing.get("progress", 0)}
    _download_tasks[model_id] = {
        "status": "starting",
        "progress": 0.0,
        "downloaded": 0,
        "total": 0,
        "error": ""
    }
    t = threading.Thread(target=download_worker, args=(model_id, filename), daemon=True)
    t.start()
    return {"status": "started"}

def get_download_status(model_id):
    task = _download_tasks.get(model_id)
    if not task:
        return {"status": "none"}
    return task


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if self.path == "/api/models":
            self._serve_models()
        elif self.path == "/api/system":
            self._serve_system()
        elif self.path == "/api/flags":
            self._serve_flags()
        elif self.path.startswith("/api/exec?"):
            self._serve_exec()
        elif self.path == "/api/kill":
            self._serve_kill()
        elif self.path == "/api/exit":
            self._serve_exit()
        elif self.path == "/api/builds":
            self._serve_builds()
        elif self.path.startswith("/api/launch?"):
            self._serve_launch()
        elif self.path == "/api/status":
            self._serve_status()
        elif parsed.path == "/api/launch-log":
            self._serve_launch_log()
        elif self.path == "/api/bench-history":
            self._serve_bench_history()
        elif parsed.path == "/api/hf-search-gguf":
            q = params.get("q", [""])[0]
            page = int(params.get("page", ["1"])[0])
            self._json(search_hf_gguf(q, page))
        elif parsed.path == "/api/download-status":
            model_id = params.get("model_id", [""])[0]
            self._json(get_download_status(model_id))
        elif self.path == "/":
            self._serve_index()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/recommend":
            self._serve_recommend()
        elif self.path == "/api/optimize":
            self._serve_optimize()
        elif self.path == "/api/download":
            self._serve_download()
        elif self.path == "/api/delete-model":
            self._serve_delete_model()
        elif self.path == "/api/bench-capture":
            self._serve_bench_capture()
        elif self.path == "/api/bench-clear":
            self._serve_bench_clear()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_download(self):
        """Start background download of a GGUF model from HF Hub."""
        content_len = int(self.headers.get('Content-Length', 0))
        if content_len:
            body = json.loads(self.rfile.read(content_len))
            model_id = body.get("model_id", "")
            filename = body.get("filename", "")
        else:
            model_id = ""
            filename = ""
        if not model_id or not filename:
            self._json({"error": "Need model_id and filename", "success": False}, 400)
            return
        result = start_download(model_id, filename)
        self._json({**result, "success": True})

    def _serve_delete_model(self):
        """Delete a model from ~/models/ or HF cache."""
        content_len = int(self.headers.get('Content-Length', 0))
        if not content_len:
            self._json({"error": "No request body", "success": False}, 400)
            return
        body = json.loads(self.rfile.read(content_len))
        name = body.get("name", "")
        model_id = body.get("model_id", "")
        if not name and not model_id:
            self._json({"error": "No model specified", "success": False}, 400)
            return
        try:
            import shutil
            # HF cached model — delete the whole cache directory
            if model_id:
                cache_dir_name = f"models--{model_id.replace('/', '--')}"
                cache_dir = os.path.join(HF_CACHE_DIR, cache_dir_name)
                if not os.path.isdir(cache_dir):
                    self._json({"error": f"Model not found in HF cache: {model_id}", "success": False}, 404)
                    return
                shutil.rmtree(cache_dir)
                self._json({"success": True, "message": f"Deleted {model_id} from HF cache"})
                return
            # Regular GGUF file in ~/models/
            file_path = os.path.join(MODELS_DIR, name)
            if not os.path.isfile(file_path):
                self._json({"error": f"File not found: {name}", "success": False}, 404)
                return
            os.remove(file_path)
            self._json({"success": True, "message": f"Deleted {name}"})
        except PermissionError:
            self._json({
                "error": f"Permission denied. Try: sudo rm -rf {cache_dir if model_id else file_path}",
                "success": False
            }, 500)
        except Exception as e:
            self._json({"error": f"Failed to delete: {e}", "success": False}, 500)

    def _serve_models(self):
        models = []
        seen_names = set()
        # Scan ~/models/ for GGUF files
        if os.path.isdir(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                fp = os.path.join(MODELS_DIR, f)
                if os.path.isfile(fp) and f.endswith(".gguf"):
                    size = os.path.getsize(fp)
                    mtime = os.path.getmtime(fp)
                    rec = recommend_flags(f, size)
                    models.append({
                        "name": f,
                        "size_gb": rec["size_gb"],
                        "flags": rec["flags"],
                        "arch_hints": rec["arch_hints"],
                        "ctx": rec["ctx"],
                        "batch": rec["batch"],
                        "threads": rec["threads"],
                        "modified_time": mtime,
                        "model_id": "",
                    })
                    seen_names.add(f)

        # Scan HF cache for GGUF models not already in ~/models/
        hf_cache_map = {}  # filename → model_id for matching local models
        if os.path.isdir(HF_CACHE_DIR):
            for entry in os.listdir(HF_CACHE_DIR):
                if not entry.startswith("models--"):
                    continue
                snap_dir = os.path.join(HF_CACHE_DIR, entry, "snapshots")
                if not os.path.isdir(snap_dir):
                    continue
                # Check download status
                model_id = entry.replace("models--", "", 1).replace("--", "/", 1)
                dl_status = _download_tasks.get(model_id, {})
                for snap_id in os.listdir(snap_dir):
                    snap_path = os.path.join(snap_dir, snap_id)
                    if not os.path.isdir(snap_path):
                        continue
                    for fname in os.listdir(snap_path):
                        if fname.endswith(".gguf") and fname not in seen_names:
                            hf_cache_map[fname] = model_id
                            fp = os.path.join(snap_path, fname)
                            try:
                                size = os.path.getsize(fp)
                                mtime = os.path.getmtime(fp)
                            except (FileNotFoundError, OSError):
                                continue
                            rec = recommend_flags(fname, size)
                            models.append({
                                "name": fname,
                                "size_gb": rec["size_gb"],
                                "flags": rec["flags"],
                                "arch_hints": rec["arch_hints"],
                                "ctx": rec["ctx"],
                                "batch": rec["batch"],
                                "threads": rec["threads"],
                                "modified_time": mtime,
                                "model_id": model_id,
                                "source": "hf_cache",
                                "download_progress": dl_status.get("progress", 0) if dl_status.get("status") == "downloading" else None,
                                "download_status": dl_status.get("status", ""),
                            })
                            seen_names.add(fname)
                    break  # only process first snapshot

        # ── Orthrus GGUF model entry ──
        orthrus_base = os.path.join(MODELS_DIR, "Orthrus-Qwen3-1.7B")
        orthrus_gguf = os.path.join(orthrus_base, "orthrus-qwen3-1.7b-Q4_K_M.gguf")
        orthrus_quant = "Q4_K_M"
        if not os.path.isfile(orthrus_gguf):
            orthrus_gguf = os.path.join(orthrus_base, "orthrus-qwen3-1.7b.gguf")
            orthrus_quant = "f32"
        if os.path.isfile(orthrus_gguf):
            size = os.path.getsize(orthrus_gguf)
            mtime = os.path.getmtime(orthrus_gguf)
            models.append({
                "name": f"ORTHRUS Qwen3-1.7B build-sycl-orthrus-b1 ({orthrus_quant})",
                "size_gb": round(size / (1024**3), 2),
                "flags": [f"-m {orthrus_gguf}"],
                "arch_hints": {"reasoning": False, "uncensored": False, "quant": orthrus_quant},
                "ctx": 4096,
                "batch": 512,
                "threads": 8,
                "modified_time": mtime,
                "model_id": "chiennv/Orthrus-Qwen3-1.7B",
            })


        # ── Orthrus-MoE 35B GGUF model entry ──
        orthrus_moe_gguf = os.path.join(MODELS_DIR, 'orthrus-qwen3.5moe-35b.gguf')
        if os.path.isfile(orthrus_moe_gguf):
            size = os.path.getsize(orthrus_moe_gguf)
            mtime = os.path.getmtime(orthrus_moe_gguf)
            models.append({
                'name': 'ORTHRUS-MoE Qwen3.5-35B build-sycl-orthrus-b1',
                'size_gb': round(size / (1024**3), 2),
                'flags': [f'-m {orthrus_moe_gguf}'],
                'arch_hints': {'reasoning': False, 'uncensored': False, 'quant': 'Q4_K_M'},
                'ctx': 4096,
                'batch': 256,
                'threads': 8,
                'modified_time': mtime,
                'model_id': 'restone/orthrus-qwen3.5moe-35b',
            })
        # Match local models to HF cache entries for -hf auto-fill
        # Remove common local-only suffixes (e.g. "-MTP-" infix) when matching
        def normalize_hf_name(n):
            base = n.replace(".gguf", "").lower()
            # Strip common local-only markers like "-mtp-" appearing anywhere
            for marker in ["-mtp-", "-mtp", "mtp-", "_mtp_", "-mtp_", "_mtp-"]:
                base = base.replace(marker, "-")
            # Collapse double hyphens from marker removal
            while "--" in base:
                base = base.replace("--", "-")
            # Strip leading/trailing hyphens
            base = base.strip("-")
            return base

        hf_by_normalized = {}
        for fname, mid in hf_cache_map.items():
            hf_by_normalized[normalize_hf_name(fname)] = {"model_id": mid, "hf_name": fname}

        for m in models:
            if not m.get("model_id") and m.get("source") != "hf_cache":
                norm = normalize_hf_name(m["name"])
                matched = hf_by_normalized.get(norm)
                if not matched:
                    for hf_norm, hf_data in hf_by_normalized.items():
                        if norm in hf_norm or hf_norm in norm:
                            matched = hf_data
                            break
                if matched:
                    m["model_id"] = matched["model_id"]
                    # Include the HF cache filename for correct quant derivation
                    hf_base = matched["hf_name"].replace(".gguf", "")
                    m["hf_tag"] = f"{matched['model_id']}:{hf_base}"

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

        # Build flag reference for Hermes - compact format
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

    def _serve_exec(self):
        """Execute a shell command and return its output."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        cmd_str = params.get("cmd", [""])[0]
        if not cmd_str:
            self._json({"error": "No command specified"}, 400)
            return
        try:
            result = subprocess.run(
                cmd_str, shell=True, capture_output=True, text=True, timeout=30
            )
            output = (result.stdout or "") + (result.stderr or "")
            self._json({"output": output or "(empty)", "code": result.returncode})
        except subprocess.TimeoutExpired:
            self._json({"error": "Command timed out after 30s"}, 200)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _serve_kill(self):
        """Kill llama-server via pkill, then lsof fallback."""
        results = []
        # Step 1: pkill -9 llama-server
        try:
            r = subprocess.run(
                ["pkill", "-9", "llama-server"], capture_output=True, text=True, timeout=5
            )
            killed = r.returncode == 0 or r.returncode == 1  # 1 = no process found
            results.append(f"pkill -9 llama-server: {'killed' if r.returncode == 0 else 'no process found (ok)'}")
        except Exception as e:
            results.append(f"pkill error: {e}")

        # Step 2: lsof on port 8080 for any remaining
        try:
            r = subprocess.run(
                ["lsof", "-i", ":8080"], capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in (r.stdout or "").split("\n") if l.strip()]
            # Filter out header line and empty lines; find llama-server entries
            for line in lines:
                if "llama-server" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        try:
                            subprocess.run(
                                ["kill", "-9", pid], capture_output=True, text=True, timeout=5
                            )
                            results.append(f"lsof found llama-server PID {pid} → kill -9 sent")
                        except Exception as e2:
                            results.append(f"kill -9 {pid}: {e2}")
            if not any("llama-server" in l.lower() for l in lines):
                results.append("lsof: no llama-server on port 8080 (all clean)")
        except Exception as e:
            results.append(f"lsof error: {e}")

        self._json({"output": "\n".join(results)})

    def _serve_exit(self):
        """Close the terminal window that was opened with the server."""
        import subprocess
        import os
        try:
            # Step 1: Try PID file first
            pid_file = os.path.expanduser("~/llama-term-pid.tmp")
            terminal_pid = None
            
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, "r") as f:
                        terminal_pid = int(f.read().strip())
                except:
                    pass
            
            if terminal_pid:
                try:
                    subprocess.run(["kill", "-9", str(terminal_pid)], capture_output=True, timeout=2)
                    os.unlink(pid_file)
                    return self._json({"output": f"Closed terminal PID {terminal_pid}"})
                except Exception:
                    pass
            
            # Step 2: Fallback — find the llama-server terminal by window title
            try:
                result = subprocess.run(
                    ["wmctrl", "-l"],
                    capture_output=True, text=True, timeout=3
                )
                for line in result.stdout.strip().split("\n"):
                    if "llama-server" in line.lower() or "llama" in line.lower():
                        win_id = line.split()[0]
                        subprocess.run(["wmctrl", "-i", "-c", win_id], capture_output=True, timeout=3)
                        return self._json({"output": "Closed llama-server terminal window"})
            except Exception:
                pass
            
            # Step 3: Last resort — kill only the last xdg-terminal-exec child
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "xdg-terminal-exec"],
                    capture_output=True, text=True, timeout=3
                )
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid.strip():
                        subprocess.run(["kill", "-9", pid.strip()], capture_output=True, timeout=2)
                return self._json({"output": "Closed terminal (last resort)"})
            except Exception:
                pass
            
            return self._json({"output": "No terminal window to close"})
            
        except Exception as e:
            return self._json({"output": f"Error: {e}"})

    def _serve_launch(self):
        """Launch llama-server using the raw command via a terminal emulator."""
        from urllib.parse import urlparse, parse_qs
        import subprocess
        import os

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        cmd_str = params.get("cmd", [""])[0]
        if not cmd_str:
            self._json({"error": "No command specified", "success": False}, 400)
            return

        # Ensure --metrics is included so the /metrics endpoint is live
        if "llama-server" in cmd_str and "--metrics" not in cmd_str:
            cmd_str = cmd_str + " --metrics"

        # Capture launch stderr for error log modal
        launch_log_path = os.path.expanduser("~/llama-launch-stderr.log")
        # Clear old log before launch so stale entries don't pollute
        try:
            with open(launch_log_path, "w") as lf:
                lf.write("")
        except Exception:
            pass
        # Capture ALL stderr output to the log file, including bash-level errors.
        # Terminal emulators (ptyxis via xdg-terminal-exec) don't relay their
        # child's stderr — proc.stderr.PIPE on the emulator itself is always empty.
        # Using { cmd; } 2>>log instead of raw 2>> because:
        #   (a) plain cmd && cmd2 2>>log only redirects the LAST command
        #   (b) the {} group captures ALL commands' stderr in the chain
        #   (c) bash-level errors (segfault, command-not-found) also go to the log
        launch_cmd = f"{{ {cmd_str}; }} 2>>{launch_log_path}"

        terminal_procs = [
            (["xdg-terminal-exec", "bash", "-c", launch_cmd], "xdg-terminal-exec"),
            (["xterm", "-T", "llama-server", "-e", "bash", "-c", launch_cmd], "xterm"),
            (["gnome-terminal", "--", "bash", "-c", launch_cmd], "gnome-terminal"),
            (["kitty", "--title", "llama-server", "-e", "bash", "-c", launch_cmd], "kitty"),
            (["xfce4-terminal", "-e", "bash", "-c", launch_cmd], "xfce4-terminal"),
        ]

        proc = None
        term_name = None

        for term_cmd, name in terminal_procs:
            try:
                proc = subprocess.Popen(
                    term_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                term_name = name
                break
            except FileNotFoundError:
                continue

        if not proc:
            return self._json({
                "error": "No terminal emulator found (tried xdg-terminal-exec, xterm, gnome-terminal, kitty, xfce4-terminal)",
                "success": False
            }, 500)

        # Store terminal PID so /api/exit can close the window later
        pid_file = os.path.expanduser("~/llama-term-pid.tmp")
        try:
            with open(pid_file, "w") as f:
                f.write(str(proc.pid))
        except Exception:
            pass

        # Auto-record the launch in bench history (tok_s fills in on later capture)
        try:
            model = None
            m = re.search(r"-m\s+(\S+)", cmd_str)
            if m:
                model = os.path.basename(m.group(1).rstrip("'\""))
            build = None
            m = re.search(r"(build-[^\s/]+)/bin/llama-server", cmd_str)
            if m:
                build = m.group(1)
            ctx = None
            m = re.search(r"-c\s+(\d+)", cmd_str)
            if m:
                ctx = int(m.group(1))
            spec = None
            m = re.search(r"--spec-type\s+(\S+)", cmd_str)
            if m:
                spec = m.group(1)
            append_bench_entry({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "running": True,
                "model": model,
                "build": build,
                "tok_s": 0,
                "tg": None,
                "acceptance": None,
                "ctx": ctx,
                "spec": spec,
                "note": "launch",
            })
        except Exception:
            pass

        return self._json({
            "output": f"llama-server launch command sent to {term_name}",
            "success": True
        })

    def _serve_launch_log(self):
        """Return last N lines of launch stderr log (default: all, use ?lines=500 for tail)."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        n_lines = 500
        try:
            val = params.get("lines", [None])[0]
            if val is not None:
                n_lines = int(val)
        except (ValueError, TypeError):
            n_lines = 500

        log_path = os.path.expanduser("~/llama-launch-stderr.log")
        try:
            if os.path.isfile(log_path):
                with open(log_path, "r") as f:
                    lines = f.readlines()
                if n_lines > 0 and len(lines) > n_lines:
                    log = "".join(lines[-n_lines:])
                else:
                    log = "".join(lines)
                self._json({"logs": log or "(empty log)"})
            else:
                self._json({"logs": "(no launch log found)"})
        except Exception as e:
            self._json({"logs": f"Failed to read log: {e}"})

    def _serve_builds(self):
        """Scan ~/llama.cpp/ for build directories with llama-server.
        Only returns builds that actually exist on disk — no hardcoded extras."""
        import os, glob, subprocess

        base = os.path.expanduser("~/llama.cpp")
        builds = []

        # Scan build-* directories only — bare 'build' is ambiguous
        backend = "CPU"
        for d in sorted(glob.glob(os.path.join(base, "build-*"))):
            if not os.path.isdir(d):
                continue
            bin_path = os.path.join(d, "bin", "llama-server")
            if not os.path.isfile(bin_path):
                continue
            dirname = os.path.basename(d)
            rel_path = f"~/llama.cpp/{dirname}/bin/llama-server"

            # Detect backend from directory name (reliable, doesn't crash SYCL)
            if "sycl" in dirname.lower():
                backend = "SYCL"
            elif "vulkan" in dirname.lower():
                backend = "Vulkan"
            else:
                # Fallback: try binary --version (only works for non-SYCL)
                backend = "Unknown"
                bin_path_abs = os.path.join(d, "bin", "llama-server")
                try:
                    env = os.environ.copy()
                    env["LD_LIBRARY_PATH"] = (
                        "/opt/intel/oneapi/compiler/2025.3/lib:"
                        + env.get("LD_LIBRARY_PATH", "")
                    )
                    ver = subprocess.run([bin_path_abs, "--version"], capture_output=True, text=True, timeout=5, env=env)
                    output = ver.stdout + ver.stderr
                    if "IntelLLVM" in output:
                        backend = "SYCL"
                    elif "GNU" in output:
                        backend = "Vulkan"
                except Exception:
                    pass

            builds.append({
                "name": dirname,
                "path": rel_path,
                "desc": f"{backend} ({dirname.replace('build-', '')})"
            })

        # Build descriptions with PR/feature info
        _build_descs = {
            "build-sycl-b9780": "SYCL master b9780 (#24578 reorder fix, #24452 MoE MUL_MAT_ID, #24340 flash mtp3, oneAPI 2026.0, SYCL F16)",
            "build-sycl-b9780-xmx": "SYCL+XMX b9780 (XMX tiling enabled, same commit, prefill-optimized)",
            "build-sycl-b9780-pr25025-xmx": "SYCL+XMX b9780 + #25025 (oneMKL GEMM FA, XMX-accel prefill w/ quant KV, ~91 tg64)",
            "build-sycl-b9780-pr25064-xmx": "SYCL+XMX b9780 + #25064 (Q2_K DMMV reorder — Q2_K only, no Q4_K effect, ~91 tg64)",
            "build-sycl-b9780-pr22105": "SYCL b9780 + #22105 (DFlash spec decode — COMPILES, needs draft model + 2nd GPU to run)",
            "build-sycl-b9781": "SYCL master b9781 (#24152 split-mode tensor, #24941 softmax clamp, #24838 bf16 bin_bcast, #20793 split-sync, oneAPI 2026.0, F16)",
            "build-sycl-b9781-xmx": "SYCL+XMX b9781 (XMX tiling + #24152 split-mode tensor, #24941 softmax, oneAPI 2026.0)",
            "build-sycl-b9782": "SYCL master b9782 (#24162 DeepSeek V4 arch, #25063 DMMV perf, #25231-25264 SYCL fixes, oneAPI 2026.0, F16, MTP)",
            "build-sycl-b9795": "SYCL master b9795 (latest, test DS4 SYCL crash, oneAPI 2026.0, F16)",
            "build-sycl-b10108": "★★★ SYCL LATEST b10108 (0a50d9909, oneAPI 2026.0, F16, MTP, Jul 23 2026) ★★★",
            "build-sycl-b10121-dspark": "🔥 SYCL b10121 + #25173 DSpark (DFlash + Markov head, DS4 speculative decode, ~3x speedup target)",
            "build-sycl-b10235-dspark": "D-Spark b10235 (official #25173+#25784+#26458, DSV4 backbone + MTP + Markov head, oneAPI 2026.0, F16)",
            "build-vulkan-b10235-dspark": "VULKAN D-Spark b10235 (same DSpark master, Mesa 26.1.5 kisak, BMG G31 - speed test vs SYCL)",
            "build-sycl-mtp-b10235": "SYCL+MTP b10235 (MTP self-spec n-max4, no DSpark, 2026-08-14 vLLM-compare build)",
            "build-vulkan-b10121": "Vulkan b10121 ★DS4 (clean master, latest, Mesa 26, no PRs)",
            "build-vulkan-b9780": "Vulkan master b9780 (Mesa 26, stock Vulkan)",
            "build-vulkan-b9782": "Vulkan master b9782 (#24162 DeepSeek V4 arch, same commit as SYCL b9782, Mesa 26)",
            "build-vulkan-mtp-mainline-b9967": "VULKAN+MTP b9967 (GDNfix, 118+ tok/s Qwen3.6-35B-A3B Q4, MTP 82-88% accept)",
            "build-vulkan-mtp-mainline-b9982": "VULKAN+MTP b9982 (checkpoint fix, 120+ tok/s heretic Q4_K_M, Mesa 26.1.4)",
            "build-vulkan-fp32-b9982": "VULKAN+MTP b9982 (3-draft MTP, 140 tok/s heretic Q4_K_M, Mesa 26.1.4)",
            "build-prism-vulkan-nomtp": "PrismML Bonsai fork 38c66ad (Q1_0 native, cooperative matrix, Vulkan, No MTP)",
            "build-vulkan-b10034-bonsai": "Mainline b10034 (Bonsai GGUF compat, cooperative matrix, Vulkan, MTP for MTP models)",
        }
        for b in builds:
            if b["name"] in _build_descs:
                b["desc"] = _build_descs[b["name"]]

        # Enrich with build numbers and commit hashes from each binary
        import re
        for b in builds:
            ver = ""
            commit = ""
            # Extract build number from directory name (e.g. "build-sycl-b9159" → "9159")
            dm = re.search(r'b(\d+)', b.get("name", ""))
            if dm:
                ver = dm.group(1)
            b["build"] = ver
            b["commit"] = commit or ""
            # Detect backend from directory name per-build
            name_lower = b.get("name", "").lower()
            b["backend"] = "SYCL" if "sycl" in name_lower else "Vulkan" if "vulkan" in name_lower else "Unknown"
            b["mtp"] = False

            # Try to get version/commit from binary, but catch crashes (SYCL)
            bin_path = os.path.expanduser(b["path"])
            if os.path.isfile(bin_path):
                # Check if it's SYCL - skip binary execution, use dirname-based info
                if "sycl" in name_lower:
                    # SYCL: detect MTP from libllama-common.so (binary crashes on --help)
                    bin_dir = os.path.dirname(bin_path)
                    lib_path = os.path.join(bin_dir, "libllama-common.so")
                    try:
                        mtp_check = subprocess.run(
                            ["strings", lib_path], capture_output=True, text=True, timeout=5
                        )
                        b["mtp"] = "draft-mtp" in mtp_check.stdout
                    except Exception:
                        b["mtp"] = False
                else:
                    # Non-SYCL: can safely run --version and --help
                    try:
                        env = os.environ.copy()
                        out = subprocess.run(
                            [bin_path, "--version"],
                            capture_output=True, text=True, timeout=5,
                            env=env
                        )
                        output = out.stdout + out.stderr
                        m = re.search(r'version:\s+(\d+)', output)
                        if m:
                            b["build"] = m.group(1)
                        m = re.search(r'\(([0-9a-f]{7,})\)', output)
                        if m:
                            b["commit"] = m.group(1)
                        # Detect MTP from --help
                        help_out = subprocess.run(
                            [bin_path, "--help"],
                            capture_output=True, text=True, timeout=5,
                            env=env
                        )
                        b["mtp"] = "draft-mtp" in (help_out.stdout + help_out.stderr)
                    except Exception:
                        pass

        # ── MTP overrides for builds where --help detection is wrong ──
        _build_mtp_overrides = {
            "build-prism-vulkan-nomtp": False,  # prism fork lists draft-mtp in help but has no MTP implementation
        }
        for b in builds:
            if b["name"] in _build_mtp_overrides:
                b["mtp"] = _build_mtp_overrides[b["name"]]

        # ── Orthrus build variant ──
        orthrus_bin = os.path.expanduser("~/llama.cpp/build-sycl-orthrus/bin/llama-server")
        if os.path.isfile(orthrus_bin):
            builds.append({
                "name": "build-sycl-orthrus",
                "path": "~/llama.cpp/build-sycl-orthrus/bin/llama-server",
                "desc": "ORTHRUS Diffusion Decoder",
                "build": "b1",
                "commit": "",
                "backend": "SYCL",
                "mtp": False,
            })
            # Override MTP detection for orthrus
            try:
                import subprocess
                help_out = subprocess.run(
                    ["strings", orthrus_bin], capture_output=True, text=True, timeout=5
                )
                builds[-1]["mtp"] = "draft-mtp" in help_out.stdout
            except Exception:
                pass

        self._json({"builds": builds, "default": builds[0]["name"] if builds else "build"})

    def _serve_status(self):
        """Return server status, throughput, and selected model."""
        import subprocess
        import re
        import urllib.request
        import urllib.error
        
        # Check if llama-server is running
        running = False
        throughput = 0
        selected_model = None
        
        try:
            # Find llama-server process
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                running = True
                
                # Get model name from process arguments
                pid = result.stdout.strip().split('\n')[0]
                proc_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "args="],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if proc_result.stdout.strip():
                    args = proc_result.stdout.strip()
                    match = re.search(r'-m\s+(\S+)', args)
                    if match:
                        model_path = match.group(1).rstrip("'\"")
                        selected_model = os.path.basename(model_path)
                
                # Poll /metrics endpoint for throughput stats
                try:
                    with urllib.request.urlopen("http://localhost:8080/metrics", timeout=3) as resp:
                        metrics_text = resp.read().decode("utf-8")
                    
                    # Parse Prometheus metrics format
                    # llamacpp:tokens_predicted_total 12345
                    # llamacpp:predicted_tokens_seconds 42.5
                    # llamacpp:prompt_tokens_total 678
                    # llamacpp:prompt_tokens_seconds 150.0
                    total_tokens = 0
                    total_prompt_tokens = 0
                    predicted_tok_sec = 0
                    prompt_tok_sec = 0
                    
                    for line in metrics_text.split("\n"):
                        line = line.strip()
                        if line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        metric_name = parts[0]
                        try:
                            metric_val = float(parts[1])
                        except (ValueError, IndexError):
                            continue
                        
                        if metric_name == "llamacpp:tokens_predicted_total":
                            total_tokens = metric_val
                        elif metric_name == "llamacpp:prompt_tokens_total":
                            total_prompt_tokens = metric_val
                        elif metric_name == "llamacpp:predicted_tokens_seconds":
                            predicted_tok_sec = metric_val
                        elif metric_name == "llamacpp:prompt_tokens_seconds":
                            prompt_tok_sec = metric_val
                    
                    # Return generation throughput (tokens/sec) as the throughput metric
                    # This is the most relevant metric for the user
                    throughput = round(predicted_tok_sec, 1)
                    
                except (urllib.error.URLError, urllib.error.HTTPError, Exception):
                    # Metrics endpoint not available or unreachable — server may be starting up
                    throughput = 0
                    
        except Exception:
            pass
        
        self._json({
            "running": running,
            "throughput": throughput if isinstance(throughput, (int, float)) else 0,
            "selected_model": selected_model
        })

    def _serve_bench_history(self):
        """Return all bench history entries (newest last)."""
        self._json({"entries": load_bench_history(), "file": os.path.basename(BENCH_FILE)})

    def _serve_bench_capture(self):
        """Capture current llama-server state into bench history.
        POST body: {"note": "optional label", "ctx": optional int, "spec": optional string}
        Reads throughput from the running server's /metrics, acceptance + tg from the launch log.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_len:
            try:
                body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            except Exception:
                body = {}

        note = str(body.get("note", "") or "")
        ctx = body.get("ctx")
        spec = str(body.get("spec", "") or "")

        running, model, build = current_llama_server_state()

        # throughput from /metrics (same parse as _serve_status)
        throughput = 0
        if running:
            try:
                with urllib.request.urlopen("http://localhost:8080/metrics", timeout=3) as resp:
                    metrics_text = resp.read().decode("utf-8")
                for line in metrics_text.split("\n"):
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    if parts[0] == "llamacpp:predicted_tokens_seconds":
                        try:
                            throughput = round(float(parts[1]), 1)
                        except ValueError:
                            pass
            except Exception:
                throughput = 0

        # acceptance + tg from launch log
        log_path = os.path.expanduser("~/llama-launch-stderr.log")
        acceptance = parse_acceptance_from_log(log_path)
        tg = parse_tg_from_log(log_path)

        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": running,
            "model": model,
            "build": build,
            "tok_s": throughput,
            "tg": tg,
            "acceptance": acceptance,
            "ctx": ctx,
            "spec": spec,
            "note": note,
        }
        entries = append_bench_entry(entry)
        self._json({"success": True, "entry": entry, "entries": entries})

    def _serve_bench_clear(self):
        """Wipe bench history."""
        save_bench_history([])
        self._json({"success": True, "entries": []})

    def _serve_index(self):
        with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
            html = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
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
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 llama.cpp Launch Command Builder")
        print(f"   Open http://localhost:{PORT} in your browser")
        print(f"   Models dir: {MODELS_DIR}")
        print(f"   CPU: {SYSTEM_INFO.get('cpu', 'Unknown')}")
        print(f"   RAM: {SYSTEM_INFO.get('ram_total_gb', '?')}GB | VRAM: {SYSTEM_INFO.get('vram_gb', '?')}GB")
        print(f"   Press Ctrl+C to stop")
        httpd.serve_forever()
