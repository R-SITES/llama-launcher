# llama-launcher

A web UI that helps you launch and manage [llama.cpp](https://github.com/ggerganov/llama.cpp) inference servers. Browse your local GGUF models, configure launch flags through a visual interface, and fire up a server with one click.

```
llama.cpp is a C/C++ LLM inference engine that runs on CPU, Apple Silicon,
NVIDIA CUDA, AMD ROCm, and Intel GPUs (SYCL/Vulkan). This UI wraps its
server binary so you don't have to memorize flags.
```

## How It Works

**Architecture** — Two files, one process:

```
llama-launcher/
├── index.html      ← The web UI (what you see in your browser)
├── server.py       ← Python backend (serves the UI and manages llama.cpp)
├── flags_db.json   ← Database of all llama.cpp flags (categories, defaults)
└── Checkpoints/    ← Example saved presets from development
```

When you run `python server.py`, it starts a web server on **http://localhost:9876**. The browser loads `index.html`, which talks to the server via API calls. The server detects your hardware (CPU, RAM, GPU), scans `~/models/` for GGUF files, and serves the flag database.

### API Endpoints

| Endpoint | What it does |
|---|---|
| `GET /` | Serves the UI |
| `GET /api/models` | Lists GGUF models in `~/models/` with file sizes |
| `GET /api/system` | Returns CPU, RAM, and GPU info detected from the system |
| `GET /api/flags` | Returns the full flag database (categories, flags, defaults) |
| `GET /api/builds` | Lists available llama.cpp builds (e.g., Vulkan, SYCL, CUDA) |
| `GET /api/status` | Checks if a llama.cpp server is currently running (port 8080) |
| `GET /api/kill` | Kills the running llama.cpp server process |
| `GET /api/exit` | Stops the launcher itself |
| `POST /api/recommend` | Suggests optimal flags based on model type + hardware |
| `POST /api/optimize` | Optimizes flags for your specific hardware config |
| `POST /api/download` | Downloads a GGUF model from Hugging Face Hub |
| `POST /api/delete-model` | Removes a model from the local cache |

### What You Can Do

1. **Browse your models** — Scans `~/models/` and shows every `.gguf` file with its size
2. **Search flags** — Hundreds of llama.cpp flags organized by category (context, GPU, sampling, etc.)
3. **Build commands** — Pick flags visually, the UI builds the command string in real-time
4. **Save presets** — Save your favorite flag combinations as named presets
5. **Launch servers** — Start/stop a llama.cpp server from the UI
6. **Download models** — Search Hugging Face Hub for GGUF models and download them directly
7. **Hardware overview** — See CPU, RAM, GPU at a glance

## Quick Start

```bash
# Make sure you have a compiled llama.cpp server binary
# (the one at your chosen build path with --server enabled)

# Start the launcher
python server.py

# Open in your browser
open http://localhost:9876
```

### Requirements

- Python 3.8+
- A compiled llama.cpp server binary (e.g., `llama-server` for Vulkan, SYCL, CUDA, or CPU)
- GGUF model files in `~/models/` (or wherever your models live)

## What llama.cpp Supports

- **Inference engines**: CPU (with BLAS), NVIDIA CUDA, AMD ROCm, Intel SYCL, Vulkan, Apple Metal
- **Quantization**: GGUF format with 2-bit to 8-bit quantization (Q2_K through Q8_0)
- **Features**: Speculative decoding, MTP, KV cache quantization, batched inference, prompt caching

## License

MIT
