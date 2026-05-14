# llama-launcher

A web UI for launching and managing [llama.cpp](https://github.com/ggerganov/llama.cpp) inference servers. Browse, configure, and run GGUF models with a clean interface.

## Features

- **Flag browser** — Search and select from hundreds of llama.cpp flags organized by category
- **Preset management** — Save and load named presets with your preferred configurations
- **Build selection** — Support for multiple llama.cpp builds (SYCL, Vulkan, CUDA, CPU)
- **Real-time logs** — See server output directly in the browser
- **Template support** — Built-in chat templates and grammar files

## Quick Start

```bash
python server.py
```

Then open `http://localhost:8080` in your browser.

Requires Python 3 and a compiled llama.cpp server binary.

## Project Structure

```
llama-launcher/
├── index.html       # Web UI
├── server.py        # Python backend
├── flags_db.json    # Complete flag database
├── Checkpoints/     # Saved presets
└── llama-sq.jpg     # UI image
```

## License

MIT
