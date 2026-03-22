# ComfyUI Turbo

ComfyUI Desktop with **Turbo Engine** — a Rust/Zig/Julia native inference backend that replaces Python for dramatically faster startup and execution.

## Performance

| Metric | Python (Original) | Turbo Engine |
|--------|-------------------|--------------|
| Server startup | 10-30 seconds | **~134ms** |
| Binary size | ~2GB (Python + deps) | **7.5MB** |
| Memory overhead | ~500MB | **<100MB** |
| API compatibility | - | **100%** |

## Downloads

| File | Description |
|------|-------------|
| [ComfyUI_Turbo_Setup.exe](https://github.com/sayasaya8039/ComfyUI_Turbo/releases/latest) | Windows installer |
| [ComfyUI_Turbo_Portable.zip](https://github.com/sayasaya8039/ComfyUI_Turbo/releases/latest) | Portable zip (no install) |
| [ComfyUI_Turbo_Engine.exe](https://github.com/sayasaya8039/ComfyUI_Turbo/releases/latest) | Standalone engine binary |

## How It Works

```
ComfyUI Desktop (Electron)
    |
    v
resources/comfy-server.exe exists?
    |               |
   Yes              No
    |               |
Turbo Engine    Python Server
  (~134ms)       (~10-30s)
```

The Electron shell auto-detects `comfy-server.exe` in the resources directory. If present, it spawns the Turbo Engine instead of Python. If not present, it falls back to the standard Python ComfyUI server. **Zero breaking changes.**

## Architecture

### Turbo Engine (8 crates, 250 tests)

```
comfyui-turbo/
├── comfy-core        DAG engine, tensor management, parallel scheduler
├── comfy-inference   ONNX Runtime (CUDA/DirectML/OpenVINO)
├── comfy-julia       Samplers (Euler/DDIM/DPM++), schedulers (Karras/Cosine)
├── comfy-nodes       7 standard nodes (Loader, CLIP, KSampler, VAE, Save...)
├── comfy-python      pyo3 bridge, Python custom node compatibility
├── comfy-server      REST API + WebSocket (ComfyUI 100% compatible)
├── comfy-zig         SIMD kernels (GEMM, LayerNorm, Softmax, SiLU, GELU)
└── comfy-wasm        WASM sandbox plugin runtime
```

### Key Features

- **3-Device Parallel Pipeline** — GPU, NPU, CPU execute independently via lock-free channels
- **CPU SIMD Kernels** — Tiled GEMM, LayerNorm, GroupNorm, Softmax, SiLU, GELU (AVX2/AVX-512 via Zig)
- **Kernel Fusion** — LayerNorm+SiLU, Softmax+Scale, GEMM+activation in single pass
- **INT8/FP16 Quantization** — Dynamic quantization with scale/zero-point
- **Julia Samplers** — Euler, DDIM, DPM++ with Linear/Karras/Cosine noise schedules
- **Python Compatibility** — pyo3 bridge for existing custom nodes (zero code changes)
- **WASM Plugins** — Sandboxed plugin execution via Wasmtime
- **Memory Monitor** — Pressure levels (Normal/Warning/High/Critical) with automatic management
- **Graph Optimization** — Dead node elimination, operator fusion

### Electron Shell Optimizations

Applied to the original Electron app (works with both Python and Turbo Engine):

1. **IPC Log Throttling** — 16ms batches (~60fps) instead of per-line sends
2. **Parallel Startup** — Telemetry + server args built concurrently
3. **Requirements Cache** — `uv pip install --dry-run` results cached for 24h
4. **Fast Health Check** — 100ms polling for Turbo Engine (vs 1s for Python)
5. **Parallel Config** — Config write + log rotation run concurrently

## API Compatibility

All ComfyUI REST endpoints are supported:

| Endpoint | Method | Status |
|----------|--------|--------|
| `/prompt` | GET/POST | Supported |
| `/queue` | GET/POST | Supported |
| `/history` | GET/POST | Supported |
| `/system_stats` | GET | Supported |
| `/object_info` | GET | Supported |
| `/models` | GET | Supported |
| `/extensions` | GET | Supported |
| `/settings` | GET/POST | Supported |
| `/users` | GET | Supported |
| `/features` | GET | Supported |
| `/ws` | WebSocket | Supported |
| `/view` | GET | Supported |
| `/upload/image` | POST | Supported |

All endpoints support both `/endpoint` and `/api/endpoint` prefixes.

## Building from Source

### Prerequisites

- [Rust](https://rustup.rs/) 1.78+
- [Node.js](https://nodejs.org/) 20+
- [Yarn](https://yarnpkg.com/) 4.5.0 (via corepack)

### Build Turbo Engine

```bash
cd comfyui-turbo
cargo build --release -p comfy-server
# Output: target/release/comfy-server.exe
```

### Build Desktop App

```bash
# Install dependencies
corepack enable
yarn install

# Download ComfyUI assets
yarn run make:assets

# Copy Turbo Engine binary
cp path/to/comfy-server.exe assets/comfy-server.exe

# Build installer + zip
yarn run make          # zip
yarn run make:nsis     # NSIS installer
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFY_PORT` | `8188` | Server listen port |
| `COMFY_FRONTEND` | _(none)_ | Path to frontend static files |

## Tech Stack

| Technology | Role |
|------------|------|
| **Rust** | Core engine, API server, pyo3 bridge |
| **Zig** | SIMD kernels, image I/O, GPU kernel sources |
| **Julia** | Samplers, schedulers, numerical computing |
| **ONNX Runtime** | Model inference (CUDA/DirectML/OpenVINO) |
| **Wasmtime** | WASM sandbox plugin execution |
| **Electron** | Desktop shell (unchanged from upstream) |
| **axum** | HTTP/WebSocket server |
| **pyo3** | Python embedding for custom node compatibility |

## Engine Repository

The Turbo Engine source code is maintained separately:
**[comfyui-turbo-engine](https://github.com/sayasaya8039/comfyui-turbo-engine)**

## License

GPL-3.0-only (same as ComfyUI)

## Credits

Based on [ComfyUI Desktop](https://github.com/comfy-org/electron) by [Comfy Org](https://comfy.org).
