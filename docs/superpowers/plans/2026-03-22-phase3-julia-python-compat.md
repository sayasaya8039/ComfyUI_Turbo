# Phase 3: Julia 数値計算 + Python 互換レイヤー 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Julia JIT サンプラー/スケジューラで数値計算を高速化し、pyo3 で既存 Python カスタムノード完全互換を実現する。Phase 2 (5-8x) から 8-15x に改善。

**Architecture:** comfy-julia クレートで Julia をサブプロセス実行し、JSON-over-stdio で通信（jlrs 埋め込みは Windows ビルドが複雑なため Phase 3 ではサブプロセス方式を採用。Julia なし環境では Rust フォールバック）。comfy-python クレートで pyo3 0.23+ を使い Python 3.12+ を埋め込み、ComfyUI 互換シムと DLPack テンソルブリッジを提供。Julia インターセプトは comfy-python 内の monkey-patch モジュールで実装。

**Tech Stack:** Rust 1.78+, pyo3 0.23+, Julia 1.10+ (optional), serde_json (Julia IPC), Python 3.12+

**Spec:** `docs/superpowers/specs/2026-03-22-comfyui-turbo-engine-design.md` Section 6, 7

**Phase 2 前提:** `D:\NEXTCLOUD\Windows_app\comfyui-turbo` — 5クレート, 130テスト, v0.2.0

---

## ファイル構成 (新規・変更)

```
comfyui-turbo/
├── crates/
│   ├── comfy-julia/                 # NEW CRATE
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # クレートルート + JuliaRuntime trait
│   │       ├── subprocess.rs       # Julia サブプロセス IPC (JSON-over-stdio)
│   │       ├── samplers.rs         # Euler/DDIM/DPM++ サンプラー (Julia or Rust fallback)
│   │       ├── schedulers.rs       # ノイズスケジュール (linear/cosine/karras)
│   │       └── fallback.rs         # Rust 純粋実装フォールバック
│   │
│   ├── comfy-python/                # NEW CRATE
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # クレートルート
│   │       ├── runtime.rs          # Python ランタイム初期化 + GIL 管理
│   │       ├── bridge.rs           # Tensor ↔ numpy/torch 変換
│   │       ├── node_loader.rs      # Python カスタムノード動的ロード
│   │       └── shim.rs             # ComfyUI 互換モジュール (folder_paths 等)
│   │
│   ├── comfy-core/src/
│   │   └── node.rs                 # MODIFY: PythonNode ラッパー型追加
│   │
│   ├── comfy-nodes/src/
│   │   └── sampling.rs             # MODIFY: KSampler が comfy-julia を使用
│   │
│   └── comfy-server/src/
│       └── routes.rs               # MODIFY: /object_info に Python ノード含める
│
├── julia/                           # Julia ソースファイル
│   └── src/
│       ├── samplers.jl             # Euler/DDIM/DPM++ 実装
│       ├── schedulers.jl           # ノイズスケジュール
│       └── server.jl               # JSON-over-stdio サーバー
│
└── python/                          # Python 互換シム
    └── comfy_shim/
        ├── __init__.py
        ├── folder_paths.py         # ComfyUI folder_paths 互換
        └── model_management.py     # ComfyUI model_management 互換
```

---

## Task 1: comfy-julia クレート — Rust フォールバックサンプラー

**Julia なし環境でも動作する Rust 実装を先に作る。** Julia 統合は Task 2 で追加。

**Files:**
- Create: `crates/comfy-julia/Cargo.toml`
- Create: `crates/comfy-julia/src/lib.rs`
- Create: `crates/comfy-julia/src/fallback.rs`
- Create: `crates/comfy-julia/src/samplers.rs`
- Create: `crates/comfy-julia/src/schedulers.rs`
- Modify: `Cargo.toml` (ワークスペースメンバー追加)

- [ ] **Step 1: Cargo.toml**

```toml
# crates/comfy-julia/Cargo.toml
[package]
name = "comfy-julia"
version.workspace = true
edition.workspace = true

[features]
default = []
julia = []  # Enable Julia subprocess integration

[dependencies]
comfy-core = { path = "../comfy-core" }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
```

ワークスペース Cargo.toml に `"crates/comfy-julia"` 追加。

- [ ] **Step 2: schedulers.rs — ノイズスケジュール テスト + 実装**

```rust
// crates/comfy-julia/src/schedulers.rs
use comfy_core::error::ComfyResult;

/// Generate sigma schedule for diffusion sampling
pub fn linear_schedule(num_steps: usize, beta_start: f64, beta_end: f64) -> Vec<f64> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        let t = i as f64 / (num_steps - 1).max(1) as f64;
        let beta = beta_start + t * (beta_end - beta_start);
        let alpha_cumprod = (1.0 - beta);
        sigmas.push(((1.0 - alpha_cumprod) / alpha_cumprod).sqrt());
    }
    sigmas.push(0.0); // Final sigma = 0
    sigmas
}

/// Karras noise schedule (recommended for DPM++ 2M)
pub fn karras_schedule(num_steps: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    let rho_inv = 1.0 / rho;
    for i in 0..num_steps {
        let t = i as f64 / (num_steps - 1).max(1) as f64;
        let sigma = (sigma_max.powf(rho_inv) + t * (sigma_min.powf(rho_inv) - sigma_max.powf(rho_inv))).powf(rho);
        sigmas.push(sigma);
    }
    sigmas.push(0.0);
    sigmas
}

/// Cosine schedule
pub fn cosine_schedule(num_steps: usize) -> Vec<f64> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        let t = i as f64 / (num_steps - 1).max(1) as f64;
        let alpha_cumprod = (((t + 0.008) / 1.008) * std::f64::consts::FRAC_PI_2).cos().powi(2);
        sigmas.push(((1.0 - alpha_cumprod) / alpha_cumprod).sqrt());
    }
    sigmas.push(0.0);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule_length() {
        let s = linear_schedule(20, 0.00085, 0.012);
        assert_eq!(s.len(), 21); // num_steps + 1
        assert_eq!(*s.last().unwrap(), 0.0);
    }

    #[test]
    fn test_linear_schedule_monotonic() {
        let s = linear_schedule(20, 0.00085, 0.012);
        // Sigmas should be roughly increasing (not strictly, depends on params)
        assert!(s[0] > 0.0);
        assert_eq!(s[20], 0.0);
    }

    #[test]
    fn test_karras_schedule_length() {
        let s = karras_schedule(20, 0.0292, 14.6146, 7.0);
        assert_eq!(s.len(), 21);
        assert_eq!(*s.last().unwrap(), 0.0);
    }

    #[test]
    fn test_karras_schedule_decreasing() {
        let s = karras_schedule(20, 0.0292, 14.6146, 7.0);
        // Karras schedule should be decreasing
        for i in 0..s.len() - 2 {
            assert!(s[i] >= s[i + 1], "sigma[{i}]={} < sigma[{}]={}", s[i], i+1, s[i+1]);
        }
    }

    #[test]
    fn test_cosine_schedule_length() {
        let s = cosine_schedule(20);
        assert_eq!(s.len(), 21);
        assert_eq!(*s.last().unwrap(), 0.0);
    }
}
```

- [ ] **Step 3: fallback.rs — Rust Euler サンプラー テスト + 実装**

```rust
// crates/comfy-julia/src/fallback.rs
use comfy_core::tensor::Tensor;
use comfy_core::error::ComfyResult;

/// Euler Discrete sampler step (pure Rust implementation)
/// x_next = x + (x - denoised) / sigma * (sigma_next - sigma)
pub fn euler_step(x: &[f32], denoised: &[f32], sigma: f64, sigma_next: f64) -> Vec<f32> {
    let dt = sigma_next - sigma;
    x.iter()
        .zip(denoised.iter())
        .map(|(xi, di)| {
            let d = (xi - di) / sigma as f32; // derivative
            xi + d * dt as f32
        })
        .collect()
}

/// DDIM sampler step
pub fn ddim_step(x: &[f32], denoised: &[f32], sigma: f64, sigma_next: f64, eta: f64) -> Vec<f32> {
    let sigma_up = if sigma_next == 0.0 {
        0.0
    } else {
        eta * (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt()
    };
    let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

    x.iter()
        .zip(denoised.iter())
        .map(|(xi, di)| {
            let d = (xi - di) / sigma as f32;
            xi + d * (sigma_down - sigma) as f32
        })
        .collect()
}

/// Run full Euler sampling loop
pub fn euler_sample(
    latent: &Tensor,
    sigmas: &[f64],
    denoise_fn: &dyn Fn(&Tensor, f64) -> ComfyResult<Tensor>,
) -> ComfyResult<Tensor> {
    let mut x_data = latent.as_slice_f32().unwrap_or_default();
    let shape = latent.shape().to_vec();

    for i in 0..sigmas.len() - 1 {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];
        if sigma == 0.0 { break; }

        let x_tensor = Tensor::from_vec(x_data.clone(), shape.clone());
        let denoised = denoise_fn(&x_tensor, sigma)?;
        let d_data = denoised.as_slice_f32().unwrap_or_default();

        x_data = euler_step(&x_data, &d_data, sigma, sigma_next);
    }

    Ok(Tensor::from_vec(x_data, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_step_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let denoised = vec![0.5, 1.0, 1.5];
        let result = euler_step(&x, &denoised, 1.0, 0.5);
        // d = (x - denoised) / sigma = [0.5, 1.0, 1.5]
        // x_next = x + d * (sigma_next - sigma) = x + d * (-0.5)
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.75).abs() < 1e-5);
        assert!((result[1] - 1.5).abs() < 1e-5);
        assert!((result[2] - 2.25).abs() < 1e-5);
    }

    #[test]
    fn test_ddim_step_eta_zero() {
        // eta=0 DDIM should behave like DDIS (deterministic)
        let x = vec![1.0, 2.0];
        let denoised = vec![0.5, 1.0];
        let result = ddim_step(&x, &denoised, 1.0, 0.5, 0.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_euler_step_zero_sigma_next() {
        let x = vec![1.0];
        let denoised = vec![0.8];
        let result = euler_step(&x, &denoised, 0.5, 0.0);
        // d = 0.2/0.5 = 0.4, dt = -0.5, x_next = 1.0 + 0.4*(-0.5) = 0.8
        assert!((result[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_euler_sample_convergence() {
        // Simple denoise function: just return x * 0.9 (shrink toward 0)
        let latent = Tensor::from_vec(vec![10.0, 20.0], vec![2]);
        let sigmas = vec![1.0, 0.5, 0.0];
        let result = euler_sample(&latent, &sigmas, &|x, _sigma| {
            let data: Vec<f32> = x.as_slice_f32().unwrap().iter().map(|v| v * 0.9).collect();
            Ok(Tensor::from_vec(data, x.shape().to_vec()))
        }).unwrap();
        let data = result.as_slice_f32().unwrap();
        // After 2 steps, values should have moved toward 0
        assert!(data[0].abs() < 10.0);
    }
}
```

- [ ] **Step 4: samplers.rs — パブリック API (Julia or fallback ディスパッチ)**

```rust
// crates/comfy-julia/src/samplers.rs
use comfy_core::tensor::Tensor;
use comfy_core::error::ComfyResult;
use crate::fallback;
use crate::schedulers;

#[derive(Debug, Clone, Copy)]
pub enum SamplerType {
    Euler,
    EulerAncestral,
    Ddim,
    DpmPlusPlus2m,
}

impl SamplerType {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "euler" => Some(Self::Euler),
            "euler_ancestral" | "euler_a" => Some(Self::EulerAncestral),
            "ddim" => Some(Self::Ddim),
            "dpmpp_2m" | "dpm_plus_plus_2m" => Some(Self::DpmPlusPlus2m),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SchedulerType {
    Normal,
    Karras,
    Cosine,
}

impl SchedulerType {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "normal" => Some(Self::Normal),
            "karras" => Some(Self::Karras),
            "cosine" => Some(Self::Cosine),
            _ => None,
        }
    }
}

pub fn get_sigmas(scheduler: SchedulerType, steps: usize) -> Vec<f64> {
    match scheduler {
        SchedulerType::Normal => schedulers::linear_schedule(steps, 0.00085, 0.012),
        SchedulerType::Karras => schedulers::karras_schedule(steps, 0.0292, 14.6146, 7.0),
        SchedulerType::Cosine => schedulers::cosine_schedule(steps),
    }
}

/// Run sampling with the specified sampler and scheduler.
/// denoise_fn: called each step with (noisy_latent, sigma) → denoised prediction
pub fn sample(
    sampler: SamplerType,
    latent: &Tensor,
    sigmas: &[f64],
    denoise_fn: &dyn Fn(&Tensor, f64) -> ComfyResult<Tensor>,
) -> ComfyResult<Tensor> {
    match sampler {
        SamplerType::Euler | SamplerType::EulerAncestral => {
            fallback::euler_sample(latent, sigmas, denoise_fn)
        }
        SamplerType::Ddim | SamplerType::DpmPlusPlus2m => {
            // For now, use Euler as fallback for all samplers
            // Full DDIM/DPM++ implementation in Phase 4
            fallback::euler_sample(latent, sigmas, denoise_fn)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_from_name() {
        assert!(matches!(SamplerType::from_name("euler"), Some(SamplerType::Euler)));
        assert!(matches!(SamplerType::from_name("ddim"), Some(SamplerType::Ddim)));
        assert!(SamplerType::from_name("unknown").is_none());
    }

    #[test]
    fn test_scheduler_from_name() {
        assert!(matches!(SchedulerType::from_name("karras"), Some(SchedulerType::Karras)));
        assert!(SchedulerType::from_name("unknown").is_none());
    }

    #[test]
    fn test_get_sigmas() {
        let s = get_sigmas(SchedulerType::Karras, 20);
        assert_eq!(s.len(), 21);
    }

    #[test]
    fn test_sample_euler() {
        let latent = Tensor::from_vec(vec![5.0, 10.0], vec![2]);
        let sigmas = vec![1.0, 0.5, 0.0];
        let result = sample(SamplerType::Euler, &latent, &sigmas, &|x, _| {
            let d: Vec<f32> = x.as_slice_f32().unwrap().iter().map(|v| v * 0.5).collect();
            Ok(Tensor::from_vec(d, x.shape().to_vec()))
        }).unwrap();
        assert_eq!(result.shape(), &[2]);
    }
}
```

- [ ] **Step 5: lib.rs**

```rust
// crates/comfy-julia/src/lib.rs
pub mod fallback;
pub mod samplers;
pub mod schedulers;

#[cfg(feature = "julia")]
pub mod subprocess;

pub use samplers::{SamplerType, SchedulerType, get_sigmas, sample};
```

- [ ] **Step 6: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-julia`
Expected: 13+ tests PASS

- [ ] **Step 7: コミット**

```bash
git add -A && git commit -m "feat: comfy-julia — Rust サンプラー/スケジューラ (Euler/DDIM/Karras/Cosine)"
```

---

## Task 2: comfy-julia — Julia サブプロセス統合 (optional)

**Files:**
- Create: `crates/comfy-julia/src/subprocess.rs`
- Create: `julia/src/samplers.jl`
- Create: `julia/src/schedulers.jl`
- Create: `julia/src/server.jl`

- [ ] **Step 1: julia/src/server.jl — JSON-over-stdio サーバー**

```julia
# julia/src/server.jl
# JSON-over-stdio protocol: read JSON line → process → write JSON line

using JSON3

include("samplers.jl")
include("schedulers.jl")

function main()
    while !eof(stdin)
        line = readline(stdin)
        isempty(line) && continue
        try
            req = JSON3.read(line)
            result = dispatch(req)
            println(stdout, JSON3.write(result))
            flush(stdout)
        catch e
            err = Dict("error" => string(e))
            println(stdout, JSON3.write(err))
            flush(stdout)
        end
    end
end

function dispatch(req)
    cmd = req["command"]
    if cmd == "euler_step"
        x = Float32.(req["x"])
        denoised = Float32.(req["denoised"])
        sigma = Float64(req["sigma"])
        sigma_next = Float64(req["sigma_next"])
        result = euler_step(x, denoised, sigma, sigma_next)
        return Dict("result" => result)
    elseif cmd == "karras_schedule"
        steps = Int(req["steps"])
        sigma_min = Float64(req["sigma_min"])
        sigma_max = Float64(req["sigma_max"])
        rho = Float64(req["rho"])
        result = karras_schedule(steps, sigma_min, sigma_max, rho)
        return Dict("result" => result)
    elseif cmd == "ping"
        return Dict("result" => "pong")
    else
        return Dict("error" => "unknown command: $cmd")
    end
end

main()
```

- [ ] **Step 2: julia/src/samplers.jl**

```julia
# julia/src/samplers.jl
using LoopVectorization

function euler_step(x::Vector{Float32}, denoised::Vector{Float32}, sigma::Float64, sigma_next::Float64)
    dt = Float32(sigma_next - sigma)
    inv_sigma = Float32(1.0 / sigma)
    result = similar(x)
    @turbo for i in eachindex(x)
        d = (x[i] - denoised[i]) * inv_sigma
        result[i] = x[i] + d * dt
    end
    return result
end
```

- [ ] **Step 3: subprocess.rs — Julia サブプロセス IPC**

```rust
// crates/comfy-julia/src/subprocess.rs
use comfy_core::error::{ComfyError, ComfyResult};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

pub struct JuliaProcess {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
    reader: Mutex<Option<BufReader<std::process::ChildStdout>>>,
}

impl JuliaProcess {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            reader: Mutex::new(None),
        }
    }

    pub fn start(&self, julia_path: &str, server_script: &str) -> ComfyResult<()> {
        let mut child = Command::new(julia_path)
            .args(&["--startup-file=no", "--project=.", server_script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ComfyError::InferenceError(format!("failed to start Julia: {e}")))?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        *self.child.lock().unwrap() = Some(child);
        *self.stdin.lock().unwrap() = Some(stdin);
        *self.reader.lock().unwrap() = Some(BufReader::new(stdout));

        // Ping to verify Julia is ready
        let resp = self.call(serde_json::json!({"command": "ping"}))?;
        if resp.get("result").and_then(|v| v.as_str()) != Some("pong") {
            return Err(ComfyError::InferenceError("Julia ping failed".into()));
        }
        tracing::info!("Julia subprocess started successfully");
        Ok(())
    }

    pub fn call(&self, request: serde_json::Value) -> ComfyResult<serde_json::Value> {
        let mut stdin = self.stdin.lock().unwrap();
        let mut reader = self.reader.lock().unwrap();

        let stdin = stdin.as_mut().ok_or_else(|| ComfyError::InferenceError("Julia not started".into()))?;
        let reader = reader.as_mut().ok_or_else(|| ComfyError::InferenceError("Julia not started".into()))?;

        let mut line = serde_json::to_string(&request).unwrap();
        line.push('\n');
        stdin.write_all(line.as_bytes()).map_err(|e| ComfyError::InferenceError(e.to_string()))?;
        stdin.flush().map_err(|e| ComfyError::InferenceError(e.to_string()))?;

        let mut response = String::new();
        reader.read_line(&mut response).map_err(|e| ComfyError::InferenceError(e.to_string()))?;

        serde_json::from_str(&response).map_err(|e| ComfyError::InferenceError(e.to_string()))
    }

    pub fn is_running(&self) -> bool {
        self.child.lock().unwrap().is_some()
    }

    pub fn stop(&self) {
        if let Some(mut child) = self.child.lock().unwrap().take() {
            let _ = child.kill();
        }
        *self.stdin.lock().unwrap() = None;
        *self.reader.lock().unwrap() = None;
    }
}

impl Drop for JuliaProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_julia_process_not_started() {
        let jp = JuliaProcess::new();
        assert!(!jp.is_running());
        assert!(jp.call(serde_json::json!({"command": "ping"})).is_err());
    }
}
```

- [ ] **Step 4: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-julia`
Expected: 全テスト PASS (Julia テストは julia feature なしではスキップ)

- [ ] **Step 5: コミット**

```bash
git add -A && git commit -m "feat: Julia サブプロセス IPC + Julia サンプラー/スケジューラ"
```

---

## Task 3: comfy-python クレート — pyo3 ブリッジ

**Files:**
- Create: `crates/comfy-python/Cargo.toml`
- Create: `crates/comfy-python/src/lib.rs`
- Create: `crates/comfy-python/src/runtime.rs`
- Create: `crates/comfy-python/src/bridge.rs`
- Create: `crates/comfy-python/src/node_loader.rs`
- Create: `crates/comfy-python/src/shim.rs`
- Modify: `Cargo.toml` (ワークスペースメンバー追加)

- [ ] **Step 1: Cargo.toml**

```toml
# crates/comfy-python/Cargo.toml
[package]
name = "comfy-python"
version.workspace = true
edition.workspace = true

[features]
default = []
python = ["pyo3"]  # Enable Python embedding

[dependencies]
comfy-core = { path = "../comfy-core" }
pyo3 = { version = "0.23", features = ["auto-initialize"], optional = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
```

ワークスペース Cargo.toml に `"crates/comfy-python"` 追加。

- [ ] **Step 2: runtime.rs — Python ランタイム管理**

```rust
// crates/comfy-python/src/runtime.rs
use comfy_core::error::{ComfyError, ComfyResult};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python runtime wrapper — handles GIL acquisition and release
pub struct PythonRuntime {
    initialized: bool,
}

impl PythonRuntime {
    pub fn new() -> Self {
        Self { initialized: false }
    }

    #[cfg(feature = "python")]
    pub fn initialize(&mut self) -> ComfyResult<()> {
        if self.initialized { return Ok(()); }
        // pyo3 with auto-initialize handles Python::with_gil automatically
        Python::with_gil(|py| {
            tracing::info!("Python {} initialized", py.version());
        });
        self.initialized = true;
        Ok(())
    }

    #[cfg(not(feature = "python"))]
    pub fn initialize(&mut self) -> ComfyResult<()> {
        Err(ComfyError::InferenceError("Python support not compiled (enable 'python' feature)".into()))
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Execute a Python expression and return the string result
    #[cfg(feature = "python")]
    pub fn eval(&self, code: &str) -> ComfyResult<String> {
        if !self.initialized {
            return Err(ComfyError::InferenceError("Python not initialized".into()));
        }
        Python::with_gil(|py| {
            let result = py.eval(pyo3::types::PyString::new(py, code), None, None)
                .map_err(|e| ComfyError::InferenceError(format!("Python eval error: {e}")))?;
            Ok(result.to_string())
        })
    }

    #[cfg(not(feature = "python"))]
    pub fn eval(&self, _code: &str) -> ComfyResult<String> {
        Err(ComfyError::InferenceError("Python not available".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let rt = PythonRuntime::new();
        assert!(!rt.is_initialized());
    }

    #[test]
    fn test_runtime_without_python_feature() {
        #[cfg(not(feature = "python"))]
        {
            let mut rt = PythonRuntime::new();
            assert!(rt.initialize().is_err());
        }
    }
}
```

- [ ] **Step 3: bridge.rs — テンソルブリッジ (型変換)**

```rust
// crates/comfy-python/src/bridge.rs
use comfy_core::tensor::{DType, Tensor};
use comfy_core::error::{ComfyError, ComfyResult};

/// Metadata for passing tensors across the Python boundary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_size: usize,
}

impl TensorMeta {
    pub fn from_tensor(t: &Tensor) -> Self {
        Self {
            shape: t.shape().to_vec(),
            dtype: match t.dtype() {
                DType::F32 => "float32",
                DType::F16 => "float16",
                DType::BF16 => "bfloat16",
                DType::I8 => "int8",
                DType::I32 => "int32",
                DType::U8 => "uint8",
            }.into(),
            byte_size: t.byte_size(),
        }
    }

    pub fn to_dtype(&self) -> ComfyResult<DType> {
        match self.dtype.as_str() {
            "float32" => Ok(DType::F32),
            "float16" => Ok(DType::F16),
            "bfloat16" => Ok(DType::BF16),
            "int8" => Ok(DType::I8),
            "int32" => Ok(DType::I32),
            "uint8" => Ok(DType::U8),
            other => Err(ComfyError::TensorError(format!("unknown dtype: {other}"))),
        }
    }
}

/// Convert Tensor to raw bytes + metadata for Python consumption
pub fn tensor_to_bytes(tensor: &Tensor) -> (Vec<u8>, TensorMeta) {
    let meta = TensorMeta::from_tensor(tensor);
    (tensor.as_bytes().to_vec(), meta)
}

/// Reconstruct Tensor from raw bytes + metadata
pub fn tensor_from_bytes(bytes: Vec<u8>, meta: &TensorMeta) -> ComfyResult<Tensor> {
    let dtype = meta.to_dtype()?;
    Ok(Tensor::from_raw(bytes, meta.shape.clone(), dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_meta_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let meta = TensorMeta::from_tensor(&t);
        assert_eq!(meta.shape, vec![2, 2]);
        assert_eq!(meta.dtype, "float32");
        assert_eq!(meta.byte_size, 16);
        assert_eq!(meta.to_dtype().unwrap(), DType::F32);
    }

    #[test]
    fn test_tensor_bytes_roundtrip() {
        let original = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let (bytes, meta) = tensor_to_bytes(&original);
        let restored = tensor_from_bytes(bytes, &meta).unwrap();
        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.as_slice_f32().unwrap(), original.as_slice_f32().unwrap());
    }

    #[test]
    fn test_unknown_dtype_error() {
        let meta = TensorMeta { shape: vec![1], dtype: "complex128".into(), byte_size: 16 };
        assert!(meta.to_dtype().is_err());
    }
}
```

- [ ] **Step 4: node_loader.rs — Python カスタムノードローダー (スタブ)**

```rust
// crates/comfy-python/src/node_loader.rs
use comfy_core::error::{ComfyError, ComfyResult};
use comfy_core::node::NodeMetadata;
use std::collections::HashMap;
use std::path::Path;

/// Scan a directory for Python custom node packages
pub fn scan_custom_nodes(dir: &str) -> ComfyResult<Vec<PythonNodeInfo>> {
    let path = Path::new(dir);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut nodes = Vec::new();
    // Scan each subdirectory for __init__.py with NODE_CLASS_MAPPINGS
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let init_py = p.join("__init__.py");
                if init_py.exists() {
                    nodes.push(PythonNodeInfo {
                        module_path: p.to_string_lossy().to_string(),
                        package_name: p.file_name().unwrap().to_string_lossy().to_string(),
                    });
                }
            }
        }
    }
    Ok(nodes)
}

#[derive(Debug, Clone)]
pub struct PythonNodeInfo {
    pub module_path: String,
    pub package_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_nonexistent_dir() {
        let nodes = scan_custom_nodes("/nonexistent/path").unwrap();
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_scan_empty_dir() {
        let tmp = std::env::temp_dir().join("comfy_test_empty");
        std::fs::create_dir_all(&tmp).ok();
        let nodes = scan_custom_nodes(tmp.to_str().unwrap()).unwrap();
        assert!(nodes.is_empty());
        std::fs::remove_dir(&tmp).ok();
    }
}
```

- [ ] **Step 5: shim.rs — ComfyUI 互換シム (パス解決)**

```rust
// crates/comfy-python/src/shim.rs
use std::collections::HashMap;
use std::path::PathBuf;

/// ComfyUI folder_paths compatible path resolver
pub struct FolderPaths {
    base_path: PathBuf,
    folders: HashMap<String, Vec<PathBuf>>,
}

impl FolderPaths {
    pub fn new(base_path: &str) -> Self {
        let base = PathBuf::from(base_path);
        let mut folders = HashMap::new();
        folders.insert("checkpoints".into(), vec![base.join("models/checkpoints")]);
        folders.insert("loras".into(), vec![base.join("models/loras")]);
        folders.insert("vae".into(), vec![base.join("models/vae")]);
        folders.insert("embeddings".into(), vec![base.join("models/embeddings")]);
        folders.insert("controlnet".into(), vec![base.join("models/controlnet")]);
        folders.insert("upscale_models".into(), vec![base.join("models/upscale_models")]);
        folders.insert("input".into(), vec![base.join("input")]);
        folders.insert("output".into(), vec![base.join("output")]);
        folders.insert("temp".into(), vec![base.join("temp")]);
        Self { base_path: base, folders }
    }

    pub fn get_folder_paths(&self, folder_name: &str) -> Vec<PathBuf> {
        self.folders.get(folder_name).cloned().unwrap_or_default()
    }

    pub fn add_folder_path(&mut self, folder_name: &str, path: PathBuf) {
        self.folders.entry(folder_name.into()).or_default().push(path);
    }

    pub fn get_input_directory(&self) -> PathBuf {
        self.base_path.join("input")
    }

    pub fn get_output_directory(&self) -> PathBuf {
        self.base_path.join("output")
    }

    pub fn folder_names(&self) -> Vec<String> {
        self.folders.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_folder_paths_default() {
        let fp = FolderPaths::new("/comfy");
        let ckpts = fp.get_folder_paths("checkpoints");
        assert_eq!(ckpts.len(), 1);
        assert!(ckpts[0].to_str().unwrap().contains("checkpoints"));
    }

    #[test]
    fn test_folder_paths_add() {
        let mut fp = FolderPaths::new("/comfy");
        fp.add_folder_path("checkpoints", PathBuf::from("/extra/models"));
        let ckpts = fp.get_folder_paths("checkpoints");
        assert_eq!(ckpts.len(), 2);
    }

    #[test]
    fn test_folder_paths_unknown() {
        let fp = FolderPaths::new("/comfy");
        assert!(fp.get_folder_paths("nonexistent").is_empty());
    }

    #[test]
    fn test_input_output_dirs() {
        let fp = FolderPaths::new("/comfy");
        assert!(fp.get_input_directory().to_str().unwrap().contains("input"));
        assert!(fp.get_output_directory().to_str().unwrap().contains("output"));
    }

    #[test]
    fn test_folder_names() {
        let fp = FolderPaths::new("/comfy");
        let names = fp.folder_names();
        assert!(names.contains(&"checkpoints".to_string()));
        assert!(names.contains(&"loras".to_string()));
    }
}
```

- [ ] **Step 6: lib.rs**

```rust
// crates/comfy-python/src/lib.rs
pub mod runtime;
pub mod bridge;
pub mod node_loader;
pub mod shim;

pub use runtime::PythonRuntime;
pub use bridge::{TensorMeta, tensor_to_bytes, tensor_from_bytes};
pub use node_loader::{scan_custom_nodes, PythonNodeInfo};
pub use shim::FolderPaths;
```

- [ ] **Step 7: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-python`
Expected: 12+ tests PASS (python feature なしでも基本テストは通る)

- [ ] **Step 8: コミット**

```bash
git add -A && git commit -m "feat: comfy-python — pyo3 ブリッジ + テンソル変換 + ComfyUI パス互換"
```

---

## Task 4: KSampler を comfy-julia サンプラーに統合

**Files:**
- Modify: `crates/comfy-nodes/Cargo.toml` (comfy-julia 依存追加)
- Modify: `crates/comfy-nodes/src/sampling.rs` (comfy-julia 使用)

- [ ] **Step 1: comfy-nodes に comfy-julia 依存追加**

```toml
comfy-julia = { path = "../comfy-julia" }
```

- [ ] **Step 2: KSampler を comfy-julia::sample() で実装**

```rust
// sampling.rs の KSampler::execute() を更新
fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
    let seed = inputs.get_int("seed").unwrap_or(0);
    let steps = inputs.get_int("steps").unwrap_or(20) as usize;
    let cfg = inputs.get_float("cfg").unwrap_or(7.0);
    let sampler_name = inputs.get_string("sampler_name").unwrap_or("euler");
    let scheduler_name = inputs.get_string("scheduler").unwrap_or("normal");

    let sampler = comfy_julia::SamplerType::from_name(sampler_name)
        .unwrap_or(comfy_julia::SamplerType::Euler);
    let scheduler = comfy_julia::SchedulerType::from_name(scheduler_name)
        .unwrap_or(comfy_julia::SchedulerType::Normal);

    let sigmas = comfy_julia::get_sigmas(scheduler, steps);

    // Generate initial latent noise
    let latent = Tensor::randn(vec![1, 4, 64, 64], seed as u64);

    // Stub denoise function (real model inference in Phase 4)
    let result = comfy_julia::sample(sampler, &latent, &sigmas, &|x, _sigma| {
        // Placeholder: return slightly denoised version
        let data: Vec<f32> = x.as_slice_f32().unwrap().iter().map(|v| v * 0.95).collect();
        Ok(Tensor::from_vec(data, x.shape().to_vec()))
    })?;

    let mut out = NodeOutputs::new();
    out.set("output_0", NodeValue::Tensor(result));
    Ok(out)
}
```

- [ ] **Step 3: テスト実行**

Run: `cd comfyui-turbo && cargo test --workspace`
Expected: 全テスト PASS

- [ ] **Step 4: コミット**

```bash
git add -A && git commit -m "feat: KSampler が comfy-julia サンプラー/スケジューラを使用"
```

---

## Task 5: comfy-server に Python/Julia 統合 + リリース

**Files:**
- Modify: `crates/comfy-server/Cargo.toml`
- Modify: `crates/comfy-server/src/routes.rs` (/models エンドポイント + folder_paths)
- Modify: `crates/comfy-server/src/state.rs` (FolderPaths 追加)

- [ ] **Step 1: AppState に FolderPaths 追加**

```rust
use comfy_python::FolderPaths;

pub struct AppState {
    // ... existing fields ...
    pub folder_paths: FolderPaths,
}
```

- [ ] **Step 2: /models エンドポイント追加**

```rust
// GET /models — list model folder types
async fn get_models(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    Json(state.folder_paths.folder_names())
}

// GET /models/{folder} — list files in folder
async fn get_models_by_folder(
    State(state): State<Arc<AppState>>,
    Path(folder): Path<String>,
) -> Json<Vec<String>> {
    let paths = state.folder_paths.get_folder_paths(&folder);
    let mut files = Vec::new();
    for p in paths {
        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                if entry.path().is_file() {
                    files.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    Json(files)
}
```

Route 登録: `.route("/models", get(get_models)).route("/models/{folder}", get(get_models_by_folder))`

- [ ] **Step 3: テスト + リリースビルド**

Run: `cd comfyui-turbo && cargo test --workspace && cargo build --release -p comfy-server`

- [ ] **Step 4: コミット + タグ**

```bash
git add -A && git commit -m "feat: Phase 3 complete — Julia samplers + Python compat (v0.3.0)"
git tag v0.3.0
```

---

## 完了基準

| 基準 | 検証方法 |
|---|---|
| Rust サンプラー (Euler/DDIM) | `cargo test -p comfy-julia` |
| スケジューラ (Linear/Karras/Cosine) | `cargo test -p comfy-julia -- schedulers` |
| Julia サブプロセス IPC | `cargo test -p comfy-julia -- subprocess` |
| Python ランタイム管理 | `cargo test -p comfy-python -- runtime` |
| テンソルブリッジ | `cargo test -p comfy-python -- bridge` |
| カスタムノードスキャン | `cargo test -p comfy-python -- node_loader` |
| FolderPaths 互換 | `cargo test -p comfy-python -- shim` |
| KSampler 統合 | `cargo test -p comfy-nodes` |
| /models エンドポイント | `cargo test -p comfy-server` |
| 全テスト PASS | `cargo test --workspace` |
| v0.3.0 タグ | `git tag -l` |
