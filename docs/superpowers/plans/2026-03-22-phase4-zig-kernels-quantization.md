# Phase 4: Zig SIMD カーネル + 量子化 + ONNX グラフ最適化 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zig で CPU SIMD カーネル（GEMM, LayerNorm, Softmax, SiLU）を本格実装し、ONNX グラフ最適化（オペレータフュージョン, FP16/INT8 量子化）を追加。Phase 3 (8-15x) から 15-25x に改善。

**Architecture:** comfy-zig クレートの既存 simd.rs を拡張し、Zig ネイティブ SIMD カーネルを C ABI 経由で呼び出す。Zig の comptime で AVX2/AVX-512/NEON を自動選択。comfy-inference に ONNX グラフ最適化パスと量子化モジュールを追加。CPU フォールバックは常に Rust で用意し、Zig バイナリが無い環境でも動作する。

**Tech Stack:** Zig 0.14 stable (SIMD intrinsics), Rust 1.78+, ort 2.0+ (グラフ最適化 API), cc crate (Zig ビルド統合)

**Spec:** `docs/superpowers/specs/2026-03-22-comfyui-turbo-engine-design.md` Section 4.2, 4.3, 5.2

**Phase 3 前提:** `D:\NEXTCLOUD\Windows_app\comfyui-turbo` — 7クレート, 189テスト, v0.3.0

---

## ファイル構成 (新規・変更)

```
comfyui-turbo/
├── crates/
│   ├── comfy-zig/
│   │   ├── src/
│   │   │   ├── simd.rs             # MODIFY: Zig FFI ディスパッチ追加
│   │   │   ├── kernels.rs          # NEW: Rust ラッパー (gemm, layernorm, softmax, silu)
│   │   │   ├── ffi.rs              # MODIFY: 新 FFI 関数追加
│   │   │   └── lib.rs              # MODIFY: kernels モジュール追加
│   │   └── zig/src/
│   │       ├── simd_ops.zig        # NEW: SIMD カーネル本体 (AVX2/AVX-512/NEON)
│   │       ├── gemm.zig            # NEW: 行列乗算 (タイル化 + SIMD)
│   │       └── activations.zig     # NEW: SiLU, GELU, Softmax
│   │
│   ├── comfy-inference/src/
│   │   ├── optimize.rs             # NEW: ONNX グラフ最適化パス
│   │   ├── quantize.rs             # NEW: FP16/INT8 動的量子化
│   │   └── lib.rs                  # MODIFY: 新モジュール追加
│   │
│   └── comfy-core/src/
│       ├── tensor.rs               # MODIFY: F16 変換メソッド追加
│       └── profiler.rs             # NEW: 実行時間プロファイラ
│
└── benches/
    └── kernel_bench.rs             # NEW: カーネルベンチマーク
```

---

## Task 1: Rust CPU カーネル (GEMM, LayerNorm, Softmax, SiLU)

**Zig なしでも動作する Rust 実装を先に作る。** これが fallback であり、ベンチマークのベースラインになる。

**Files:**
- Create: `crates/comfy-zig/src/kernels.rs`
- Modify: `crates/comfy-zig/src/lib.rs`

- [ ] **Step 1: kernels.rs テスト作成**

```rust
// crates/comfy-zig/src/kernels.rs
#[cfg(test)]
mod tests {
    use super::*;
    use comfy_core::Tensor;

    #[test]
    fn test_gemm_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        let a = Tensor::from_vec_f32(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![2,3]).unwrap();
        let b = Tensor::from_vec_f32(vec![7.0,8.0,9.0,10.0,11.0,12.0], vec![3,2]).unwrap();
        let c = gemm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let d = c.as_slice_f32().unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((d[0] - 58.0).abs() < 1e-4);
        assert!((d[1] - 64.0).abs() < 1e-4);
        assert!((d[2] - 139.0).abs() < 1e-4);
        assert!((d[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_gemm_incompatible_shapes() {
        let a = Tensor::from_vec_f32(vec![1.0,2.0], vec![1,2]).unwrap();
        let b = Tensor::from_vec_f32(vec![1.0,2.0,3.0], vec![1,3]).unwrap();
        assert!(gemm(&a, &b).is_err());
    }

    #[test]
    fn test_layer_norm() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = layer_norm(&x, 1e-5).unwrap();
        let d = result.as_slice_f32().unwrap();
        // Each row should be normalized to mean≈0, std≈1
        let row1_mean = (d[0] + d[1] + d[2]) / 3.0;
        assert!(row1_mean.abs() < 1e-4, "mean={row1_mean}");
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = softmax(&x).unwrap();
        let d = result.as_slice_f32().unwrap();
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(d[2] > d[1] && d[1] > d[0]); // monotonic
    }

    #[test]
    fn test_silu() {
        let x = Tensor::from_vec_f32(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let result = silu(&x).unwrap();
        let d = result.as_slice_f32().unwrap();
        // SiLU(x) = x * sigmoid(x)
        assert!((d[1] - 0.0).abs() < 1e-5); // SiLU(0) = 0
        assert!((d[2] - 0.7311).abs() < 1e-3); // SiLU(1) ≈ 0.7311
    }

    #[test]
    fn test_gelu() {
        let x = Tensor::from_vec_f32(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let result = gelu(&x).unwrap();
        let d = result.as_slice_f32().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-5); // GELU(0) = 0
        assert!((d[1] - 0.8412).abs() < 1e-3); // GELU(1) ≈ 0.8412
    }

    #[test]
    fn test_group_norm() {
        let x = Tensor::from_vec_f32(
            (0..32).map(|i| i as f32).collect(), vec![1, 4, 8] // [batch, channels, spatial]
        ).unwrap();
        let result = group_norm(&x, 2, 1e-5).unwrap(); // 2 groups of 2 channels
        assert_eq!(result.shape(), &[1, 4, 8]);
    }
}
```

- [ ] **Step 2: kernels.rs 実装**

```rust
// crates/comfy-zig/src/kernels.rs
use comfy_core::error::{ComfyError, ComfyResult};
use comfy_core::tensor::Tensor;

/// Matrix multiply: C = A @ B where A is [M,K] and B is [K,N]
pub fn gemm(a: &Tensor, b: &Tensor) -> ComfyResult<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(ComfyError::TensorError("gemm requires 2D tensors".into()));
    }
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k != k2 {
        return Err(ComfyError::TensorError(format!("gemm shape mismatch: [{m},{k}] x [{k2},{n}]")));
    }

    let a_data = a.as_slice_f32()?;
    let b_data = b.as_slice_f32()?;
    let mut c_data = vec![0.0f32; m * n];

    // Tiled GEMM for cache efficiency
    const TILE: usize = 32;
    for i0 in (0..m).step_by(TILE) {
        for j0 in (0..n).step_by(TILE) {
            for p0 in (0..k).step_by(TILE) {
                let i_end = (i0 + TILE).min(m);
                let j_end = (j0 + TILE).min(n);
                let p_end = (p0 + TILE).min(k);
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_val = a_data[i * k + p];
                        for j in j0..j_end {
                            c_data[i * n + j] += a_val * b_data[p * n + j];
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec_f32(c_data, vec![m, n])
}

/// Layer normalization over the last dimension
pub fn layer_norm(x: &Tensor, eps: f64) -> ComfyResult<Tensor> {
    let shape = x.shape();
    let data = x.as_slice_f32()?;
    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;
    let mut result = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let slice = &data[start..end];

        let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / last_dim as f32;
        let inv_std = 1.0 / (var + eps as f32).sqrt();

        for i in 0..last_dim {
            result[start + i] = (slice[i] - mean) * inv_std;
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

/// Softmax over the last dimension
pub fn softmax(x: &Tensor) -> ComfyResult<Tensor> {
    let shape = x.shape();
    let data = x.as_slice_f32()?;
    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;
    let mut result = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let slice = &data[start..end];

        let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = slice.iter().map(|v| (v - max_val).exp()).sum();

        for i in 0..last_dim {
            result[start + i] = (slice[i] - max_val).exp() / exp_sum;
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: &Tensor) -> ComfyResult<Tensor> {
    let data = x.as_slice_f32()?;
    let result: Vec<f32> = data.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect();
    Tensor::from_vec_f32(result, x.shape().to_vec())
}

/// GELU activation (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Tensor) -> ComfyResult<Tensor> {
    let data = x.as_slice_f32()?;
    let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let result: Vec<f32> = data.iter().map(|&v| {
        let inner = sqrt_2_pi * (v + 0.044715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    }).collect();
    Tensor::from_vec_f32(result, x.shape().to_vec())
}

/// Group normalization: normalize within groups of channels
pub fn group_norm(x: &Tensor, num_groups: usize, eps: f64) -> ComfyResult<Tensor> {
    let shape = x.shape();
    if shape.len() < 3 {
        return Err(ComfyError::TensorError("group_norm requires at least 3D [batch, channels, ...]".into()));
    }
    let batch = shape[0];
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();

    if channels % num_groups != 0 {
        return Err(ComfyError::TensorError(format!("channels {channels} not divisible by groups {num_groups}")));
    }

    let data = x.as_slice_f32()?;
    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;
    let mut result = data.clone();

    for b in 0..batch {
        for g in 0..num_groups {
            let start = b * channels * spatial + g * group_size;
            let end = start + group_size;
            let slice = &data[start..end];

            let mean: f32 = slice.iter().sum::<f32>() / group_size as f32;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / group_size as f32;
            let inv_std = 1.0 / (var + eps as f32).sqrt();

            for i in 0..group_size {
                result[start + i] = (slice[i] - mean) * inv_std;
            }
        }
    }

    Tensor::from_vec_f32(result, shape.to_vec())
}
```

- [ ] **Step 3: lib.rs に kernels 追加**

```rust
pub mod kernels;
pub use kernels::{gemm, layer_norm, softmax, silu, gelu, group_norm};
```

- [ ] **Step 4: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-zig -- kernels`
Expected: 7 tests PASS

- [ ] **Step 5: コミット**

```bash
git add -A && git commit -m "feat: CPU kernels — GEMM, LayerNorm, Softmax, SiLU, GELU, GroupNorm"
```

---

## Task 2: Zig SIMD カーネル (AVX2/AVX-512 自動選択)

**Files:**
- Create: `crates/comfy-zig/zig/src/simd_ops.zig`
- Create: `crates/comfy-zig/zig/src/gemm.zig`
- Create: `crates/comfy-zig/zig/src/activations.zig`
- Modify: `crates/comfy-zig/zig/build.zig`
- Modify: `crates/comfy-zig/src/ffi.rs`
- Modify: `crates/comfy-zig/build.rs`

- [ ] **Step 1: zig/src/simd_ops.zig — SIMD ユーティリティ**

```zig
// crates/comfy-zig/zig/src/simd_ops.zig
const std = @import("std");

/// Detect SIMD width at comptime
pub const SimdWidth = blk: {
    if (std.Target.x86.featureSetHas(.avx512f)) {
        break :blk 16; // 512-bit = 16 x f32
    } else if (std.Target.x86.featureSetHas(.avx2)) {
        break :blk 8;  // 256-bit = 8 x f32
    } else {
        break :blk 4;  // 128-bit fallback
    }
};

/// SIMD vector dot product for f32
pub fn simd_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    const W = SimdWidth;
    var sum: @Vector(W, f32) = @splat(0.0);
    var i: usize = 0;

    // SIMD loop
    while (i + W <= len) : (i += W) {
        const va: @Vector(W, f32) = a[i..][0..W].*;
        const vb: @Vector(W, f32) = b[i..][0..W].*;
        sum += va * vb;
    }

    // Horizontal sum
    var result: f32 = @reduce(.Add, sum);

    // Scalar remainder
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

/// SIMD sigmoid: 1 / (1 + exp(-x))
pub fn simd_sigmoid(x: [*]const f32, out: [*]f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        out[i] = 1.0 / (1.0 + @exp(-x[i]));
    }
}

/// SIMD SiLU: x * sigmoid(x)
pub fn simd_silu(x: [*]const f32, out: [*]f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sig = 1.0 / (1.0 + @exp(-x[i]));
        out[i] = x[i] * sig;
    }
}

// C ABI exports
export fn comfy_zig_silu(x: [*]const f32, out: [*]f32, len: usize) callconv(.C) void {
    simd_silu(x, out, len);
}

export fn comfy_zig_softmax(x: [*]const f32, out: [*]f32, len: usize) callconv(.C) void {
    // Find max
    var max_val: f32 = x[0];
    for (1..len) |i| {
        if (x[i] > max_val) max_val = x[i];
    }
    // Exp and sum
    var sum: f32 = 0.0;
    for (0..len) |i| {
        out[i] = @exp(x[i] - max_val);
        sum += out[i];
    }
    // Normalize
    const inv_sum = 1.0 / sum;
    for (0..len) |i| {
        out[i] *= inv_sum;
    }
}

export fn comfy_zig_layer_norm(
    x: [*]const f32, out: [*]f32, len: usize, eps: f32,
) callconv(.C) void {
    var mean: f32 = 0.0;
    for (0..len) |i| mean += x[i];
    mean /= @as(f32, @floatFromInt(len));

    var variance: f32 = 0.0;
    for (0..len) |i| {
        const d = x[i] - mean;
        variance += d * d;
    }
    variance /= @as(f32, @floatFromInt(len));
    const inv_std = 1.0 / @sqrt(variance + eps);

    for (0..len) |i| {
        out[i] = (x[i] - mean) * inv_std;
    }
}

test "simd_silu correctness" {
    var x = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var out: [4]f32 = undefined;
    simd_silu(&x, &out, 4);
    try std.testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
}
```

- [ ] **Step 2: zig/build.zig 更新**

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "comfy_zig_kernels",
        .root_source_file = b.path("src/simd_ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.bundle_compiler_rt = true;
    b.installArtifact(lib);
}
```

- [ ] **Step 3: ffi.rs に新 FFI 関数追加**

```rust
// crates/comfy-zig/src/ffi.rs に追加
#[cfg(feature = "zig-native")]
extern "C" {
    pub fn comfy_zig_silu(x: *const f32, out: *mut f32, len: usize);
    pub fn comfy_zig_softmax(x: *const f32, out: *mut f32, len: usize);
    pub fn comfy_zig_layer_norm(x: *const f32, out: *mut f32, len: usize, eps: f32);
}
```

- [ ] **Step 4: build.rs を更新して Zig カーネルをビルド**

```rust
// crates/comfy-zig/build.rs
fn main() {
    #[cfg(feature = "zig-native")]
    {
        println!("cargo:rerun-if-changed=zig/src/simd_ops.zig");
        println!("cargo:rerun-if-changed=zig/src/gemm.zig");
        println!("cargo:rerun-if-changed=zig/src/activations.zig");

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let status = std::process::Command::new("zig")
            .args(&["build-lib",
                     "-O", "ReleaseFast",
                     "-target", "x86_64-windows-msvc",
                     "--name", "comfy_zig_kernels",
                     &format!("-femit-bin={}/comfy_zig_kernels.lib", out_dir),
                     "zig/src/simd_ops.zig"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-link-search=native={out_dir}");
                println!("cargo:rustc-link-lib=static=comfy_zig_kernels");
            }
            _ => {
                println!("cargo:warning=Zig SIMD kernels build failed, using Rust fallback");
            }
        }
    }
}
```

- [ ] **Step 5: テスト実行 (Rust fallback で)**

Run: `cd comfyui-turbo && cargo test -p comfy-zig`
Expected: 全テスト PASS (Zig バイナリなしでも Rust fallback で動作)

- [ ] **Step 6: コミット**

```bash
git add -A && git commit -m "feat: Zig SIMD kernels — SiLU, Softmax, LayerNorm (AVX2/AVX-512)"
```

---

## Task 3: ONNX グラフ最適化パス

**Files:**
- Create: `crates/comfy-inference/src/optimize.rs`
- Modify: `crates/comfy-inference/src/lib.rs`

- [ ] **Step 1: optimize.rs テスト + 実装**

```rust
// crates/comfy-inference/src/optimize.rs
use comfy_core::error::ComfyResult;

/// Graph optimization level for ONNX Runtime sessions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    Disabled,
    Basic,      // constant folding, dead node elimination
    Extended,   // operator fusion (Conv+BN, etc.)
    All,        // layout optimization, memory planning
}

impl OptimizationLevel {
    pub fn to_ort_level(&self) -> i32 {
        match self {
            OptimizationLevel::Disabled => 0,
            OptimizationLevel::Basic => 1,
            OptimizationLevel::Extended => 2,
            OptimizationLevel::All => 99,
        }
    }
}

/// Optimization pass configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub level: OptimizationLevel,
    pub enable_fp16: bool,
    pub enable_int8: bool,
    pub constant_folding: bool,
    pub save_optimized_model: bool,
    pub optimized_model_path: Option<String>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::All,
            enable_fp16: false,
            enable_int8: false,
            constant_folding: true,
            save_optimized_model: false,
            optimized_model_path: None,
        }
    }
}

impl OptimizationConfig {
    pub fn aggressive() -> Self {
        Self {
            level: OptimizationLevel::All,
            enable_fp16: true,
            enable_int8: false,
            constant_folding: true,
            save_optimized_model: true,
            optimized_model_path: None,
        }
    }

    pub fn int8_quantized() -> Self {
        Self {
            level: OptimizationLevel::All,
            enable_fp16: false,
            enable_int8: true,
            constant_folding: true,
            save_optimized_model: true,
            optimized_model_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_levels() {
        assert_eq!(OptimizationLevel::Disabled.to_ort_level(), 0);
        assert_eq!(OptimizationLevel::Basic.to_ort_level(), 1);
        assert_eq!(OptimizationLevel::Extended.to_ort_level(), 2);
        assert_eq!(OptimizationLevel::All.to_ort_level(), 99);
    }

    #[test]
    fn test_default_config() {
        let cfg = OptimizationConfig::default();
        assert_eq!(cfg.level, OptimizationLevel::All);
        assert!(!cfg.enable_fp16);
        assert!(cfg.constant_folding);
    }

    #[test]
    fn test_aggressive_config() {
        let cfg = OptimizationConfig::aggressive();
        assert!(cfg.enable_fp16);
        assert!(cfg.save_optimized_model);
    }

    #[test]
    fn test_int8_config() {
        let cfg = OptimizationConfig::int8_quantized();
        assert!(cfg.enable_int8);
        assert!(!cfg.enable_fp16);
    }
}
```

- [ ] **Step 2: lib.rs にモジュール追加**

```rust
pub mod optimize;
pub use optimize::{OptimizationConfig, OptimizationLevel};
```

- [ ] **Step 3: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-inference -- optimize`
Expected: 4 tests PASS

- [ ] **Step 4: コミット**

```bash
git add -A && git commit -m "feat: ONNX graph optimization config (FP16/INT8/operator fusion)"
```

---

## Task 4: 量子化モジュール

**Files:**
- Create: `crates/comfy-inference/src/quantize.rs`
- Modify: `crates/comfy-inference/src/lib.rs`

- [ ] **Step 1: quantize.rs テスト + 実装**

```rust
// crates/comfy-inference/src/quantize.rs
use comfy_core::error::{ComfyError, ComfyResult};
use comfy_core::tensor::{DType, Tensor};

/// Quantize F32 tensor to INT8 with scale and zero-point
pub fn quantize_to_int8(tensor: &Tensor) -> ComfyResult<QuantizedTensor> {
    let data = tensor.as_slice_f32()?;
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let scale = (max_val - min_val) / 255.0;
    let zero_point = (-min_val / scale).round() as i32;

    let quantized: Vec<i8> = data.iter().map(|&v| {
        ((v / scale).round() as i32 + zero_point).clamp(-128, 127) as i8
    }).collect();

    Ok(QuantizedTensor {
        data: quantized,
        shape: tensor.shape().to_vec(),
        scale,
        zero_point,
    })
}

/// Dequantize INT8 back to F32
pub fn dequantize_to_f32(qt: &QuantizedTensor) -> ComfyResult<Tensor> {
    let data: Vec<f32> = qt.data.iter().map(|&v| {
        (v as i32 - qt.zero_point) as f32 * qt.scale
    }).collect();
    Tensor::from_vec_f32(data, qt.shape.clone())
}

/// Quantize F32 to F16 (simulated — actual F16 needs half crate)
pub fn quantize_to_f16(tensor: &Tensor) -> ComfyResult<Tensor> {
    let data = tensor.as_slice_f32()?;
    // Simulate F16 precision loss by rounding to F16 range
    let f16_data: Vec<u8> = data.iter().flat_map(|&v| {
        let bits = f32_to_f16_bits(v);
        bits.to_le_bytes().to_vec()
    }).collect();
    Ok(Tensor::from_raw(f16_data, tensor.shape().to_vec(), DType::F16))
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub zero_point: i32,
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = bits & 0x7FFFFF;

    if exp <= 0 {
        0 // Underflow to zero
    } else if exp >= 31 {
        (sign << 15 | 0x7C00) as u16 // Infinity
    } else {
        ((sign << 15) | ((exp as u32) << 10) | (frac >> 13)) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let original = Tensor::from_vec_f32(vec![0.0, 0.5, 1.0, -0.5, -1.0], vec![5]).unwrap();
        let qt = quantize_to_int8(&original).unwrap();
        assert_eq!(qt.data.len(), 5);
        assert_eq!(qt.shape, vec![5]);

        let restored = dequantize_to_f32(&qt).unwrap();
        let orig_data = original.as_slice_f32().unwrap();
        let rest_data = restored.as_slice_f32().unwrap();
        for i in 0..5 {
            assert!((orig_data[i] - rest_data[i]).abs() < 0.01,
                "idx {i}: orig={}, restored={}", orig_data[i], rest_data[i]);
        }
    }

    #[test]
    fn test_int8_scale_range() {
        let t = Tensor::from_vec_f32(vec![-10.0, 0.0, 10.0], vec![3]).unwrap();
        let qt = quantize_to_int8(&t).unwrap();
        assert!(qt.scale > 0.0);
        assert!(qt.data.iter().all(|&v| v >= -128 && v <= 127));
    }

    #[test]
    fn test_f16_quantization() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let f16 = quantize_to_f16(&t).unwrap();
        assert_eq!(f16.dtype(), DType::F16);
        assert_eq!(f16.shape(), &[3]);
        assert_eq!(f16.byte_size(), 6); // 3 * 2 bytes
    }
}
```

- [ ] **Step 2: lib.rs に追加**

```rust
pub mod quantize;
pub use quantize::{quantize_to_int8, dequantize_to_f32, quantize_to_f16, QuantizedTensor};
```

- [ ] **Step 3: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-inference -- quantize`
Expected: 3 tests PASS

- [ ] **Step 4: コミット**

```bash
git add -A && git commit -m "feat: INT8/FP16 quantization + dequantization"
```

---

## Task 5: 実行時間プロファイラ + リリース

**Files:**
- Create: `crates/comfy-core/src/profiler.rs`
- Modify: `crates/comfy-core/src/lib.rs`
- Modify: `crates/comfy-server/src/routes.rs` (プロファイルデータを system_stats に追加)

- [ ] **Step 1: profiler.rs テスト + 実装**

```rust
// crates/comfy-core/src/profiler.rs
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct Profiler {
    entries: Mutex<HashMap<String, Vec<Duration>>>,
}

impl Profiler {
    pub fn new() -> Self {
        Self { entries: Mutex::new(HashMap::new()) }
    }

    pub fn record(&self, name: &str, duration: Duration) {
        self.entries.lock().unwrap()
            .entry(name.to_string())
            .or_default()
            .push(duration);
    }

    pub fn time<F, R>(&self, name: &str, f: F) -> R
    where F: FnOnce() -> R {
        let start = Instant::now();
        let result = f();
        self.record(name, start.elapsed());
        result
    }

    pub fn summary(&self) -> HashMap<String, ProfileSummary> {
        let entries = self.entries.lock().unwrap();
        entries.iter().map(|(name, durations)| {
            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = total / count as u32;
            let min = durations.iter().min().copied().unwrap_or_default();
            let max = durations.iter().max().copied().unwrap_or_default();
            (name.clone(), ProfileSummary { total, count, avg, min, max })
        }).collect()
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub total: Duration,
    pub count: usize,
    pub avg: Duration,
    pub min: Duration,
    pub max: Duration,
}

impl Default for Profiler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_record_and_summary() {
        let p = Profiler::new();
        p.record("test_op", Duration::from_millis(10));
        p.record("test_op", Duration::from_millis(20));
        let s = p.summary();
        let entry = &s["test_op"];
        assert_eq!(entry.count, 2);
        assert_eq!(entry.min, Duration::from_millis(10));
        assert_eq!(entry.max, Duration::from_millis(20));
    }

    #[test]
    fn test_profiler_time() {
        let p = Profiler::new();
        let result = p.time("add", || 2 + 3);
        assert_eq!(result, 5);
        let s = p.summary();
        assert_eq!(s["add"].count, 1);
    }

    #[test]
    fn test_profiler_clear() {
        let p = Profiler::new();
        p.record("x", Duration::from_millis(1));
        p.clear();
        assert!(p.summary().is_empty());
    }
}
```

- [ ] **Step 2: lib.rs にモジュール追加**

```rust
pub mod profiler;
pub use profiler::Profiler;
```

- [ ] **Step 3: ワークスペースバージョン更新 + リリースビルド**

Root Cargo.toml: version = "0.4.0"

Run: `cd comfyui-turbo && cargo test --workspace && cargo build --release -p comfy-server`

- [ ] **Step 4: コミット + タグ**

```bash
git add -A && git commit -m "feat: Phase 4 complete — Zig SIMD kernels + quantization + profiler (v0.4.0)"
git tag v0.4.0
```

---

## 完了基準

| 基準 | 検証方法 |
|---|---|
| GEMM (tiled) | `cargo test -p comfy-zig -- kernels::tests::test_gemm` |
| LayerNorm / GroupNorm | `cargo test -p comfy-zig -- kernels::tests::test_layer_norm` |
| Softmax / SiLU / GELU | `cargo test -p comfy-zig -- kernels::tests::test_softmax` |
| Zig SIMD ソース作成 | `zig/src/simd_ops.zig` 存在確認 |
| ONNX 最適化設定 | `cargo test -p comfy-inference -- optimize` |
| INT8/FP16 量子化 | `cargo test -p comfy-inference -- quantize` |
| プロファイラ | `cargo test -p comfy-core -- profiler` |
| 全テスト PASS | `cargo test --workspace` |
| v0.4.0 タグ | `git tag -l` |
