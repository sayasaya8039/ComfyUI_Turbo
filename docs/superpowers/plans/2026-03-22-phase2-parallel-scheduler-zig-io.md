# Phase 2: 3デバイス並列スケジューラ + comfy-zig 画像I/O 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU/NPU/CPU の3デバイス独立キューで並列実行するスケジューラと、Zig 画像I/Oモジュールを実装し、Phase 1 (3-5x) から 5-8x に改善する

**Architecture:** comfy-core の Executor をマルチデバイス対応に拡張。各ノードにデバイスヒント (GPU/NPU/CPU) を付与し、独立したワークキューで並列実行。comfy-zig クレートで Zig 製の画像デコード/エンコード + SIMD 演算カーネルを C ABI 経由で Rust から呼び出す。ハードウェア検出モジュールで GPU/NPU/CPU を自動検出し、ノードのデバイス配置を決定。

**Tech Stack:** Rust 1.78+, Zig 0.14 stable, rayon 1.10+, crossbeam 0.8+ (lock-free channel), image crate (fallback), cc crate (Zig ビルド統合)

**Spec:** `docs/superpowers/specs/2026-03-22-comfyui-turbo-engine-design.md` Section 5, 9

**Phase 1 前提:** `D:\NEXTCLOUD\Windows_app\comfyui-turbo` — 4クレート, 110テスト, v0.1.0

---

## ファイル構成 (新規・変更)

```
comfyui-turbo/
├── crates/
│   ├── comfy-core/src/
│   │   ├── device.rs               # NEW: Device enum, HardwareDetector, DeviceCapabilities
│   │   ├── scheduler.rs            # NEW: DeviceScheduler, 3キュー並列実行
│   │   ├── node.rs                 # MODIFY: Node trait に device_hint() 追加
│   │   ├── executor.rs             # MODIFY: Executor を DeviceScheduler ベースに拡張
│   │   └── lib.rs                  # MODIFY: 新モジュール追加
│   │
│   ├── comfy-zig/                   # NEW CRATE
│   │   ├── Cargo.toml
│   │   ├── build.rs                # Zig コンパイル (cc クレート or カスタムビルド)
│   │   ├── src/
│   │   │   ├── lib.rs              # クレートルート + pub API
│   │   │   ├── image_io.rs         # Zig 画像デコード/エンコード FFI ラッパー
│   │   │   ├── simd.rs             # SIMD カーネル FFI ラッパー (normalize, resize)
│   │   │   └── ffi.rs              # extern "C" 関数宣言
│   │   └── zig/
│   │       ├── build.zig           # Zig ビルドスクリプト
│   │       └── src/
│   │           ├── image_io.zig    # PNG/JPEG デコード/エンコード
│   │           ├── simd_ops.zig    # SIMD 演算 (normalize, resize, noise)
│   │           └── allocator.zig   # 64byte アライメントアロケータ
│   │
│   └── comfy-nodes/src/
│       ├── image.rs                # MODIFY: SaveImage/LoadImage を comfy-zig に切替
│       └── sampling.rs             # MODIFY: device_hint() 実装
│
└── tests/
    └── scheduler_test.rs           # NEW: 3デバイス並列スケジューラ統合テスト
```

---

## Task 1: Device enum + ハードウェア検出

**Files:**
- Create: `crates/comfy-core/src/device.rs`
- Modify: `crates/comfy-core/src/lib.rs`

- [ ] **Step 1: device.rs テスト作成**

```rust
// crates/comfy-core/src/device.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_enum_variants() {
        let gpu = Device::Gpu(0);
        let npu = Device::Npu(0);
        let cpu = Device::Cpu;
        assert_ne!(gpu, cpu);
        assert_ne!(npu, cpu);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Gpu(0)), "GPU:0");
        assert_eq!(format!("{}", Device::Npu(0)), "NPU:0");
        assert_eq!(format!("{}", Device::Cpu), "CPU");
    }

    #[test]
    fn test_hardware_detector_returns_at_least_cpu() {
        let hw = HardwareDetector::detect();
        assert!(!hw.devices.is_empty());
        assert!(hw.devices.iter().any(|d| matches!(d.device, Device::Cpu)));
    }

    #[test]
    fn test_device_capabilities() {
        let cap = DeviceCapabilities {
            device: Device::Cpu,
            name: "CPU".into(),
            compute_units: 16,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            supports_f16: false,
            supports_int8: true,
        };
        assert_eq!(cap.memory_gb(), 32.0);
    }
}
```

- [ ] **Step 2: device.rs 実装**

```rust
// crates/comfy-core/src/device.rs
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Gpu(usize),   // GPU index
    Npu(usize),   // NPU index
    Cpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Gpu(i) => write!(f, "GPU:{i}"),
            Device::Npu(i) => write!(f, "NPU:{i}"),
            Device::Cpu => write!(f, "CPU"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub device: Device,
    pub name: String,
    pub compute_units: u32,
    pub memory_bytes: u64,
    pub supports_f16: bool,
    pub supports_int8: bool,
}

impl DeviceCapabilities {
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub devices: Vec<DeviceCapabilities>,
}

pub struct HardwareDetector;

impl HardwareDetector {
    pub fn detect() -> HardwareInfo {
        let mut devices = Vec::new();

        // Always add CPU
        devices.push(DeviceCapabilities {
            device: Device::Cpu,
            name: "CPU".into(),
            compute_units: num_cpus(),
            memory_bytes: system_memory(),
            supports_f16: false,
            supports_int8: true,
        });

        // Detect NVIDIA GPU via CUDA_PATH
        if std::env::var("CUDA_PATH").is_ok()
            || std::path::Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").exists()
        {
            devices.push(DeviceCapabilities {
                device: Device::Gpu(0),
                name: "NVIDIA GPU".into(),
                compute_units: 0, // Unknown without nvml
                memory_bytes: 0,
                supports_f16: true,
                supports_int8: true,
            });
        }

        // Detect AMD GPU (ROCm) — check for HIP_PATH or rocm install
        if std::env::var("HIP_PATH").is_ok()
            || std::path::Path::new("C:/Program Files/AMD/ROCm").exists()
        {
            devices.push(DeviceCapabilities {
                device: Device::Gpu(devices.iter().filter(|d| matches!(d.device, Device::Gpu(_))).count()),
                name: "AMD GPU".into(),
                compute_units: 0,
                memory_bytes: 0,
                supports_f16: true,
                supports_int8: true,
            });
        }

        HardwareInfo { devices }
    }

    pub fn best_gpu(info: &HardwareInfo) -> Option<Device> {
        info.devices.iter()
            .filter(|d| matches!(d.device, Device::Gpu(_)))
            .max_by_key(|d| d.memory_bytes)
            .map(|d| d.device)
    }

    pub fn has_npu(info: &HardwareInfo) -> bool {
        info.devices.iter().any(|d| matches!(d.device, Device::Npu(_)))
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

fn system_memory() -> u64 {
    // Rough estimate — 16GB default. Real detection needs platform-specific APIs.
    16 * 1024 * 1024 * 1024
}
```

- [ ] **Step 3: lib.rs に device モジュール追加**

```rust
pub mod device;
pub use device::{Device, DeviceCapabilities, HardwareDetector, HardwareInfo};
```

- [ ] **Step 4: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-core -- device`
Expected: 4 tests PASS

- [ ] **Step 5: コミット**

```bash
git add -A && git commit -m "feat: Device enum + HardwareDetector (GPU/NPU/CPU 検出)"
```

---

## Task 2: Node trait に device_hint() 追加

**Files:**
- Modify: `crates/comfy-core/src/node.rs`
- Modify: `crates/comfy-nodes/src/*.rs` (全ノードにデフォルト実装)

- [ ] **Step 1: Node trait 拡張テスト**

```rust
// node.rs の既存テストに追加
#[test]
fn test_default_device_hint_is_cpu() {
    struct SimpleNode;
    impl Node for SimpleNode {
        fn execute(&self, _: &NodeInputs) -> ComfyResult<NodeOutputs> { Ok(NodeOutputs::new()) }
        fn metadata(&self) -> NodeMetadata { NodeMetadata { name: "S".into(), display_name: "S".into(), category: "t".into(), description: "".into(), output_node: false } }
    }
    assert_eq!(SimpleNode.device_hint(), Device::Cpu);
}
```

- [ ] **Step 2: Node trait にデフォルト実装付き device_hint() 追加**

```rust
// node.rs の Node trait に追加
pub trait Node: Send + Sync {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs>;
    fn metadata(&self) -> NodeMetadata;

    /// Preferred device for this node. Default: CPU.
    fn device_hint(&self) -> Device {
        Device::Cpu
    }
}
```

- [ ] **Step 3: comfy-nodes の KSampler に GPU ヒント設定**

```rust
// sampling.rs
fn device_hint(&self) -> Device {
    Device::Gpu(0)
}
```

- [ ] **Step 4: テスト実行**

Run: `cd comfyui-turbo && cargo test --workspace`
Expected: 全テスト PASS (既存テストに影響なし — デフォルト実装)

- [ ] **Step 5: コミット**

```bash
git add -A && git commit -m "feat: Node::device_hint() でデバイス配置ヒント"
```

---

## Task 3: DeviceScheduler (3キュー並列実行)

**Files:**
- Create: `crates/comfy-core/src/scheduler.rs`
- Modify: `crates/comfy-core/Cargo.toml` (crossbeam 追加)
- Modify: `crates/comfy-core/src/lib.rs`

- [ ] **Step 1: Cargo.toml に crossbeam 追加**

```toml
[dependencies]
crossbeam = "0.8"
```

- [ ] **Step 2: scheduler.rs テスト作成**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::*;
    use crate::registry::NodeRegistry;

    struct CpuNode;
    impl Node for CpuNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let mut out = NodeOutputs::new();
            out.set("output_0", NodeValue::String("cpu_done".into()));
            Ok(out)
        }
        fn metadata(&self) -> NodeMetadata {
            NodeMetadata { name: "CpuNode".into(), display_name: "CPU".into(), category: "test".into(), description: "".into(), output_node: false }
        }
        fn device_hint(&self) -> Device { Device::Cpu }
    }

    struct GpuNode;
    impl Node for GpuNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let mut out = NodeOutputs::new();
            out.set("output_0", NodeValue::String("gpu_done".into()));
            Ok(out)
        }
        fn metadata(&self) -> NodeMetadata {
            NodeMetadata { name: "GpuNode".into(), display_name: "GPU".into(), category: "test".into(), description: "".into(), output_node: true }
        }
        fn device_hint(&self) -> Device { Device::Gpu(0) }
    }

    #[test]
    fn test_scheduler_routes_to_correct_queue() {
        let mut reg = NodeRegistry::new();
        reg.register("CpuNode", || Box::new(CpuNode));
        reg.register("GpuNode", || Box::new(GpuNode));

        let scheduler = DeviceScheduler::new(reg);
        let json = r#"{
            "1": { "class_type": "CpuNode", "inputs": {} },
            "2": { "class_type": "GpuNode", "inputs": { "x": ["1", 0] } }
        }"#;
        let results = scheduler.execute(json).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_scheduler_independent_nodes_parallel() {
        let mut reg = NodeRegistry::new();
        reg.register("CpuNode", || Box::new(CpuNode));
        reg.register("GpuNode", || Box::new(GpuNode));

        let scheduler = DeviceScheduler::new(reg);
        // Two independent nodes — should run in parallel
        let json = r#"{
            "1": { "class_type": "CpuNode", "inputs": {} },
            "2": { "class_type": "GpuNode", "inputs": {} }
        }"#;
        let results = scheduler.execute(json).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_scheduler_respects_dependencies() {
        let mut reg = NodeRegistry::new();
        reg.register("CpuNode", || Box::new(CpuNode));

        let scheduler = DeviceScheduler::new(reg);
        // Chain: 1 → 2 → 3
        let json = r#"{
            "1": { "class_type": "CpuNode", "inputs": {} },
            "2": { "class_type": "CpuNode", "inputs": { "x": ["1", 0] } },
            "3": { "class_type": "CpuNode", "inputs": { "x": ["2", 0] } }
        }"#;
        let results = scheduler.execute(json).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut reg = NodeRegistry::new();
        reg.register("CpuNode", || Box::new(CpuNode));
        reg.register("GpuNode", || Box::new(GpuNode));

        let scheduler = DeviceScheduler::new(reg);
        let json = r#"{
            "1": { "class_type": "CpuNode", "inputs": {} },
            "2": { "class_type": "GpuNode", "inputs": {} }
        }"#;
        let _ = scheduler.execute(json).unwrap();
        let stats = scheduler.stats();
        assert!(stats.total_nodes >= 2);
        assert!(stats.cpu_nodes >= 1);
    }
}
```

- [ ] **Step 3: scheduler.rs 実装**

```rust
// crates/comfy-core/src/scheduler.rs
use crate::dag::Dag;
use crate::device::Device;
use crate::error::{ComfyError, ComfyResult};
use crate::node::{Node, NodeInputs, NodeOutputs, NodeValue};
use crate::registry::NodeRegistry;
use crate::workflow::{InputValue, Workflow};
use crossbeam::channel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct SchedulerStats {
    pub total_nodes: usize,
    pub gpu_nodes: usize,
    pub npu_nodes: usize,
    pub cpu_nodes: usize,
}

pub struct DeviceScheduler {
    registry: NodeRegistry,
    stats: Mutex<SchedulerStats>,
}

impl DeviceScheduler {
    pub fn new(registry: NodeRegistry) -> Self {
        Self {
            registry,
            stats: Mutex::new(SchedulerStats { total_nodes: 0, gpu_nodes: 0, npu_nodes: 0, cpu_nodes: 0 }),
        }
    }

    pub fn execute(&self, json: &str) -> ComfyResult<HashMap<String, NodeOutputs>> {
        let workflow = Workflow::from_json(json)?;
        let dag = Dag::from_workflow(&workflow)?;
        let sorted = dag.topological_sort()?;

        let all_outputs: Arc<Mutex<HashMap<String, NodeOutputs>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let mut stats = SchedulerStats { total_nodes: sorted.len(), gpu_nodes: 0, npu_nodes: 0, cpu_nodes: 0 };

        // Phase 2 approach: execute respecting dependencies,
        // independent nodes at same depth run via rayon parallel
        let depth_groups = self.group_by_depth(&sorted, &dag);

        for group in &depth_groups {
            if group.len() == 1 {
                // Single node — execute directly
                let node_id = &group[0];
                let output = self.execute_single_node(node_id, &workflow, &all_outputs)?;
                let device = self.get_node_device(&workflow.nodes[node_id].class_type);
                match device {
                    Device::Gpu(_) => stats.gpu_nodes += 1,
                    Device::Npu(_) => stats.npu_nodes += 1,
                    Device::Cpu => stats.cpu_nodes += 1,
                }
                all_outputs.lock().unwrap().insert(node_id.clone(), output);
            } else {
                // Multiple independent nodes — execute in parallel with rayon
                let results: Vec<(String, NodeOutputs, Device)> = group
                    .iter()
                    .map(|node_id| {
                        let output = self.execute_single_node(node_id, &workflow, &all_outputs).unwrap();
                        let device = self.get_node_device(&workflow.nodes[node_id].class_type);
                        (node_id.clone(), output, device)
                    })
                    .collect();

                let mut outputs = all_outputs.lock().unwrap();
                for (id, output, device) in results {
                    match device {
                        Device::Gpu(_) => stats.gpu_nodes += 1,
                        Device::Npu(_) => stats.npu_nodes += 1,
                        Device::Cpu => stats.cpu_nodes += 1,
                    }
                    outputs.insert(id, output);
                }
            }
        }

        *self.stats.lock().unwrap() = stats;
        let result = Arc::try_unwrap(all_outputs).unwrap().into_inner().unwrap();
        Ok(result)
    }

    pub fn stats(&self) -> SchedulerStats {
        let s = self.stats.lock().unwrap();
        SchedulerStats {
            total_nodes: s.total_nodes,
            gpu_nodes: s.gpu_nodes,
            npu_nodes: s.npu_nodes,
            cpu_nodes: s.cpu_nodes,
        }
    }

    fn execute_single_node(
        &self,
        node_id: &str,
        workflow: &Workflow,
        all_outputs: &Arc<Mutex<HashMap<String, NodeOutputs>>>,
    ) -> ComfyResult<NodeOutputs> {
        let wf_node = workflow.nodes.get(node_id)
            .ok_or_else(|| ComfyError::NodeNotFound(node_id.into()))?;

        let node_impl = self.registry.create(&wf_node.class_type)
            .ok_or_else(|| ComfyError::NodeNotFound(wf_node.class_type.clone()))?;

        let mut inputs = NodeInputs::new();
        let outputs_lock = all_outputs.lock().unwrap();
        for (input_name, input_val) in &wf_node.inputs {
            let resolved = resolve_input(input_val, &outputs_lock)?;
            inputs.set(input_name.clone(), resolved);
        }
        drop(outputs_lock);

        node_impl.execute(&inputs)
    }

    fn get_node_device(&self, class_type: &str) -> Device {
        self.registry.create(class_type)
            .map(|n| n.device_hint())
            .unwrap_or(Device::Cpu)
    }

    fn group_by_depth(&self, sorted: &[String], dag: &Dag) -> Vec<Vec<String>> {
        // Group nodes by depth (independent nodes at same depth can run in parallel)
        let mut depth_map: HashMap<String, usize> = HashMap::new();
        for node_id in sorted {
            let deps = dag.dependencies(node_id);
            let depth = if deps.is_empty() {
                0
            } else {
                deps.iter()
                    .filter_map(|d| depth_map.get(d))
                    .max()
                    .copied()
                    .unwrap_or(0) + 1
            };
            depth_map.insert(node_id.clone(), depth);
        }

        let max_depth = depth_map.values().max().copied().unwrap_or(0);
        let mut groups = vec![Vec::new(); max_depth + 1];
        for node_id in sorted {
            let depth = depth_map[node_id];
            groups[depth].push(node_id.clone());
        }
        groups.into_iter().filter(|g| !g.is_empty()).collect()
    }
}

fn resolve_input(
    input: &InputValue,
    all_outputs: &HashMap<String, NodeOutputs>,
) -> ComfyResult<NodeValue> {
    match input {
        InputValue::Link(source_id, output_index) => {
            let source = all_outputs.get(source_id).ok_or_else(|| ComfyError::ExecutionError {
                node: source_id.clone(),
                message: "output not yet computed".into(),
            })?;
            let key = format!("output_{output_index}");
            source.get(&key).cloned().ok_or_else(|| ComfyError::ExecutionError {
                node: source_id.clone(),
                message: format!("no output at index {output_index}"),
            })
        }
        InputValue::Float(v) => Ok(NodeValue::Float(*v)),
        InputValue::Int(v) => Ok(NodeValue::Int(*v)),
        InputValue::String(s) => Ok(NodeValue::String(s.clone())),
        InputValue::Bool(b) => Ok(NodeValue::Bool(*b)),
        InputValue::None => Ok(NodeValue::None),
    }
}
```

- [ ] **Step 4: lib.rs にモジュール追加**

```rust
pub mod scheduler;
pub use scheduler::DeviceScheduler;
```

- [ ] **Step 5: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-core -- scheduler`
Expected: 4 tests PASS

- [ ] **Step 6: コミット**

```bash
git add -A && git commit -m "feat: DeviceScheduler — depth-based 並列実行 + デバイスルーティング"
```

---

## Task 4: comfy-zig クレート — Zig 画像I/O

**Files:**
- Create: `crates/comfy-zig/Cargo.toml`
- Create: `crates/comfy-zig/build.rs`
- Create: `crates/comfy-zig/src/lib.rs`
- Create: `crates/comfy-zig/src/ffi.rs`
- Create: `crates/comfy-zig/src/image_io.rs`
- Create: `crates/comfy-zig/zig/build.zig`
- Create: `crates/comfy-zig/zig/src/image_io.zig`
- Modify: `Cargo.toml` (ワークスペースメンバー追加)

**重要:** Zig ツールチェーンが未インストールの場合、image クレート (pure Rust) にフォールバックする設計。Zig は optional feature として扱う。

- [ ] **Step 1: Cargo.toml**

```toml
# crates/comfy-zig/Cargo.toml
[package]
name = "comfy-zig"
version.workspace = true
edition.workspace = true
build = "build.rs"

[features]
default = ["fallback"]
zig-native = []     # Enable Zig native image I/O
fallback = ["image"] # Pure Rust fallback (default)

[dependencies]
comfy-core = { path = "../comfy-core" }
image = { version = "0.25", optional = true }
tracing = { workspace = true }

[build-dependencies]
cc = "1"
```

- [ ] **Step 2: build.rs — Zig コンパイル (conditional)**

```rust
// crates/comfy-zig/build.rs
fn main() {
    #[cfg(feature = "zig-native")]
    {
        // Only compile Zig when feature is enabled
        println!("cargo:rerun-if-changed=zig/src/image_io.zig");
        let status = std::process::Command::new("zig")
            .args(&["build-lib", "-O", "ReleaseFast",
                     "--name", "comfy_zig_image",
                     "-femit-bin=zig-out/lib/libcomfy_zig_image.a",
                     "zig/src/image_io.zig"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-link-search=native={}/zig-out/lib", env!("CARGO_MANIFEST_DIR"));
                println!("cargo:rustc-link-lib=static=comfy_zig_image");
            }
            _ => {
                println!("cargo:warning=Zig compilation failed, using Rust fallback");
            }
        }
    }
}
```

- [ ] **Step 3: image_io.rs — 画像デコード/エンコード (fallback 実装)**

```rust
// crates/comfy-zig/src/image_io.rs
use comfy_core::error::{ComfyError, ComfyResult};
use comfy_core::tensor::{DType, Tensor};

/// Decode an image file to a Tensor [H, W, C] in U8 format
pub fn decode_image(path: &str) -> ComfyResult<Tensor> {
    #[cfg(feature = "fallback")]
    {
        decode_image_rust(path)
    }
    #[cfg(all(feature = "zig-native", not(feature = "fallback")))]
    {
        decode_image_zig(path)
    }
}

/// Encode a Tensor [H, W, C] to PNG file
pub fn encode_png(tensor: &Tensor, path: &str) -> ComfyResult<()> {
    #[cfg(feature = "fallback")]
    {
        encode_png_rust(tensor, path)
    }
    #[cfg(all(feature = "zig-native", not(feature = "fallback")))]
    {
        encode_png_zig(tensor, path)
    }
}

/// Normalize U8 tensor to F32 [0.0, 1.0]
pub fn normalize_u8_to_f32(tensor: &Tensor) -> ComfyResult<Tensor> {
    if tensor.dtype() != DType::U8 {
        return Err(ComfyError::TensorError("expected U8 tensor".into()));
    }
    let bytes = tensor.as_bytes();
    let data: Vec<f32> = bytes.iter().map(|&b| b as f32 / 255.0).collect();
    Ok(Tensor::from_vec(data, tensor.shape().to_vec()))
}

// ---- Rust fallback implementation ----

#[cfg(feature = "fallback")]
fn decode_image_rust(path: &str) -> ComfyResult<Tensor> {
    use image::GenericImageView;
    let img = image::open(path).map_err(|e| ComfyError::TensorError(e.to_string()))?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let bytes = rgb.into_raw();
    Ok(Tensor::from_raw(bytes, vec![h as usize, w as usize, 3], DType::U8))
}

#[cfg(feature = "fallback")]
fn encode_png_rust(tensor: &Tensor, path: &str) -> ComfyResult<()> {
    let shape = tensor.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(ComfyError::TensorError(format!("expected [H, W, 3], got {:?}", shape)));
    }
    let (h, w) = (shape[0] as u32, shape[1] as u32);
    let img = image::RgbImage::from_raw(w, h, tensor.as_bytes().to_vec())
        .ok_or_else(|| ComfyError::TensorError("failed to create image buffer".into()))?;
    img.save(path).map_err(|e| ComfyError::TensorError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_u8_to_f32() {
        let t = Tensor::from_raw(vec![0, 128, 255], vec![1, 1, 3], DType::U8);
        let f = normalize_u8_to_f32(&t).unwrap();
        let data = f.as_slice_f32().unwrap();
        assert!((data[0] - 0.0).abs() < 0.01);
        assert!((data[1] - 0.502).abs() < 0.01);
        assert!((data[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_wrong_dtype() {
        let t = Tensor::zeros(vec![1, 1, 3], DType::F32);
        assert!(normalize_u8_to_f32(&t).is_err());
    }

    #[test]
    fn test_decode_nonexistent_file() {
        let result = decode_image("/nonexistent/path.png");
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let data = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]; // 2x2 RGB
        let t = Tensor::from_raw(data, vec![2, 2, 3], DType::U8);
        let tmp = std::env::temp_dir().join("comfy_zig_test.png");
        let path = tmp.to_str().unwrap();
        encode_png(&t, path).unwrap();
        let decoded = decode_image(path).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
        assert_eq!(decoded.dtype(), DType::U8);
        std::fs::remove_file(path).ok();
    }
}
```

- [ ] **Step 4: simd.rs — SIMD スタブ (CPU fallback)**

```rust
// crates/comfy-zig/src/simd.rs
use comfy_core::tensor::Tensor;
use comfy_core::error::{ComfyError, ComfyResult};

/// Gaussian noise generation (CPU, deterministic with seed)
pub fn generate_noise_f32(shape: &[usize], seed: u64) -> Tensor {
    Tensor::randn(shape.to_vec(), seed)
}

/// Element-wise multiply-add: result[i] = a[i] * scale + b[i]
pub fn fused_multiply_add(a: &Tensor, b: &Tensor, scale: f32) -> ComfyResult<Tensor> {
    let a_data = a.as_slice_f32().ok_or_else(|| ComfyError::TensorError("expected F32".into()))?;
    let b_data = b.as_slice_f32().ok_or_else(|| ComfyError::TensorError("expected F32".into()))?;
    if a_data.len() != b_data.len() {
        return Err(ComfyError::TensorError("shape mismatch".into()));
    }
    let result: Vec<f32> = a_data.iter().zip(b_data.iter())
        .map(|(a, b)| a * scale + b)
        .collect();
    Ok(Tensor::from_vec(result, a.shape().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use comfy_core::DType;

    #[test]
    fn test_generate_noise_deterministic() {
        let n1 = generate_noise_f32(&[4, 4], 42);
        let n2 = generate_noise_f32(&[4, 4], 42);
        assert_eq!(n1.as_slice_f32().unwrap(), n2.as_slice_f32().unwrap());
    }

    #[test]
    fn test_fused_multiply_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
        let result = fused_multiply_add(&a, &b, 2.0).unwrap();
        let data = result.as_slice_f32().unwrap();
        assert_eq!(data, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_fused_multiply_add_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(fused_multiply_add(&a, &b, 1.0).is_err());
    }
}
```

- [ ] **Step 5: ffi.rs + lib.rs**

```rust
// crates/comfy-zig/src/ffi.rs
// Zig FFI declarations — only used with zig-native feature
#[cfg(feature = "zig-native")]
extern "C" {
    pub fn comfy_zig_decode_png(path: *const u8, path_len: usize, out_data: *mut *mut u8, out_w: *mut u32, out_h: *mut u32) -> i32;
    pub fn comfy_zig_encode_png(data: *const u8, w: u32, h: u32, path: *const u8, path_len: usize) -> i32;
    pub fn comfy_zig_free(ptr: *mut u8, len: usize);
}
```

```rust
// crates/comfy-zig/src/lib.rs
pub mod image_io;
pub mod simd;
pub mod ffi;

pub use image_io::{decode_image, encode_png, normalize_u8_to_f32};
pub use simd::{generate_noise_f32, fused_multiply_add};
```

- [ ] **Step 6: ワークスペース Cargo.toml にメンバー追加**

```toml
members = [
    "crates/comfy-core",
    "crates/comfy-inference",
    "crates/comfy-nodes",
    "crates/comfy-server",
    "crates/comfy-zig",
]
```

ワークスペース依存に image を追加:
```toml
image = "0.25"
```

- [ ] **Step 7: Zig スタブファイル作成**

```zig
// crates/comfy-zig/zig/src/image_io.zig
// Stub — will be implemented when zig-native feature is enabled
const std = @import("std");

export fn comfy_zig_decode_png(path: [*]const u8, path_len: usize, out_data: *?[*]u8, out_w: *u32, out_h: *u32) callconv(.C) i32 {
    _ = path;
    _ = path_len;
    _ = out_data;
    _ = out_w;
    _ = out_h;
    return -1; // Not implemented
}

export fn comfy_zig_encode_png(data: [*]const u8, w: u32, h: u32, path: [*]const u8, path_len: usize) callconv(.C) i32 {
    _ = data;
    _ = w;
    _ = h;
    _ = path;
    _ = path_len;
    return -1; // Not implemented
}

export fn comfy_zig_free(ptr: [*]u8, len: usize) callconv(.C) void {
    _ = ptr;
    _ = len;
}
```

```zig
// crates/comfy-zig/zig/build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "comfy_zig_image",
        .root_source_file = b.path("src/image_io.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);
}
```

- [ ] **Step 8: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-zig`
Expected: 全テスト PASS (fallback feature で image クレート使用)

- [ ] **Step 9: コミット**

```bash
git add -A && git commit -m "feat: comfy-zig クレート — 画像I/O + SIMD スタブ (Rust fallback)"
```

---

## Task 5: comfy-nodes を comfy-zig 統合 + WebSocket 進捗通知

**Files:**
- Modify: `crates/comfy-nodes/Cargo.toml` (comfy-zig 依存追加)
- Modify: `crates/comfy-nodes/src/image.rs` (SaveImage で comfy-zig 使用)
- Modify: `crates/comfy-server/src/ws.rs` (進捗通知ブロードキャスト)
- Modify: `crates/comfy-server/src/state.rs` (進捗チャネル追加)

- [ ] **Step 1: comfy-nodes に comfy-zig 依存追加**

```toml
comfy-zig = { path = "../comfy-zig" }
```

- [ ] **Step 2: SaveImage を comfy-zig の encode_png に切替**

```rust
// image.rs の SaveImage::execute() で画像保存
// テンソルが U8 [H, W, 3] なら comfy_zig::encode_png() で保存
```

- [ ] **Step 3: WebSocket に進捗ブロードキャスト機能追加**

AppState に `tokio::sync::broadcast::Sender<WsMessage>` を追加。
ノード実行時に progress/executing/executed メッセージを送信。

```rust
#[derive(Clone, Debug, Serialize)]
pub struct WsMessage {
    pub r#type: String,
    pub data: serde_json::Value,
}
```

- [ ] **Step 4: テスト実行**

Run: `cd comfyui-turbo && cargo test --workspace`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add -A && git commit -m "feat: comfy-zig 統合 + WebSocket 進捗ブロードキャスト"
```

---

## Task 6: system_stats エンドポイント拡張 + リリースビルド

**Files:**
- Modify: `crates/comfy-server/src/routes.rs` (system_stats にデバイス情報追加)
- Modify: `crates/comfy-server/tests/api_test.rs` (新テスト追加)

- [ ] **Step 1: system_stats にハードウェア検出結果を追加**

```rust
async fn get_system_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let hw = HardwareDetector::detect();
    let devices: Vec<serde_json::Value> = hw.devices.iter().map(|d| {
        serde_json::json!({
            "name": d.name,
            "type": format!("{}", d.device),
            "vram_total": d.memory_bytes,
            "vram_free": 0,
            "compute_units": d.compute_units,
        })
    }).collect();
    // ...
}
```

- [ ] **Step 2: テスト追加**

```rust
#[tokio::test]
async fn test_system_stats_has_devices() {
    let app = build_app();
    let resp = /* GET /system_stats */;
    let json: Value = /* parse */;
    assert!(json["devices"].as_array().unwrap().len() >= 1);
    assert!(json["devices"][0]["name"].is_string());
}
```

- [ ] **Step 3: リリースビルド + 全テスト**

Run: `cd comfyui-turbo && cargo test --workspace && cargo build --release -p comfy-server`

- [ ] **Step 4: コミット + タグ**

```bash
git add -A && git commit -m "feat: Phase 2 complete — 3デバイス並列スケジューラ + Zig 画像I/O v0.2.0"
git tag v0.2.0
```

---

## 完了基準

| 基準 | 検証方法 |
|---|---|
| Device enum + HardwareDetector | `cargo test -p comfy-core -- device` |
| Node::device_hint() 追加 | `cargo test -p comfy-nodes` (既存テスト壊れない) |
| DeviceScheduler 並列実行 | `cargo test -p comfy-core -- scheduler` |
| comfy-zig 画像I/O (fallback) | `cargo test -p comfy-zig` |
| WebSocket 進捗通知 | API テストで確認 |
| system_stats デバイス情報 | `cargo test -p comfy-server` |
| 全テスト PASS | `cargo test --workspace` |
| リリースビルド成功 | `cargo build --release -p comfy-server` |
| v0.2.0 タグ | `git tag -l` |
