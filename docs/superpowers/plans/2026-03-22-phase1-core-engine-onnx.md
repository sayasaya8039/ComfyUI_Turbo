# Phase 1: ComfyUI Turbo Engine — Core Engine + ONNX Runtime 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ComfyUI Python サーバーと 100% API 互換の Rust 推論エンジンを構築し、ONNX Runtime で 3-5x 高速化を達成する

**Architecture:** Rust ワークスペースに 4 クレート (comfy-core, comfy-inference, comfy-nodes, comfy-server) を構成。comfy-core が DAG 実行エンジンとテンソル管理、comfy-inference が ONNX Runtime ラッパー、comfy-nodes が標準ノードのネイティブ実装、comfy-server が axum ベースの REST API + WebSocket サーバー。Electron Shell は変更なし。

**Tech Stack:** Rust 1.78+ (Edition 2021), axum 0.8+, ort 2.0+ (ONNX Runtime), rayon 1.10+, crossbeam 0.8+, serde/serde_json, tokio, tower-http (CORS)

**Spec:** `docs/superpowers/specs/2026-03-22-comfyui-turbo-engine-design.md`

---

## ファイル構成

```
comfyui-turbo/
├── Cargo.toml                          # ワークスペース定義
├── crates/
│   ├── comfy-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # クレートルート、pub mod 宣言
│   │       ├── tensor.rs               # Tensor 型、DType、Shape、メモリ管理
│   │       ├── node.rs                 # Node トレイト、NodeInput/NodeOutput 型
│   │       ├── registry.rs             # NodeRegistry: ノード名→ファクトリのマップ
│   │       ├── dag.rs                  # DAG 構築、トポロジカルソート、依存解決
│   │       ├── executor.rs             # ワークフロー実行エンジン (Rayon 並列)
│   │       ├── workflow.rs             # ワークフロー JSON パース (ComfyUI 形式)
│   │       ├── cache.rs                # LRU テンソルキャッシュ
│   │       └── error.rs                # エラー型定義
│   │
│   ├── comfy-inference/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # クレートルート
│   │       ├── session.rs              # ONNX Runtime セッション管理 + キャッシュ
│   │       ├── provider.rs             # EP 自動選択 (CUDA/DirectML/CPU)
│   │       └── convert.rs              # Tensor ↔ ort::Value 変換
│   │
│   ├── comfy-nodes/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # クレートルート + register_all_nodes()
│   │       ├── loaders.rs              # CheckpointLoader (safetensors → ONNX 変換)
│   │       ├── conditioning.rs         # CLIPTextEncode
│   │       ├── sampling.rs             # KSampler (Euler Discrete)
│   │       ├── latent.rs               # EmptyLatentImage, VAEDecode, VAEEncode
│   │       ├── image.rs                # SaveImage, LoadImage, PreviewImage
│   │       └── utils.rs                # ノード共通ユーティリティ
│   │
│   └── comfy-server/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # エントリポイント + CLI args
│           ├── routes.rs               # REST API ルート定義
│           ├── ws.rs                   # WebSocket ハンドラ + メッセージ型
│           ├── queue.rs                # PromptQueue (優先度キュー + 実行管理)
│           ├── history.rs              # 実行履歴管理
│           ├── state.rs                # AppState (共有状態)
│           └── upload.rs               # ファイルアップロード + /view
│
└── tests/
    ├── api_compat_test.rs              # API 互換性テスト (全エンドポイント)
    └── workflow_test.rs                # ワークフロー実行 E2E テスト
```

---

## Task 0: ワークスペースセットアップ

**Files:**
- Create: `comfyui-turbo/` ディレクトリ構造全体
- Create: `comfyui-turbo/.gitignore`

**前提条件:**
- Rust 1.78+ インストール済み (`rustup update stable`)
- ONNX Runtime: ort クレートが自動ダウンロードする (初回ビルド時)。手動インストール不要。
  - GPU 使用時は CUDA 12.0+ が必要 (`CUDA_PATH` 環境変数設定済み)
  - CPU のみの場合は追加設定不要

- [ ] **Step 1: プロジェクトディレクトリ作成**

```bash
mkdir -p comfyui-turbo/crates/{comfy-core/src,comfy-inference/src,comfy-nodes/src,comfy-server/src}
mkdir -p comfyui-turbo/tests
cd comfyui-turbo
```

- [ ] **Step 2: .gitignore 作成**

```gitignore
/target
*.onnx
*.safetensors
*.ckpt
.env
```

- [ ] **Step 3: git init + 初回コミット**

```bash
cd comfyui-turbo && git init && git add .gitignore && git commit -m "chore: init comfyui-turbo workspace"
```

---

## Task 1: ワークスペース + comfy-core 基本型

**Files:**
- Create: `comfyui-turbo/Cargo.toml`
- Create: `comfyui-turbo/crates/comfy-core/Cargo.toml`
- Create: `comfyui-turbo/crates/comfy-core/src/lib.rs`
- Create: `comfyui-turbo/crates/comfy-core/src/error.rs`
- Create: `comfyui-turbo/crates/comfy-core/src/tensor.rs`

- [ ] **Step 1: ワークスペース Cargo.toml 作成**

```toml
# comfyui-turbo/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/comfy-core",
    "crates/comfy-inference",
    "crates/comfy-nodes",
    "crates/comfy-server",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-only"

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

- [ ] **Step 2: comfy-core Cargo.toml 作成**

```toml
# crates/comfy-core/Cargo.toml
[package]
name = "comfy-core"
version.workspace = true
edition.workspace = true

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
```

- [ ] **Step 3: error.rs — エラー型のテスト作成**

```rust
// crates/comfy-core/src/error.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ComfyError::NodeNotFound("KSampler".into());
        assert!(err.to_string().contains("KSampler"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ComfyError>();
    }
}
```

- [ ] **Step 4: error.rs — 実装**

```rust
// crates/comfy-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ComfyError {
    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("invalid input for node '{node}': {message}")]
    InvalidInput { node: String, message: String },

    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("execution error in node '{node}': {message}")]
    ExecutionError { node: String, message: String },

    #[error("workflow has no output nodes")]
    NoOutputNodes,

    #[error("cycle detected in workflow graph")]
    CycleDetected,

    #[error("tensor error: {0}")]
    TensorError(String),

    #[error("inference error: {0}")]
    InferenceError(String),

    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

pub type ComfyResult<T> = Result<T, ComfyError>;

// テストは上の Step 3 で定義済み
```

- [ ] **Step 5: tensor.rs — テスト作成**

```rust
// crates/comfy-core/src/tensor.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_f32() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 3);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_slice_f32().unwrap().iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn test_tensor_randn() {
        let t = Tensor::randn(vec![4, 4], 42);
        assert_eq!(t.shape(), &[4, 4]);
        assert_eq!(t.numel(), 16);
        // ランダム値が 0 でないことを確認
        assert!(t.as_slice_f32().unwrap().iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_tensor_byte_size() {
        let t = Tensor::zeros(vec![2, 3], DType::F16);
        assert_eq!(t.byte_size(), 12); // 6 elements * 2 bytes
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::I32.byte_size(), 4);
    }
}
```

- [ ] **Step 6: tensor.rs — 実装**

```rust
// crates/comfy-core/src/tensor.rs
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I32,
    U8,
}

impl DType {
    pub fn byte_size(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Arc<Vec<u8>>,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Arc::new(bytes),
            shape,
            dtype: DType::F32,
        }
    }

    pub fn from_raw(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            dtype,
        }
    }

    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let bytes = vec![0u8; numel * dtype.byte_size()];
        Self {
            data: Arc::new(bytes),
            shape,
            dtype,
        }
    }

    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        let numel: usize = shape.iter().product();
        // Simple xoshiro256++ PRNG + Box-Muller transform
        let mut state = [seed, seed.wrapping_mul(6364136223846793005), seed ^ 0xdeadbeef, seed.wrapping_add(1)];
        let mut values = Vec::with_capacity(numel);
        for _ in 0..numel {
            let u1 = next_f64(&mut state);
            let u2 = next_f64(&mut state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            values.push(z as f32);
        }
        Self::from_vec(values, shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }

    pub fn as_slice_f32(&self) -> Option<Vec<f32>> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(
            self.data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

fn next_f64(state: &mut [u64; 4]) -> f64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = state[3].rotate_left(45);
    (result >> 11) as f64 / (1u64 << 53) as f64
}

// テストは Step 5 で定義済み
```

- [ ] **Step 7: lib.rs 作成**

```rust
// crates/comfy-core/src/lib.rs
pub mod error;
pub mod tensor;

pub use error::{ComfyError, ComfyResult};
pub use tensor::{DType, Tensor};
```

- [ ] **Step 8: ビルド + テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-core`
Expected: 全テスト PASS

- [ ] **Step 9: コミット**

```bash
git add -A && git commit -m "feat: comfy-core 基盤 — Tensor型、DType、エラー型"
```

---

## Task 2: Node トレイト + NodeRegistry

**Files:**
- Create: `crates/comfy-core/src/node.rs`
- Create: `crates/comfy-core/src/registry.rs`
- Modify: `crates/comfy-core/src/lib.rs`

- [ ] **Step 1: node.rs — テスト作成**

```rust
// crates/comfy-core/src/node.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    struct AddOneNode;

    impl Node for AddOneNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let t = inputs.get_tensor("value")?;
            let data: Vec<f32> = t.as_slice_f32().unwrap().iter().map(|v| v + 1.0).collect();
            let out = Tensor::from_vec(data, t.shape().to_vec());
            let mut outputs = NodeOutputs::new();
            outputs.set("result", NodeValue::Tensor(out));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "AddOne".into(),
                display_name: "Add One".into(),
                category: "math".into(),
                description: "Adds 1 to input tensor".into(),
                output_node: false,
            }
        }
    }

    #[test]
    fn test_node_execute() {
        let node = AddOneNode;
        let mut inputs = NodeInputs::new();
        inputs.set("value", NodeValue::Tensor(Tensor::from_vec(vec![1.0, 2.0], vec![2])));
        let outputs = node.execute(&inputs).unwrap();
        let result = outputs.get_tensor("result").unwrap();
        assert_eq!(result.as_slice_f32().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_node_value_types() {
        let v = NodeValue::Float(3.14);
        assert!(matches!(v, NodeValue::Float(_)));
        let v = NodeValue::String("hello".into());
        assert!(matches!(v, NodeValue::String(_)));
    }
}
```

- [ ] **Step 2: node.rs — 実装**

```rust
// crates/comfy-core/src/node.rs
use crate::error::{ComfyError, ComfyResult};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub name: String,
    pub display_name: String,
    pub category: String,
    pub description: String,
    pub output_node: bool,
}

#[derive(Debug, Clone)]
pub enum NodeValue {
    Tensor(Tensor),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<NodeValue>),
    None,
}

#[derive(Debug, Clone, Default)]
pub struct NodeInputs {
    values: HashMap<String, NodeValue>,
}

impl NodeInputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, value: NodeValue) {
        self.values.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> ComfyResult<&NodeValue> {
        self.values.get(key).ok_or_else(|| ComfyError::InvalidInput {
            node: String::new(),
            message: format!("missing input: {key}"),
        })
    }

    pub fn get_tensor(&self, key: &str) -> ComfyResult<&Tensor> {
        match self.get(key)? {
            NodeValue::Tensor(t) => Ok(t),
            other => Err(ComfyError::TypeMismatch {
                expected: "Tensor".into(),
                got: format!("{other:?}"),
            }),
        }
    }

    pub fn get_float(&self, key: &str) -> ComfyResult<f64> {
        match self.get(key)? {
            NodeValue::Float(v) => Ok(*v),
            NodeValue::Int(v) => Ok(*v as f64),
            other => Err(ComfyError::TypeMismatch {
                expected: "Float".into(),
                got: format!("{other:?}"),
            }),
        }
    }

    pub fn get_int(&self, key: &str) -> ComfyResult<i64> {
        match self.get(key)? {
            NodeValue::Int(v) => Ok(*v),
            other => Err(ComfyError::TypeMismatch {
                expected: "Int".into(),
                got: format!("{other:?}"),
            }),
        }
    }

    pub fn get_string(&self, key: &str) -> ComfyResult<&str> {
        match self.get(key)? {
            NodeValue::String(s) => Ok(s),
            other => Err(ComfyError::TypeMismatch {
                expected: "String".into(),
                got: format!("{other:?}"),
            }),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NodeOutputs {
    values: HashMap<String, NodeValue>,
}

impl NodeOutputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, value: NodeValue) {
        self.values.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> ComfyResult<&NodeValue> {
        self.values.get(key).ok_or_else(|| ComfyError::InvalidInput {
            node: String::new(),
            message: format!("missing output: {key}"),
        })
    }

    pub fn get_tensor(&self, key: &str) -> ComfyResult<&Tensor> {
        match self.get(key)? {
            NodeValue::Tensor(t) => Ok(t),
            other => Err(ComfyError::TypeMismatch {
                expected: "Tensor".into(),
                got: format!("{other:?}"),
            }),
        }
    }
}

pub trait Node: Send + Sync {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs>;
    fn metadata(&self) -> NodeMetadata;
}

// テストは Step 1 で定義済み
```

- [ ] **Step 3: registry.rs — テスト作成**

```rust
// crates/comfy-core/src/registry.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_create() {
        let mut reg = NodeRegistry::new();
        reg.register("TestNode", || Box::new(crate::node::tests::helper_node()));
        assert!(reg.create("TestNode").is_some());
        assert!(reg.create("Unknown").is_none());
    }

    #[test]
    fn test_list_nodes() {
        let mut reg = NodeRegistry::new();
        reg.register("A", || Box::new(crate::node::tests::helper_node()));
        reg.register("B", || Box::new(crate::node::tests::helper_node()));
        let names = reg.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"A".to_string()));
        assert!(names.contains(&"B".to_string()));
    }
}
```

- [ ] **Step 4: registry.rs — 実装**

```rust
// crates/comfy-core/src/registry.rs
use crate::node::Node;
use std::collections::HashMap;

type NodeFactory = Box<dyn Fn() -> Box<dyn Node> + Send + Sync>;

pub struct NodeRegistry {
    factories: HashMap<String, NodeFactory>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn Node> + Send + Sync + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn Node>> {
        self.factories.get(name).map(|f| f())
    }

    pub fn list(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// テストは Step 3 で定義済み
```

- [ ] **Step 5: node.rs にテスト用ヘルパーを追加**

node.rs の tests モジュール内に追加:
```rust
pub(crate) fn helper_node() -> impl Node {
    struct NoopNode;
    impl Node for NoopNode {
        fn execute(&self, _: &NodeInputs) -> ComfyResult<NodeOutputs> {
            Ok(NodeOutputs::new())
        }
        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "Noop".into(),
                display_name: "Noop".into(),
                category: "test".into(),
                description: "".into(),
                output_node: false,
            }
        }
    }
    NoopNode
}
```

- [ ] **Step 6: lib.rs 更新**

```rust
// crates/comfy-core/src/lib.rs
pub mod error;
pub mod tensor;
pub mod node;
pub mod registry;

pub use error::{ComfyError, ComfyResult};
pub use tensor::{DType, Tensor};
pub use node::{Node, NodeInputs, NodeOutputs, NodeValue, NodeMetadata};
pub use registry::NodeRegistry;
```

- [ ] **Step 7: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-core`
Expected: 全テスト PASS

- [ ] **Step 8: コミット**

```bash
git add -A && git commit -m "feat: Node トレイト + NodeRegistry"
```

---

## Task 3: DAG 構築 + ワークフローパーサー

**Files:**
- Create: `crates/comfy-core/src/workflow.rs`
- Create: `crates/comfy-core/src/dag.rs`
- Modify: `crates/comfy-core/src/lib.rs`

- [ ] **Step 1: workflow.rs — テスト作成**

```rust
// crates/comfy-core/src/workflow.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_workflow() {
        let json = r#"{
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": { "ckpt_name": "model.safetensors" }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": { "text": "a cat", "clip": ["1", 1] }
            }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        assert_eq!(wf.nodes.len(), 2);
        assert_eq!(wf.nodes["1"].class_type, "CheckpointLoaderSimple");
        assert_eq!(wf.nodes["2"].class_type, "CLIPTextEncode");
    }

    #[test]
    fn test_parse_link() {
        let json = r#"{
            "1": {
                "class_type": "A",
                "inputs": { "x": ["2", 0] }
            },
            "2": {
                "class_type": "B",
                "inputs": { "y": 3.14 }
            }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let inputs = &wf.nodes["1"].inputs;
        match &inputs["x"] {
            InputValue::Link(node_id, output_idx) => {
                assert_eq!(node_id, "2");
                assert_eq!(*output_idx, 0);
            }
            _ => panic!("expected Link"),
        }
    }

    #[test]
    fn test_parse_literal_values() {
        let json = r#"{
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 123,
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "denoise": 1.0
                }
            }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let inputs = &wf.nodes["1"].inputs;
        assert!(matches!(inputs["seed"], InputValue::Int(123)));
        assert!(matches!(inputs["cfg"], InputValue::Float(f) if (f - 7.5).abs() < 0.01));
        assert!(matches!(&inputs["sampler_name"], InputValue::String(s) if s == "euler"));
    }
}
```

- [ ] **Step 2: workflow.rs — 実装**

```rust
// crates/comfy-core/src/workflow.rs
use crate::error::{ComfyError, ComfyResult};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum InputValue {
    Link(String, usize),  // (node_id, output_index)
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    None,
}

#[derive(Debug, Clone)]
pub struct WorkflowNode {
    pub id: String,
    pub class_type: String,
    pub inputs: HashMap<String, InputValue>,
}

#[derive(Debug, Clone)]
pub struct Workflow {
    pub nodes: HashMap<String, WorkflowNode>,
}

impl Workflow {
    pub fn from_json(json: &str) -> ComfyResult<Self> {
        let root: HashMap<String, Value> = serde_json::from_str(json)?;
        let mut nodes = HashMap::new();

        for (id, node_val) in &root {
            let class_type = node_val["class_type"]
                .as_str()
                .ok_or_else(|| ComfyError::InvalidInput {
                    node: id.clone(),
                    message: "missing class_type".into(),
                })?
                .to_string();

            let mut inputs = HashMap::new();
            if let Some(inputs_obj) = node_val["inputs"].as_object() {
                for (key, val) in inputs_obj {
                    inputs.insert(key.clone(), parse_input_value(val));
                }
            }

            nodes.insert(
                id.clone(),
                WorkflowNode {
                    id: id.clone(),
                    class_type,
                    inputs,
                },
            );
        }

        Ok(Self { nodes })
    }
}

fn parse_input_value(val: &Value) -> InputValue {
    match val {
        Value::Array(arr) if arr.len() == 2 => {
            // Link: [node_id, output_index]
            let node_id = match &arr[0] {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                _ => return InputValue::None,
            };
            let output_idx = arr[1].as_u64().unwrap_or(0) as usize;
            InputValue::Link(node_id, output_idx)
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                // JSON の整数は小数点なし
                if n.is_f64() && n.to_string().contains('.') {
                    InputValue::Float(n.as_f64().unwrap())
                } else {
                    InputValue::Int(i)
                }
            } else {
                InputValue::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => InputValue::String(s.clone()),
        Value::Bool(b) => InputValue::Bool(*b),
        _ => InputValue::None,
    }
}

// テストは Step 1 で定義済み
```

- [ ] **Step 3: dag.rs — テスト作成**

```rust
// crates/comfy-core/src/dag.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::Workflow;

    #[test]
    fn test_topo_sort_linear() {
        let json = r#"{
            "1": { "class_type": "A", "inputs": {} },
            "2": { "class_type": "B", "inputs": { "x": ["1", 0] } },
            "3": { "class_type": "C", "inputs": { "x": ["2", 0] } }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&wf).unwrap();
        let order = dag.topological_sort().unwrap();
        let pos_1 = order.iter().position(|id| id == "1").unwrap();
        let pos_2 = order.iter().position(|id| id == "2").unwrap();
        let pos_3 = order.iter().position(|id| id == "3").unwrap();
        assert!(pos_1 < pos_2);
        assert!(pos_2 < pos_3);
    }

    #[test]
    fn test_topo_sort_diamond() {
        // 1 → 2, 1 → 3, 2 → 4, 3 → 4
        let json = r#"{
            "1": { "class_type": "A", "inputs": {} },
            "2": { "class_type": "B", "inputs": { "x": ["1", 0] } },
            "3": { "class_type": "C", "inputs": { "x": ["1", 0] } },
            "4": { "class_type": "D", "inputs": { "a": ["2", 0], "b": ["3", 0] } }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&wf).unwrap();
        let order = dag.topological_sort().unwrap();
        assert_eq!(order.len(), 4);
        let pos_1 = order.iter().position(|id| id == "1").unwrap();
        let pos_4 = order.iter().position(|id| id == "4").unwrap();
        assert!(pos_1 < pos_4);
    }

    #[test]
    fn test_cycle_detection() {
        let json = r#"{
            "1": { "class_type": "A", "inputs": { "x": ["2", 0] } },
            "2": { "class_type": "B", "inputs": { "x": ["1", 0] } }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&wf).unwrap();
        assert!(dag.topological_sort().is_err());
    }

    #[test]
    fn test_dependencies() {
        let json = r#"{
            "1": { "class_type": "A", "inputs": {} },
            "2": { "class_type": "B", "inputs": { "x": ["1", 0] } }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&wf).unwrap();
        assert!(dag.dependencies("1").is_empty());
        assert_eq!(dag.dependencies("2"), vec!["1".to_string()]);
    }
}
```

- [ ] **Step 4: dag.rs — 実装**

```rust
// crates/comfy-core/src/dag.rs
use crate::error::{ComfyError, ComfyResult};
use crate::workflow::{InputValue, Workflow};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug)]
pub struct DagEdge {
    pub from_node: String,
    pub from_output: usize,
    pub to_node: String,
    pub to_input: String,
}

#[derive(Debug)]
pub struct Dag {
    pub node_ids: Vec<String>,
    pub edges: Vec<DagEdge>,
    adjacency: HashMap<String, Vec<String>>,      // node → dependents
    reverse_adj: HashMap<String, Vec<String>>,     // node → dependencies
}

impl Dag {
    pub fn from_workflow(wf: &Workflow) -> ComfyResult<Self> {
        let node_ids: Vec<String> = wf.nodes.keys().cloned().collect();
        let mut edges = Vec::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        let mut reverse_adj: HashMap<String, Vec<String>> = HashMap::new();

        for id in &node_ids {
            adjacency.entry(id.clone()).or_default();
            reverse_adj.entry(id.clone()).or_default();
        }

        for (id, node) in &wf.nodes {
            for (input_name, val) in &node.inputs {
                if let InputValue::Link(from_id, from_output) = val {
                    edges.push(DagEdge {
                        from_node: from_id.clone(),
                        from_output: *from_output,
                        to_node: id.clone(),
                        to_input: input_name.clone(),
                    });
                    adjacency.entry(from_id.clone()).or_default().push(id.clone());
                    reverse_adj.entry(id.clone()).or_default().push(from_id.clone());
                }
            }
        }

        Ok(Self {
            node_ids,
            edges,
            adjacency,
            reverse_adj,
        })
    }

    pub fn topological_sort(&self) -> ComfyResult<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for id in &self.node_ids {
            in_degree.insert(id, self.reverse_adj.get(id.as_str()).map_or(0, |v| v.len()));
        }

        let mut queue: VecDeque<String> = self
            .node_ids
            .iter()
            .filter(|id| in_degree[id.as_str()] == 0)
            .cloned()
            .collect();

        let mut result = Vec::with_capacity(self.node_ids.len());

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());
            if let Some(dependents) = self.adjacency.get(&node) {
                for dep in dependents {
                    let deg = in_degree.get_mut(dep.as_str()).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        if result.len() != self.node_ids.len() {
            return Err(ComfyError::CycleDetected);
        }

        Ok(result)
    }

    pub fn dependencies(&self, node_id: &str) -> Vec<String> {
        self.reverse_adj
            .get(node_id)
            .cloned()
            .unwrap_or_default()
    }
}

// テストは Step 3 で定義済み
```

- [ ] **Step 5: lib.rs 更新**

```rust
pub mod workflow;
pub mod dag;
```
を lib.rs に追加。

- [ ] **Step 6: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-core`
Expected: 全テスト PASS

- [ ] **Step 7: コミット**

```bash
git add -A && git commit -m "feat: DAG 構築 + ワークフロー JSON パーサー"
```

---

## Task 4: Executor（ワークフロー実行エンジン）

**Files:**
- Create: `crates/comfy-core/src/executor.rs`
- Create: `crates/comfy-core/src/cache.rs`
- Modify: `crates/comfy-core/src/lib.rs`
- Modify: `crates/comfy-core/Cargo.toml` (rayon 追加)

- [ ] **Step 1: Cargo.toml に rayon + crossbeam 追加**

```toml
[dependencies]
rayon = "1.10"
crossbeam = "0.8"
```

- [ ] **Step 2: cache.rs — テスト作成**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_cache_put_get() {
        let mut cache = TensorCache::new(10);
        let t = Tensor::zeros(vec![2, 2], crate::DType::F32);
        cache.put("node1:0", t.clone());
        assert!(cache.get("node1:0").is_some());
        assert!(cache.get("node2:0").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = TensorCache::new(2);
        cache.put("a", Tensor::zeros(vec![1], crate::DType::F32));
        cache.put("b", Tensor::zeros(vec![1], crate::DType::F32));
        cache.put("c", Tensor::zeros(vec![1], crate::DType::F32));
        // "a" should be evicted
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }
}
```

- [ ] **Step 3: cache.rs — 実装**

```rust
use crate::tensor::Tensor;
use std::collections::{HashMap, VecDeque};

pub struct TensorCache {
    map: HashMap<String, Tensor>,
    order: VecDeque<String>,
    max_entries: usize,
}

impl TensorCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.map.get(key)
    }

    pub fn put(&mut self, key: &str, tensor: Tensor) {
        if self.map.contains_key(key) {
            self.map.insert(key.to_string(), tensor);
            return;
        }
        while self.order.len() >= self.max_entries {
            if let Some(old_key) = self.order.pop_front() {
                self.map.remove(&old_key);
            }
        }
        self.order.push_back(key.to_string());
        self.map.insert(key.to_string(), tensor);
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}
```

- [ ] **Step 4: executor.rs — テスト作成**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::*;
    use crate::registry::NodeRegistry;
    use crate::Tensor;

    struct SourceNode(f32);
    impl Node for SourceNode {
        fn execute(&self, _inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let mut out = NodeOutputs::new();
            out.set("output", NodeValue::Tensor(Tensor::from_vec(vec![self.0], vec![1])));
            Ok(out)
        }
        fn metadata(&self) -> NodeMetadata {
            NodeMetadata { name: "Source".into(), display_name: "Source".into(), category: "test".into(), description: "".into(), output_node: false }
        }
    }

    struct DoubleNode;
    impl Node for DoubleNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let t = inputs.get_tensor("input")?;
            let data: Vec<f32> = t.as_slice_f32().unwrap().iter().map(|v| v * 2.0).collect();
            let mut out = NodeOutputs::new();
            out.set("output", NodeValue::Tensor(Tensor::from_vec(data, t.shape().to_vec())));
            Ok(out)
        }
        fn metadata(&self) -> NodeMetadata {
            NodeMetadata { name: "Double".into(), display_name: "Double".into(), category: "test".into(), description: "".into(), output_node: true }
        }
    }

    #[test]
    fn test_executor_simple_chain() {
        let mut reg = NodeRegistry::new();
        reg.register("Source", || Box::new(SourceNode(5.0)));
        reg.register("Double", || Box::new(DoubleNode));

        let json = r#"{
            "1": { "class_type": "Source", "inputs": {} },
            "2": { "class_type": "Double", "inputs": { "input": ["1", 0] } }
        }"#;

        let mut executor = Executor::new(reg);
        let results = executor.execute_workflow(json).unwrap();
        let t = results["2"].get_tensor("output").unwrap();
        assert_eq!(t.as_slice_f32().unwrap(), vec![10.0]);
    }
}
```

- [ ] **Step 5: executor.rs — 実装**

```rust
use crate::cache::TensorCache;
use crate::dag::Dag;
use crate::error::{ComfyError, ComfyResult};
use crate::node::{Node, NodeInputs, NodeOutputs, NodeValue};
use crate::registry::NodeRegistry;
use crate::workflow::{InputValue, Workflow};
use std::collections::HashMap;

pub struct Executor {
    registry: NodeRegistry,
    cache: TensorCache,
}

impl Executor {
    pub fn new(registry: NodeRegistry) -> Self {
        Self {
            registry,
            cache: TensorCache::new(1024),
        }
    }

    pub fn execute_workflow(&mut self, json: &str) -> ComfyResult<HashMap<String, NodeOutputs>> {
        let workflow = Workflow::from_json(json)?;
        let dag = Dag::from_workflow(&workflow)?;
        let order = dag.topological_sort()?;

        let mut all_outputs: HashMap<String, NodeOutputs> = HashMap::new();

        for node_id in &order {
            let wf_node = &workflow.nodes[node_id];

            let node_impl = self.registry.create(&wf_node.class_type).ok_or_else(|| {
                ComfyError::NodeNotFound(wf_node.class_type.clone())
            })?;

            // Build inputs by resolving links
            let mut inputs = NodeInputs::new();
            for (key, val) in &wf_node.inputs {
                let resolved = match val {
                    InputValue::Link(from_id, from_output) => {
                        let from_outputs = all_outputs.get(from_id).ok_or_else(|| {
                            ComfyError::ExecutionError {
                                node: node_id.clone(),
                                message: format!("dependency {from_id} not yet executed"),
                            }
                        })?;
                        // Find output by index (use "output" as default key for index 0)
                        let output_keys: Vec<&String> = from_outputs.values.keys().collect();
                        if let Some(out_key) = output_keys.get(*from_output) {
                            from_outputs.get(out_key)?.clone()
                        } else if let Ok(v) = from_outputs.get("output") {
                            v.clone()
                        } else {
                            return Err(ComfyError::ExecutionError {
                                node: node_id.clone(),
                                message: format!("output index {from_output} not found in {from_id}"),
                            });
                        }
                    }
                    InputValue::Float(v) => NodeValue::Float(*v),
                    InputValue::Int(v) => NodeValue::Int(*v),
                    InputValue::String(s) => NodeValue::String(s.clone()),
                    InputValue::Bool(b) => NodeValue::Bool(*b),
                    InputValue::None => NodeValue::None,
                };
                inputs.set(key, resolved);
            }

            let outputs = node_impl.execute(&inputs)?;
            all_outputs.insert(node_id.clone(), outputs);
        }

        Ok(all_outputs)
    }
}
```

- [ ] **Step 6: NodeOutputs に values フィールドを pub(crate) に変更**

node.rs の `NodeOutputs` で `values` を `pub(crate)` に修正:
```rust
pub struct NodeOutputs {
    pub(crate) values: HashMap<String, NodeValue>,
}
```

NodeValue に Clone を derive 追加:
```rust
#[derive(Debug, Clone)]
pub enum NodeValue { ... }
```

- [ ] **Step 7: lib.rs 更新**

```rust
pub mod cache;
pub mod executor;
```
を lib.rs に追加。

- [ ] **Step 8: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-core`
Expected: 全テスト PASS

- [ ] **Step 9: コミット**

```bash
git add -A && git commit -m "feat: Executor ワークフロー実行エンジン + LRU キャッシュ"
```

---

## Task 5: comfy-server — REST API サーバー

**Files:**
- Create: `crates/comfy-server/Cargo.toml`
- Create: `crates/comfy-server/src/main.rs`
- Create: `crates/comfy-server/src/state.rs`
- Create: `crates/comfy-server/src/routes.rs`
- Create: `crates/comfy-server/src/queue.rs`
- Create: `crates/comfy-server/src/history.rs`
- Create: `crates/comfy-server/src/ws.rs`
- Create: `crates/comfy-server/src/upload.rs`

- [ ] **Step 1: Cargo.toml**

```toml
[package]
name = "comfy-server"
version.workspace = true
edition.workspace = true

[dependencies]
comfy-core = { path = "../comfy-core" }
axum = { version = "0.8", features = ["multipart", "ws"] }
tokio = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tower-http = { version = "0.6", features = ["cors", "fs"] }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
uuid = { version = "1", features = ["v4"] }
```

- [ ] **Step 2: state.rs — AppState**

```rust
// crates/comfy-server/src/state.rs
use comfy_core::{NodeRegistry, NodeMetadata};
use crate::queue::PromptQueue;
use crate::history::HistoryStore;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

pub struct AppState {
    pub registry: Arc<NodeRegistry>,
    pub queue: Arc<Mutex<PromptQueue>>,
    pub history: Arc<Mutex<HistoryStore>>,
    pub base_path: String,
}
```

- [ ] **Step 3: queue.rs — PromptQueue**

```rust
// crates/comfy-server/src/queue.rs
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub number: f64,
    pub prompt_id: String,
    pub prompt: serde_json::Value,
    pub client_id: Option<String>,
}

pub struct PromptQueue {
    pub running: Vec<QueueItem>,
    pub pending: VecDeque<QueueItem>,
    counter: f64,
}

impl PromptQueue {
    pub fn new() -> Self {
        Self { running: Vec::new(), pending: VecDeque::new(), counter: 0.0 }
    }

    pub fn enqueue(&mut self, prompt: serde_json::Value, prompt_id: String, client_id: Option<String>, front: bool) -> f64 {
        self.counter += 1.0;
        let number = if front { -self.counter } else { self.counter };
        let item = QueueItem { number, prompt_id, prompt, client_id };
        if front {
            self.pending.push_front(item);
        } else {
            self.pending.push_back(item);
        }
        number
    }

    pub fn dequeue(&mut self) -> Option<QueueItem> {
        let item = self.pending.pop_front()?;
        self.running.push(item.clone());
        Some(item)
    }

    pub fn finish(&mut self, prompt_id: &str) {
        self.running.retain(|i| i.prompt_id != prompt_id);
    }

    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    pub fn delete(&mut self, ids: &[String]) {
        self.pending.retain(|i| !ids.contains(&i.prompt_id));
    }

    pub fn remaining(&self) -> usize {
        self.pending.len() + self.running.len()
    }
}
```

- [ ] **Step 4: history.rs — HistoryStore**

```rust
// crates/comfy-server/src/history.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct HistoryEntry {
    pub prompt: serde_json::Value,
    pub outputs: HashMap<String, serde_json::Value>,
    pub status: HistoryStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryStatus {
    pub status_str: String,
    pub completed: bool,
}

pub struct HistoryStore {
    entries: Vec<(String, HistoryEntry)>,
    max_size: usize,
}

impl HistoryStore {
    pub fn new(max_size: usize) -> Self {
        Self { entries: Vec::new(), max_size }
    }

    pub fn add(&mut self, prompt_id: String, entry: HistoryEntry) {
        if self.entries.len() >= self.max_size {
            self.entries.remove(0);
        }
        self.entries.push((prompt_id, entry));
    }

    pub fn get(&self, prompt_id: &str) -> Option<&HistoryEntry> {
        self.entries.iter().find(|(id, _)| id == prompt_id).map(|(_, e)| e)
    }

    pub fn list(&self, max_items: usize, offset: i64) -> HashMap<String, &HistoryEntry> {
        let start = if offset < 0 {
            self.entries.len().saturating_sub(max_items)
        } else {
            offset as usize
        };
        self.entries[start..]
            .iter()
            .take(max_items)
            .map(|(id, e)| (id.clone(), e))
            .collect()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn delete(&mut self, ids: &[String]) {
        self.entries.retain(|(id, _)| !ids.contains(id));
    }
}
```

- [ ] **Step 5: routes.rs — REST API ルート**

```rust
// crates/comfy-server/src/routes.rs
use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        // /prompt
        .route("/prompt", get(get_prompt))
        .route("/prompt", post(post_prompt))
        .route("/api/prompt", get(get_prompt))
        .route("/api/prompt", post(post_prompt))
        // /queue
        .route("/queue", get(get_queue))
        .route("/queue", post(post_queue))
        .route("/api/queue", get(get_queue))
        .route("/api/queue", post(post_queue))
        // /interrupt
        .route("/interrupt", post(post_interrupt))
        .route("/api/interrupt", post(post_interrupt))
        // /history
        .route("/history", get(get_history))
        .route("/history", post(post_history))
        .route("/history/{prompt_id}", get(get_history_by_id))
        .route("/api/history", get(get_history))
        .route("/api/history", post(post_history))
        .route("/api/history/{prompt_id}", get(get_history_by_id))
        // /system_stats
        .route("/system_stats", get(get_system_stats))
        .route("/api/system_stats", get(get_system_stats))
        // /object_info
        .route("/object_info", get(get_object_info))
        .route("/api/object_info", get(get_object_info))
        // /embeddings
        .route("/embeddings", get(get_embeddings))
        .route("/api/embeddings", get(get_embeddings))
}

async fn get_prompt(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let q = state.queue.lock().unwrap();
    Json(serde_json::json!({
        "exec_info": { "queue_remaining": q.remaining() }
    }))
}

#[derive(Deserialize)]
struct PromptRequest {
    prompt: serde_json::Value,
    #[serde(default)]
    prompt_id: Option<String>,
    #[serde(default)]
    front: bool,
    #[serde(default)]
    client_id: Option<String>,
}

async fn post_prompt(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let prompt_id = req.prompt_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let mut q = state.queue.lock().unwrap();
    let number = q.enqueue(req.prompt, prompt_id.clone(), req.client_id, req.front);
    Ok(Json(serde_json::json!({
        "prompt_id": prompt_id,
        "number": number,
        "node_errors": {}
    })))
}

async fn get_queue(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let q = state.queue.lock().unwrap();
    Json(serde_json::json!({
        "queue_running": q.running.iter().take(5).map(|i| serde_json::json!([i.number, i.prompt_id])).collect::<Vec<_>>(),
        "queue_pending": q.pending.iter().take(5).map(|i| serde_json::json!([i.number, i.prompt_id])).collect::<Vec<_>>()
    }))
}

#[derive(Deserialize)]
struct QueueAction {
    #[serde(default)]
    clear: bool,
    #[serde(default)]
    delete: Vec<String>,
}

async fn post_queue(State(state): State<Arc<AppState>>, Json(action): Json<QueueAction>) -> StatusCode {
    let mut q = state.queue.lock().unwrap();
    if action.clear { q.clear_pending(); }
    if !action.delete.is_empty() { q.delete(&action.delete); }
    StatusCode::OK
}

async fn post_interrupt() -> StatusCode {
    // TODO: interrupt currently executing prompt
    StatusCode::OK
}

#[derive(Deserialize)]
struct HistoryQuery {
    #[serde(default = "default_max_items")]
    max_items: usize,
    #[serde(default = "default_offset")]
    offset: i64,
}
fn default_max_items() -> usize { 200 }
fn default_offset() -> i64 { -1 }

async fn get_history(State(state): State<Arc<AppState>>, Query(q): Query<HistoryQuery>) -> Json<serde_json::Value> {
    let h = state.history.lock().unwrap();
    let entries = h.list(q.max_items, q.offset);
    Json(serde_json::to_value(entries).unwrap_or_default())
}

async fn get_history_by_id(State(state): State<Arc<AppState>>, Path(prompt_id): Path<String>) -> Json<serde_json::Value> {
    let h = state.history.lock().unwrap();
    match h.get(&prompt_id) {
        Some(entry) => Json(serde_json::json!({ prompt_id: entry })),
        None => Json(serde_json::json!({})),
    }
}

#[derive(Deserialize)]
struct HistoryAction {
    #[serde(default)]
    clear: bool,
    #[serde(default)]
    delete: Vec<String>,
}

async fn post_history(State(state): State<Arc<AppState>>, Json(action): Json<HistoryAction>) -> StatusCode {
    let mut h = state.history.lock().unwrap();
    if action.clear { h.clear(); }
    if !action.delete.is_empty() { h.delete(&action.delete); }
    StatusCode::OK
}

async fn get_system_stats() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "system": {
            "os": std::env::consts::OS,
            "ram_total": 0,
            "ram_free": 0,
            "comfyui_version": "0.8.24-turbo",
            "python_version": "N/A (native engine)",
            "pytorch_version": "N/A (ONNX Runtime)",
            "embedded_python": false
        },
        "devices": []
    }))
}

async fn get_object_info(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let names = state.registry.list();
    let mut info = serde_json::Map::new();
    for name in names {
        if let Some(node) = state.registry.create(&name) {
            let meta = node.metadata();
            info.insert(name, serde_json::json!({
                "name": meta.name,
                "display_name": meta.display_name,
                "category": meta.category,
                "description": meta.description,
                "output_node": meta.output_node,
                "input": { "required": {} },
                "output": [],
                "output_name": []
            }));
        }
    }
    Json(serde_json::Value::Object(info))
}

async fn get_embeddings() -> Json<Vec<String>> {
    Json(vec![])
}
```

- [ ] **Step 6: ws.rs — WebSocket ハンドラ**

```rust
// crates/comfy-server/src/ws.rs
use crate::state::AppState;
use axum::{
    extract::{ws::{Message, WebSocket, WebSocketUpgrade}, Query, State},
    response::Response,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct WsQuery {
    #[serde(rename = "clientId")]
    pub client_id: Option<String>,
}

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    State(state): State<Arc<AppState>>,
) -> Response {
    let client_id = query.client_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    ws.on_upgrade(move |socket| handle_socket(socket, client_id, state))
}

async fn handle_socket(mut socket: WebSocket, client_id: String, state: Arc<AppState>) {
    // Send initial status
    let q = state.queue.lock().unwrap();
    let remaining = q.remaining();
    drop(q);

    let status = serde_json::json!({
        "type": "status",
        "data": {
            "status": { "exec_info": { "queue_remaining": remaining } },
            "sid": client_id
        }
    });

    if socket.send(Message::Text(status.to_string().into())).await.is_err() {
        return;
    }

    // Keep connection alive, handle incoming messages
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                // Handle feature_flags etc.
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                    if val["type"] == "feature_flags" {
                        let resp = serde_json::json!({
                            "type": "feature_flags",
                            "data": {}
                        });
                        if socket.send(Message::Text(resp.to_string().into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}
```

- [ ] **Step 7: upload.rs — ファイルアップロード + /view (スタブ)**

```rust
// crates/comfy-server/src/upload.rs
use axum::{
    extract::Query,
    http::StatusCode,
    response::Json,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ViewQuery {
    pub filename: String,
    #[serde(default = "default_type")]
    pub r#type: String,
    #[serde(default)]
    pub subfolder: String,
}
fn default_type() -> String { "output".into() }

pub async fn view_image(Query(q): Query<ViewQuery>) -> Result<Vec<u8>, StatusCode> {
    // TODO: read and return actual image file
    Err(StatusCode::NOT_FOUND)
}

pub async fn upload_image() -> Json<serde_json::Value> {
    // TODO: handle multipart upload
    Json(serde_json::json!({"name": "", "subfolder": "", "type": "input"}))
}
```

- [ ] **Step 8: main.rs — エントリポイント**

```rust
// crates/comfy-server/src/main.rs
mod state;
mod routes;
mod queue;
mod history;
mod ws;
mod upload;

use state::AppState;
use queue::PromptQueue;
use history::HistoryStore;
use comfy_core::NodeRegistry;
use axum::Router;
use axum::routing::{get, post};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let registry = NodeRegistry::new();
    // ノード登録は Task 7 で comfy-nodes 統合時に追加する

    let state = Arc::new(AppState {
        registry: Arc::new(registry),
        queue: Arc::new(Mutex::new(PromptQueue::new())),
        history: Arc::new(Mutex::new(HistoryStore::new(10000))),
        base_path: ".".into(),
    });

    let app = routes::api_routes()
        .route("/ws", get(ws::ws_handler))
        .route("/view", get(upload::view_image))
        .route("/upload/image", post(upload::upload_image))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "127.0.0.1:8188";
    tracing::info!("ComfyUI Turbo Engine listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

- [ ] **Step 9: ビルド確認**

Run: `cd comfyui-turbo && cargo build -p comfy-server`
Expected: コンパイル成功

- [ ] **Step 10: コミット**

```bash
git add -A && git commit -m "feat: comfy-server REST API + WebSocket (ComfyUI 互換)"
```

---

## Task 6: comfy-inference — ONNX Runtime ラッパー

**Files:**
- Create: `crates/comfy-inference/Cargo.toml`
- Create: `crates/comfy-inference/src/lib.rs`
- Create: `crates/comfy-inference/src/session.rs`
- Create: `crates/comfy-inference/src/provider.rs`
- Create: `crates/comfy-inference/src/convert.rs`

- [ ] **Step 1: Cargo.toml**

```toml
[package]
name = "comfy-inference"
version.workspace = true
edition.workspace = true

[dependencies]
comfy-core = { path = "../comfy-core" }
ort = { version = "2", features = ["load-dynamic"] }
thiserror = { workspace = true }
tracing = { workspace = true }
```

- [ ] **Step 2: provider.rs — EP 自動選択テスト + 実装**

```rust
// crates/comfy-inference/src/provider.rs
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionProvider {
    Cuda,
    TensorRt,
    DirectMl,
    OpenVino,
    Cpu,
}

impl ExecutionProvider {
    pub fn auto_detect() -> Self {
        // Try providers in order of preference
        if Self::cuda_available() {
            info!("CUDA execution provider selected");
            ExecutionProvider::Cuda
        } else if Self::directml_available() {
            info!("DirectML execution provider selected");
            ExecutionProvider::DirectMl
        } else {
            info!("CPU execution provider selected (fallback)");
            ExecutionProvider::Cpu
        }
    }

    fn cuda_available() -> bool {
        // Check for NVIDIA GPU via environment
        std::env::var("CUDA_PATH").is_ok() || std::path::Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").exists()
    }

    fn directml_available() -> bool {
        cfg!(target_os = "windows")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_returns_valid_provider() {
        let ep = ExecutionProvider::auto_detect();
        assert!(matches!(ep, ExecutionProvider::Cuda | ExecutionProvider::DirectMl | ExecutionProvider::Cpu));
    }
}
```

- [ ] **Step 3: session.rs — セッション管理テスト + 実装**

```rust
// crates/comfy-inference/src/session.rs
use crate::provider::ExecutionProvider;
use comfy_core::error::{ComfyError, ComfyResult};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub struct SessionCache {
    sessions: Mutex<HashMap<String, Arc<ort::session::Session>>>,
    provider: ExecutionProvider,
}

impl SessionCache {
    pub fn new(provider: ExecutionProvider) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            provider,
        }
    }

    pub fn get_or_load(&self, model_path: &str) -> ComfyResult<Arc<ort::session::Session>> {
        let mut cache = self.sessions.lock().unwrap();
        if let Some(session) = cache.get(model_path) {
            return Ok(Arc::clone(session));
        }

        let session = self.create_session(model_path)?;
        let session = Arc::new(session);
        cache.insert(model_path.to_string(), Arc::clone(&session));
        Ok(session)
    }

    fn create_session(&self, model_path: &str) -> ComfyResult<ort::session::Session> {
        let builder = ort::session::Session::builder()
            .map_err(|e| ComfyError::InferenceError(e.to_string()))?;

        builder
            .commit_from_file(model_path)
            .map_err(|e| ComfyError::InferenceError(format!("failed to load {model_path}: {e}")))
    }

    pub fn clear(&self) {
        self.sessions.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_cache_creation() {
        let cache = SessionCache::new(ExecutionProvider::Cpu);
        // Loading a non-existent model should return an error
        assert!(cache.get_or_load("nonexistent.onnx").is_err());
    }
}
```

- [ ] **Step 4: convert.rs — テンソル変換テスト + 実装**

```rust
// crates/comfy-inference/src/convert.rs
use comfy_core::tensor::{DType, Tensor};
use comfy_core::error::{ComfyError, ComfyResult};

/// Convert our Tensor to ort input format (Vec<f32> + shape)
pub fn tensor_to_f32_vec(tensor: &Tensor) -> ComfyResult<(Vec<f32>, Vec<usize>)> {
    let data = tensor.as_slice_f32().ok_or_else(|| {
        ComfyError::TensorError("cannot convert non-F32 tensor".into())
    })?;
    Ok((data, tensor.shape().to_vec()))
}

/// Convert ort output (f32 slice + shape) back to our Tensor
pub fn f32_slice_to_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_roundtrip() {
        let original = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let (data, shape) = tensor_to_f32_vec(&original).unwrap();
        let restored = f32_slice_to_tensor(&data, shape);
        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.as_slice_f32().unwrap(), original.as_slice_f32().unwrap());
    }

    #[test]
    fn test_non_f32_returns_error() {
        let t = Tensor::zeros(vec![2], DType::I8);
        assert!(tensor_to_f32_vec(&t).is_err());
    }
}
```

- [ ] **Step 5: lib.rs**

```rust
pub mod session;
pub mod provider;
pub mod convert;
```

- [ ] **Step 6: テスト実行**

Run: `cd comfyui-turbo && cargo test -p comfy-inference`
Expected: 全テスト PASS

- [ ] **Step 7: コミット**

```bash
git add -A && git commit -m "feat: comfy-inference — ONNX Runtime セッション管理 + EP 自動選択"
```

---

## Task 7: comfy-nodes — 標準ノード (スタブ実装)

**Files:**
- Create: `crates/comfy-nodes/Cargo.toml`
- Create: `crates/comfy-nodes/src/lib.rs`
- Create: `crates/comfy-nodes/src/loaders.rs`
- Create: `crates/comfy-nodes/src/conditioning.rs`
- Create: `crates/comfy-nodes/src/sampling.rs`
- Create: `crates/comfy-nodes/src/latent.rs`
- Create: `crates/comfy-nodes/src/image.rs`

- [ ] **Step 1: Cargo.toml**

```toml
[package]
name = "comfy-nodes"
version.workspace = true
edition.workspace = true

[dependencies]
comfy-core = { path = "../comfy-core" }
comfy-inference = { path = "../comfy-inference" }
tracing = { workspace = true }
```

- [ ] **Step 2: loaders.rs — CheckpointLoaderSimple**

```rust
use comfy_core::*;

pub struct CheckpointLoaderSimple;

impl Node for CheckpointLoaderSimple {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let ckpt_name = inputs.get_string("ckpt_name")?;
        tracing::info!("Loading checkpoint: {ckpt_name}");
        // Stub: return placeholder tensors
        let mut out = NodeOutputs::new();
        out.set("MODEL", NodeValue::String(ckpt_name.to_string()));
        out.set("CLIP", NodeValue::String(format!("{ckpt_name}:clip")));
        out.set("VAE", NodeValue::String(format!("{ckpt_name}:vae")));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "CheckpointLoaderSimple".into(),
            display_name: "Load Checkpoint".into(),
            category: "loaders".into(),
            description: "Loads a diffusion model checkpoint".into(),
            output_node: false,
        }
    }
}
```

- [ ] **Step 3: conditioning.rs — CLIPTextEncode**

```rust
// crates/comfy-nodes/src/conditioning.rs
use comfy_core::*;

pub struct CLIPTextEncode;

impl Node for CLIPTextEncode {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let text = inputs.get_string("text")?;
        tracing::info!("CLIP encoding: {text}");
        // Stub: return text as conditioning placeholder
        let mut out = NodeOutputs::new();
        out.set("CONDITIONING", NodeValue::String(format!("cond:{text}")));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "CLIPTextEncode".into(),
            display_name: "CLIP Text Encode".into(),
            category: "conditioning".into(),
            description: "Encodes text with CLIP model".into(),
            output_node: false,
        }
    }
}
```

- [ ] **Step 4: sampling.rs — KSampler**

```rust
// crates/comfy-nodes/src/sampling.rs
use comfy_core::*;

pub struct KSampler;

impl Node for KSampler {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let seed = inputs.get_int("seed").unwrap_or(0);
        let steps = inputs.get_int("steps").unwrap_or(20);
        let cfg = inputs.get_float("cfg").unwrap_or(7.0);
        let sampler_name = inputs.get_string("sampler_name").unwrap_or("euler");
        tracing::info!("KSampler: seed={seed}, steps={steps}, cfg={cfg}, sampler={sampler_name}");
        // Stub: return random latent
        let latent = Tensor::randn(vec![1, 4, 64, 64], seed as u64);
        let mut out = NodeOutputs::new();
        out.set("LATENT", NodeValue::Tensor(latent));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "KSampler".into(),
            display_name: "KSampler".into(),
            category: "sampling".into(),
            description: "Samples latent using specified sampler and scheduler".into(),
            output_node: false,
        }
    }
}
```

- [ ] **Step 5: latent.rs — EmptyLatentImage + VAEDecode**

```rust
// crates/comfy-nodes/src/latent.rs
use comfy_core::*;

pub struct EmptyLatentImage;

impl Node for EmptyLatentImage {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let width = inputs.get_int("width").unwrap_or(512) as usize;
        let height = inputs.get_int("height").unwrap_or(512) as usize;
        let batch = inputs.get_int("batch_size").unwrap_or(1) as usize;
        let latent = Tensor::zeros(vec![batch, 4, height / 8, width / 8], DType::F32);
        let mut out = NodeOutputs::new();
        out.set("LATENT", NodeValue::Tensor(latent));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "EmptyLatentImage".into(),
            display_name: "Empty Latent Image".into(),
            category: "latent".into(),
            description: "Creates blank latent image".into(),
            output_node: false,
        }
    }
}

pub struct VAEDecode;

impl Node for VAEDecode {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let latent = inputs.get_tensor("samples")?;
        tracing::info!("VAE decoding latent {:?}", latent.shape());
        // Stub: upscale latent 8x to image dimensions, 3 channels
        let shape = latent.shape();
        let h = shape.get(2).copied().unwrap_or(64) * 8;
        let w = shape.get(3).copied().unwrap_or(64) * 8;
        let image = Tensor::zeros(vec![shape[0], 3, h, w], DType::U8);
        let mut out = NodeOutputs::new();
        out.set("IMAGE", NodeValue::Tensor(image));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "VAEDecode".into(),
            display_name: "VAE Decode".into(),
            category: "latent".into(),
            description: "Decodes latent to pixel image using VAE".into(),
            output_node: false,
        }
    }
}
```

- [ ] **Step 6: image.rs — SaveImage + PreviewImage**

```rust
// crates/comfy-nodes/src/image.rs
use comfy_core::*;

pub struct SaveImage;

impl Node for SaveImage {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let prefix = inputs.get_string("filename_prefix").unwrap_or("ComfyUI");
        let image = inputs.get_tensor("images")?;
        tracing::info!("Saving image: prefix={prefix}, shape={:?}", image.shape());
        // Stub: return filename as output
        let filename = format!("{prefix}_00001_.png");
        let mut out = NodeOutputs::new();
        out.set("ui", NodeValue::String(
            serde_json::json!({"images": [{"filename": filename, "type": "output"}]}).to_string()
        ));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "SaveImage".into(),
            display_name: "Save Image".into(),
            category: "image".into(),
            description: "Saves image to output directory".into(),
            output_node: true,
        }
    }
}

pub struct PreviewImage;

impl Node for PreviewImage {
    fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
        let image = inputs.get_tensor("images")?;
        tracing::info!("Preview image: shape={:?}", image.shape());
        let mut out = NodeOutputs::new();
        out.set("ui", NodeValue::String(
            serde_json::json!({"images": [{"filename": "preview.png", "type": "temp"}]}).to_string()
        ));
        Ok(out)
    }

    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            name: "PreviewImage".into(),
            display_name: "Preview Image".into(),
            category: "image".into(),
            description: "Shows image preview".into(),
            output_node: true,
        }
    }
}
```

- [ ] **Step 7: lib.rs — register_all_nodes()**

```rust
// crates/comfy-nodes/src/lib.rs
use comfy_core::NodeRegistry;

mod loaders;
mod conditioning;
mod sampling;
mod latent;
mod image;

pub fn register_all_nodes(registry: &mut NodeRegistry) {
    registry.register("CheckpointLoaderSimple", || Box::new(loaders::CheckpointLoaderSimple));
    registry.register("CLIPTextEncode", || Box::new(conditioning::CLIPTextEncode));
    registry.register("KSampler", || Box::new(sampling::KSampler));
    registry.register("EmptyLatentImage", || Box::new(latent::EmptyLatentImage));
    registry.register("VAEDecode", || Box::new(latent::VAEDecode));
    registry.register("SaveImage", || Box::new(image::SaveImage));
    registry.register("PreviewImage", || Box::new(image::PreviewImage));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_all_nodes() {
        let mut reg = NodeRegistry::new();
        register_all_nodes(&mut reg);
        assert!(reg.contains("CheckpointLoaderSimple"));
        assert!(reg.contains("CLIPTextEncode"));
        assert!(reg.contains("KSampler"));
        assert!(reg.contains("EmptyLatentImage"));
        assert!(reg.contains("VAEDecode"));
        assert!(reg.contains("SaveImage"));
        assert!(reg.contains("PreviewImage"));
        assert_eq!(reg.list().len(), 7);
    }
}
```

- [ ] **Step 8: comfy-server の main.rs を更新して comfy-nodes を統合**

comfy-server/Cargo.toml に依存追加:
```toml
comfy-nodes = { path = "../comfy-nodes" }
```

comfy-server/src/main.rs のレジストリ初期化を変更:
```rust
    let mut registry = NodeRegistry::new();
    comfy_nodes::register_all_nodes(&mut registry);
```

- [ ] **Step 9: テスト実行**

Run: `cd comfyui-turbo && cargo test --workspace`
Expected: 全テスト PASS

- [ ] **Step 10: ビルド + 起動テスト**

Run: `cd comfyui-turbo && cargo run -p comfy-server --release`
Expected: `ComfyUI Turbo Engine listening on 127.0.0.1:8188`

別ターミナルで: `curl http://127.0.0.1:8188/system_stats`
Expected: JSON レスポンス（`"comfyui_version": "0.8.24-turbo"` を含む）

別ターミナルで: `curl http://127.0.0.1:8188/object_info`
Expected: 7ノード分の定義が返る

- [ ] **Step 11: コミット**

```bash
git add -A && git commit -m "feat: comfy-nodes 標準ノード7種 + サーバー統合"
```

---

## Task 8: E2E API 互換性テスト

**Files:**
- Create: `tests/api_compat_test.rs`

- [ ] **Step 1: API 互換性テスト作成**

tests/Cargo.toml はワークスペースルートに統合テスト用依存を追加:
```toml
# comfyui-turbo/Cargo.toml の末尾に追加
[dev-dependencies]
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

```rust
// tests/api_compat_test.rs
use serde_json::Value;

const BASE: &str = "http://127.0.0.1:18188"; // テスト用ポート

/// テスト前に comfy-server をバックグラウンド起動する必要がある:
/// COMFY_PORT=18188 cargo run -p comfy-server &

#[tokio::test]
async fn test_get_prompt_returns_queue_info() {
    let resp: Value = reqwest::get(format!("{BASE}/prompt"))
        .await.unwrap().json().await.unwrap();
    assert!(resp["exec_info"]["queue_remaining"].is_number());
}

#[tokio::test]
async fn test_post_prompt_returns_prompt_id() {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "prompt": {
            "1": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": "test.safetensors" } }
        }
    });
    let resp: Value = client.post(format!("{BASE}/prompt"))
        .json(&body).send().await.unwrap().json().await.unwrap();
    assert!(resp["prompt_id"].is_string());
    assert!(resp["number"].is_number());
    assert!(resp["node_errors"].is_object());
}

#[tokio::test]
async fn test_get_queue_format() {
    let resp: Value = reqwest::get(format!("{BASE}/queue"))
        .await.unwrap().json().await.unwrap();
    assert!(resp["queue_running"].is_array());
    assert!(resp["queue_pending"].is_array());
}

#[tokio::test]
async fn test_get_system_stats_format() {
    let resp: Value = reqwest::get(format!("{BASE}/system_stats"))
        .await.unwrap().json().await.unwrap();
    assert!(resp["system"]["os"].is_string());
    assert!(resp["system"]["comfyui_version"].is_string());
    assert!(resp["devices"].is_array());
}

#[tokio::test]
async fn test_get_object_info_returns_nodes() {
    let resp: Value = reqwest::get(format!("{BASE}/object_info"))
        .await.unwrap().json().await.unwrap();
    assert!(resp["CheckpointLoaderSimple"].is_object());
    assert!(resp["KSampler"].is_object());
    assert!(resp["SaveImage"].is_object());
    assert_eq!(resp["SaveImage"]["output_node"], true);
}

#[tokio::test]
async fn test_api_prefix_works() {
    // /api/prompt と /prompt が同じ結果を返す
    let resp1: Value = reqwest::get(format!("{BASE}/prompt"))
        .await.unwrap().json().await.unwrap();
    let resp2: Value = reqwest::get(format!("{BASE}/api/prompt"))
        .await.unwrap().json().await.unwrap();
    assert_eq!(resp1["exec_info"]["queue_remaining"], resp2["exec_info"]["queue_remaining"]);
}

#[tokio::test]
async fn test_history_crud() {
    let client = reqwest::Client::new();
    // GET history should return object
    let resp: Value = reqwest::get(format!("{BASE}/history"))
        .await.unwrap().json().await.unwrap();
    assert!(resp.is_object());

    // POST clear history
    let status = client.post(format!("{BASE}/history"))
        .json(&serde_json::json!({"clear": true}))
        .send().await.unwrap().status();
    assert_eq!(status, 200);
}

#[tokio::test]
async fn test_queue_crud() {
    let client = reqwest::Client::new();
    // POST clear queue
    let status = client.post(format!("{BASE}/queue"))
        .json(&serde_json::json!({"clear": true}))
        .send().await.unwrap().status();
    assert_eq!(status, 200);
}
```

- [ ] **Step 2: テスト実行**

Run: `cd comfyui-turbo && cargo test --test api_compat_test`
Expected: 全テスト PASS

- [ ] **Step 3: コミット**

```bash
git add -A && git commit -m "test: E2E API 互換性テスト"
```

---

## Task 9: Electron Shell との統合テスト

- [ ] **Step 1: comfy-server をリリースビルド**

Run: `cd comfyui-turbo && cargo build -p comfy-server --release`

- [ ] **Step 2: Electron Shell から接続テスト**

ComfyUI Desktop の comfyServer.ts が `http://127.0.0.1:8188/queue` をポーリングして起動完了を検出する。Turbo Engine を起動した状態で Electron Shell が正常にロードされることを確認。

- [ ] **Step 3: コミット + タグ**

```bash
git add -A && git commit -m "feat: Phase 1 完了 — ComfyUI Turbo Engine v0.1.0"
git tag v0.1.0
```

---

## 完了基準

| 基準 | 検証方法 |
|---|---|
| comfy-core テスト全 PASS | `cargo test -p comfy-core` |
| comfy-inference テスト全 PASS | `cargo test -p comfy-inference` |
| comfy-server ビルド成功 | `cargo build -p comfy-server --release` |
| REST API 全エンドポイント応答 | `curl` で各エンドポイント確認 |
| WebSocket 接続成功 | wscat で接続テスト |
| Electron Shell がバックエンドを認識 | Electron 起動でロード画面通過 |
| E2E テスト全 PASS | `cargo test --test api_compat_test` |
