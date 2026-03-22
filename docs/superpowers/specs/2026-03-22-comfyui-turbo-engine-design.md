# ComfyUI Turbo Engine — 推論エンジン高速化設計書

**日付:** 2026-03-22
**バージョン:** 1.1 (レビュー指摘対応版)
**ステータス:** 承認済み

---

## 1. 概要

ComfyUI Desktop 0.8.24 の Python 推論エンジンを、Rust + Zig + Julia + ONNX Runtime でフルリライトし、可能な限りの高速化を実現する。既存の Electron シェルはそのまま維持し、REST API 完全互換でバックエンドを差し替える。

### 目標

- **性能:** Python ComfyUI 比で上限なしの最適化（段階的に 3x → 15x → 25x+）
- **互換性:** 既存 Python カスタムノード完全互換（pyo3 組み込み）
- **ハードウェア:** NVIDIA (CUDA) + AMD (ROCm/XDNA) + Intel (oneAPI/NPU) 3社対応
- **アーキテクチャ:** GPU / NPU / CPU 完全並列パイプライン

### 技術スタック

| 技術 | 役割 |
|---|---|
| **Rust** | コアエンジン（DAG実行、テンソル管理、API サーバー、pyo3 ブリッジ） |
| **Zig** | 画像I/O、テンソルアロケータ、SIMD カーネル、GPU カスタムカーネル |
| **Julia** | サンプラー、スケジューラ、数値計算、Python numpy/scipy/PIL 差替え |
| **ONNX Runtime** | 推論バックエンド（Phase A）→ 段階的に Zig カーネルで置換 |
| **WASM (Wasmtime)** | 次世代プラグインサンドボックス |
| **NPU** | Intel OpenVINO / AMD XDNA / Qualcomm QNN で軽量推論オフロード |

---

## 2. アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Electron Shell (既存 UI そのまま)                │
│                      http://127.0.0.1:8188/api/*                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP/WebSocket (ComfyUI 互換 API)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  REST API サーバー (Rust: axum or actix-web)                        │
│  /prompt  /queue  /history  /system_stats  /interrupt  /ws          │
│  → ComfyUI Python サーバーと 100% API 互換                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Layer 5: WASM サンドボックスプラグイン (Wasmtime)                    │
│                                                                      │
│  Layer 4: Python 互換 (pyo3) + Julia インターセプト                   │
│           numpy/scipy/PIL → Julia JIT 自動差替え                     │
│                                                                      │
│  Layer 3: Julia 数値計算 (jlrs 埋め込み)                              │
│           サンプラー / スケジューラ / ノイズ / 画像処理                 │
│                                                                      │
│  Layer 2: 推論バックエンド (3段階進化)                                │
│           Phase A: ONNX RT + TensorRT + OpenVINO                     │
│           Phase B: Zig カスタムカーネル置換                            │
│           Phase C: 極限最適化 (フュージョン/FP8/パイプライン並列)      │
│                                                                      │
│  Layer 1: Rust コアエンジン                                           │
│           DAG 実行 / グラフ最適化 / テンソル管理                       │
│           3デバイス並列スケジューラ (GPU/NPU/CPU 独立キュー)           │
│           Zig テンソルアロケータ / 画像 I/O                           │
│                                                                      │
└─────────────┬──────────────────┬──────────────────┬─────────────────┘
              │                  │                  │
    ┌─────────▼───────┐ ┌───────▼────────┐ ┌──────▼──────────┐
    │   GPU Queue      │ │  NPU Queue     │ │  CPU Queue      │
    │ NVIDIA: CUDA     │ │ Intel: OpenVINO│ │ Zig SIMD        │
    │   +TensorRT      │ │ AMD: XDNA      │ │  AVX-512/AMX    │
    │ AMD: ROCm        │ │ Qualcomm: QNN  │ │ Julia JIT       │
    │ Intel: Level Zero│ │                │ │ Rayon pool      │
    └─────────────────┘ └────────────────┘ └─────────────────┘
              │                  │                  │
              └──────────────────┴──────────────────┘
                        共有テンソルバス
                   (ゼロコピー / ピンドメモリ / DMA)
```

---

## 3. Layer 1: Rust コアエンジン

### 3.1 DAG 実行エンジン

ワークフロー JSON をパースし、トポロジカルソートで依存グラフを構築。work-stealing スケジューラで独立ノードを並列実行する。

**Python ComfyUI との差分:**

| Python (現状) | Rust エンジン |
|---|---|
| GIL でシングルスレッド実行 | work-stealing 並列（ノード間 + ノード内） |
| 毎ノードで dict コピー | ゼロコピー Arc<Tensor> 受渡し |
| eager evaluation | 遅延評価 + グラフ最適化 |
| Python dict でテンソル管理 | 型付きテンソルレジストリ + LRU キャッシュ |

### 3.2 グラフ最適化パス

1. 定数畳み込み（静的パラメータを事前計算）
2. デッドノード除去（未使用出力の削除）
3. オペレータフュージョン（Conv+BN+ReLU → FusedConvBNReLU）
4. メモリ割当て計画（テンソルのライフタイム解析 → 再利用）
5. デバイス配置（GPU/NPU/CPU の自動割り当て）
6. 実行順序最適化（転送レイテンシ最小化）

### 3.3 テンソル管理

```
TensorPool
├── VramArena     (GPU メモリプール、断片化防止)
├── PinnedPool    (ページロック CPU メモリ、DMA 転送用)
├── SystemPool    (通常 CPU メモリ、Zig アロケータ)
└── UnifiedPool   (NPU/APU 用 共有メモリ)

特徴:
- アライメント保証 (64byte = キャッシュライン境界)
- NUMA 対応 (マルチソケット CPU)
- 参照カウント + ゼロコピー受渡し
- フレーム単位アリーナアロケータ
```

---

## 4. Layer 2: 推論バックエンド

### 4.1 Phase A: ONNX Runtime + TensorRT（即効 3-5x）

モデルロード時に .safetensors/.ckpt を ONNX 変換・キャッシュし、プロバイダ自動選択で実行。

| GPU/デバイス | プロバイダ |
|---|---|
| NVIDIA | TensorRT EP (FP16/INT8 グラフ最適化) |
| AMD | DirectML EP / MIGraphX EP |
| Intel GPU | DirectML EP |
| Intel NPU | OpenVINO EP |
| AMD NPU | Vitis AI EP |
| CPU | Zig SIMD カーネル |

TensorRT 固有最適化: レイヤーフュージョン、FP16/INT8 自動量子化、Dynamic shape 対応、エンジンキャッシュ。

### 4.2 Phase B: Zig カスタムカーネル（10-20x）

ホットパスから順に ONNX Runtime を Zig カーネルで置換。

**置換優先順位:**
1. UNet/DiT Attention（推論時間の 60-70%）
2. Conv2D（推論時間の 15-20%）
3. VAE Decode
4. CLIP Text Encode
5. その他ノード

**GPU バインディング:**

| GPU | バインディング | カーネル言語 |
|---|---|---|
| NVIDIA | Zig → C interop → CUDA Driver API | PTX (cuModuleLoad) |
| AMD | Zig → C interop → HIP Runtime API | GCN ISA / HIP C++ |
| Intel | Zig → C interop → Level Zero API | SPIR-V |
| CPU | Zig ネイティブ SIMD | AVX-512 / AVX2 / NEON / AMX |

**Flash Attention v2 実装 (Zig comptime):**

```zig
pub fn attention_kernel(comptime backend: Backend) type {
    return struct {
        pub fn forward(q: Tensor, k: Tensor, v: Tensor) Tensor {
            switch (backend) {
                .cuda  => cuda_flash_attn_v2(q, k, v),
                .rocm  => hip_flash_attn(q, k, v),
                .spirv => vulkan_attention(q, k, v),
                .cpu   => simd_attention(q, k, v),
            }
        }
    };
}
```

### 4.3 Phase C: 極限最適化（20x+）

- **カーネルフュージョン:** Attention + LayerNorm → 1カーネル、中間テンソルを SRAM 保持
- **混合精度:** FP8 (Attention Q/K), FP16 (Attention V, Conv), FP32 (アキュムレータ), INT4 (LoRA)
- **パイプライン並列:** UNet 前半/後半を別ストリーム、VAE decode を推論中に先行開始
- **メモリ最適化:** Activation checkpointing, VRAM↔RAM スワップ, テンソル再利用
- **comptime ハードウェア適応:** SM バージョン別タイルサイズ・共有メモリ自動最適化

---

## 5. Layer 2.5: Zig アクセラレーションモジュール

### 5.1 画像 I/O パイプライン

```
入力画像 → Zig デコーダ → ピンドメモリに直接デコード → DMA で VRAM アップロード
                          → GPU カーネルで uint8→FP16 + normalize

Python: PIL.open → numpy → torch.tensor → .to(device)  [4コピー]
Zig:    decode → pinned_mem → DMA → GPU                [1コピー]
改善: 3-8x
```

### 5.2 CPU SIMD カーネル

Zig comptime SIMD ディスパッチで AVX-512 / AVX2 / NEON / AMX を自動選択。

実装カーネル:
1. GEMM（行列乗算）— Intel AMX 対応で CPU でも高スループット
2. Conv2D
3. LayerNorm / GroupNorm
4. SiLU / GELU / Swish
5. Softmax
6. 画像リサイズ (Lanczos)
7. ガウシアンノイズ生成

### 5.3 Zig テンソルアロケータ

```
ZigTensorAllocator
├── PageAllocator     (大テンソル用)
├── SlabAllocator     (小テンソル用、断片化ゼロ)
├── ArenaAllocator    (推論フレーム単位でまとめて解放)
└── PoolAllocator     (頻繁な alloc/free を高速化)

64byte アライメント、NUMA 対応、メモリ使用量トラッキング
```

---

## 6. Layer 3: Julia 数値計算

### 6.1 Julia 置換対象

| 処理 | Python 現状 | Julia 置換 | 期待改善 |
|---|---|---|---|
| サンプラー (Euler/DDIM/DPM++) | NumPy + for ループ | JIT + SIMD 自動ベクトル化 | 5-20x |
| ノイズスケジューラ | Python float 演算 | Julia Float64 JIT | 3-10x |
| CFG | torch.lerp on CPU | Julia ベクトル演算 | 2-5x |
| 画像後処理 | PIL/NumPy | Julia Images.jl | 3-8x |
| LoRA weight merge | Python dict ループ | Julia 行列演算 (BLAS) | 5-15x |
| 統計計算 (SSIM/PSNR) | NumPy/scipy | Julia Statistics.jl | 3-10x |
| マスク演算 | NumPy ブロードキャスト | Julia zero-alloc broadcast | 2-5x |

### 6.2 Julia モジュール構成

```
julia/src/
├── samplers/          euler_discrete, ddim, dpm_plus_plus_2m, uni_pc, lcm
├── schedulers/        noise_schedule, timestep_spacing, cfg_guidance
├── image/             color_transform, blend_modes, resize, mask_ops
├── math/              lora_merge, quantize, statistics
├── numpy_compat/      Python monkey-patch 用 Julia 高速実装
└── gpu/               CUDA.jl, AMDGPU.jl, oneAPI.jl
```

### 6.3 Rust ↔ Julia 統合

- **主方式:** jlrs 埋め込み（Rust プロセス内、ゼロコピー、< 1μs/call）
- **フォールバック:** サブプロセス + 共有メモリ（クラッシュ分離）
- **Julia なし環境:** Rust 実装のサンプラー/スケジューラにフォールバック

---

## 7. Layer 4: Python 互換レイヤー

### 7.1 pyo3 ブリッジ

Python 3.12+ ランタイムを pyo3 0.23+ で埋め込み。ComfyUI 互換シム（comfy.utils, comfy.model_management, comfy.sd, comfy.samplers, comfy.nodes, folder_paths）を提供。

### 7.2 ゼロコピー テンソルブリッジ

```
ネイティブテンソル (Rust/Zig)
    → DLPack 経由 → Python torch.Tensor (ゼロコピー)
    → カスタムノードが処理
    → DLPack 経由 → ネイティブテンソルに戻す (ゼロコピー)
オーバーヘッド: ~5μs per ノード呼び出し
```

### 7.3 Julia インターセプト

Python カスタムノード内の高負荷処理を monkey-patch で透過的に Julia に差替え:

- `numpy.*` 演算 → Julia 配列演算
- `scipy.*` 関数 → Julia 数値計算
- `torch` CPU 演算 → Julia BLAS/LAPACK
- `PIL` 画像処理 → Julia Images.jl
- Python `for` ループ演算 → Julia JIT

**torch GPU 演算はそのまま通過。カスタムノードのコード変更不要。**

### 7.4 Python GIL 対策

Python ノードは専用キューで GIL 制御下に実行。ネイティブノードは GIL 無関係で GPU/NPU/CPU キューでフル並列続行。ノード内の torch GPU 演算中は GIL 自動解放。

---

## 8. Layer 5: WASM サンドボックスプラグイン

### 8.1 概要

Wasmtime ランタイムで WASM バイナリをサンドボックス実行。WASI Preview 2 + Component Model で型安全なノード API を定義。任意の言語（Rust/Zig/C++/Go）から WASM にコンパイルして配布可能。

### 8.2 ノード API (WIT)

```wit
interface comfy-node {
    record tensor {
        data: list<f32>,
        shape: list<u32>,
        dtype: dtype-enum,
    }
    process: func(inputs: list<tensor>) -> list<tensor>;
    metadata: func() -> node-metadata;
}
```

### 8.3 位置づけ

Python 互換は維持しつつ、新規ノードは WASM 推奨。サンドボックスによりファイルシステム/ネットワークアクセスを制御。

---

## 9. 3デバイス完全並列パイプライン

### 9.1 独立ワークキュー

```
DAG スケジューラ
    ├── GPU Queue (専用スレッド) — UNet, Upscale, 大規模推論
    ├── NPU Queue (専用スレッド) — CLIP, ControlNet前処理, VAE Decode, 顔検出
    └── CPU Queue (スレッドプール) — LoRA merge, ノイズ生成, スケジューラ計算, 画像I/O
```

各キューは完全独立。ロックフリーチャネル (crossbeam) で同期。テンソル完了通知で依存先ノードを即座にキュー投入。

### 9.2 並列実行パターン例（txt2img）

```
GPU:  ................[==== UNet Step 1 ====][==== Step 2 ====][==== Step 3 ====]
NPU:  [= CLIP Encode =][= ControlNet Prep =][= 顔検出(先読み) =][= VAE Decode =]
CPU:  [LoRA merge][σ計算][ノイズ生成 Step1][ノイズ生成 Step2][画像保存]
```

GPU が UNet を回す間、NPU と CPU は一切アイドルにならない。

### 9.3 ワークスティーリング

アイドルデバイスが他キューからタスクを奪う:
- GPU アイドル → NPU キューの VAE Decode を GPU で実行
- NPU アイドル → CPU キューの軽量タスクを NPU で実行
- 判定: `steal_benefit = target_speed - source_speed - transfer_cost > threshold`

### 9.4 NPU 3社対応

| ベンダー | NPU | API | TOPS |
|---|---|---|---|
| Intel | Core Ultra / Lunar Lake | OpenVINO EP / Level Zero | 11-48 TOPS |
| AMD | Ryzen AI (XDNA/XDNA2) | Vitis AI EP / XRT Driver | 最大 50 TOPS |
| Qualcomm | Snapdragon X Elite | QNN EP | 最大 45 TOPS |

NPU オフロード適性: CLIP Encode ◎、VAE Decode ◎、ControlNet 前処理 ◎、顔検出/セグメント ◎、UNet △（GPU 優先）

### 9.5 共有テンソルバス

```
TensorBus
├── GPU VRAM Mirror
├── NPU Memory Mirror
└── CPU RAM (Primary)

コヒーレンシ: Exclusive / Shared / Invalid 状態管理
APU/統合メモリ → ゼロコピー
ディスクリート GPU → DMA ダブルバッファ
転送コスト > 節約分 → オフロードしない（自動判定）
```

### 9.6 プロファイラ内蔵

実行時に GPU/NPU/CPU 稼働率を計測。100回実行ごとにルーティングテーブルを自動再最適化。

---

## 10. 4つの実行パス

```
① ネイティブノード (Rust/Zig/Julia) → GPU/NPU/CPU キューに直接投入 [最速]
② ONNX ノード (TensorRT/OpenVINO/DirectML) → ONNX Runtime セッション実行
③ Python ノード (pyo3 + Julia インターセプト) → numpy/scipy/PIL を Julia に差替え
④ WASM ノード (Wasmtime) → サンドボックス内で安全に実行

速度順: ① >> ② > ③+Julia > ④ > ③素Python
```

---

## 11. ComfyUI API 完全互換

既存の ComfyUI クライアントがコード変更なしで動作:

| エンドポイント | 動作 |
|---|---|
| POST /prompt | ワークフロー投入 → DAG 構築 → 実行 |
| GET /queue | 実行キューの状態取得 |
| GET /history | 実行履歴 |
| GET /system_stats | GPU/NPU/CPU 稼働率、VRAM 使用量 |
| POST /interrupt | 実行中ワークフロー中断 |
| WS /ws | 進捗通知 (execution_start/progress/executed) |
| GET /view | 生成画像取得 |
| GET /object_info | ノード定義一覧 |
| POST /upload/image | 画像アップロード |

WebSocket メッセージ形式も完全互換。Electron Shell はバックエンドが Python か Rust か区別できない。

---

## 12. クレート構成

```
comfyui-turbo/
├── crates/
│   ├── comfy-core/         Layer 1: DAG エンジン + テンソル管理 + 3デバイススケジューラ
│   ├── comfy-inference/    Layer 2: ONNX Runtime / TensorRT / OpenVINO / DirectML
│   ├── comfy-zig/          Layer 2.5: Zig カーネル + SIMD + 画像I/O + アロケータ
│   ├── comfy-julia/        Layer 3: Julia サンプラー + スケジューラ + numpy互換
│   ├── comfy-python/       Layer 4: pyo3 ブリッジ + ComfyUI シム + Julia インターセプト
│   ├── comfy-wasm/         Layer 5: Wasmtime ランタイム + WIT API + サンドボックス
│   ├── comfy-nodes/        標準ノード (ネイティブ実装)
│   └── comfy-server/       REST API サーバー (axum) + WebSocket
├── zig/                    Zig ソース
├── julia/                  Julia ソース
└── python/                 Python 互換シム
```

---

## 13. 段階デリバリー計画

| Phase | 内容 | 期待改善 |
|---|---|---|
| Phase 1 | comfy-core + comfy-server + comfy-inference (ONNX RT) | 3-5x |
| Phase 2 | 3デバイススケジューラ + comfy-zig 画像I/O | 5-8x |
| Phase 3 | comfy-julia + comfy-python + Julia インターセプト | 8-15x |
| Phase 4 | comfy-zig GPU カーネル (Attention/Conv2D) | 15-25x |
| Phase 5 | 極限最適化 + comfy-wasm + カーネルフュージョン + FP8 | 25x+ |

---

## 14. 設計判断の根拠

| 判断 | 理由 |
|---|---|
| Rust をコアに選択 | メモリ安全性、ゼロコスト抽象化、pyo3 による Python 互換 |
| Zig でカーネル実装 | comptime でハードウェア固有コード生成、C ABI 互換、SIMD 制御 |
| Julia で数値計算 | JIT で Python 比 10-50x、CUDA.jl/AMDGPU.jl で GPU 直接制御 |
| ONNX RT を Phase A に | 即効性、プロバイダ切替でマルチHW対応、段階的に Zig で置換 |
| 3デバイス独立キュー | GPU/NPU/CPU が互いをブロックしない完全並列 |
| Python monkey-patch | カスタムノードのコード変更なしで Julia 高速化 |
| WASM プラグイン | 言語無関係、サンドボックス安全性、次世代配布形式 |

---

## 15. 依存関係マトリックス

### 15.1 コア依存関係

| 依存関係 | 最低バージョン | 推奨バージョン | 備考 |
|---|---|---|---|
| **Rust** | 1.78+ (Edition 2021) | stable latest | MSRV は pyo3 の要求に準拠 |
| **Zig** | 0.14.0 (stable) | 0.14.x stable | 1.0 GA まで stable リリースのみ使用 |
| **Julia** | 1.10 LTS | 1.11+ | LTS で安定性確保、jlrs 互換性 |
| **Python** | 3.12 | 3.12-3.13 | pyo3 0.23+ の対応範囲 |
| **PyTorch** | 2.2+ | 2.4+ | DLPack 互換、torch.onnx.export 対応 |

### 15.2 推論バックエンド

| 依存関係 | 最低バージョン | 対応 Phase | 備考 |
|---|---|---|---|
| **ONNX Runtime** | 1.18+ | Phase A | opset 18+ 対応 |
| **TensorRT** | 10.0+ | Phase A | FP16/INT8 量子化、Dynamic shape |
| **OpenVINO** | 2024.3+ | Phase A | NPU 対応、Multi-device scheduling |
| **CUDA Toolkit** | 12.0+ | Phase A-C | Compute Capability 7.0+ (Volta 以降) |
| **ROCm** | 6.0+ | Phase A-B | gfx900+ (Vega 以降) |
| **Level Zero** | 1.8+ | Phase B | Intel GPU/NPU 直接制御 |
| **AMD XRT** | 2.16+ | Phase A | XDNA/XDNA2 NPU 制御 |

### 15.3 ランタイム・ライブラリ

| 依存関係 | バージョン | 用途 |
|---|---|---|
| **pyo3** | 0.23+ | Python 埋め込み |
| **jlrs** | 0.22+ | Julia 埋め込み |
| **ort (Rust)** | 2.0+ | ONNX Runtime Rust バインディング |
| **axum** | 0.8+ | REST API サーバー |
| **rayon** | 1.10+ | CPU 並列 (work-stealing) |
| **crossbeam** | 0.8+ | ロックフリーチャネル |
| **wasmtime** | 27.0+ | WASM ランタイム (Component Model 対応) |
| **dlpack** | 1.0 | テンソルゼロコピー規格 |

### 15.4 Phase 別依存関係の変遷

```
Phase 1: Rust + ort + axum + pyo3 + crossbeam + rayon
Phase 2: + Zig 0.14 (画像I/O, アロケータ) + NPU SDK (OpenVINO/XRT)
Phase 3: + Julia 1.10 + jlrs
Phase 4: + CUDA Driver API + HIP + Level Zero (Zig C interop)
Phase 5: + wasmtime + カスタム PTX/SPIR-V
```

---

## 16. エラーハンドリング・フォールバック戦略

### 16.1 グレースフルデグラデーション原則

**すべての非コアコンポーネントは失敗しても推論を継続できる。** フォールバックチェーンで段階的に機能を縮退する。

### 16.2 フォールバックチェーン

| 失敗シナリオ | フォールバック | ユーザー通知 |
|---|---|---|
| **ONNX 変換失敗** | PyTorch モデルを pyo3 経由で直接実行（Python パス）| WARNING ログ + WebSocket 通知 |
| **TensorRT エンジンビルド失敗** | ONNX Runtime CPU/CUDA EP にフォールバック | WARNING ログ |
| **Julia 初期化失敗** | Rust 実装のサンプラー/スケジューラを使用 | INFO ログ（性能は 0.7-0.9x） |
| **Julia JIT 初回コンパイル遅延** | Precompile 済み sysimage を使用。未コンパイル関数は Rust フォールバック | 初回のみ遅延（以降キャッシュ） |
| **NPU デバイス未検出** | GPU または CPU にタスク再割当て | INFO ログ |
| **GPU VRAM 不足 (OOM)** | ① Activation checkpointing 有効化 → ② テンソルを CPU RAM にスピル → ③ バッチサイズ縮小 → ④ 解像度縮小提案 | ERROR + 段階的リカバリ試行 |
| **Python カスタムノードクラッシュ** | ノード単位で失敗を隔離。ワークフロー全体は中断しない | ERROR + 失敗ノード情報 |
| **WASM サンドボックスクラッシュ** | ノード単位で隔離。リソースリミット超過なら通知 | ERROR + リソース超過情報 |
| **pyo3 Python panic** | Rust 側で catch_unwind、Python exception に変換 | ERROR ログ |
| **デバイス間テンソル転送失敗** | ソースデバイスで実行継続（オフロードキャンセル） | WARNING ログ |

### 16.3 VRAM メモリ管理

```
VRAM 使用量監視:
  < 70%: 通常動作
  70-85%: LRU キャッシュから古いテンソルを解放
  85-95%: Activation checkpointing 自動有効化
  > 95%: テンソルを CPU RAM にスピル (ピンドメモリ経由)
  OOM 発生: 全キャッシュクリア → リトライ → 失敗なら解像度縮小提案

スピル戦略:
  1. 最も古いキャッシュテンソルから CPU に移動
  2. 中間 Activation を再計算可能な場合は解放（checkpointing）
  3. モデル重みは最後に CPU 退避（推論速度が大幅に低下するため）
```

### 16.4 Python GIL + Julia 相互作用

```
Python ノード実行時の Julia 呼び出しフロー:

1. pyo3: Python::with_gil(|py| { ... })  — GIL 取得
2. Python ノードコード実行開始
3. numpy 関数呼び出し → monkey-patch が Julia 関数にディスパッチ
4. Julia 呼び出し前に py.allow_threads(|| { ... }) で GIL 一時解放
5. Julia 関数が JIT 実行（GIL なしで並列可能）
6. Julia 完了後、GIL 再取得して Python に戻る

→ Julia 実行中は他の Python ノードも GIL 取得可能
→ torch C extension (cuDNN 等) は内部で GIL 解放済みなので問題なし
```

---

## 17. パフォーマンスベンチマーク基準

### 17.1 基準モデル・条件

| ベンチマーク | モデル | 解像度 | ステップ数 | バッチ | 基準HW |
|---|---|---|---|---|---|
| **B1: SD 1.5 標準** | SD 1.5 (FP16) | 512x512 | 20 (Euler) | 1 | RTX 4090 24GB |
| **B2: SDXL 標準** | SDXL 1.0 (FP16) | 1024x1024 | 20 (DPM++ 2M) | 1 | RTX 4090 24GB |
| **B3: Flux 標準** | Flux.1-dev (FP16) | 1024x1024 | 20 (Euler) | 1 | RTX 4090 24GB |
| **B4: バッチ生成** | SD 1.5 (FP16) | 512x512 | 20 (Euler) | 4 | RTX 4090 24GB |
| **B5: ControlNet** | SD 1.5 + ControlNet (Canny) | 512x512 | 20 | 1 | RTX 4090 24GB |
| **B6: CPU のみ** | SD 1.5 (FP32) | 512x512 | 20 (Euler) | 1 | i9-14900K |
| **B7: NPU オフロード** | SD 1.5 (INT8) | 512x512 | 20 | 1 | Core Ultra 9 |

### 17.2 Phase 別達成基準

| Phase | B1 目標 | B2 目標 | B6 目標 | 測定方法 |
|---|---|---|---|---|
| **Python baseline** | ~4.5 it/s | ~1.2 it/s | ~0.05 it/s | ComfyUI 0.18.0 + PyTorch 2.4 |
| **Phase 1** (3-5x) | 13-22 it/s | 3.6-6 it/s | 0.15-0.25 it/s | ONNX RT + TensorRT |
| **Phase 2** (5-8x) | 22-36 it/s | 6-10 it/s | 0.25-0.4 it/s | + 3デバイス並列 |
| **Phase 3** (8-15x) | 36-67 it/s | 10-18 it/s | 0.4-0.75 it/s | + Julia サンプラー |
| **Phase 4** (15-25x) | 67-112 it/s | 18-30 it/s | 0.75-1.25 it/s | + Zig カーネル |
| **Phase 5** (25x+) | 112+ it/s | 30+ it/s | 1.25+ it/s | + 極限最適化 |

**注:** 上記は理想条件下での目標値。実測値は HW 構成、モデルサイズ、VRAM 制約により変動する。各 Phase 完了時にベンチマークを実行し、未達の場合はボトルネック分析後に追加最適化を検討する。

### 17.3 追加測定指標

| 指標 | 目標 |
|---|---|
| サーバー起動時間 | < 3秒（モデルロード除く） |
| メモリ使用量 (エンジンのみ) | < 100MB RAM (モデル・テンソル除く) |
| 最初の画像生成までの時間 (TTFI) | < 10秒（モデル初回ロード込み） |
| ワークフロー投入→実行開始レイテンシ | < 5ms |

---

## 18. テスト・検証計画

### 18.1 Phase 別テスト要件

| Phase | 単体テスト | 統合テスト | 互換性テスト | ベンチマーク |
|---|---|---|---|---|
| **Phase 1** | DAG エンジン、ONNX ラッパー、API サーバー | API 互換 (全エンドポイント) | ComfyUI 標準ノード 30+ 個 | B1, B2, B3 |
| **Phase 2** | スケジューラ、テンソルバス、画像I/O | 3デバイス並列動作確認 | Phase 1 テスト全パス | B1-B5, B7 |
| **Phase 3** | Julia サンプラー、pyo3 ブリッジ、インターセプト | Python カスタムノード互換 | 人気ノード Top 50 | B1-B7 |
| **Phase 4** | Zig カーネル (各 GPU バックエンド) | 推論精度検証 (PSNR > 40dB) | Phase 1-3 テスト全パス | 全ベンチマーク |
| **Phase 5** | WASM ランタイム、カーネルフュージョン | WASM ノード動作確認 | 全テスト回帰 | 全ベンチマーク |

### 18.2 互換性テスト対象（Python カスタムノード）

```
Tier 1 (必須互換、Phase 1 で確認):
  - ComfyUI 標準ノード全種 (~80 個)
  - KSampler, CLIPTextEncode, VAEDecode, ImageSave 等

Tier 2 (重要、Phase 3 で確認):
  - ComfyUI-Impact-Pack
  - ComfyUI-AnimateDiff
  - ComfyUI-IPAdapter
  - ComfyUI-ControlNet-Aux
  - ComfyUI-KJNodes
  - 他 人気 Top 50 ノードパック

Tier 3 (ベストエフォート、Phase 5):
  - その他全カスタムノード（pyo3 互換で原理的に動作するはず）
```

### 18.3 推論精度検証

同一モデル・同一シード・同一パラメータで Python ComfyUI と Turbo Engine の出力画像を比較:
- **PSNR > 40dB** (ほぼ同一)
- **SSIM > 0.99**
- 許容誤差: FP16 丸め誤差のみ。INT8 量子化時は PSNR > 30dB

---

## 19. WASM テンソル I/O 最適化

### 19.1 問題

Section 8.2 の `list<f32>` は大規模テンソル (4K 画像 = ~100MB) でコピーボトルネックになる。

### 19.2 解決: 2段階テンソル受渡し

```
小テンソル (< 1MB):
  → list<f32> で直接コピー（オーバーヘッド無視可能）

大テンソル (>= 1MB):
  → shared memory 経由:
    1. ホスト側でテンソルを mmap 共有メモリ領域に配置
    2. WASM ノードに共有メモリのファイルディスクリプタを渡す
    3. WASM 側で memory.grow() せずに直接アクセス
    4. ゼロコピー（コヒーレンシはホスト側が管理）

WIT 拡張:
  interface comfy-node-v2 {
      // 小テンソル用
      record tensor-inline {
          data: list<f32>,
          shape: list<u32>,
          dtype: dtype-enum,
      }
      // 大テンソル用 (shared memory)
      record tensor-shared {
          shm-handle: u64,
          offset: u64,
          byte-length: u64,
          shape: list<u32>,
          dtype: dtype-enum,
      }
      type tensor = variant { inline(tensor-inline), shared(tensor-shared) };
  }
```

### 19.3 WASM の推奨用途

GPU メモリアクセスが必要な推論カーネルには不向き。以下に限定:
- 前処理 (画像フィルタ、色変換、リサイズ)
- 後処理 (合成、テキスト描画、メタデータ付与)
- ユーティリティ (フォーマット変換、バリデーション)
- 軽量推論 (CPU のみの小規模モデル)

---

## 20. リスク評価と対策

| ID | リスク | 影響度 | 確率 | 対策 |
|---|---|---|---|---|
| R1 | ONNX 変換が一部モデルで失敗 | 高 | 60% | pyo3 経由 PyTorch 直接実行にフォールバック。変換成功率を Tier テストで計測 |
| R2 | Julia JIT 初回コンパイルが遅い (100ms-1s) | 中 | 50% | PackageCompiler.jl で sysimage 事前ビルド。未コンパイル関数は Rust フォールバック |
| R3 | pyo3 + GIL + C extension の相互作用が未知 | 中-高 | 40% | Phase 1 で PoC 実施: カスタムノード 10 個で動作・性能検証 |
| R4 | Zig GPU カーネルの3社対応が複雑 | 高 | 60% | Phase 1-3 は ONNX RT に依存。Phase 4 は NVIDIA 先行、AMD/Intel は順次対応 |
| R5 | Zig コンパイラの安定性 (0.14 stable) | 中 | 30% | Zig stable のみ使用。C ABI fallback を常に用意 |
| R6 | 3デバイス並列のワークスティーリングが期待通り機能しない | 中 | 50% | Phase 2 は静的デバイス割当てで開始。プロファイラデータ蓄積後に動的 steal を有効化 |
| R7 | WASM Component Model の成熟度不足 | 低 | 30% | Phase 5 まで延期。それまでは Python + ネイティブのみ |
