# HANDOFF — ComfyUI Turbo 次セッション引継ぎ

## 現状

### 動作するもの
- Turbo Engine (comfy-server.exe) — 134ms 起動、8クレート、250テスト
- Electron Desktop 統合 — comfy-server.exe 自動検出
- ハイブリッドモード — Turbo Engine で即起動 → Python サーバーに自動切替 (28秒後)
- テンプレート — Python venv から966アセットを起動時にロード
- 全 REST API エンドポイント (ComfyUI 互換)
- WebSocket 進捗通知
- プロセス cleanup (taskkill /T)

### 未解決の課題

#### 最重要: Turbo Engine ⇔ Python 並走プロキシ
現在は「Turbo → Python 完全切替」方式。これだと Turbo Engine の高速カーネルが使われない。

**あるべき姿:**
```
Turbo Engine (port 8000) ← フロントエンド常駐
    ↓ リクエスト振り分け
    ├── 自分で処理 → DAG実行、SIMD カーネル、テンソル管理
    └── Python にプロキシ → /prompt 実行、object_info、カスタムノード、テンプレートDL
Python Server (port 8001) ← バックグラウンド常駐
```

実装すべきこと:
1. `comfy-server` に HTTP プロキシ機能追加 (reqwest or hyper)
2. `/prompt` POST → Python にフォワード
3. `/object_info` → Python から取得してマージ (Turbo ノード + Python ノード)
4. テンプレートDL → Python の ComfyUI-Manager にフォワード
5. Turbo Engine 独自のノード実行は将来的に対応

#### その他
- `object_info` が7ノードしかない → Python の全ノード定義を起動時にキャッシュ
- ワークフロー実行 → 実モデル推論は ONNX Runtime + 実モデルロードが必要
- ゾンビプロセス問題 → Windows で taskkill が効かないケースがある

## リポジトリ

| リポジトリ | URL |
|---|---|
| Desktop (Electron) | https://github.com/sayasaya8039/ComfyUI_Turbo |
| Engine (Rust) | https://github.com/sayasaya8039/comfyui-turbo-engine |

## ファイル構成

### Turbo Engine (D:\NEXTCLOUD\Windows_app\comfyui-turbo)
- 8クレート: comfy-core, comfy-inference, comfy-julia, comfy-nodes, comfy-python, comfy-server, comfy-zig, comfy-wasm
- 250テスト、v1.0.0
- `target/release/comfy-server.exe` — 7.5MB

### Desktop (D:\NEXTCLOUD\Windows_app\ComfyUI_desktop_0.8.24)
- 変更ファイル: `src/main-process/comfyServer.ts`, `src/desktopApp.ts`, `builder-debug.config.ts`
- `dist/win-unpacked/ComfyUI.exe` — ビルド済み

## 環境変数

| 変数 | 用途 |
|---|---|
| `COMFY_PORT` | サーバーポート (default: 8188) |
| `COMFY_FRONTEND` | フロントエンド静的ファイルパス |
| `COMFY_VENV` | Python venv パス (テンプレート用) |
