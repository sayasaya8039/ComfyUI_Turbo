import { app } from 'electron';
import log from 'electron-log/main';
import { ChildProcess, execSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import { getAppResourcesPath } from '../install/resourcePaths';

export interface JuliaCommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

/**
 * Julia ランタイム環境マネージャ。
 * Python の virtualEnvironment.ts に相当する Julia 側の管理クラス。
 */
export class JuliaEnvironment {
  private static instance: JuliaEnvironment | null = null;

  /** Julia 実行パス */
  readonly juliaPath: string;

  /** Julia スクリプトディレクトリ */
  readonly scriptDir: string;

  /** Julia プロジェクトディレクトリ */
  readonly projectDir: string;

  /** Julia が利用可能か */
  private _available: boolean | null = null;

  /** Julia バージョンキャッシュ */
  private _version: string | null = null;

  private constructor() {
    this.scriptDir = path.join(getAppResourcesPath(), 'julia', 'src');
    this.projectDir = path.join(getAppResourcesPath(), 'julia');
    this.juliaPath = this.detectJuliaPath();
  }

  static getInstance(): JuliaEnvironment {
    if (!JuliaEnvironment.instance) {
      JuliaEnvironment.instance = new JuliaEnvironment();
    }
    return JuliaEnvironment.instance;
  }

  /** Julia の実行パスを検出 */
  private detectJuliaPath(): string {
    // 1. 環境変数 JULIA_PATH
    const envPath = process.env.JULIA_PATH;
    if (envPath && fs.existsSync(envPath)) return envPath;

    // 2. アプリ同梱の Julia
    const bundledPath = path.join(
      getAppResourcesPath(),
      'julia',
      'bin',
      process.platform === 'win32' ? 'julia.exe' : 'julia'
    );
    if (fs.existsSync(bundledPath)) return bundledPath;

    // 3. システム Julia
    const systemNames = process.platform === 'win32' ? ['julia.exe'] : ['julia'];
    for (const name of systemNames) {
      try {
        const which = process.platform === 'win32' ? 'where' : 'which';
        const result = execSync(`${which} ${name}`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
        const found = result.trim().split('\n')[0]?.trim();
        if (found && fs.existsSync(found)) return found;
      } catch {
        // not found
      }
    }

    // 4. Windows のよくあるパス
    if (process.platform === 'win32') {
      const commonPaths = [
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Julia', 'bin', 'julia.exe'),
        path.join(process.env.LOCALAPPDATA || '', 'juliaup', 'bin', 'julia.exe'),
        'C:\\Julia\\bin\\julia.exe',
      ];
      for (const p of commonPaths) {
        if (fs.existsSync(p)) return p;
      }
    }

    return 'julia'; // フォールバック
  }

  /** Julia が利用可能かチェック */
  async isAvailable(): Promise<boolean> {
    if (this._available !== null) return this._available;

    try {
      const result = await this.runCommand(['--version']);
      this._available = result.exitCode === 0;
      if (this._available) {
        this._version = result.stdout.trim().replace('julia version ', '');
        log.info(`Julia detected: v${this._version} at ${this.juliaPath}`);
      }
    } catch {
      this._available = false;
      log.debug('Julia not available');
    }

    return this._available;
  }

  get version(): string | null {
    return this._version;
  }

  /** Julia コマンドを実行 */
  async runCommand(args: string[], env: Record<string, string> = {}): Promise<JuliaCommandResult> {
    return new Promise((resolve) => {
      let stdout = '';
      let stderr = '';

      const proc = spawn(this.juliaPath, args, {
        env: { ...process.env, ...env },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on('error', (err) => {
        resolve({ exitCode: -1, stdout, stderr: stderr + err.message });
      });

      proc.on('exit', (code) => {
        resolve({ exitCode: code, stdout, stderr });
      });
    });
  }

  /** Julia スクリプトを実行（プロジェクト環境付き） */
  async runScript(
    scriptName: string,
    args: string[] = [],
    env: Record<string, string> = {}
  ): Promise<JuliaCommandResult> {
    const scriptPath = path.join(this.scriptDir, scriptName);

    if (!fs.existsSync(scriptPath)) {
      return {
        exitCode: -1,
        stdout: '',
        stderr: `Julia script not found: ${scriptPath}`,
      };
    }

    const juliaArgs = ['--project=' + this.projectDir, '--startup-file=no', scriptPath, ...args];

    return this.runCommand(juliaArgs, env);
  }

  /** Julia スクリプトを実行し、JSON 結果をパース */
  async runScriptJSON<T = Record<string, unknown>>(
    scriptName: string,
    args: string[] = [],
    env: Record<string, string> = {}
  ): Promise<{ success: boolean; data?: T; error?: string }> {
    const result = await this.runScript(scriptName, args, env);

    if (result.exitCode !== 0 && result.exitCode !== 1) {
      return { success: false, error: `Exit code ${result.exitCode}: ${result.stderr}` };
    }

    try {
      const data = JSON.parse(result.stdout.trim()) as T;
      return { success: true, data };
    } catch {
      return { success: false, error: `Failed to parse JSON: ${result.stdout}` };
    }
  }

  /** Julia パッケージをインストール（初回セットアップ） */
  async installPackages(): Promise<boolean> {
    if (!(await this.isAvailable())) return false;

    log.info('Installing Julia packages...');
    const result = await this.runCommand(['--project=' + this.projectDir, '-e', 'using Pkg; Pkg.instantiate()']);

    if (result.exitCode === 0) {
      log.info('Julia packages installed successfully');
      return true;
    }

    log.error('Julia package installation failed:', result.stderr);
    return false;
  }
}
