import log from 'electron-log/main';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileSync } from 'tmp';

import { pathAccessible } from '@/utils';

import { getAppResourcesPath } from '../install/resourcePaths';
import { ProcessCallbacks, VirtualEnvironment } from '../virtualEnvironment';
import { JuliaEnvironment } from './juliaEnvironment';
import { HasTelemetry, ITelemetry, trackEvent } from './telemetry';

/**
 * ComfyUI-Manager CLI ラッパー。
 * Julia を優先使用し、利用不可の場合は Python (cm-cli.py) にフォールバック。
 */
export class CmCli implements HasTelemetry {
  private readonly cliPath: string;
  private readonly moduleName = 'comfyui_manager.cm_cli';
  private readonly julia: JuliaEnvironment;

  constructor(
    private readonly virtualEnvironment: VirtualEnvironment,
    readonly telemetry: ITelemetry
  ) {
    this.cliPath = path.join(getAppResourcesPath(), 'ComfyUI', 'custom_nodes', 'ComfyUI-Manager', 'cm-cli.py');
    this.julia = JuliaEnvironment.getInstance();
  }

  /** Julia の manager_cli.jl が使えるか */
  private async canUseJulia(): Promise<boolean> {
    return await this.julia.isAvailable();
  }

  /** Julia 経由でスナップショット保存 */
  private async saveSnapshotViaJulia(comfyDir: string, outFile: string): Promise<boolean> {
    const result = await this.julia.runScriptJSON('manager_cli.jl', ['save-snapshot', comfyDir, outFile]);
    if (result.success && result.data && (result.data as Record<string, unknown>).status === 'success') {
      log.info(`Julia snapshot saved: ${outFile}`);
      return true;
    }
    log.warn('Julia snapshot failed:', result.error);
    return false;
  }

  /** Julia 経由でスナップショット復元 */
  private async restoreSnapshotViaJulia(snapshotFile: string, targetDir: string): Promise<boolean> {
    const result = await this.julia.runScriptJSON('manager_cli.jl', ['restore-snapshot', snapshotFile, targetDir]);
    if (result.success && result.data && (result.data as Record<string, unknown>).status === 'success') {
      const data = result.data as Record<string, unknown>;
      log.info(`Julia restore: ${data.restored_count} nodes restored, ${data.failed_count} failed`);
      return true;
    }
    log.warn('Julia restore failed:', result.error);
    return false;
  }

  /** Julia 経由でカスタムノードをスキャン */
  public async scanCustomNodes(customNodesDir: string): Promise<unknown[]> {
    if (await this.canUseJulia()) {
      const result = await this.julia.runScriptJSON<{ nodes: unknown[] }>('manager_cli.jl', ['scan', customNodesDir]);
      if (result.success && result.data) {
        return result.data.nodes;
      }
    }
    return [];
  }

  /** Julia 経由で更新チェック */
  public async checkUpdates(customNodesDir: string): Promise<unknown> {
    if (await this.canUseJulia()) {
      const result = await this.julia.runScriptJSON('manager_cli.jl', ['check-updates', customNodesDir]);
      if (result.success && result.data) {
        return result.data;
      }
    }
    return null;
  }

  private async buildCommandArgs(args: string[]): Promise<string[]> {
    if (await pathAccessible(this.cliPath)) {
      return [this.cliPath, ...args];
    }
    return ['-m', this.moduleName, ...args];
  }

  public async runCommandAsync(
    args: string[],
    callbacks?: ProcessCallbacks,
    env: Record<string, string> = {},
    checkExit: boolean = true,
    cwd?: string
  ) {
    let output = '';
    let error = '';
    const ENV = {
      COMFYUI_PATH: this.virtualEnvironment.basePath,
      ...env,
    };
    const commandArgs = await this.buildCommandArgs(args);
    const { exitCode } = await this.virtualEnvironment.runPythonCommandAsync(
      commandArgs,
      {
        onStdout: (message) => {
          output += message;
          callbacks?.onStdout?.(message);
        },
        onStderr: (message) => {
          console.warn('[warn]', message);
          error += message;
          callbacks?.onStderr?.(message);
        },
      },
      ENV,
      cwd
    );

    if (checkExit && exitCode !== 0) {
      throw new Error(`Error calling cm-cli: \nExit code: ${exitCode}\nOutput:${output}\n\nError:${error}`);
    }

    return output;
  }

  @trackEvent('migrate_flow:migrate_custom_nodes')
  public async restoreCustomNodes(fromComfyDir: string, callbacks: ProcessCallbacks) {
    const tmpFile = fileSync({ postfix: '.json' });
    try {
      log.debug('Using temp file:', tmpFile.name);

      // Julia 優先でスナップショット保存
      let juliaUsed = false;
      if (await this.canUseJulia()) {
        callbacks.onStdout?.('Using Julia for snapshot save...\n');
        juliaUsed = await this.saveSnapshotViaJulia(fromComfyDir, tmpFile.name);
      }

      if (!juliaUsed) {
        // Python フォールバック
        callbacks.onStdout?.('Using Python cm-cli for snapshot save...\n');
        await this.saveSnapshot(fromComfyDir, tmpFile.name, callbacks);
      }

      // Julia 優先でスナップショット復元
      const targetDir = path.join(this.virtualEnvironment.basePath, 'custom_nodes');
      let juliaRestored = false;
      if (juliaUsed && (await this.canUseJulia())) {
        callbacks.onStdout?.('Using Julia for snapshot restore...\n');
        juliaRestored = await this.restoreSnapshotViaJulia(tmpFile.name, targetDir);
      }

      if (!juliaRestored) {
        // Python フォールバック
        callbacks.onStdout?.('Using Python cm-cli for snapshot restore...\n');
        await this.restoreSnapshot(tmpFile.name, targetDir, callbacks);
      }

      // Remove extra ComfyUI-Manager directory that was created by the migration.
      const managerPath = path.join(this.virtualEnvironment.basePath, 'custom_nodes', 'ComfyUI-Manager');
      if (await pathAccessible(managerPath)) {
        await fs.rm(managerPath, { recursive: true, force: true });
        log.info('Removed extra ComfyUI-Manager directory:', managerPath);
      }
    } finally {
      tmpFile?.removeCallback();
    }
  }

  public async saveSnapshot(fromComfyDir: string, outFile: string, callbacks: ProcessCallbacks): Promise<void> {
    const output = await this.runCommandAsync(
      ['save-snapshot', '--output', outFile, '--no-full-snapshot'],
      callbacks,
      {
        COMFYUI_PATH: fromComfyDir,
        PYTHONPATH: fromComfyDir,
      },
      true,
      fromComfyDir
    );
    log.info(output);
  }

  public async restoreSnapshot(snapshotFile: string, toComfyDir: string, callbacks: ProcessCallbacks) {
    log.info('Restoring snapshot', snapshotFile);
    const output = await this.runCommandAsync(
      ['restore-snapshot', snapshotFile, '--restore-to', toComfyDir],
      callbacks,
      {
        COMFYUI_PATH: path.join(getAppResourcesPath(), 'ComfyUI'),
      }
    );
    log.info(output);
  }
}
