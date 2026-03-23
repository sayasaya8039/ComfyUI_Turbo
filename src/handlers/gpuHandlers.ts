import { exec } from 'node:child_process';
import { promisify } from 'node:util';

import log from 'electron-log/main';

import { strictIpcMain as ipcMain } from '@/infrastructure/ipcChannels';

import { IPC_CHANNELS } from '../constants';
import { JuliaEnvironment } from '../services/juliaEnvironment';

const execAsync = promisify(exec);

/** Julia system_info.jl の GPU レスポンス型 */
interface JuliaSystemInfo {
  gpu: {
    is_blackwell: boolean;
    architecture: string;
    nvidia_available: boolean;
  };
}

/** Julia 経由で Blackwell 検出 */
async function checkBlackwellViaJulia(): Promise<boolean | null> {
  const julia = JuliaEnvironment.getInstance();
  if (!(await julia.isAvailable())) return null;

  const result = await julia.runScriptJSON<JuliaSystemInfo>('system_info.jl');
  if (result.success && result.data) {
    log.debug(`GPU detected via Julia: ${result.data.gpu.architecture} (Blackwell: ${result.data.gpu.is_blackwell})`);
    return result.data.gpu.is_blackwell;
  }
  return null;
}

/**
 * Handles GPU-related IPC channels.
 * Julia を優先使用し、利用不可の場合は nvidia-smi にフォールバック。
 */
// Note: GET_GPU is handled in appInfoHandlers.ts
export function registerGpuHandlers() {
  ipcMain.handle(IPC_CHANNELS.CHECK_BLACKWELL, async () => {
    // Julia で検出を試行
    try {
      const juliaResult = await checkBlackwellViaJulia();
      if (juliaResult !== null) return juliaResult;
    } catch (err) {
      log.debug('Julia GPU detection failed, falling back to nvidia-smi:', err);
    }

    // nvidia-smi フォールバック
    try {
      const { stdout } = await execAsync('nvidia-smi -q');
      return /Product Architecture\s*:\s*Blackwell/.test(stdout);
    } catch {
      return false;
    }
  });
}
