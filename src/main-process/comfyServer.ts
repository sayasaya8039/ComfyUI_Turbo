import { app } from 'electron';
import log from 'electron-log/main';
import { ChildProcess, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import waitOn from 'wait-on';

import { removeAnsiCodesTransform } from '@/infrastructure/structuredLogging';

import { ComfyServerConfig } from '../config/comfyServerConfig';
import { ComfySettings } from '../config/comfySettings';
import { IPC_CHANNELS, LogFile, ServerArgs } from '../constants';
import { getAppResourcesPath } from '../install/resourcePaths';
import { HasTelemetry, ITelemetry, trackEvent } from '../services/telemetry';
import { rotateLogFiles } from '../utils';
import { VirtualEnvironment } from '../virtualEnvironment';
import { AppWindow } from './appWindow';

/** Throttle IPC log messages to reduce renderer overhead */
class LogThrottle {
  private buffer: string[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  constructor(
    private readonly flush: (msg: string) => void,
    private readonly intervalMs = 16 // ~60fps
  ) {}

  push(data: string) {
    this.buffer.push(data);
    if (!this.timer) {
      this.timer = setTimeout(() => {
        const batch = this.buffer.join('');
        this.buffer = [];
        this.timer = null;
        this.flush(batch);
      }, this.intervalMs);
    }
  }

  drain() {
    if (this.timer) clearTimeout(this.timer);
    if (this.buffer.length > 0) {
      this.flush(this.buffer.join(''));
      this.buffer = [];
    }
    this.timer = null;
  }
}

/** Known server start errors. */
type ServerStartError = 'ModuleNotFoundError';

/**
 * A class that manages the ComfyUI server.
 *
 * This class is responsible for starting and stopping the ComfyUI server,
 * as well as handling the server's lifecycle events.
 *
 * isRunning: The server process is running.
 * timedOutWhilstStarting: The server process failed to start within the timeout. The process may still be running.
 */
export class ComfyServer implements HasTelemetry {
  /**
   * The maximum amount of time to wait for the server to start.
   * Installing custom nodes dependencies like ffmpeg can take a long time,
   * so we need to give it a long timeout.
   */
  public static readonly MAX_FAIL_WAIT = 30 * 60 * 1000; // 30 minutes

  /**
   * The interval to check if the server is ready.
   */
  public static readonly CHECK_INTERVAL = 1000; // Check every second

  /** The path to the ComfyUI main python script. */
  readonly mainScriptPath = path.join(getAppResourcesPath(), 'ComfyUI', 'main.py');

  /** The path to the Turbo Engine binary (if available). */
  readonly turboEnginePath = path.join(getAppResourcesPath(), 'comfy-server.exe');

  /** Whether to use the native Turbo Engine instead of Python. */
  get useTurboEngine(): boolean {
    return fs.existsSync(this.turboEnginePath);
  }

  /**
   * The path to the ComfyUI web root. This directory should host compiled
   * ComfyUI web assets.
   */
  readonly webRootPath = path.join(getAppResourcesPath(), 'ComfyUI', 'web_custom_versions', 'desktop_app');

  readonly userDirectoryPath: string;
  readonly inputDirectoryPath: string;
  readonly outputDirectoryPath: string;

  /** Whether the server failed to report started within the start timeout. */
  timedOutWhilstStarting = false;

  private comfyServerProcess: ChildProcess | null = null;

  private lastStdErr?: string;

  constructor(
    readonly basePath: string,
    readonly serverArgs: ServerArgs,
    readonly virtualEnvironment: VirtualEnvironment,
    readonly appWindow: AppWindow,
    readonly telemetry: ITelemetry
  ) {
    this.userDirectoryPath = path.join(this.basePath, 'user');
    this.inputDirectoryPath = path.join(this.basePath, 'input');
    this.outputDirectoryPath = path.join(this.basePath, 'output');
  }

  /** Whether the server is expected to be running. */
  get isRunning() {
    return !!this.comfyServerProcess;
  }

  get baseUrl() {
    return `http://${this.serverArgs.listen}:${this.serverArgs.port}`;
  }

  private resolveDatabasePath(userDirectoryPath: string): string {
    if (process.platform === 'win32') {
      return path.win32.resolve(userDirectoryPath, 'comfyui.db');
    }

    return path.resolve(userDirectoryPath, 'comfyui.db');
  }

  private get databaseUrl(): string {
    const dbPath = this.resolveDatabasePath(this.userDirectoryPath);
    const normalizedDbPath = process.platform === 'win32' ? dbPath.replaceAll('\\', '/') : dbPath;
    return `sqlite:///${normalizedDbPath}`;
  }

  /**
   * Core arguments to pass to the ComfyUI server to ensure electron app
   * works as expected.
   */
  get coreLaunchArgs() {
    return {
      'user-directory': this.userDirectoryPath,
      'input-directory': this.inputDirectoryPath,
      'output-directory': this.outputDirectoryPath,
      'front-end-root': this.webRootPath,
      'base-directory': this.basePath,
      'database-url': this.databaseUrl,
      'extra-model-paths-config': ComfyServerConfig.configPath,
      'log-stdout': '',
    };
  }

  /**
   * Builds CLI arguments from an object of key-value pairs.
   * @param args Object key-value pairs of CLI arguments.
   * @returns A string array of CLI arguments.
   */
  static buildLaunchArgs(args: Record<string, string>) {
    // Empty string values are ignored. e.g. { cpu: '' } => '--cpu'
    return Object.entries(args)
      .flatMap(([key, value]) => [`--${key}`, value])
      .filter((value) => value !== '');
  }

  get launchArgs() {
    const args = ComfyServer.buildLaunchArgs({
      ...this.coreLaunchArgs,
      ...this.serverArgs,
    });
    return [this.mainScriptPath, ...args];
  }

  /**
   * Attempts to parse the type of the last error.
   * @returns The last error type, if it can be parsed.
   */
  parseLastError(): ServerStartError | undefined {
    return this.lastStdErr?.match(/(^|\n)ModuleNotFoundError: /) ? 'ModuleNotFoundError' : undefined;
  }

  @trackEvent('comfyui:server_start')
  async start() {
    if (this.isRunning) {
      const message = 'ComfyUI server is already running';
      log.error(message);
      throw new Error(message);
    }

    ComfySettings.lockWrites();
    // Run config and log rotation in parallel (neither blocks server start)
    await Promise.all([
      ComfyServerConfig.addAppBundledCustomNodesToConfig(),
      rotateLogFiles(app.getPath('logs'), LogFile.ComfyUI, 50),
    ]);
    return new Promise<void>((resolve, reject) => {
      const comfyUILog = log.create({ logId: 'comfyui' });
      comfyUILog.transports.file.fileName = LogFile.ComfyUI;

      comfyUILog.transports.file.transforms.unshift(removeAnsiCodesTransform);

      this.timedOutWhilstStarting = false;

      // Throttle IPC log messages to ~60fps batches
      const logThrottle = new LogThrottle((msg) => {
        this.appWindow.send(IPC_CHANNELS.LOG_MESSAGE, msg);
      });

      let comfyServerProcess: ChildProcess;

      if (this.useTurboEngine) {
        // Turbo Engine: spawn comfy-server.exe directly (no Python needed)
        log.info(`Using Turbo Engine: ${this.turboEnginePath}`);
        const venvPath = path.join(this.basePath, '.venv');
        comfyServerProcess = spawn(this.turboEnginePath, [], {
          cwd: this.basePath,
          env: {
            ...process.env,
            COMFY_PORT: this.serverArgs.port,
            COMFY_FRONTEND: this.webRootPath,
            COMFY_VENV: venvPath,
          },
        });
        log.info(`Frontend root: ${this.webRootPath}`);
        log.info(`Venv path: ${venvPath}`);
        comfyServerProcess.stdout?.on('data', (data: Buffer) => {
          comfyUILog.info(data.toString());
          logThrottle.push(data.toString());
        });
        comfyServerProcess.stderr?.on('data', (data: Buffer) => {
          comfyUILog.error(data.toString());
          this.lastStdErr = data.toString();
          logThrottle.push(data.toString());
        });
      } else {
        // Python mode: use virtual environment
        comfyServerProcess = this.virtualEnvironment.runPythonCommand(this.launchArgs, {
          onStdout: (data) => {
            comfyUILog.info(data);
            logThrottle.push(data);
          },
          onStderr: (data) => {
            comfyUILog.error(data);
            this.lastStdErr = data;
            logThrottle.push(data);
          },
        });
      }

      const rejectOnError = (err: Error) => {
        this.comfyServerProcess = null;
        log.error('Failed to start ComfyUI:', err);
        reject(err);
      };
      comfyServerProcess.on('error', rejectOnError);

      comfyServerProcess.on('exit', (code, signal) => {
        this.comfyServerProcess = null;
        if (code !== 0) {
          log.error(`Python process exited with code ${code} and signal ${signal}`);
          reject(new Error(`Python process exited with code ${code} and signal ${signal}`));
        } else {
          log.info(`Python process exited successfully`);
          resolve();
        }
      });

      this.comfyServerProcess = comfyServerProcess;

      waitOn({
        resources: [`${this.baseUrl}/queue`],
        timeout: ComfyServer.MAX_FAIL_WAIT,
        interval: this.useTurboEngine ? 100 : ComfyServer.CHECK_INTERVAL, // Turbo: 100ms, Python: 1s
      })
        .then(() => {
          log.info(this.useTurboEngine ? 'Turbo Engine is ready' : 'Python server is ready');
          comfyServerProcess.off('error', rejectOnError);
          resolve();
        })
        .catch((error) => {
          this.timedOutWhilstStarting = true;
          log.error('Server failed to start within timeout:', error);
          reject(new Error('Python server failed to start within timeout.'));
        });
    });
  }

  async kill() {
    return new Promise<void>((resolve, reject) => {
      if (!this.comfyServerProcess) {
        log.info('No server process to kill');
        resolve();
        return;
      }

      const pid = this.comfyServerProcess.pid;
      log.info(`Killing server process (PID: ${pid}, turbo: ${this.useTurboEngine})`);

      const timeout = setTimeout(() => {
        // Force kill via taskkill as last resort
        if (pid && process.platform === 'win32') {
          log.warn(`Process ${pid} did not exit gracefully, force killing with taskkill`);
          try {
            require('node:child_process').execSync(`taskkill /F /T /PID ${pid}`, { stdio: 'ignore' });
          } catch {
            // Process may already be dead
          }
        }
        this.comfyServerProcess = null;
        resolve();
      }, 5_000);

      this.comfyServerProcess.once('exit', () => {
        clearTimeout(timeout);
        this.comfyServerProcess = null;
        log.info('Server process exited');
        resolve();
      });

      // On Windows, use taskkill /T to kill the entire process tree
      if (pid && process.platform === 'win32') {
        try {
          require('node:child_process').execSync(`taskkill /F /T /PID ${pid}`, { stdio: 'ignore' });
        } catch {
          // Fallback to Node.js kill
          this.comfyServerProcess.kill('SIGKILL');
        }
      } else {
        this.comfyServerProcess.kill();
      }
    });
  }
}
