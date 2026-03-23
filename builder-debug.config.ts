import { Configuration } from 'electron-builder';

const debugConfig: Configuration = {
  files: ['node_modules', 'package.json', '.vite/**'],
  extraResources: [
    { from: './assets/ComfyUI', to: 'ComfyUI' },
    { from: './assets/uv', to: 'uv' },
    { from: './assets/UI', to: 'UI' },
    { from: './assets/desktop-ui', to: 'desktop-ui' },
    { from: './assets/comfy-server.exe', to: 'comfy-server.exe' },
    { from: './assets/julia', to: 'julia' },
  ],
  beforeBuild: './scripts/preMake.js',
  win: {
    icon: './assets/UI/Comfy_Logo.ico',
    target: ['nsis', 'zip'],
    signtoolOptions: null,
  },
  nsis: {
    oneClick: false,
    perMachine: false,
    allowToChangeInstallationDirectory: true,
    installerIcon: './assets/UI/Comfy_Logo.ico',
    uninstallerIcon: './assets/UI/Comfy_Logo.ico',
    installerHeaderIcon: './assets/UI/Comfy_Logo.ico',
    artifactName: 'ComfyUI-Turbo-${version}-setup.${ext}',
  },
  mac: {
    icon: './assets/UI/Comfy_Logo.icns',
    target: 'zip',
    identity: null,
  },
  linux: {
    icon: './assets/UI/Comfy_Logo_x256.png',
    target: 'appimage',
  },
  asarUnpack: ['**/node_modules/node-pty/**/*'],
};

export default debugConfig;
