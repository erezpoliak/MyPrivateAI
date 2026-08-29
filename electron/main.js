// Electron main process. Its only job is process management: spawn the
// bundled Python/FastAPI backend as a sidecar, show a splash screen while
// models load, then point a normal window at it. All real app logic lives
// in the FastAPI + React app the backend serves — this file has no UI logic
// of its own beyond the splash.
"use strict";

const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const net = require("net");
const http = require("http");
const { spawn } = require("child_process");

// Force a consistent app name (and therefore userData path) whether this is
// run via `electron .` in dev or as the packaged app — npm package names
// must be lowercase, but electron-builder's productName is "MyPrivateAI",
// so without this dev and packaged runs would use different data dirs.
app.setName("MyPrivateAI");

let splashWindow = null;
let mainWindow = null;
let backendProcess = null;
let backendPort = null;
let backendReady = false;

// ---------------------------------------------------------------------------
// Paths — dev runs straight out of the repo (using the build/ artifacts
// scripts/build-python.sh and scripts/stage-models.sh already produced),
// packaged runs out of Contents/Resources. Same shape either way so this is
// also how you smoke-test the full backend lifecycle before cutting a DMG.
// ---------------------------------------------------------------------------
function getPaths() {
  const repoRoot = path.join(__dirname, "..");
  if (app.isPackaged) {
    const res = process.resourcesPath;
    return {
      pythonBin: path.join(res, "python", "bin", "python3.13"),
      backendCwd: res, // Resources/app/{__init__.py,backend/} — importable as `app.backend`
      hfHome: path.join(res, "models", "hf"),
      nltkData: path.join(res, "models", "nltk"),
      resourceDir: path.join(res, "models"),
      frontendDir: path.join(res, "frontend"),
    };
  }
  return {
    pythonBin: path.join(repoRoot, "build", "python", "bin", "python3.13"),
    backendCwd: repoRoot, // repo root already has app/backend importable as `app.backend`
    hfHome: path.join(repoRoot, "build", "models", "hf"),
    nltkData: path.join(repoRoot, "build", "models", "nltk"),
    resourceDir: path.join(repoRoot, "build", "models"),
    frontendDir: path.join(repoRoot, "app", "frontend", "dist"),
  };
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function waitForHealth(port, { intervalMs = 500, timeoutMs = 120000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(
        { host: "127.0.0.1", port, path: "/health", timeout: 2000 },
        (res) => {
          res.resume();
          if (res.statusCode === 200) resolve();
          else retry();
        }
      );
      req.on("error", retry);
      req.on("timeout", () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() > deadline) {
        reject(new Error("Backend did not become healthy in time."));
        return;
      }
      setTimeout(tick, intervalMs);
    };
    tick();
  });
}

// ---------------------------------------------------------------------------
// Splash window
// ---------------------------------------------------------------------------
function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 460,
    height: 320,
    resizable: false,
    frame: false,
    show: true,
    backgroundColor: "#111214",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  splashWindow.loadFile(path.join(__dirname, "splash.html"));
}

function sendStatus(text) {
  splashWindow?.webContents.send("splash:status", text);
}
function sendLog(line) {
  splashWindow?.webContents.send("splash:log", line);
}
function sendFatal(message) {
  splashWindow?.webContents.send("splash:fatal", message);
}

// ---------------------------------------------------------------------------
// Backend process
// ---------------------------------------------------------------------------
function pipeToSplashLog(stream) {
  stream.setEncoding("utf8");
  let buf = "";
  stream.on("data", (chunk) => {
    buf += chunk;
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 1);
      if (line.trim()) sendLog(line);
    }
  });
}

async function startBackend() {
  const paths = getPaths();
  backendPort = await getFreePort();

  const env = {
    ...process.env,
    HOST: "127.0.0.1",
    PORT: String(backendPort),
    HF_HOME: paths.hfHome,
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    NLTK_DATA: paths.nltkData,
    MYPRIVATEAI_DATA_DIR: app.getPath("userData"),
    MYPRIVATEAI_RESOURCE_DIR: paths.resourceDir,
    MYPRIVATEAI_FRONTEND_DIR: paths.frontendDir,
  };

  sendStatus("Loading models…");

  backendProcess = spawn(paths.pythonBin, ["-m", "app.backend.server"], {
    cwd: paths.backendCwd,
    env,
  });

  let exitedEarly = false;
  backendProcess.on("exit", (code, signal) => {
    if (!backendReady) {
      exitedEarly = true;
      sendFatal(`Backend exited before starting (code ${code}, signal ${signal}). See log below.`);
    }
    backendProcess = null;
  });

  pipeToSplashLog(backendProcess.stdout);
  pipeToSplashLog(backendProcess.stderr);

  try {
    await waitForHealth(backendPort);
  } catch (err) {
    if (!exitedEarly) sendFatal(err.message);
    throw err;
  }
}

function killBackend() {
  if (!backendProcess) return;
  const proc = backendProcess;
  backendProcess = null;
  proc.kill("SIGTERM");
  const killTimer = setTimeout(() => {
    try {
      proc.kill("SIGKILL");
    } catch {
      /* already exited */
    }
  }, 5000);
  proc.once("exit", () => clearTimeout(killTimer));
}

// ---------------------------------------------------------------------------
// Main window
// ---------------------------------------------------------------------------
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    show: false,
    backgroundColor: "#111214",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.once("ready-to-show", () => {
    splashWindow?.close();
    splashWindow = null;
    mainWindow.show();
  });
  mainWindow.loadURL(`http://127.0.0.1:${backendPort}/`);
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------
app.whenReady().then(async () => {
  createSplashWindow();
  try {
    await startBackend();
    backendReady = true;
    createMainWindow();
  } catch (err) {
    // Splash is already showing the fatal message via sendFatal(); leaving
    // it open (with its Quit button) instead of crashing the whole app.
    console.error("Backend failed to start:", err);
  }
});

ipcMain.on("splash:quit", () => app.quit());

// The backend holds a multi-GB model in memory and must not be left running
// as an orphan, so — unlike the usual mac convention of staying alive in the
// dock — this app fully quits whenever its one window closes.
app.on("window-all-closed", () => {
  killBackend();
  app.quit();
});

app.on("before-quit", () => {
  killBackend();
});
