const { contextBridge, ipcRenderer } = require("electron");

// Minimal bridge for the splash window only — the real app UI is plain
// FastAPI-served HTML/JS loaded from localhost and gets no bridge at all.
contextBridge.exposeInMainWorld("splash", {
  onStatus: (cb) => ipcRenderer.on("splash:status", (_e, text) => cb(text)),
  onLog: (cb) => ipcRenderer.on("splash:log", (_e, line) => cb(line)),
  onFatal: (cb) => ipcRenderer.on("splash:fatal", (_e, message) => cb(message)),
  quit: () => ipcRenderer.send("splash:quit"),
});
