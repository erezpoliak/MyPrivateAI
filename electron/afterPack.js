// electron-builder afterPack hook.
//
// Electron's own binary ships with an ad-hoc signature already baked in.
// electron-builder then copies our extraResources (python/, models/,
// app/backend/ — ~7.6GB) into Contents/Resources *after* that signature was
// sealed, so the seal no longer matches the bundle's actual contents. With
// `identity: null` (unsigned build) electron-builder skips its own signing
// step entirely, leaving that stale/mismatched signature in place — which
// macOS Gatekeeper reports as "'MyPrivateAI' is damaged and can't be
// opened," not the friendlier "unidentified developer" prompt, because the
// bundle genuinely doesn't match what's sealed.
//
// Fix: re-sign the whole bundle ad-hoc (`-s -`) and deep (nested
// binaries — the python interpreter, .so files, Electron helpers) after all
// resources are in place, so the seal covers what's actually shipped.
"use strict";

const { execFileSync } = require("child_process");
const path = require("path");

module.exports = async function afterPack(context) {
  if (context.electronPlatformName !== "darwin") return;

  const appPath = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  console.log(`[afterPack] ad-hoc deep-signing ${appPath} so the seal matches the final bundle contents…`);

  execFileSync("codesign", ["--force", "--deep", "--sign", "-", appPath], {
    stdio: "inherit",
  });

  console.log("[afterPack] done.");
};
