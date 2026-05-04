/**
 * esbuild config for the Cognithor VS Code extension.
 *
 * Bundles src/extension.ts → dist/extension.js as a single CommonJS
 * file targeting Node 20 (the VS Code 1.85+ runtime). The vscode
 * module is marked external because the host environment provides
 * it at runtime.
 *
 * Pass --watch for the dev loop or --production for the
 * marketplace-bound build (minified, no sourcemap).
 */

import { build, context } from "esbuild";

const args = new Set(process.argv.slice(2));
const watch = args.has("--watch");
const production = args.has("--production");

const baseConfig = {
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "dist/extension.js",
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  target: "node20",
  sourcemap: !production,
  minify: production,
  logLevel: "info",
};

async function run() {
  if (watch) {
    const ctx = await context(baseConfig);
    await ctx.watch();
    console.log("[esbuild] watching for changes...");
    return;
  }
  await build(baseConfig);
  console.log("[esbuild] build complete");
}

run().catch((err) => {
  console.error("[esbuild] build failed:", err);
  process.exit(1);
});
