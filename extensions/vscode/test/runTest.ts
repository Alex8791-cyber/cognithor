/**
 * Integration test entry point — boots a real VS Code instance via
 * `@vscode/test-electron` and runs the mocha suite at
 * `test/suite/index.ts` against the live API.
 *
 * The unit suite under `test/unit/` does NOT need this — it runs in
 * plain Node via `npm run test:unit`. This entry point is for tests
 * that genuinely require the VS Code extension host (commands, tree
 * views, real activation lifecycle).
 */

import * as path from "path";
import { runTests } from "@vscode/test-electron";

async function main(): Promise<void> {
  try {
    const extensionDevelopmentPath = path.resolve(__dirname, "..", "..");
    const extensionTestsPath = path.resolve(__dirname, "suite", "index.js");
    await runTests({ extensionDevelopmentPath, extensionTestsPath });
  } catch (err) {
    console.error("Failed to run tests:", err);
    process.exit(1);
  }
}

void main();
