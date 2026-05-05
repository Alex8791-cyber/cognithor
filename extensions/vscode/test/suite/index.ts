/**
 * Mocha suite loader — invoked inside the VS Code extension host
 * by `runTest.ts`. Discovers `*.test.js` files under `out/test/suite/`
 * and runs them through mocha against the live `vscode` API.
 *
 * Unit tests (vscode mocked, no Electron) live under `test/unit/`
 * instead and are run by `npm run test:unit`.
 */

import * as path from "path";
import Mocha from "mocha";
import { glob } from "glob";

export async function run(): Promise<void> {
  const mocha = new Mocha({ ui: "bdd", color: true, timeout: 20_000 });
  const testsRoot = path.resolve(__dirname, "..");

  const files = await glob("**/*.test.js", { cwd: testsRoot });
  for (const f of files) {
    mocha.addFile(path.resolve(testsRoot, f));
  }

  await new Promise<void>((resolve, reject) => {
    try {
      mocha.run((failures: number) => {
        if (failures > 0) {
          reject(new Error(`${failures} tests failed.`));
        } else {
          resolve();
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}
