import { strict as assert } from "assert";
import * as vscode from "vscode";

suite("integration · smoke", () => {
  test("vscode API is reachable", () => {
    assert.ok(vscode.version, "vscode.version should be populated inside the host");
  });

  test("Cognithor extension is present", () => {
    const ext = vscode.extensions.getExtension("cognithor.cognithor-vscode");
    assert.ok(ext, "extension should be discoverable in the test host");
  });
});
