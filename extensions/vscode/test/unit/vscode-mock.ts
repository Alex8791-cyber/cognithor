/**
 * Minimal in-process mock of the `vscode` module so unit tests can
 * exercise extension logic without booting the Electron host.
 *
 * Only the surface used by `cost_gutter.ts` and `decision_hover.ts`
 * is implemented. New unit suites that touch fresh API should extend
 * this file rather than reach into VS Code's runtime.
 *
 * Wired up through mocha's `--require` so the mock is registered in
 * Node's module cache before any source under test imports `vscode`.
 */

import * as Module from "module";
import * as path from "path";
import * as sinon from "sinon";

class Position {
  constructor(public readonly line: number, public readonly character: number) {}
}

class Range {
  constructor(public readonly start: Position, public readonly end: Position) {}
}

class MarkdownString {
  public value = "";
  public isTrusted = false;
  public supportThemeIcons = false;
  appendMarkdown(s: string): MarkdownString {
    this.value += s;
    return this;
  }
}

class Hover {
  constructor(public readonly contents: unknown) {}
}

class TextEditorDecorationType {
  public disposed = false;
  constructor(public readonly options: Record<string, unknown> = {}) {}
  dispose(): void {
    this.disposed = true;
  }
}

class StatusBarItem {
  public text = "";
  public tooltip: unknown = "";
  public visible = false;
  public disposed = false;
  show(): void {
    this.visible = true;
  }
  hide(): void {
    this.visible = false;
  }
  dispose(): void {
    this.disposed = true;
  }
}

class OutputChannel {
  public lines: string[] = [];
  appendLine(s: string): void {
    this.lines.push(s);
  }
  show(): void {
    /* no-op */
  }
  dispose(): void {
    /* no-op */
  }
}

interface ConfigStore {
  [key: string]: unknown;
}

let configStore: ConfigStore = {};

class WorkspaceConfiguration {
  constructor(private readonly section: string) {}
  get<T>(key: string, defaultValue: T): T {
    const full = `${this.section}.${key}`;
    if (full in configStore) {
      return configStore[full] as T;
    }
    return defaultValue;
  }
}

const overviewRulerLane = { Left: 1, Center: 2, Right: 4, Full: 7 };
const statusBarAlignment = { Left: 1, Right: 2 };

const window = {
  activeTextEditor: undefined as unknown,
  createTextEditorDecorationType: sinon.stub().callsFake(
    (opts: Record<string, unknown>) => new TextEditorDecorationType(opts),
  ),
  createStatusBarItem: sinon
    .stub()
    .callsFake((_alignment: number, _priority: number) => new StatusBarItem()),
  createOutputChannel: sinon.stub().callsFake((_name: string) => new OutputChannel()),
  showWarningMessage: sinon.stub().resolves(undefined),
  showErrorMessage: sinon.stub().resolves(undefined),
  showInformationMessage: sinon.stub().resolves(undefined),
  onDidChangeActiveTextEditor: sinon.stub().returns({ dispose: () => undefined }),
};

const workspace = {
  getConfiguration: (section: string) => new WorkspaceConfiguration(section),
  onDidChangeTextDocument: sinon.stub().returns({ dispose: () => undefined }),
  onDidChangeConfiguration: sinon.stub().returns({ dispose: () => undefined }),
};

const languages = {
  registerHoverProvider: sinon.stub().returns({ dispose: () => undefined }),
};

const vscodeMock = {
  Position,
  Range,
  Hover,
  MarkdownString,
  OverviewRulerLane: overviewRulerLane,
  StatusBarAlignment: statusBarAlignment,
  window,
  workspace,
  languages,
};

export function setConfig(values: ConfigStore): void {
  configStore = { ...configStore, ...values };
}

export function resetConfig(): void {
  configStore = {};
}

export function getMock(): typeof vscodeMock {
  return vscodeMock;
}

export function resetMockSpies(): void {
  (window.createTextEditorDecorationType as sinon.SinonStub).resetHistory();
  (window.createStatusBarItem as sinon.SinonStub).resetHistory();
  (workspace.onDidChangeTextDocument as sinon.SinonStub).resetHistory();
  (workspace.onDidChangeConfiguration as sinon.SinonStub).resetHistory();
  (window.onDidChangeActiveTextEditor as sinon.SinonStub).resetHistory();
  (languages.registerHoverProvider as sinon.SinonStub).resetHistory();
}

// Node 20 made `Module._resolveFilename` non-writable in some
// contexts, so instead of swapping the resolver we override the
// instance-level `require` hook on `Module.prototype` and short-
// circuit the `vscode` request before it ever hits the resolver.
//
// Side-effect on import: any subsequent `require("vscode")` from
// this Node process returns our in-memory mock.
interface PrototypeRequire {
  (this: NodeJS.Module, id: string): unknown;
}

const proto = (Module as unknown as { prototype: { require: PrototypeRequire } }).prototype;
const originalRequire = proto.require;
proto.require = function (this: NodeJS.Module, id: string): unknown {
  if (id === "vscode") {
    return vscodeMock;
  }
  return originalRequire.call(this, id);
};

// Keep the import path used so the tooling that warns about
// __dirname/path being unused doesn't fire.
void path;
