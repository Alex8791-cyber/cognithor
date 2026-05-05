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

class ThemeIcon {
  constructor(public readonly id: string) {}
}

const treeItemCollapsibleState = { None: 0, Collapsed: 1, Expanded: 2 };
const viewColumn = { Active: -1, Beside: -2, One: 1, Two: 2, Three: 3 };

class TreeItem {
  public tooltip: unknown;
  public contextValue: unknown;
  public iconPath: unknown;
  public command: unknown;
  constructor(
    public readonly label: string,
    public readonly collapsibleState: number = treeItemCollapsibleState.None,
  ) {}
}

type EventEmitterListener<T> = (e: T) => void;

class EventEmitter<T> {
  public listeners: EventEmitterListener<T>[] = [];
  public disposed = false;
  readonly event = (listener: EventEmitterListener<T>): { dispose: () => void } => {
    this.listeners.push(listener);
    return {
      dispose: () => {
        const idx = this.listeners.indexOf(listener);
        if (idx !== -1) this.listeners.splice(idx, 1);
      },
    };
  };
  fire(payload: T): void {
    for (const l of [...this.listeners]) l(payload);
  }
  dispose(): void {
    this.disposed = true;
    this.listeners = [];
  }
}

class FakeWebview {
  public html = "";
}

class FakeWebviewPanel {
  public webview = new FakeWebview();
  public title: string;
  public visible = true;
  public disposed = false;
  public revealCalls: { column: number; preserveFocus: boolean }[] = [];
  private disposeCallbacks: Array<() => void> = [];
  constructor(
    public readonly viewType: string,
    title: string,
    public readonly viewColumn: number,
    public readonly options: Record<string, unknown>,
  ) {
    this.title = title;
  }
  reveal(column?: number, preserveFocus?: boolean): void {
    this.revealCalls.push({ column: column ?? -1, preserveFocus: preserveFocus ?? false });
    this.visible = true;
  }
  onDidDispose(cb: () => void): { dispose: () => void } {
    this.disposeCallbacks.push(cb);
    return { dispose: () => undefined };
  }
  dispose(): void {
    this.disposed = true;
    this.visible = false;
    for (const cb of this.disposeCallbacks) cb();
  }
}

class FakeMemento {
  public store: Record<string, unknown> = {};
  get<T>(key: string, defaultValue?: T): T | undefined {
    if (key in this.store) return this.store[key] as T;
    return defaultValue;
  }
  update(key: string, value: unknown): Promise<void> {
    if (value === undefined) {
      delete this.store[key];
    } else {
      this.store[key] = value;
    }
    return Promise.resolve();
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
  createWebviewPanel: sinon.stub().callsFake(
    (
      viewType: string,
      title: string,
      column: number,
      options: Record<string, unknown> = {},
    ) => new FakeWebviewPanel(viewType, title, column, options),
  ),
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
  ThemeIcon,
  TreeItem,
  TreeItemCollapsibleState: treeItemCollapsibleState,
  ViewColumn: viewColumn,
  EventEmitter,
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
  (window.createWebviewPanel as sinon.SinonStub).resetHistory();
  (window.showWarningMessage as sinon.SinonStub).resetHistory();
  (window.showErrorMessage as sinon.SinonStub).resetHistory();
}

export { FakeMemento, FakeWebviewPanel };

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
