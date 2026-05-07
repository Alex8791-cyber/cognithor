import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./",
  testMatch: /.*\.journey\.ts/,
  fullyParallel: false, // single ephemeral Cognithor instance per worker
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"], ["html", { outputFolder: "playwright-report" }]],
  use: {
    baseURL: process.env.COGNITHOR_BASE_URL ?? "http://localhost:8741",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  timeout: 60_000,
  expect: { timeout: 10_000 },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    // Add Firefox/WebKit when smoke is stable.
  ],
  webServer: process.env.SKIP_WEBSERVER
    ? undefined
    : {
        // Caller is expected to launch Cognithor + Flutter Web before
        // running the journeys (or set SKIP_WEBSERVER=1 to point at
        // a running instance). Playwright's webServer config can be
        // wired here once the Flutter Web build pipeline is parameterised.
        command: "echo 'webServer disabled — start Cognithor manually'",
        url: process.env.COGNITHOR_BASE_URL ?? "http://localhost:8741/health",
        reuseExistingServer: true,
        timeout: 30_000,
      },
});
