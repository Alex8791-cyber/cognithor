import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

import { defaultAxeOptions, screensToAudit } from "./axe.config";

/**
 * a11y audit suite — Sprint 3.3.
 *
 * Per-PR gate: any new violation in `serious` or `critical` impact
 * fails the build. `moderate` is reported but not blocking;
 * `minor` is logged.
 *
 * Run::
 *
 *     npx playwright test integration_test/a11y/
 */

for (const screen of screensToAudit) {
  test.describe(`a11y — ${screen.name}`, () => {
    test(`WCAG 2.1 AA — ${screen.name}`, async ({ page }) => {
      await page.goto(screen.url);
      await page.waitForLoadState("networkidle");

      const builder = new AxeBuilder({ page });
      const opts = { ...defaultAxeOptions, ...(screen.overrides ?? {}) };
      if (opts.runOnly) builder.withTags(
        Array.isArray(opts.runOnly) ? opts.runOnly : (opts.runOnly as { values: string[] }).values,
      );

      const results = await builder.analyze();

      const blocking = results.violations.filter((v) =>
        ["serious", "critical"].includes(v.impact ?? ""),
      );
      const reported = results.violations.filter((v) => v.impact === "moderate");

      // eslint-disable-next-line no-console
      if (reported.length) console.warn(
        `[${screen.name}] ${reported.length} moderate-impact a11y issue(s):\n` +
          reported.map((v) => `  - ${v.id}: ${v.help}`).join("\n"),
      );

      expect(
        blocking,
        `${blocking.length} serious/critical a11y violations on ${screen.name}:\n` +
          blocking
            .map((v) => `  - ${v.id} (${v.impact}): ${v.help}\n      ${v.helpUrl}`)
            .join("\n"),
      ).toEqual([]);
    });

    test(`Contrast — ${screen.name} dark theme`, async ({ page }) => {
      // Force dark theme via app-specific URL parameter (matches Flutter
      // theme provider behaviour — see ThemeProvider.toggle).
      await page.goto(`${screen.url}?theme=dark`);
      await page.waitForLoadState("networkidle");

      const results = await new AxeBuilder({ page })
        .withRules(["color-contrast"])
        .analyze();
      const blocking = results.violations.filter((v) =>
        ["serious", "critical"].includes(v.impact ?? ""),
      );
      expect(blocking, `Dark-theme contrast issues on ${screen.name}`).toEqual([]);
    });
  });
}

test.describe("a11y — RTL layout (AR locale)", () => {
  test("Chat renders right-to-left when locale=ar", async ({ page }) => {
    await page.goto("/?locale=ar");
    await page.waitForLoadState("networkidle");

    const dir = await page.evaluate(() => document.documentElement.dir);
    expect(dir).toBe("rtl");

    // The send button should now be on the LEFT in RTL layout
    const sendButton = page.getByRole("button", { name: /send|إرسال/i });
    if (await sendButton.count()) {
      const box = await sendButton.boundingBox();
      const viewport = page.viewportSize();
      expect(box, "send button has bounding box").toBeTruthy();
      expect(viewport, "viewport size known").toBeTruthy();
      expect(box!.x).toBeLessThan(viewport!.width / 2);
    }
  });
});
