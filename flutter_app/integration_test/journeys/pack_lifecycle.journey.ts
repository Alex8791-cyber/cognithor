import { test, expect } from "@playwright/test";

/**
 * Pack lifecycle — Sprint 3.2.
 *
 * Install → use → rollback. The pack rollback feature (TRUST-4) is
 * the highest-stakes operator action in normal usage; this journey
 * makes sure it's reachable without a console.
 */

test.describe("Pack lifecycle", () => {
  test("operator can reach pack-management screen", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /settings|einstellungen/i }).click();
    const packs = page.getByRole("link", { name: /packs?/i });
    await expect(packs).toBeVisible({ timeout: 10_000 });
    await packs.click();

    // Pack list page
    await expect(page.getByRole("heading", { name: /packs/i })).toBeVisible();
  });

  test("rollback button shows confirmation dialog", async ({ page }) => {
    // Pre-condition: at least one installed pack with a previous version.
    // In a hermetic test environment we'd seed this via a fixture HTTP call.
    test.skip(
      !process.env.HAS_INSTALLED_PACK,
      "Set HAS_INSTALLED_PACK=1 to run with a fixture-installed pack",
    );

    await page.goto("/settings/packs");
    const firstPack = page
      .getByRole("listitem")
      .filter({ hasText: /installed/i })
      .first();
    await firstPack.getByRole("button", { name: /rollback/i }).click();

    const dialog = page.getByRole("dialog", { name: /rollback/i });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/cannot be undone/i)).toBeVisible();
  });
});
