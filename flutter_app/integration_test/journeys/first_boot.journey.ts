import { test, expect } from "@playwright/test";

/**
 * First-boot journey — Sprint 3.2.
 *
 * Covers: install → first-boot wizard → backend handshake → chat-page reach.
 * Asserts: the user can reach the chat page within the documented
 * "5 minutes from download" promise (here: 30 s simulated).
 */

test.describe("First boot", () => {
  test("user reaches chat page within first-boot SLO", async ({ page }) => {
    await page.goto("/");
    // Splash / connection guard — must reach a non-error state quickly
    await expect(page).toHaveTitle(/Cognithor/i, { timeout: 15_000 });

    // First-boot wizard, if shown
    const wizard = page.getByRole("dialog", { name: /first.?boot/i });
    if (await wizard.count()) {
      await wizard.getByRole("button", { name: /weiter|next/i }).click();
    }

    // Chat page is the documented default — should be reachable
    const chatHeader = page.getByRole("heading", { name: /chat/i });
    await expect(chatHeader).toBeVisible({ timeout: 30_000 });
  });

  test("backend version-mismatch overlay does not appear in happy path", async ({
    page,
  }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    const overlay = page.getByText(/version mismatch/i);
    await expect(overlay).toHaveCount(0);
  });
});
