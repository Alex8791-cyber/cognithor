import { test, expect } from "@playwright/test";
import path from "node:path";

/**
 * Chat-with-video journey — Sprint 3.2.
 *
 * Hits the VLM-router-driven path end-to-end: paperclip menu → upload
 * a small fixture clip → ask a one-line question → assert a non-empty
 * response. This is the highest-value journey because it exercises:
 *   - upload pipeline (media_api.py)
 *   - VLM routing (vlm_router + video.routing)
 *   - vLLM backend round-trip
 *   - chat WebSocket streaming
 *
 * Skips when VLM-router and a running vLLM aren't available.
 */

test.describe("Chat with video attachment", () => {
  test.skip(
    !process.env.VLM_AVAILABLE,
    "Set VLM_AVAILABLE=1 with a running vLLM for video tests",
  );

  test("uploads short clip + receives description", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("link", { name: /chat/i }).click();

    const fileChooserPromise = page.waitForEvent("filechooser");
    await page.getByRole("button", { name: /attach|paperclip/i }).click();
    const chooser = await fileChooserPromise;
    const fixturePath = path.resolve(
      __dirname,
      "../fixtures/short-clip.mp4",
    );
    await chooser.setFiles(fixturePath);

    // Wait for the upload-thumbnail to render
    await expect(page.getByTestId("video-attachment-thumb")).toBeVisible({
      timeout: 30_000,
    });

    // Send prompt
    const input = page.getByRole("textbox", { name: /message/i });
    await input.fill("Beschreibe den Clip in einem Satz.");
    await page.getByRole("button", { name: /send|senden/i }).click();

    // Assistant response within 60 s (premium tier may take longer)
    const response = page.getByTestId("assistant-message").last();
    await expect(response).toBeVisible({ timeout: 60_000 });
    await expect(response).not.toBeEmpty();
  });
});
