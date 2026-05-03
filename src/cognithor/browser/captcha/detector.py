"""CAPTCHA detector — scans a page for CAPTCHA elements via JavaScript injection."""

from __future__ import annotations

from typing import Any

from cognithor.browser.captcha.models import CaptchaChallenge, CaptchaType
from cognithor.utils.logging import get_logger

__all__ = ["DETECT_JS", "detect_captcha"]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# JavaScript injected into the page to locate CAPTCHA widgets.
# Returns a list of {type, selector, sitekey} objects.
# ---------------------------------------------------------------------------
DETECT_JS = """
(function() {
    var results = [];

    function sitekey(el) {
        return (
            el.getAttribute('data-sitekey') ||
            el.getAttribute('data-pkey') ||
            ''
        );
    }

    // reCAPTCHA v2 checkbox
    var rcv2 = document.querySelectorAll('.g-recaptcha, [data-sitekey]');
    rcv2.forEach(function(el) {
        // Distinguish v3 by render=explicit on script or grecaptcha.execute usage —
        // fall back to v2 if we cannot determine otherwise.
        var sk = sitekey(el);
        results.push({type: 'recaptcha_v2', selector: '.g-recaptcha', sitekey: sk});
    });

    // hCaptcha
    var hc = document.querySelectorAll('.h-captcha');
    hc.forEach(function(el) {
        results.push({type: 'hcaptcha', selector: '.h-captcha', sitekey: sitekey(el)});
    });

    // Cloudflare Turnstile
    var ts = document.querySelectorAll('.cf-turnstile');
    ts.forEach(function(el) {
        results.push({type: 'turnstile', selector: '.cf-turnstile', sitekey: sitekey(el)});
    });

    // FunCaptcha / Arkose Labs
    var fc = document.querySelectorAll('#FunCaptcha, [data-pkey], iframe[src*="arkoselabs"]');
    fc.forEach(function(el) {
        var sel = el.id === 'FunCaptcha' ? '#FunCaptcha' :
                  el.hasAttribute('data-pkey') ? '[data-pkey]' :
                  'iframe[src*="arkoselabs"]';
        results.push({type: 'funcaptcha', selector: sel, sitekey: sitekey(el)});
    });

    // Text / image CAPTCHA (img with captcha in class/src/alt near a text input)
    var imgs = document.querySelectorAll('img');
    imgs.forEach(function(img) {
        var cls = (img.className || '').toLowerCase();
        var src = (img.getAttribute('src') || '').toLowerCase();
        var alt = (img.getAttribute('alt') || '').toLowerCase();
        if (cls.indexOf('captcha') !== -1 ||
            src.indexOf('captcha') !== -1 || alt.indexOf('captcha') !== -1) {
            // Check that there is a nearby text input
            var parent = img.parentElement;
            var hasInput = parent && (
                parent.querySelector('input[type="text"]') ||
                parent.querySelector('input:not([type])') ||
                parent.querySelector('input[type="number"]')
            );
            if (hasInput) {
                var cls_parts = img.className.trim().split(/\\s+/).join('.');
                var selector = img.className ? 'img.' + cls_parts : 'img';
                results.push({type: 'text', selector: selector, sitekey: ''});
            }
        }
    });

    // De-duplicate by selector
    var seen = {};
    var deduped = [];
    results.forEach(function(r) {
        var key = r.type + '|' + r.selector;
        if (!seen[key]) {
            seen[key] = true;
            deduped.push(r);
        }
    });

    return deduped;
})();
"""

# ---------------------------------------------------------------------------
# Mapping from JS type strings to CaptchaType enum values.
# ---------------------------------------------------------------------------
_TYPE_MAP: dict[str, CaptchaType] = {
    "recaptcha_v2": CaptchaType.RECAPTCHA_V2_CHECKBOX,
    "recaptcha_v3": CaptchaType.RECAPTCHA_V3,
    "hcaptcha": CaptchaType.HCAPTCHA,
    "turnstile": CaptchaType.TURNSTILE,
    "funcaptcha": CaptchaType.FUNCAPTCHA,
    "text": CaptchaType.TEXT,
}


async def detect_captcha(page) -> list[CaptchaChallenge]:
    """Scan *page* for CAPTCHA widgets and return a list of CaptchaChallenge objects.

    Parameters
    ----------
    page:
        A Playwright ``Page`` (or compatible mock with an ``evaluate`` coroutine
        and a ``url`` attribute).

    Returns
    -------
    list[CaptchaChallenge]
        Detected challenges, possibly empty.  Never raises — exceptions from
        JavaScript evaluation are logged and an empty list is returned.
    """
    try:
        raw: list[dict[str, Any]] = await page.evaluate(DETECT_JS)
    except Exception as exc:
        logger.debug("CAPTCHA detection failed on %s: %s", getattr(page, "url", "?"), exc)
        return []

    challenges: list[CaptchaChallenge] = []
    page_url: str = getattr(page, "url", "")

    for item in raw:
        js_type: str = item.get("type", "")
        captcha_type = _TYPE_MAP.get(js_type, CaptchaType.UNKNOWN)
        challenges.append(
            CaptchaChallenge(
                captcha_type=captcha_type,
                selector=item.get("selector", ""),
                sitekey=item.get("sitekey", ""),
                page_url=page_url,
            )
        )

    logger.debug("Detected %d CAPTCHA(s) on %s", len(challenges), page_url)
    return challenges
