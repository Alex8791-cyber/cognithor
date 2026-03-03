import { useState, useEffect, useCallback } from "react";
import { I } from "../utils/icons";

const THEME_KEY = "cc-theme";

const DARK_VARS = {
  "--accent": "#00d4ff",
  "--accent2": "#6c63ff",
  "--bg": "#08080f",
  "--bg2": "#10101a",
  "--bg3": "#181825",
  "--border": "#1e1e30",
  "--text": "#e0e0e8",
  "--text2": "#8888a0",
};

const LIGHT_VARS = {
  "--accent": "#0077cc",
  "--accent2": "#5b52d5",
  "--bg": "#f5f5f8",
  "--bg2": "#ffffff",
  "--bg3": "#eeeef2",
  "--border": "#d0d0dd",
  "--text": "#1a1a2e",
  "--text2": "#666688",
};

function applyTheme(theme) {
  const vars = theme === "light" ? LIGHT_VARS : DARK_VARS;
  const root = document.querySelector(".cc-root");
  if (root) {
    for (const [key, val] of Object.entries(vars)) {
      root.style.setProperty(key, val);
    }
  }
  // Update meta theme-color for browser chrome
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = vars["--bg2"];
}

export function useTheme() {
  const [theme, setThemeState] = useState(() => {
    try {
      return localStorage.getItem(THEME_KEY) || "dark";
    } catch {
      return "dark";
    }
  });

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setTheme = useCallback((t) => {
    setThemeState(t);
    try { localStorage.setItem(THEME_KEY, t); } catch {}
  }, []);

  const toggle = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  return { theme, setTheme, toggle };
}

export function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      className="cc-theme-toggle"
      onClick={onToggle}
      title={theme === "dark" ? "Helles Design" : "Dunkles Design"}
      aria-label={theme === "dark" ? "Zu hellem Design wechseln" : "Zu dunklem Design wechseln"}
      type="button"
    >
      {theme === "dark" ? I.sun : I.moon}
    </button>
  );
}
