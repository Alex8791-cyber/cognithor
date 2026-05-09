using System;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;
using System.Windows.Forms;

namespace Cognithor;

sealed class AppShell : IDisposable
{
    readonly ProcessManager _processes;
    readonly HealthChecker _health;
    readonly LauncherSettings _settings;
    readonly NotifyIcon _trayIcon;
    readonly ToolStripMenuItem _statusItem;
    readonly ToolStripMenuItem _defaultToggle;
    bool _didAutoOpen;

    static readonly string Version = typeof(AppShell).Assembly
        .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
        ?.InformationalVersion ?? "dev";

    public AppShell(string[] args)
    {
        _processes = new ProcessManager();
        _health = new HealthChecker();
        _settings = LauncherSettings.Load();

        var hasDesktop = _processes.FindDesktopUi() != null;

        // --- Menu items ---
        var openBrowserItem = new ToolStripMenuItem("Open in Browser");
        openBrowserItem.Click += (_, _) => OpenBrowser();

        var openDesktopItem = new ToolStripMenuItem("Open Desktop App");
        openDesktopItem.Click += (_, _) => _processes.StartDesktopUi();
        openDesktopItem.Enabled = hasDesktop;

        _defaultToggle = new ToolStripMenuItem(DefaultToggleLabel);
        _defaultToggle.Click += (_, _) =>
        {
            _settings.TogglePreferredUi();
            _defaultToggle.Text = DefaultToggleLabel;
        };
        _defaultToggle.Enabled = hasDesktop;

        _statusItem = new ToolStripMenuItem("Status: Starting...") { Enabled = false };

        var restartItem = new ToolStripMenuItem("Restart Backend");
        restartItem.Click += (_, _) =>
        {
            SetStatus(AppStatus.Starting);
            _processes.RestartBackend();
        };

        var quitItem = new ToolStripMenuItem("Quit Cognithor");
        quitItem.Click += (_, _) => Shutdown();

        var versionItem = new ToolStripMenuItem($"v{Version}") { Enabled = false };

        var menu = new ContextMenuStrip();
        menu.Items.Add(openBrowserItem);
        menu.Items.Add(openDesktopItem);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(_defaultToggle);
        menu.Items.Add(_statusItem);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(restartItem);
        menu.Items.Add(quitItem);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(versionItem);

        // Bold the default action
        (hasDesktop && _settings.PreferDesktop ? openDesktopItem : openBrowserItem)
            .Font = new Font(openBrowserItem.Font, FontStyle.Bold);

        _trayIcon = new NotifyIcon
        {
            Icon = LoadIcon(),
            Text = "Cognithor - Starting...",
            ContextMenuStrip = menu,
            Visible = true,
        };
        _trayIcon.DoubleClick += (_, _) => OpenDefault();

        _health.StatusChanged += (prev, next) =>
        {
            SetStatus(next);
            // Auto-open the default UI the first time the backend
            // becomes healthy after launch — otherwise users land in
            // the system tray invisible-icon trap and assume "nothing
            // happened" when in fact the backend is humming along on
            // port 8741.
            if (next == AppStatus.Healthy && !_didAutoOpen)
            {
                _didAutoOpen = true;
                _trayIcon.ShowBalloonTip(4000, "Cognithor",
                    "Backend ready. Opening UI…",
                    ToolTipIcon.Info);
                OpenDefault();
            }
            if (next == AppStatus.Stopped && prev != AppStatus.Starting)
            {
                _trayIcon.ShowBalloonTip(5000, "Cognithor",
                    "Backend has stopped. Right-click to restart.",
                    ToolTipIcon.Error);
            }
        };

        _processes.StartAll();
        _health.Start();

        // Surface a missing backend immediately so the user isn't left
        // wondering why the tray icon stays on "Starting..." forever.
        if (_processes.PythonMissing)
        {
            SetStatus(AppStatus.Stopped);
            _statusItem.Text = "Status: Backend not installed";
            _trayIcon.Text = "Cognithor - Backend not installed";
            _trayIcon.ShowBalloonTip(
                8000,
                "Cognithor",
                "Python backend not found. Reinstall Cognithor or check your PATH.",
                ToolTipIcon.Error);
        }
    }

    string DefaultToggleLabel =>
        _settings.PreferDesktop ? "Default: Desktop App  \u2713" : "Default: Browser  \u2713";

    void OpenDefault()
    {
        if (_settings.PreferDesktop && _processes.FindDesktopUi() != null)
            _processes.StartDesktopUi();
        else
            OpenBrowser();
    }

    void SetStatus(AppStatus status)
    {
        var (text, label) = status switch
        {
            AppStatus.Starting  => ("Cognithor - Starting...",     "Status: Starting..."),
            AppStatus.Healthy   => ("Cognithor - Running",         "Status: Running"),
            AppStatus.Unhealthy => ("Cognithor - Backend error",   "Status: Error"),
            AppStatus.Stopped   => ("Cognithor - Backend stopped", "Status: Stopped"),
            _ => ("Cognithor", "Status: Unknown"),
        };
        _trayIcon.Text = text;
        _statusItem.Text = label;
    }

    static Icon LoadIcon()
    {
        var path = System.IO.Path.Combine(AppContext.BaseDirectory, "Resources", "app_icon.ico");
        if (System.IO.File.Exists(path))
            return new Icon(path);
        return SystemIcons.Application;
    }

    static void OpenBrowser()
    {
        Process.Start(new ProcessStartInfo
        {
            FileName = "http://localhost:8741",
            UseShellExecute = true,
        });
    }

    void Shutdown()
    {
        _health.Stop();
        _processes.KillAll();
        _trayIcon.Visible = false;
        Application.Exit();
    }

    public void Dispose()
    {
        _health.Dispose();
        _processes.Dispose();
        _trayIcon.Dispose();
    }
}
