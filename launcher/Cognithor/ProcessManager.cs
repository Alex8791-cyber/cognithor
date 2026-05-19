using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace Cognithor;

sealed class ProcessManager : IDisposable
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string? lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr hJob, int jobInfoClass,
        ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, int cbInfoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr hObject);

    const int JobObjectExtendedLimitInformation = 9;
    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    readonly IntPtr _jobHandle;
    readonly string _baseDir;
    Process? _pythonProcess;
    Process? _ollamaProcess;
    Process? _desktopUiProcess;

    public ProcessManager()
    {
        _baseDir = AppContext.BaseDirectory;
        _jobHandle = CreateJobObject(IntPtr.Zero, "CognithorJobObject");
        if (_jobHandle == IntPtr.Zero)
            throw new Win32Exception(Marshal.GetLastWin32Error());

        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(_jobHandle, JobObjectExtendedLimitInformation,
                ref info, Marshal.SizeOf(info)))
            throw new Win32Exception(Marshal.GetLastWin32Error());
    }

    string? FindPython()
    {
        var local = Path.Combine(_baseDir, "python", "python.exe");
        if (File.Exists(local)) return local;
        return FindOnPath("python.exe");
    }

    /// <summary>
    /// True if an earlier <see cref="StartPython"/> call did not find the
    /// bundled interpreter. The tray UI uses this to show a "Backend not
    /// installed" status instead of silently appearing healthy.
    /// </summary>
    public bool PythonMissing { get; private set; }

    string? FindOllama()
    {
        var local = Path.Combine(_baseDir, "ollama", "ollama.exe");
        if (File.Exists(local)) return local;

        var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var system = Path.Combine(appData, "Programs", "Ollama", "ollama.exe");
        if (File.Exists(system)) return system;

        return FindOnPath("ollama.exe");
    }

    static string? FindOnPath(string exe)
    {
        var pathVar = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var dir in pathVar.Split(Path.PathSeparator))
        {
            var full = Path.Combine(dir.Trim(), exe);
            if (File.Exists(full)) return full;
        }
        return null;
    }

    Process SpawnAndAssign(string exe, string args, bool createWindow = false)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
            WorkingDirectory = _baseDir,
            UseShellExecute = false,
            CreateNoWindow = !createWindow,
        };
        var proc = Process.Start(psi)
            ?? throw new InvalidOperationException($"Failed to start {exe}");
        AssignProcessToJobObject(_jobHandle, proc.Handle);
        return proc;
    }

    public void StartAll()
    {
        StartOllama();
        StartPython();
    }

    void StartOllama()
    {
        var ollamaPath = FindOllama();
        if (ollamaPath == null) return;
        foreach (var p in Process.GetProcessesByName("ollama"))
            if (!p.HasExited) return;
        _ollamaProcess = SpawnAndAssign(ollamaPath, "serve");
    }

    void StartPython()
    {
        var pythonPath = FindPython();
        if (pythonPath is null)
        {
            // Don't kill the launcher — the tray icon stays up and the user
            // can read the status to understand what's wrong. The HealthChecker
            // will report Unhealthy because the API won't answer, which surfaces
            // in the tray tooltip.
            PythonMissing = true;
            return;
        }
        PythonMissing = false;
        _pythonProcess = SpawnAndAssign(pythonPath, "-m cognithor --no-cli --api-port 8741");
    }

    public bool IsPythonRunning => _pythonProcess is { HasExited: false };
    public bool IsDesktopUiRunning => _desktopUiProcess is { HasExited: false };

    public string? FindDesktopUi()
    {
        // Flutter desktop runner ships as jarvis_ui.exe (legacy name — see
        // installer/cognithor.iss [Files]/[Icons]), NOT cognithor_ui.exe.
        var local = Path.Combine(_baseDir, "flutter_app", "jarvis_ui.exe");
        if (File.Exists(local)) return local;
        return null;
    }

    public bool StartDesktopUi()
    {
        if (IsDesktopUiRunning)
        {
            BringDesktopToFront();
            return true;
        }

        var uiPath = FindDesktopUi();
        if (uiPath == null) return false;

        _desktopUiProcess = SpawnAndAssign(uiPath, "", createWindow: true);
        return true;
    }

    void BringDesktopToFront()
    {
        if (_desktopUiProcess is not { HasExited: false }) return;
        try
        {
            var hwnd = _desktopUiProcess.MainWindowHandle;
            if (hwnd != IntPtr.Zero)
            {
                SetForegroundWindow(hwnd);
                ShowWindow(hwnd, 9); // SW_RESTORE
            }
        }
        catch { }
    }

    [DllImport("user32.dll")]
    static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    public void RestartBackend()
    {
        if (_pythonProcess is { HasExited: false })
        {
            try { _pythonProcess.Kill(entireProcessTree: true); } catch { }
            _pythonProcess.WaitForExit(5000);
        }
        StartPython();
    }

    public void KillAll()
    {
        if (_desktopUiProcess is { HasExited: false })
            try { _desktopUiProcess.Kill(entireProcessTree: true); } catch { }
        if (_pythonProcess is { HasExited: false })
            try { _pythonProcess.Kill(entireProcessTree: true); } catch { }
        if (_ollamaProcess is { HasExited: false })
            try { _ollamaProcess.Kill(entireProcessTree: true); } catch { }
    }

    public void Dispose()
    {
        KillAll();
        if (_jobHandle != IntPtr.Zero)
            CloseHandle(_jobHandle);
    }
}
