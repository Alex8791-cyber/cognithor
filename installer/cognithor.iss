; Cognithor Installer — Inno Setup Script
; Built by build_installer.py
;
; Inspired by Git for Windows installer architecture.
; Uses embedded Python (no system Python required).

#ifndef MyAppVersion
  #define MyAppVersion "0.80.1"
#endif

#ifndef BuildDir
  #define BuildDir SourcePath + "\build"
#endif

#ifndef ProjectRoot
  #define ProjectRoot SourcePath + "\.."
#endif

#ifndef PythonDir
  #define PythonDir BuildDir + "\python"
#endif

#ifndef OllamaDir
  #define OllamaDir BuildDir + "\ollama"
#endif

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-COGNITHOR0001}
AppName=Cognithor
AppVersion={#MyAppVersion}
AppVerName=Cognithor {#MyAppVersion}
AppPublisher=Alexander Soellner
AppPublisherURL=https://github.com/Alex8791-cyber/cognithor
AppSupportURL=https://github.com/Alex8791-cyber/cognithor/issues
DefaultDirName={localappdata}\Cognithor
DefaultGroupName=Cognithor
AllowNoIcons=yes
LicenseFile={#ProjectRoot}\LICENSE
OutputDir=dist
OutputBaseFilename=CognithorSetup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile={#ProjectRoot}\flutter_app\windows\runner\resources\app_icon.ico
UninstallDisplayIcon={app}\app_icon.ico
UninstallDisplayName=Cognithor {#MyAppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Types]
Name: "full"; Description: "Full installation (recommended)"
Name: "compact"; Description: "Cognithor only (no Ollama, no UI)"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Cognithor Core (Python + Dependencies)"; Types: full compact custom; Flags: fixed
Name: "ollama"; Description: "Ollama (Local LLM Runtime)"; Types: full custom
Name: "flutter"; Description: "Flutter Command Center (Web UI)"; Types: full custom
Name: "addpath"; Description: "Add cognithor to PATH"; Types: full custom

[Files]
; Core: Embedded Python + cognithor
Source: "{#PythonDir}\*"; DestDir: "{app}\python"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs

; Launcher
Source: "{#BuildDir}\cognithor.bat"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; App shell (tray launcher)
Source: "{#BuildDir}\Cognithor.exe"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; Ollama
Source: "{#OllamaDir}\*"; DestDir: "{app}\ollama"; Components: ollama; Flags: ignoreversion recursesubdirs createallsubdirs

; ffmpeg (LGPL) — video input support. step_ffmpeg flattens the BtbN archive
; to build\ffmpeg\bin\ so the paths below are literal (no directory wildcard).
Source: "build\ffmpeg\bin\ffmpeg.exe"; DestDir: "{localappdata}\Cognithor\ffmpeg"; Flags: ignoreversion
Source: "build\ffmpeg\bin\ffprobe.exe"; DestDir: "{localappdata}\Cognithor\ffmpeg"; Flags: ignoreversion
Source: "build\ffmpeg\bin\*.dll"; DestDir: "{localappdata}\Cognithor\ffmpeg"; Flags: ignoreversion skipifsourcedoesntexist

; Flutter UI — web build (served via backend)
Source: "{#BuildDir}\flutter_web\*"; DestDir: "{app}\flutter_app\web"; Components: flutter; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Flutter Desktop app (native Windows UI)
Source: "{#BuildDir}\flutter_desktop\*"; DestDir: "{app}\flutter_app"; Components: flutter; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; First-run setup script
Source: "{#ProjectRoot}\installer\first_run.py"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; Auto-upgrade script (syncs installed version with source tree)
Source: "{#ProjectRoot}\installer\auto_upgrade.py"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; Default agents config
Source: "{#ProjectRoot}\installer\agents.yaml.default"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; Config template
Source: "{#ProjectRoot}\config.yaml.example"; DestDir: "{app}"; DestName: "config.yaml"; Flags: onlyifdoesntexist

; App icon for shortcuts
Source: "{#ProjectRoot}\flutter_app\windows\runner\resources\app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; AppUserModelID lets Windows treat each shortcut as the canonical identity
; for the running process — required for proper taskbar pinning, jump-list
; grouping, and Start-menu/Search result deduplication.
Name: "{group}\Cognithor"; Filename: "{app}\Cognithor.exe"; IconFilename: "{app}\app_icon.ico"; Comment: "Start Cognithor"; AppUserModelID: "Cognithor.AgentOS"
Name: "{group}\Cognithor Desktop App"; Filename: "{app}\flutter_app\jarvis_ui.exe"; IconFilename: "{app}\app_icon.ico"; Comment: "Cognithor Command Center (native UI)"; AppUserModelID: "Cognithor.AgentOS.Desktop"; Components: flutter
Name: "{group}\Cognithor CLI"; Filename: "cmd.exe"; Parameters: "/k ""{app}\cognithor.bat"""; IconFilename: "{app}\app_icon.ico"; Comment: "Cognithor Command Line"; AppUserModelID: "Cognithor.AgentOS.CLI"
Name: "{group}\Uninstall Cognithor"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Cognithor"; Filename: "{app}\Cognithor.exe"; IconFilename: "{app}\app_icon.ico"; Comment: "Start Cognithor"; AppUserModelID: "Cognithor.AgentOS"

[Registry]
; Add to PATH if selected
Root: HKCU; Subkey: "Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Components: addpath; Check: NeedsAddPath('{app}')

; App Paths registration — lets Win+R "Cognithor" (or just typing "Cognithor" in the
; Start menu) launch the app directly without a fully-qualified path. Also makes the
; binary discoverable to ShellExecute callers across the system.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\Cognithor.exe"; \
    ValueType: string; ValueData: "{app}\Cognithor.exe"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\Cognithor.exe"; \
    ValueType: string; ValueName: "Path"; ValueData: "{app}"
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\Cognithor.exe"; \
    ValueType: string; ValueName: "FriendlyAppName"; ValueData: "Cognithor"

; Flutter desktop binary (still named jarvis_ui.exe for legacy reasons — surface a
; friendly name so search results show "Cognithor Desktop").
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\jarvis_ui.exe"; \
    ValueType: string; ValueData: "{app}\flutter_app\jarvis_ui.exe"; \
    Components: flutter; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\jarvis_ui.exe"; \
    ValueType: string; ValueName: "FriendlyAppName"; ValueData: "Cognithor Desktop"; \
    Components: flutter

[Run]
; Post-install: offer to start Cognithor
Filename: "{app}\Cognithor.exe"; Description: "Start Cognithor"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python\__pycache__"
Type: filesandordirs; Name: "{app}\python\Lib\site-packages\__pycache__"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\ollama"
Type: filesandordirs; Name: "{app}\flutter_app"
Type: files; Name: "{app}\cognithor.bat"
Type: files; Name: "{app}\Cognithor.exe"
Type: files; Name: "{app}\first_run.py"
Type: files; Name: "{app}\agents.yaml.default"
Type: files; Name: "{app}\config.yaml"

[Code]
// Compare version strings: returns -1, 0, or 1
function CompareVersions(V1, V2: string): Integer;
var
  P1, P2: Integer;
  N1, N2: Integer;
  S1, S2: string;
begin
  S1 := V1;
  S2 := V2;
  Result := 0;
  while (Length(S1) > 0) or (Length(S2) > 0) do
  begin
    // Extract next numeric part from S1
    P1 := Pos('.', S1);
    if P1 > 0 then begin
      N1 := StrToIntDef(Copy(S1, 1, P1 - 1), 0);
      S1 := Copy(S1, P1 + 1, Length(S1));
    end else begin
      N1 := StrToIntDef(S1, 0);
      S1 := '';
    end;
    // Extract next numeric part from S2
    P2 := Pos('.', S2);
    if P2 > 0 then begin
      N2 := StrToIntDef(Copy(S2, 1, P2 - 1), 0);
      S2 := Copy(S2, P2 + 1, Length(S2));
    end else begin
      N2 := StrToIntDef(S2, 0);
      S2 := '';
    end;
    if N1 < N2 then begin Result := -1; exit; end;
    if N1 > N2 then begin Result := 1; exit; end;
  end;
end;

// Block downgrades: check marker file for installed version
function InitializeSetup(): Boolean;
var
  MarkerPath: string;
  MarkerContent: AnsiString;
  InstalledVer: string;
  P1, P2: Integer;
begin
  Result := True;
  MarkerPath := ExpandConstant('{%USERPROFILE}\.cognithor\.cognithor_initialized');
  if FileExists(MarkerPath) then
  begin
    if LoadStringFromFile(MarkerPath, MarkerContent) then
    begin
      // Extract version from JSON: {"version": "X.Y.Z", ...}
      P1 := Pos('"version"', MarkerContent);
      if P1 > 0 then
      begin
        P2 := Pos(':', Copy(MarkerContent, P1, Length(MarkerContent)));
        if P2 > 0 then
        begin
          InstalledVer := Copy(MarkerContent, P1 + P2, Length(MarkerContent));
          // Strip to just the version string
          P1 := Pos('"', InstalledVer);
          if P1 > 0 then
          begin
            InstalledVer := Copy(InstalledVer, P1 + 1, Length(InstalledVer));
            P2 := Pos('"', InstalledVer);
            if P2 > 0 then
              InstalledVer := Copy(InstalledVer, 1, P2 - 1);
          end;

          if CompareVersions('{#MyAppVersion}', InstalledVer) < 0 then
          begin
            if MsgBox(
              'Downgrade detected!' + #13#10 + #13#10 +
              'Currently installed: v' + InstalledVer + #13#10 +
              'This installer: v{#MyAppVersion}' + #13#10 + #13#10 +
              'Installing an older version may cause data loss.' + #13#10 +
              'Do you want to continue anyway?',
              mbConfirmation, MB_YESNO) = IDNO then
            begin
              Result := False;
            end;
          end;
        end;
      end;
    end;
  end;
end;

// Check if directory is already in PATH
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER,
    'Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

// Issue #114: Persist the installer's language choice so first-run setup and
// bootstrap pick it up instead of falling back to OS-locale detection.
// Writes %USERPROFILE%\.cognithor\install_language.txt with "en" or "de".
// The marker is consumed (and deleted) on first run.
procedure CurStepChanged(CurStep: TSetupStep);
var
  LangCode: string;
  JarvisDir: string;
  MarkerPath: string;
begin
  if CurStep = ssPostInstall then
  begin
    if ActiveLanguage() = 'german' then
      LangCode := 'de'
    else
      LangCode := 'en';
    JarvisDir := ExpandConstant('{%USERPROFILE}\.cognithor');
    if not DirExists(JarvisDir) then
      ForceDirectories(JarvisDir);
    MarkerPath := JarvisDir + '\install_language.txt';
    SaveStringToFile(MarkerPath, LangCode, False);
  end;
end;

// Remove from PATH on uninstall + optional user data cleanup
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path: string;
  AppDir: string;
  JarvisHome: string;
  ResultCode: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // Remove from PATH
    AppDir := ExpandConstant('{app}');
    if RegQueryStringValue(HKEY_CURRENT_USER,
      'Environment',
      'Path', Path) then
    begin
      StringChangeEx(Path, ';' + AppDir, '', True);
      StringChangeEx(Path, AppDir + ';', '', True);
      StringChangeEx(Path, AppDir, '', True);
      RegWriteStringValue(HKEY_CURRENT_USER,
        'Environment',
        'Path', Path);
    end;

    // Stop Ollama if running
    Exec('taskkill', '/F /IM ollama.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

    // Ask user: remove user data?
    JarvisHome := ExpandConstant('{%USERPROFILE}\.jarvis');
    if DirExists(JarvisHome) then
    begin
      if MsgBox(
        'Do you want to remove all Cognithor user data?' + #13#10 +
        '(Memory, Vault, Skills, Configuration, Databases)' + #13#10 + #13#10 +
        'Location: ' + JarvisHome + #13#10 + #13#10 +
        'Click "No" to keep your data for a future reinstallation.',
        mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(JarvisHome, True, True, True);
      end;
    end;

    // Clean up install directory
    DelTree(AppDir, True, True, True);
  end;
end;
