using System;
using System.IO;
using System.Text.Json;

namespace Cognithor;

sealed class LauncherSettings
{
    static readonly string SettingsPath = Path.Combine(
        AppContext.BaseDirectory, "launcher-settings.json");

    public string PreferredUi { get; set; } = "desktop";

    public bool PreferDesktop => PreferredUi == "desktop";

    public void TogglePreferredUi()
    {
        PreferredUi = PreferDesktop ? "browser" : "desktop";
        Save();
    }

    public void Save()
    {
        try
        {
            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            });
            File.WriteAllText(SettingsPath, json);
        }
        catch { }
    }

    public static LauncherSettings Load()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                return JsonSerializer.Deserialize<LauncherSettings>(json,
                    new JsonSerializerOptions
                    {
                        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                    }) ?? new LauncherSettings();
            }
        }
        catch { }
        return new LauncherSettings();
    }
}
