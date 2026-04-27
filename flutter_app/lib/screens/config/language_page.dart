import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/locale_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';
import 'package:cognithor_ui/widgets/cognithor_toast.dart';

class LanguagePage extends StatefulWidget {
  const LanguagePage({super.key});

  @override
  State<LanguagePage> createState() => _LanguagePageState();
}

class _LanguagePageState extends State<LanguagePage> {
  bool _translating = false;

  Future<void> _translatePrompts(String targetLocale) async {
    setState(() => _translating = true);
    final api = context.read<ConnectionProvider>().api;
    final cfg = context.read<ConfigProvider>();

    // Collect current prompts to send for translation
    final prompts = <String, String>{};
    for (final key in ['plannerSystem', 'replanPrompt', 'escalationPrompt']) {
      final val = (cfg.prompts[key] ?? '').toString();
      if (val.isNotEmpty) prompts[key] = val;
    }

    // If no custom prompts, fetch defaults from backend first
    if (prompts.isEmpty) {
      try {
        final defaults = await api.get('config/prompts');
        for (final key in [
          'plannerSystem',
          'replanPrompt',
          'escalationPrompt',
        ]) {
          final val = (defaults[key] ?? '').toString();
          if (val.isNotEmpty) prompts[key] = val;
        }
      } catch (_) {}
    }

    if (prompts.isEmpty) {
      setState(() => _translating = false);
      if (mounted) {
        CognithorToast.show(
          context,
          'No prompts to translate',
          type: ToastType.warning,
        );
      }
      return;
    }

    final res = await api.translatePrompts({
      'target_locale': targetLocale,
      'method': 'ollama',
      'prompts': prompts,
    });

    setState(() => _translating = false);
    if (!mounted) return;

    final l = AppLocalizations.of(context);
    if (res.containsKey('error')) {
      CognithorToast.show(
        context,
        res['error'].toString(),
        type: ToastType.error,
        duration: const Duration(seconds: 5),
      );
    } else {
      // Apply translated prompts back to config
      final translations = res['translations'] as Map<String, dynamic>? ?? {};
      for (final entry in translations.entries) {
        cfg.prompts[entry.key] = entry.value.toString();
      }
      cfg.notify();
      CognithorToast.show(
        context,
        l.promptsTranslated,
        type: ToastType.success,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final lang = (cfg.cfg['language'] ?? 'de').toString();
        // Ensure the value is one of the supported codes
        final effectiveLang = LocaleProvider.supportedCodes.contains(lang)
            ? lang
            : 'de';
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorSelectField(
              label: l.configPageLanguage,
              value: effectiveLang,
              options: [
                SelectOption(value: 'en', label: l.languageEnglish),
                SelectOption(value: 'de', label: l.languageGerman),
                SelectOption(value: 'zh', label: l.languageChinese),
                SelectOption(value: 'ar', label: l.languageArabic),
              ],
              onChanged: (v) {
                cfg.set('language', v);
                context.read<LocaleProvider>().setLocale(v);
              },
              description: l.uiAndPromptLanguage,
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _translating
                  ? null
                  : () => _translatePrompts(effectiveLang),
              icon: _translating
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.translate, size: 18),
              label: Text(_translating ? l.translating : l.translatePrompts),
            ),
          ],
        );
      },
    );
  }
}
