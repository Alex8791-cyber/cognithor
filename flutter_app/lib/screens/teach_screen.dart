import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/neon_glow.dart';

/// "Teach Cognithor" screen -- lets users upload files, paste URLs, and submit
/// YouTube links so the assistant can learn from them.
class TeachScreen extends StatefulWidget {
  const TeachScreen({super.key});

  @override
  State<TeachScreen> createState() => _TeachScreenState();
}

class _TeachScreenState extends State<TeachScreen> {
  // Controllers
  final _urlController = TextEditingController();
  final _youtubeController = TextEditingController();

  // State
  String? _selectedFilename;
  List<int>? _selectedFileBytes;
  bool _fileUploading = false;
  String _priority = 'normal';
  bool _urlProcessing = false;
  bool _youtubeProcessing = false;

  // Results
  String? _fileResult;
  bool? _fileSuccess;
  String? _urlResult;
  bool? _urlSuccess;
  String? _youtubeResult;
  bool? _youtubeSuccess;

  // Queue
  List<Map<String, dynamic>> _queue = [];

  // History
  List<Map<String, dynamic>> _history = [];
  bool _historyLoading = false;
  String _historyFilter = 'all'; // all, file, url, youtube

  @override
  void initState() {
    super.initState();
    _loadHistory();
    _loadQueue();
  }

  @override
  void dispose() {
    _urlController.dispose();
    _youtubeController.dispose();
    super.dispose();
  }

  Future<void> _loadHistory() async {
    setState(() => _historyLoading = true);
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.getLearnHistory();
      final items = res['items'] as List<dynamic>? ?? [];
      setState(() {
        _history = items.whereType<Map<String, dynamic>>().toList();
        _historyLoading = false;
      });
    } catch (_) {
      setState(() => _historyLoading = false);
    }
  }

  Future<void> _loadQueue() async {
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.getLearnQueue();
      if (mounted) {
        setState(() {
          _queue =
              (res['queue'] as List<dynamic>?)
                  ?.map((e) => e as Map<String, dynamic>)
                  .toList() ??
              [];
        });
      }
    } catch (_) {}
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: [
        'pdf',
        'docx',
        'txt',
        'md',
        'png',
        'jpg',
        'jpeg',
        'csv',
        'json',
      ],
      withData: true,
    );
    if (result != null && result.files.isNotEmpty) {
      final file = result.files.first;
      setState(() {
        _selectedFilename = file.name;
        _selectedFileBytes = file.bytes;
        _fileResult = null;
        _fileSuccess = null;
      });
    }
  }

  Future<void> _uploadFile() async {
    if (_selectedFileBytes == null || _selectedFilename == null) return;
    setState(() {
      _fileUploading = true;
      _fileResult = null;
      _fileSuccess = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.learnFromFile(
        _selectedFileBytes!,
        _selectedFilename!,
        priority: _priority,
      );
      if (res.containsKey('error')) {
        setState(() {
          _fileResult = res['error'] as String;
          _fileSuccess = false;
        });
      } else {
        final chunks = res['chunks'] ?? res['chunk_count'] ?? '?';
        final deepStatus = res['deep_learn_status'] ?? '';
        setState(() {
          _fileResult = deepStatus == 'queued'
              ? '${AppLocalizations.of(context).deepLearningQueued} ($chunks chunks)'
              : '$chunks chunks';
          _fileSuccess = true;
        });
        _loadHistory();
        _loadQueue();
      }
    } catch (e) {
      setState(() {
        _fileResult = e.toString();
        _fileSuccess = false;
      });
    } finally {
      setState(() => _fileUploading = false);
    }
  }

  bool _isYoutubeUrl(String url) {
    return url.contains('youtube.com') || url.contains('youtu.be');
  }

  Future<void> _learnFromUrl() async {
    final url = _urlController.text.trim();
    if (url.isEmpty) return;

    // Auto-route YouTube URLs
    if (_isYoutubeUrl(url)) {
      _youtubeController.text = url;
      _urlController.clear();
      await _learnFromYoutube();
      return;
    }

    setState(() {
      _urlProcessing = true;
      _urlResult = null;
      _urlSuccess = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.learnFromUrl(url, priority: _priority);
      if (res.containsKey('error')) {
        setState(() {
          _urlResult = res['error'] as String;
          _urlSuccess = false;
        });
      } else {
        final chunks = res['chunks'] ?? res['chunk_count'] ?? '?';
        setState(() {
          _urlResult = '$chunks';
          _urlSuccess = true;
        });
        _urlController.clear();
        _loadHistory();
        _loadQueue();
      }
    } catch (e) {
      setState(() {
        _urlResult = e.toString();
        _urlSuccess = false;
      });
    } finally {
      setState(() => _urlProcessing = false);
    }
  }

  Future<void> _learnFromYoutube() async {
    final url = _youtubeController.text.trim();
    if (url.isEmpty) return;
    setState(() {
      _youtubeProcessing = true;
      _youtubeResult = null;
      _youtubeSuccess = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.learnFromYoutube(url, priority: _priority);
      if (res.containsKey('error')) {
        setState(() {
          _youtubeResult = res['error'] as String;
          _youtubeSuccess = false;
        });
      } else {
        final chunks = res['chunks'] ?? res['chunk_count'] ?? '?';
        final title = res['title'] as String?;
        setState(() {
          _youtubeResult = title != null ? '$title ($chunks)' : '$chunks';
          _youtubeSuccess = true;
        });
        _youtubeController.clear();
        _loadHistory();
        _loadQueue();
      }
    } catch (e) {
      setState(() {
        _youtubeResult = e.toString();
        _youtubeSuccess = false;
      });
    } finally {
      setState(() => _youtubeProcessing = false);
    }
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final isWide = MediaQuery.sizeOf(context).width > 800;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.school, size: 24),
            const SizedBox(width: 8),
            Text(l.teachCognithor),
          ],
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(CognithorTheme.spacing),
        children: [
          // Input cards
          if (isWide)
            IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: _buildFileCard(l)),
                  const SizedBox(width: CognithorTheme.spacing),
                  Expanded(child: _buildUrlCard(l)),
                  const SizedBox(width: CognithorTheme.spacing),
                  Expanded(child: _buildYoutubeCard(l)),
                ],
              ),
            )
          else ...[
            _buildFileCard(l),
            const SizedBox(height: CognithorTheme.spacing),
            _buildUrlCard(l),
            const SizedBox(height: CognithorTheme.spacing),
            _buildYoutubeCard(l),
          ],

          // Queue section
          if (_queue.isNotEmpty) ...[
            const SizedBox(height: 16),
            Text(
              l.deepLearningQueue,
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            ..._queue.map(
              (item) => ListTile(
                dense: true,
                leading: const Icon(Icons.hourglass_top, size: 18),
                title: Text(
                  item['source']?.toString() ?? '',
                  overflow: TextOverflow.ellipsis,
                ),
                trailing: Chip(
                  label: Text(
                    item['priority']?.toString() ?? '',
                    style: const TextStyle(fontSize: 10),
                  ),
                  visualDensity: VisualDensity.compact,
                ),
              ),
            ),
          ],

          const SizedBox(height: CognithorTheme.spacingLg),

          // History section
          _buildHistorySection(l),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // File upload card
  // ---------------------------------------------------------------------------

  Widget _buildFileCard(AppLocalizations l) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return NeonCard(
      tint: CognithorTheme.sectionDashboard,
      padding: const EdgeInsets.all(CognithorTheme.spacing),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(
                Icons.cloud_upload,
                color: CognithorTheme.sectionDashboard,
              ),
              const SizedBox(width: 8),
              Text(l.uploadFile, style: theme.textTheme.titleMedium),
            ],
          ),
          const SizedBox(height: CognithorTheme.spacing),

          // Drop zone
          GestureDetector(
            onTap: _fileUploading ? null : _pickFile,
            child: Container(
              height: 120,
              decoration: BoxDecoration(
                border: Border.all(
                  color: colorScheme.outline.withValues(alpha: 0.4),
                  width: 1.5,
                  strokeAlign: BorderSide.strokeAlignInside,
                ),
                borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
              ),
              child: Center(
                child: _fileUploading
                    ? const CircularProgressIndicator()
                    : Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.cloud_upload_outlined,
                            size: 36,
                            color: CognithorTheme.textSecondary,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            l.dropFilesHere,
                            textAlign: TextAlign.center,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: CognithorTheme.textSecondary,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'PDF, DOCX, TXT, MD, PNG, JPG, CSV, JSON',
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: CognithorTheme.textTertiary,
                            ),
                          ),
                        ],
                      ),
              ),
            ),
          ),

          if (_selectedFilename != null) ...[
            const SizedBox(height: CognithorTheme.spacingSm),
            Row(
              children: [
                const Icon(Icons.insert_drive_file, size: 16),
                const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    _selectedFilename!,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodySmall,
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton.tonal(
                  onPressed: _fileUploading ? null : _uploadFile,
                  child: Text(l.uploadFile),
                ),
              ],
            ),
            const SizedBox(height: CognithorTheme.spacingSm),
            Row(
              children: [
                Text(l.learnPriority, style: theme.textTheme.bodySmall),
                const SizedBox(width: 8),
                DropdownButton<String>(
                  value: _priority,
                  items: [
                    DropdownMenuItem(value: 'low', child: Text(l.priorityLow)),
                    DropdownMenuItem(
                      value: 'normal',
                      child: Text(l.priorityNormal),
                    ),
                    DropdownMenuItem(
                      value: 'high',
                      child: Text(l.priorityHigh),
                    ),
                  ],
                  onChanged: (v) => setState(() => _priority = v ?? 'normal'),
                ),
              ],
            ),
          ],

          // Result
          if (_fileResult != null) ...[
            const SizedBox(height: CognithorTheme.spacingSm),
            _buildResultBadge(
              _fileSuccess!,
              _fileSuccess!
                  ? l.chunksLearned(_fileResult!)
                  : '${l.learnFailed}: $_fileResult',
            ),
          ],
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // URL card
  // ---------------------------------------------------------------------------

  Widget _buildUrlCard(AppLocalizations l) {
    final theme = Theme.of(context);

    return NeonCard(
      tint: CognithorTheme.sectionDashboard,
      padding: const EdgeInsets.all(CognithorTheme.spacing),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(Icons.link, color: CognithorTheme.sectionDashboard),
              const SizedBox(width: 8),
              Text(l.learnFromUrl, style: theme.textTheme.titleMedium),
            ],
          ),
          const SizedBox(height: CognithorTheme.spacing),
          TextField(
            controller: _urlController,
            decoration: InputDecoration(
              prefixIcon: const Icon(Icons.link, size: 20),
              hintText: l.enterUrl,
              border: const OutlineInputBorder(),
            ),
            onSubmitted: (_) => _learnFromUrl(),
          ),
          const SizedBox(height: CognithorTheme.spacingSm),
          NeonGlow(
            color: CognithorTheme.sectionDashboard,
            intensity: 0.2,
            blurRadius: 8,
            child: FilledButton.icon(
              onPressed: _urlProcessing ? null : _learnFromUrl,
              icon: _urlProcessing
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.school),
              label: Text(
                _urlProcessing ? l.processingContent : l.learnFromUrl,
              ),
            ),
          ),
          const SizedBox(height: CognithorTheme.spacingSm),
          Row(
            children: [
              Text(l.learnPriority, style: theme.textTheme.bodySmall),
              const SizedBox(width: 8),
              DropdownButton<String>(
                value: _priority,
                items: [
                  DropdownMenuItem(value: 'low', child: Text(l.priorityLow)),
                  DropdownMenuItem(
                    value: 'normal',
                    child: Text(l.priorityNormal),
                  ),
                  DropdownMenuItem(value: 'high', child: Text(l.priorityHigh)),
                ],
                onChanged: (v) => setState(() => _priority = v ?? 'normal'),
              ),
            ],
          ),

          // Result
          if (_urlResult != null) ...[
            const SizedBox(height: CognithorTheme.spacingSm),
            _buildResultBadge(
              _urlSuccess!,
              _urlSuccess!
                  ? l.chunksLearned(_urlResult!)
                  : '${l.learnFailed}: $_urlResult',
            ),
          ],
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // YouTube card
  // ---------------------------------------------------------------------------

  Widget _buildYoutubeCard(AppLocalizations l) {
    final theme = Theme.of(context);

    return NeonCard(
      tint: CognithorTheme.sectionDashboard,
      padding: const EdgeInsets.all(CognithorTheme.spacing),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(
                Icons.play_circle,
                color: CognithorTheme.sectionDashboard,
              ),
              const SizedBox(width: 8),
              Text(l.learnFromYoutube, style: theme.textTheme.titleMedium),
            ],
          ),
          const SizedBox(height: CognithorTheme.spacing),
          TextField(
            controller: _youtubeController,
            decoration: InputDecoration(
              prefixIcon: const Icon(Icons.play_circle_outline, size: 20),
              hintText: l.enterYoutubeUrl,
              border: const OutlineInputBorder(),
            ),
            onSubmitted: (_) => _learnFromYoutube(),
          ),
          const SizedBox(height: CognithorTheme.spacingSm),
          FilledButton.icon(
            onPressed: _youtubeProcessing ? null : _learnFromYoutube,
            icon: _youtubeProcessing
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.school),
            label: Text(
              _youtubeProcessing ? l.processingContent : l.learnFromYoutube,
            ),
          ),
          const SizedBox(height: CognithorTheme.spacingSm),
          Row(
            children: [
              Text(l.learnPriority, style: theme.textTheme.bodySmall),
              const SizedBox(width: 8),
              DropdownButton<String>(
                value: _priority,
                items: [
                  DropdownMenuItem(value: 'low', child: Text(l.priorityLow)),
                  DropdownMenuItem(
                    value: 'normal',
                    child: Text(l.priorityNormal),
                  ),
                  DropdownMenuItem(value: 'high', child: Text(l.priorityHigh)),
                ],
                onChanged: (v) => setState(() => _priority = v ?? 'normal'),
              ),
            ],
          ),

          // Result
          if (_youtubeResult != null) ...[
            const SizedBox(height: CognithorTheme.spacingSm),
            _buildResultBadge(
              _youtubeSuccess!,
              _youtubeSuccess!
                  ? (_youtubeResult!.contains('(')
                        ? '${l.learnSuccess} $_youtubeResult'
                        : l.chunksLearned(_youtubeResult!))
                  : '${l.learnFailed}: $_youtubeResult',
            ),
          ],
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // History section
  // ---------------------------------------------------------------------------

  Widget _buildHistorySection(AppLocalizations l) {
    final theme = Theme.of(context);

    final filtered = _historyFilter == 'all'
        ? _history
        : _history.where((h) => h['source'] == _historyFilter).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header row
        Row(
          children: [
            Icon(Icons.history, color: CognithorTheme.textSecondary),
            const SizedBox(width: 8),
            Text(l.learningHistory, style: theme.textTheme.titleMedium),
            const Spacer(),
            // Filter chips
            ..._buildFilterChips(l),
            const SizedBox(width: 8),
            IconButton(
              icon: const Icon(Icons.refresh, size: 20),
              tooltip: l.refresh,
              onPressed: _loadHistory,
            ),
          ],
        ),
        const SizedBox(height: CognithorTheme.spacingSm),

        if (_historyLoading)
          const Center(child: CircularProgressIndicator())
        else if (filtered.isEmpty)
          Center(
            child: Padding(
              padding: const EdgeInsets.all(CognithorTheme.spacingLg),
              child: Text(
                l.noData,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: CognithorTheme.textSecondary,
                ),
              ),
            ),
          )
        else
          ...filtered.map((item) => _buildHistoryTile(item, l)),
      ],
    );
  }

  List<Widget> _buildFilterChips(AppLocalizations l) {
    final filters = <String, String>{
      'all': l.viewAll,
      'file': l.uploadFile,
      'url': 'URL',
      'youtube': 'YouTube',
    };
    return filters.entries.map((e) {
      final selected = _historyFilter == e.key;
      return Padding(
        padding: const EdgeInsets.only(left: 4),
        child: FilterChip(
          label: Text(e.value),
          selected: selected,
          onSelected: (_) => setState(() => _historyFilter = e.key),
          visualDensity: VisualDensity.compact,
        ),
      );
    }).toList();
  }

  Widget _buildHistoryTile(Map<String, dynamic> item, AppLocalizations l) {
    final source = item['source'] as String? ?? 'file';
    final name = item['name'] as String? ?? item['url'] as String? ?? '?';
    final status = item['status'] as String? ?? 'unknown';
    final chunks = item['chunks'] ?? item['chunk_count'];
    final ts = item['timestamp'] as String?;

    final icon = switch (source) {
      'youtube' => Icons.play_circle,
      'url' => Icons.link,
      _ => Icons.insert_drive_file,
    };

    final isOk = status == 'success' || status == 'done';

    return ListTile(
      leading: Icon(icon, color: CognithorTheme.textSecondary),
      title: Text(name, maxLines: 1, overflow: TextOverflow.ellipsis),
      subtitle: ts != null
          ? Text(ts, style: const TextStyle(fontSize: 12))
          : null,
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (chunks != null)
            Text(
              l.chunksLearned('$chunks'),
              style: TextStyle(
                fontSize: 12,
                color: CognithorTheme.textSecondary,
              ),
            ),
          const SizedBox(width: 8),
          Icon(
            isOk ? Icons.check_circle : Icons.error,
            size: 18,
            color: isOk ? CognithorTheme.green : CognithorTheme.red,
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  Widget _buildResultBadge(bool success, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: CognithorTheme.spacingSm,
        vertical: CognithorTheme.spacingXs,
      ),
      decoration: BoxDecoration(
        color: (success ? CognithorTheme.green : CognithorTheme.red).withValues(
          alpha: 0.12,
        ),
        borderRadius: BorderRadius.circular(CognithorTheme.buttonRadius),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            success ? Icons.check_circle : Icons.error,
            size: 16,
            color: success ? CognithorTheme.green : CognithorTheme.red,
          ),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 12,
                color: success ? CognithorTheme.green : CognithorTheme.red,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
