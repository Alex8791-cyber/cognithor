// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'Jarvis';

  @override
  String get chat => 'Chat';

  @override
  String get settings => 'Settings';

  @override
  String get identity => 'Identity';

  @override
  String get workflows => 'Workflows';

  @override
  String get memory => 'Memory';

  @override
  String get monitoring => 'Monitoring';

  @override
  String get skills => 'Skills';

  @override
  String get config => 'Configuration';

  @override
  String get sendMessage => 'Send a message...';

  @override
  String get send => 'Send';

  @override
  String get cancel => 'Cancel';

  @override
  String get approve => 'Approve';

  @override
  String get reject => 'Reject';

  @override
  String get retry => 'Retry';

  @override
  String get close => 'Close';

  @override
  String get save => 'Save';

  @override
  String get delete => 'Delete';

  @override
  String get loading => 'Loading...';

  @override
  String get connecting => 'Connecting...';

  @override
  String get approvalTitle => 'Approval Required';

  @override
  String approvalBody(String tool) {
    return 'The tool $tool wants to execute:';
  }

  @override
  String approvalReason(String reason) {
    return 'Reason: $reason';
  }

  @override
  String get statusThinking => 'Thinking...';

  @override
  String get statusExecuting => 'Executing...';

  @override
  String get statusFinishing => 'Finishing...';

  @override
  String get voiceMessage => 'Voice message';

  @override
  String fileUpload(String name) {
    return 'File: $name';
  }

  @override
  String get connectionError => 'Cannot reach backend';

  @override
  String connectionErrorDetail(String url) {
    return 'Check that the Jarvis backend is running at $url';
  }

  @override
  String get authFailed => 'Authentication failed';

  @override
  String get tokenExpired => 'Session expired. Reconnecting...';

  @override
  String get serverUrl => 'Server URL';

  @override
  String get serverUrlHint => 'http://localhost:8741';

  @override
  String version(String version) {
    return 'Version $version';
  }

  @override
  String get errorGeneric => 'Something went wrong';

  @override
  String get errorNetwork => 'Network error. Check your connection.';

  @override
  String get errorTimeout => 'Request timed out';

  @override
  String get errorUnauthorized => 'Unauthorized. Please reconnect.';

  @override
  String get errorServerDown => 'Backend is not reachable';

  @override
  String get identityNotAvailable => 'Identity Layer not available';

  @override
  String get identityInstallHint =>
      'Install with: pip install cognithor[identity]';

  @override
  String get identityEnergy => 'Energy';

  @override
  String get identityInteractions => 'Interactions';

  @override
  String get identityMemories => 'Memories';

  @override
  String get identityCharacterStrength => 'Character Strength';

  @override
  String get identityFrozen => 'Frozen';

  @override
  String get identityActive => 'Active';

  @override
  String get identityDream => 'Dream Cycle';

  @override
  String get identityFreeze => 'Freeze';

  @override
  String get identityUnfreeze => 'Unfreeze';

  @override
  String get identityReset => 'Soft Reset';

  @override
  String get identityResetConfirm => 'Reset identity? Memories will be lost.';

  @override
  String get pipelinePlan => 'Planning';

  @override
  String get pipelineGate => 'Gatekeeper';

  @override
  String get pipelineExecute => 'Executing';

  @override
  String get pipelineReplan => 'Replanning';

  @override
  String get pipelineComplete => 'Complete';

  @override
  String get canvasTitle => 'Canvas';

  @override
  String get canvasClose => 'Close canvas';

  @override
  String get models => 'Models';

  @override
  String get channels => 'Channels';

  @override
  String get security => 'Security';

  @override
  String get reload => 'Reload';

  @override
  String get reloading => 'Reloading...';

  @override
  String get configSaved => 'Configuration reloaded';

  @override
  String get configError => 'Configuration error';

  @override
  String get uptime => 'Uptime';

  @override
  String get activeSessions => 'Active Sessions';

  @override
  String get totalRequests => 'Total Requests';

  @override
  String get events => 'Events';

  @override
  String get noEvents => 'No events recorded';

  @override
  String get severity => 'Severity';

  @override
  String get refreshing => 'Auto-refresh: 10s';

  @override
  String get noData => 'No data available';

  @override
  String get notAvailable => 'Not available';

  @override
  String get dashboard => 'Dashboard';

  @override
  String get systemOverview => 'System Overview';

  @override
  String get cpuUsage => 'CPU Usage';

  @override
  String get memoryUsage => 'Memory Usage';

  @override
  String get responseTime => 'Response Time';

  @override
  String get toolExecutions => 'Tool Executions';

  @override
  String get successRate => 'Success Rate';

  @override
  String get recentEvents => 'Recent Events';

  @override
  String get lastUpdated => 'Last Updated';

  @override
  String get systemHealth => 'System Health';

  @override
  String get performance => 'Performance';

  @override
  String get trends => 'Trends';

  @override
  String get marketplace => 'Marketplace';

  @override
  String get featured => 'Featured';

  @override
  String get trending => 'Trending';

  @override
  String get categories => 'Categories';

  @override
  String get searchSkills => 'Search skills...';

  @override
  String get installed => 'Installed';

  @override
  String get installSkill => 'Install';

  @override
  String get uninstallSkill => 'Uninstall';

  @override
  String get installing => 'Installing...';

  @override
  String get skillDetails => 'Skill Details';

  @override
  String get reviews => 'Reviews';

  @override
  String get noSkills => 'No skills found';

  @override
  String get browseMarketplace => 'Browse Marketplace';

  @override
  String get verified => 'Verified';

  @override
  String get downloads => 'Downloads';

  @override
  String get rating => 'Rating';

  @override
  String get memoryTitle => 'Memory';

  @override
  String get knowledgeGraph => 'Knowledge Graph';

  @override
  String get entities => 'Entities';

  @override
  String get relations => 'Relations';

  @override
  String get hygiene => 'Hygiene';

  @override
  String get quarantine => 'Quarantine';

  @override
  String get scanMemory => 'Scan';

  @override
  String get scanning => 'Scanning...';

  @override
  String get explainability => 'Explainability';

  @override
  String get decisionTrails => 'Decision Trails';

  @override
  String get lowTrust => 'Low Trust';

  @override
  String get graphStats => 'Graph Statistics';

  @override
  String get noEntities => 'No entities';

  @override
  String get noTrails => 'No trails';

  @override
  String get scanComplete => 'Scan complete';

  @override
  String get threats => 'Threats';

  @override
  String get threatRate => 'Threat Rate';

  @override
  String get totalScans => 'Total Scans';

  @override
  String get integrity => 'Integrity';

  @override
  String get securityTitle => 'Security';

  @override
  String get complianceTitle => 'Compliance';

  @override
  String get rolesTitle => 'Roles';

  @override
  String get permissions => 'Permissions';

  @override
  String get auditLog => 'Audit Log';

  @override
  String get redTeam => 'Red Team';

  @override
  String get scanStatus => 'Scan Status';

  @override
  String get complianceReport => 'Compliance Report';

  @override
  String get decisionsTitle => 'Decisions';

  @override
  String get remediations => 'Remediations';

  @override
  String get openStatus => 'Open';

  @override
  String get inProgressStatus => 'In Progress';

  @override
  String get resolvedStatus => 'Resolved';

  @override
  String get overdueStatus => 'Overdue';

  @override
  String get approvalRate => 'Approval Rate';

  @override
  String get flaggedCount => 'Flagged';

  @override
  String get transparency => 'Transparency';

  @override
  String get euAiAct => 'EU AI Act';

  @override
  String get dsgvo => 'GDPR';

  @override
  String get runScan => 'Run Scan';

  @override
  String get adminTitle => 'Administration';

  @override
  String get agentsTitle => 'Agents';

  @override
  String get modelsTitle => 'Models';

  @override
  String get systemTitle => 'System';

  @override
  String get workflowsTitle => 'Workflows';

  @override
  String get vaultTitle => 'Vault';

  @override
  String get credentialsTitle => 'Credentials';

  @override
  String get bindingsTitle => 'Bindings';

  @override
  String get connectorsTitle => 'Connectors';

  @override
  String get commandsTitle => 'Commands';

  @override
  String get isolationTitle => 'Isolation';

  @override
  String get sandboxTitle => 'Sandbox';

  @override
  String get circlesTitle => 'Circles';

  @override
  String get wizardsTitle => 'Wizards';

  @override
  String get systemStatus => 'System Status';

  @override
  String get shutdownServer => 'Shutdown Server';

  @override
  String get shutdownConfirm =>
      'Are you sure you want to shut down the server?';

  @override
  String get startComponent => 'Start';

  @override
  String get stopComponent => 'Stop';

  @override
  String get selectTemplate => 'Select Template';

  @override
  String get workflowStarted => 'Workflow started';

  @override
  String get noWorkflows => 'No workflows';

  @override
  String get templates => 'Templates';

  @override
  String get running => 'Running';

  @override
  String get vaultStats => 'Vault Statistics';

  @override
  String get totalEntries => 'Total Entries';

  @override
  String get agentVaults => 'Agent Vaults';

  @override
  String get noVaults => 'No vaults';

  @override
  String get availableModels => 'Available Models';

  @override
  String get modelStats => 'Model Statistics';

  @override
  String get providers => 'Providers';

  @override
  String get capabilities => 'Capabilities';

  @override
  String get plannerModel => 'Planner';

  @override
  String get executorModel => 'Executor';

  @override
  String get coderModel => 'Coder';

  @override
  String get embeddingModel => 'Embedding';

  @override
  String get configured => 'Configured';

  @override
  String get modelWarnings => 'Warnings';

  @override
  String get identityDreamCycle => 'Dream Cycle';

  @override
  String get identityGenesisAnchors => 'Genesis Anchors';

  @override
  String get identityNoAnchors => 'No genesis anchors';

  @override
  String get identityPersonality => 'Personality';

  @override
  String get identityCognitive => 'Cognitive State';

  @override
  String get identityEmotional => 'Emotional State';

  @override
  String get identitySomatic => 'Somatic State';

  @override
  String get identityNarrative => 'Narrative';

  @override
  String get identityExistential => 'Existential';

  @override
  String get identityPredictive => 'Predictive';

  @override
  String get identityEpistemic => 'Epistemic';

  @override
  String get identityBiases => 'Active Biases';

  @override
  String get search => 'Search';

  @override
  String get filter => 'Filter';

  @override
  String get sortBy => 'Sort by';

  @override
  String get refresh => 'Refresh';

  @override
  String get export => 'Export';

  @override
  String get viewAll => 'View All';

  @override
  String get details => 'Details';

  @override
  String get back => 'Back';

  @override
  String get confirm => 'Confirm';

  @override
  String get actions => 'Actions';

  @override
  String get statusLabel => 'Status';

  @override
  String get enabled => 'Enabled';

  @override
  String get disabled => 'Disabled';

  @override
  String get total => 'Total';

  @override
  String get count => 'Count';

  @override
  String get rate => 'Rate';

  @override
  String get average => 'Average';

  @override
  String get duration => 'Duration';

  @override
  String get timestamp => 'Timestamp';

  @override
  String get severityLabel => 'Severity';

  @override
  String get critical => 'Critical';

  @override
  String get errorLabel => 'Error';

  @override
  String get warningLabel => 'Warning';

  @override
  String get infoLabel => 'Info';

  @override
  String get successLabel => 'Success';

  @override
  String get unknownLabel => 'Unknown';

  @override
  String get notConfigured => 'Not configured';

  @override
  String get comingSoon => 'Coming Soon';

  @override
  String get beta => 'Beta';

  @override
  String get copyToClipboard => 'Copy to clipboard';

  @override
  String get copied => 'Copied!';

  @override
  String get chatSettings => 'Chat Settings';

  @override
  String get clearChat => 'Clear Chat';

  @override
  String get voiceMode => 'Voice Mode';

  @override
  String get fileUploadAction => 'Upload File';

  @override
  String get planDetails => 'Plan Details';

  @override
  String get noMessages => 'No messages yet';

  @override
  String get typeMessage => 'Type a message...';

  @override
  String get settingsTitle => 'Settings';

  @override
  String get language => 'Language';

  @override
  String get theme => 'Theme';

  @override
  String get about => 'About';

  @override
  String get licenses => 'Licenses';

  @override
  String get clearCache => 'Clear Cache';

  @override
  String get adminConfigSubtitle => 'Manage configuration';

  @override
  String get adminAgentsSubtitle => 'Agents & profiles';

  @override
  String get adminModelsSubtitle => 'LLM models';

  @override
  String get adminSecuritySubtitle => 'Security & compliance';

  @override
  String get adminWorkflowsSubtitle => 'Automations';

  @override
  String get adminMemorySubtitle => 'Knowledge graph';

  @override
  String get adminVaultSubtitle => 'Secrets & keys';

  @override
  String get adminSystemSubtitle => 'System status';

  @override
  String get dashboardRefreshing => 'Auto-refresh: 15s';

  @override
  String get backendVersion => 'Backend Version';

  @override
  String get modelInfo => 'Model Info';

  @override
  String get confidence => 'Confidence';

  @override
  String get rolesAccess => 'Roles & Access';

  @override
  String get loadMore => 'Load More';

  @override
  String get actor => 'Actor';

  @override
  String get noAuditEntries => 'No audit entries';

  @override
  String get allSeverities => 'All Severities';

  @override
  String get allActions => 'All Actions';

  @override
  String get scanNotAvailable => 'Scan not available';

  @override
  String get lastScan => 'Last Scan';

  @override
  String get scanResults => 'Scan Results';

  @override
  String get compliant => 'Compliant';

  @override
  String get nonCompliant => 'Non-Compliant';

  @override
  String get model => 'Model';

  @override
  String get temperature => 'Temperature';

  @override
  String get priority => 'Priority';

  @override
  String get allowedTools => 'Allowed Tools';

  @override
  String get blockedTools => 'Blocked Tools';

  @override
  String get noAgents => 'No agents configured';

  @override
  String get description => 'Description';

  @override
  String get provider => 'Provider';

  @override
  String get noModels => 'No models available';

  @override
  String get owner => 'Owner';

  @override
  String get llmBackend => 'LLM Backend';

  @override
  String get components => 'Components';

  @override
  String get dangerZone => 'Danger Zone';

  @override
  String get reloadConfig => 'Reload Config';

  @override
  String get runtimeInfo => 'Runtime Info';

  @override
  String get startWorkflow => 'Start Workflow';

  @override
  String get noCategories => 'No categories';

  @override
  String templateCount(String count) {
    return '$count templates';
  }

  @override
  String get entityTypes => 'Entity Types';

  @override
  String get activeTrails => 'Active Trails';

  @override
  String get completedTrails => 'Completed';

  @override
  String get lastAccessed => 'Last Accessed';

  @override
  String get author => 'Author';

  @override
  String get noQuarantine => 'No quarantined items';

  @override
  String get totalVaults => 'Total Vaults';

  @override
  String get scanNow => 'Scan Now';

  @override
  String get startConversation => 'Start a conversation';

  @override
  String get attachFile => 'Attach file';

  @override
  String get voiceModeHint => 'Voice mode coming soon';

  @override
  String get canvasLabel => 'Canvas';

  @override
  String get configGeneral => 'General';

  @override
  String get configLanguage => 'Language';

  @override
  String get configProviders => 'Providers';

  @override
  String get configModels => 'Models';

  @override
  String get configPlanner => 'Planner';

  @override
  String get configExecutor => 'Executor';

  @override
  String get configMemory => 'Memory';

  @override
  String get configChannels => 'Channels';

  @override
  String get configSecurity => 'Security';

  @override
  String get configWeb => 'Web';

  @override
  String get configMcp => 'MCP';

  @override
  String get configCron => 'Cron';

  @override
  String get configDatabase => 'Database';

  @override
  String get configLogging => 'Logging';

  @override
  String get configPrompts => 'Prompts';

  @override
  String get configAgents => 'Agents';

  @override
  String get configBindings => 'Bindings';

  @override
  String get configSystem => 'System';

  @override
  String get ownerName => 'Owner Name';

  @override
  String get operationMode => 'Operation Mode';

  @override
  String get costTracking => 'Cost Tracking';

  @override
  String get dailyBudget => 'Daily Budget';

  @override
  String get monthlyBudget => 'Monthly Budget';

  @override
  String get apiKey => 'API Key';

  @override
  String get baseUrl => 'Base URL';

  @override
  String get maxTokens => 'Max Tokens';

  @override
  String get timeout => 'Timeout';

  @override
  String get keepAlive => 'Keep Alive';

  @override
  String get contextWindow => 'Context Window';

  @override
  String get vramGb => 'VRAM (GB)';

  @override
  String get topP => 'Top P';

  @override
  String get maxIterations => 'Max Iterations';

  @override
  String get escalationAfter => 'Escalation After';

  @override
  String get responseBudget => 'Response Token Budget';

  @override
  String get policiesDir => 'Policies Directory';

  @override
  String get defaultRiskLevel => 'Default Risk Level';

  @override
  String get maxBlockedRetries => 'Max Blocked Retries';

  @override
  String get sandboxLevel => 'Sandbox Level';

  @override
  String get maxMemoryMb => 'Max Memory (MB)';

  @override
  String get maxCpuSeconds => 'Max CPU Seconds';

  @override
  String get allowedPaths => 'Allowed Paths';

  @override
  String get networkAccess => 'Network Access';

  @override
  String get envVars => 'Environment Variables';

  @override
  String get defaultTimeout => 'Default Timeout';

  @override
  String get maxOutputChars => 'Max Output Chars';

  @override
  String get maxRetries => 'Max Retries';

  @override
  String get backoffDelay => 'Backoff Delay';

  @override
  String get maxParallelTools => 'Max Parallel Tools';

  @override
  String get chunkSize => 'Chunk Size';

  @override
  String get chunkOverlap => 'Chunk Overlap';

  @override
  String get searchTopK => 'Search Top K';

  @override
  String get searchWeights => 'Search Weights';

  @override
  String get vectorWeight => 'Vector Weight';

  @override
  String get bm25Weight => 'BM25 Weight';

  @override
  String get graphWeight => 'Graph Weight';

  @override
  String get recencyHalfLife => 'Recency Half-Life';

  @override
  String get compactionThreshold => 'Compaction Threshold';

  @override
  String get compactionKeepLast => 'Compaction Keep Last';

  @override
  String get episodicRetention => 'Episodic Retention';

  @override
  String get dynamicWeighting => 'Dynamic Weighting';

  @override
  String get voiceEnabled => 'Voice Enabled';

  @override
  String get ttsBackend => 'TTS Backend';

  @override
  String get piperVoice => 'Piper Voice';

  @override
  String get piperLengthScale => 'Piper Length Scale';

  @override
  String get wakeWordEnabled => 'Wake Word Enabled';

  @override
  String get wakeWord => 'Wake Word';

  @override
  String get wakeWordBackend => 'Wake Word Backend';

  @override
  String get talkMode => 'Talk Mode';

  @override
  String get autoListen => 'Auto-Listen';

  @override
  String get blockedCommands => 'Blocked Commands';

  @override
  String get credentialPatterns => 'Credential Patterns';

  @override
  String get maxSubAgentDepth => 'Max Sub-Agent Depth';

  @override
  String get searchBackends => 'Search Backends';

  @override
  String get domainFilters => 'Domain Filters';

  @override
  String get blocklist => 'Blocklist';

  @override
  String get allowlist => 'Allowlist';

  @override
  String get httpLimits => 'HTTP Limits';

  @override
  String get maxFetchBytes => 'Max Fetch Bytes';

  @override
  String get maxTextChars => 'Max Text Chars';

  @override
  String get fetchTimeout => 'Fetch Timeout';

  @override
  String get searchTimeout => 'Search Timeout';

  @override
  String get maxSearchResults => 'Max Search Results';

  @override
  String get rateLimit => 'Rate Limit';

  @override
  String get mcpServers => 'MCP Servers';

  @override
  String get a2aProtocol => 'A2A Protocol';

  @override
  String get remotes => 'Remotes';

  @override
  String get heartbeat => 'Heartbeat';

  @override
  String get intervalMinutes => 'Interval (minutes)';

  @override
  String get checklistFile => 'Checklist File';

  @override
  String get channel => 'Channel';

  @override
  String get plugins => 'Plugins';

  @override
  String get skillsDir => 'Skills Directory';

  @override
  String get autoUpdate => 'Auto Update';

  @override
  String get cronJobs => 'Cron Jobs';

  @override
  String get schedule => 'Schedule';

  @override
  String get command => 'Command';

  @override
  String get databaseBackend => 'Database Backend';

  @override
  String get encryption => 'Encryption';

  @override
  String get pgHost => 'Host';

  @override
  String get pgPort => 'Port';

  @override
  String get pgDbName => 'Database Name';

  @override
  String get pgUser => 'User';

  @override
  String get pgPassword => 'Password';

  @override
  String get pgPoolMin => 'Pool Min';

  @override
  String get pgPoolMax => 'Pool Max';

  @override
  String get logLevel => 'Log Level';

  @override
  String get jsonLogs => 'JSON Logs';

  @override
  String get consoleOutput => 'Console Output';

  @override
  String get systemPrompt => 'System Prompt';

  @override
  String get replanPrompt => 'Replan Prompt';

  @override
  String get escalationPrompt => 'Escalation Prompt';

  @override
  String get policyYaml => 'Policy YAML';

  @override
  String get heartbeatMd => 'Heartbeat Checklist';

  @override
  String get personalityPrompt => 'Personality Prompt';

  @override
  String get promptEvolution => 'Prompt Evolution';

  @override
  String get resetToDefault => 'Reset to Default';

  @override
  String get triggerPatterns => 'Trigger Patterns';

  @override
  String get channelFilter => 'Channel Filter';

  @override
  String get pattern => 'Pattern';

  @override
  String get targetAgent => 'Target Agent';

  @override
  String get restartBackend => 'Restart Backend';

  @override
  String get exportConfig => 'Export Configuration';

  @override
  String get importConfig => 'Import Configuration';

  @override
  String get factoryReset => 'Factory Reset';

  @override
  String get factoryResetConfirm =>
      'This will reset ALL configuration to factory defaults. Continue?';

  @override
  String get configurationSaved => 'Configuration saved';

  @override
  String get saveHadErrors => 'Save had errors';

  @override
  String get unsavedChanges => 'Unsaved changes';

  @override
  String get discard => 'Discard';

  @override
  String get saving => 'Saving...';

  @override
  String get voiceOff => 'Off';

  @override
  String get voiceListening => 'Listening...';

  @override
  String get voiceSpeakNow => 'Speak now';

  @override
  String get voiceProcessing => 'Processing...';

  @override
  String get voiceSpeaking => 'Speaking...';

  @override
  String get observe => 'Observe';

  @override
  String get agentLog => 'Agent Log';

  @override
  String get kanban => 'Kanban';

  @override
  String get dag => 'DAG';

  @override
  String get plan => 'Plan';

  @override
  String get toDo => 'To Do';

  @override
  String get inProgress => 'In Progress';

  @override
  String get verifying => 'Verifying';

  @override
  String get done => 'Done';

  @override
  String get searchConfigPages => 'Search config pages...';

  @override
  String get noMatchingPages => 'No matching pages';

  @override
  String get knowledgeGraphTitle => 'Knowledge Graph';

  @override
  String get searchEntities => 'Search entities...';

  @override
  String get allTypes => 'All Types';

  @override
  String get entityDetail => 'Entity Detail';

  @override
  String get attributes => 'Attributes';

  @override
  String get instances => 'Instances';

  @override
  String get dagRuns => 'DAG Runs';

  @override
  String get noInstances => 'No instances';

  @override
  String get noDagRuns => 'No DAG runs';

  @override
  String get addCredential => 'Add Credential';

  @override
  String get service => 'Service';

  @override
  String get key => 'Key';

  @override
  String get value => 'Value';

  @override
  String get noCredentials => 'No credentials';

  @override
  String get deleteCredential => 'Delete Credential';

  @override
  String get lightMode => 'Light Mode';

  @override
  String get darkMode => 'Dark Mode';

  @override
  String get globalSearch => 'Search (Ctrl+K)';

  @override
  String get configPageGeneral => 'General';

  @override
  String get configPageLanguage => 'Language';

  @override
  String get configPageProviders => 'Providers';

  @override
  String get configPageModels => 'Models';

  @override
  String get configPagePlanner => 'Planner';

  @override
  String get configPageExecutor => 'Executor';

  @override
  String get configPageMemory => 'Memory';

  @override
  String get configPageChannels => 'Channels';

  @override
  String get configPageSecurity => 'Security';

  @override
  String get configPageWeb => 'Web';

  @override
  String get configPageMcp => 'MCP';

  @override
  String get configPageCron => 'Cron';

  @override
  String get configPageDatabase => 'Database';

  @override
  String get configPageLogging => 'Logging';

  @override
  String get configPagePrompts => 'Prompts';

  @override
  String get configPageAgents => 'Agents';

  @override
  String get configPageBindings => 'Bindings';

  @override
  String get configPageSystem => 'System';

  @override
  String get configTitle => 'Configuration';

  @override
  String get reloadFromBackend => 'Reload config from backend';

  @override
  String get saveCtrlS => 'Save (Ctrl+S)';

  @override
  String savedWithErrors(String sections) {
    return 'Saved with errors in: $sections';
  }

  @override
  String get saveFailed => 'Save failed';

  @override
  String get fieldOwnerName => 'Owner Name';

  @override
  String get fieldOperationMode => 'Operation Mode';

  @override
  String get fieldCostTracking => 'Cost Tracking';

  @override
  String get fieldDailyBudget => 'Daily Budget (USD)';

  @override
  String get fieldMonthlyBudget => 'Monthly Budget (USD)';

  @override
  String get fieldLlmBackend => 'LLM Backend';

  @override
  String get fieldPrimaryProvider => 'Primary LLM provider';

  @override
  String get fieldApiKey => 'API Key';

  @override
  String get fieldBaseUrl => 'Base URL';

  @override
  String get fieldModelName => 'Model Name';

  @override
  String get fieldContextWindow => 'Context Window';

  @override
  String get fieldTemperature => 'Temperature';

  @override
  String get fieldMaxIterations => 'Max Iterations';

  @override
  String get fieldEnabled => 'Enabled';

  @override
  String get fieldPort => 'Port';

  @override
  String get fieldHost => 'Host';

  @override
  String get fieldPassword => 'Password';

  @override
  String get fieldUser => 'User';

  @override
  String get fieldTimeout => 'Timeout';

  @override
  String get fieldLevel => 'Level';

  @override
  String get sectionSearchBackends => 'Search Backends';

  @override
  String get sectionDomainFilters => 'Domain Filters';

  @override
  String get sectionFetchLimits => 'Fetch Limits';

  @override
  String get sectionSearchLimits => 'Search Limits';

  @override
  String get sectionHttpLimits => 'HTTP Request Limits';

  @override
  String get sectionVoice => 'Voice';

  @override
  String get sectionHeartbeat => 'Heartbeat';

  @override
  String get sectionPlugins => 'Plugins';

  @override
  String get sectionCronJobs => 'Cron Jobs';

  @override
  String get sectionPromptEvolution => 'Prompt Evolution';

  @override
  String get addItem => 'Add';

  @override
  String get removeItem => 'Remove';

  @override
  String get translatePrompts => 'Translate Prompts via Ollama';

  @override
  String get translating => 'Translating...';

  @override
  String get promptsTranslated => 'Prompts translated';

  @override
  String get copiedToClipboard => 'Config copied to clipboard';

  @override
  String get configImported => 'Config imported';

  @override
  String get restartInitiated => 'Restart initiated';

  @override
  String get factoryResetComplete => 'Factory reset complete';

  @override
  String get factoryResetConfirmMsg =>
      'This will reset ALL configuration to factory defaults. Continue?';

  @override
  String get languageEnglish => 'English';

  @override
  String get languageGerman => 'Deutsch';

  @override
  String get languageChinese => '中文';

  @override
  String get languageArabic => 'العربية';

  @override
  String get uiAndPromptLanguage => 'UI and prompt language';

  @override
  String get learningTitle => 'Learning';

  @override
  String get knowledgeGaps => 'Knowledge Gaps';

  @override
  String get explorationQueue => 'Exploration Queue';

  @override
  String get filesProcessed => 'Files Processed';

  @override
  String get entitiesCreated => 'Entities Created';

  @override
  String get confidenceUpdates => 'Confidence Updates';

  @override
  String get openGaps => 'Open Gaps';

  @override
  String get importance => 'Importance';

  @override
  String get curiosity => 'Curiosity';

  @override
  String get explore => 'Explore';

  @override
  String get dismiss => 'Dismiss';

  @override
  String get noGaps => 'No knowledge gaps detected';

  @override
  String get noTasks => 'No exploration tasks';

  @override
  String get confidenceHistory => 'Confidence History';

  @override
  String get feedback => 'Feedback';

  @override
  String get positive => 'Positive';

  @override
  String get negative => 'Negative';

  @override
  String get correction => 'Correction';

  @override
  String get adminLearningSubtitle => 'Active learning & curiosity';

  @override
  String get watchDirectories => 'Watch Directories';

  @override
  String get directoryExists => 'Directory exists';

  @override
  String get directoryMissing => 'Directory not found';

  @override
  String get qaKnowledgeBase => 'Q&A';

  @override
  String get lineage => 'Lineage';

  @override
  String get question => 'Question';

  @override
  String get answer => 'Answer';

  @override
  String get topic => 'Topic';

  @override
  String get addQA => 'Add Q&A';

  @override
  String get verify => 'Verify';

  @override
  String get source => 'Source';

  @override
  String get noQAPairs => 'No knowledge entries';

  @override
  String get noLineage => 'No lineage data';

  @override
  String get entityLineage => 'Entity Lineage';

  @override
  String get recentChanges => 'Recent Changes';

  @override
  String get created => 'Created';

  @override
  String get updated => 'Updated';

  @override
  String get decayed => 'Decayed';

  @override
  String get runExploration => 'Run Exploration';

  @override
  String get explorationComplete => 'Exploration complete';

  @override
  String get activityChart => 'Activity';

  @override
  String get stopped => 'Stopped';

  @override
  String get requestsOverTime => 'Requests over time';

  @override
  String get teachCognithor => 'Teach Cognithor';

  @override
  String get uploadFile => 'Upload File';

  @override
  String get learnFromUrl => 'Learn from Website';

  @override
  String get learnFromYoutube => 'Learn from Video';

  @override
  String get dropFilesHere => 'Drop files here or click to browse';

  @override
  String get learningHistory => 'Learning History';

  @override
  String chunksLearned(String count) {
    return '$count chunks learned';
  }

  @override
  String get processingContent => 'Processing content...';

  @override
  String get learnSuccess => 'Successfully learned!';

  @override
  String get learnFailed => 'Learning failed';

  @override
  String get enterUrl => 'Enter website URL...';

  @override
  String get enterYoutubeUrl => 'Enter YouTube URL...';

  @override
  String get adminTeachSubtitle => 'Upload files, URLs, videos';

  @override
  String get newSkill => 'New Skill';

  @override
  String get editSkill => 'Edit Skill';

  @override
  String get createSkill => 'Create Skill';

  @override
  String get deleteSkill => 'Delete Skill';

  @override
  String get skillName => 'Name';

  @override
  String get skillBody => 'Skill Body (Markdown)';

  @override
  String get triggerKeywords => 'Trigger Keywords';

  @override
  String get requiredTools => 'Required Tools';

  @override
  String get modelPreference => 'Model Preference';

  @override
  String get skillSaved => 'Skill saved successfully';

  @override
  String get skillCreated => 'Skill created successfully';

  @override
  String get skillDeleted => 'Skill deleted';

  @override
  String get confirmDeleteSkill =>
      'Are you sure you want to delete this skill? This cannot be undone.';

  @override
  String get discardChanges => 'Discard changes?';

  @override
  String get discardChangesBody => 'You have unsaved changes. Discard them?';

  @override
  String get totalUses => 'Total Uses';

  @override
  String get lastUsed => 'Last Used';

  @override
  String get commaSeparated => 'Comma-separated';

  @override
  String get skillBodyHint => 'Write skill instructions in Markdown...';

  @override
  String get metadata => 'Metadata';

  @override
  String get statistics => 'Statistics';

  @override
  String get builtInSkill => 'Built-in skill (read-only)';

  @override
  String get exportSkillMd => 'Export as SKILL.md';

  @override
  String get skillExported => 'Skill exported to clipboard';

  @override
  String get general => 'General';

  @override
  String get productivity => 'Productivity';

  @override
  String get research => 'Research';

  @override
  String get analysis => 'Analysis';

  @override
  String get development => 'Development';

  @override
  String get automation => 'Automation';

  @override
  String get newAgent => 'New Agent';

  @override
  String get editAgent => 'Edit Agent';

  @override
  String get deleteAgent => 'Delete Agent';

  @override
  String get confirmDeleteAgent =>
      'Are you sure you want to delete this agent? This cannot be undone.';

  @override
  String get agentCreated => 'Agent created successfully';

  @override
  String get agentSaved => 'Agent saved successfully';

  @override
  String get agentDeleted => 'Agent deleted';

  @override
  String get displayName => 'Display Name';

  @override
  String get preferredModel => 'Preferred Model';

  @override
  String get sandboxTimeout => 'Sandbox Timeout (s)';

  @override
  String get sandboxNetwork => 'Sandbox Network';

  @override
  String get canDelegateTo => 'Can Delegate To';

  @override
  String get cannotDeleteDefault => 'Cannot delete the default agent';

  @override
  String get robotOfficePipMode => 'Robot Office is in Picture-in-Picture mode';

  @override
  String get fullscreen => 'Fullscreen';

  @override
  String get pipLabel => 'PiP';

  @override
  String taskCount(int count) {
    return '$count Tasks';
  }

  @override
  String get hackerMode => 'Hacker Mode';

  @override
  String get entityVisualization => 'Entity visualization';

  @override
  String get manageSecrets => 'Manage secrets';

  @override
  String get channelToggles => 'Channel Toggles';

  @override
  String get channelSettings => 'Channel Settings';

  @override
  String get tapToSelect => 'Tap to select...';

  @override
  String get selectModel => 'Select Model';

  @override
  String get searchModels => 'Search models...';

  @override
  String get remove => 'Remove';

  @override
  String get stopBackend => 'Stop Backend';

  @override
  String get stopBackendDescription =>
      'Stop the Jarvis backend. You will need to restart it manually.';

  @override
  String get stopBackendConfirmBody =>
      'This will stop the Jarvis backend process. You will need to restart it manually from the command line.';

  @override
  String get backendStopped => 'Backend stopped. Please restart manually.';

  @override
  String get downloadConfigDesc => 'Download current config as JSON';

  @override
  String get loadConfigDesc => 'Load config from a JSON file';

  @override
  String get resetAllDesc =>
      'Reset all settings to defaults. This cannot be undone.';

  @override
  String get factoryResetNotImpl =>
      'Factory reset is not yet implemented on the backend. To reset manually, delete your config.yaml and restart Jarvis.';

  @override
  String get ok => 'OK';

  @override
  String get wizardSubtitle => 'Your Personal AI Assistant';

  @override
  String get chooseLlmProvider => 'Choose your LLM provider';

  @override
  String get localOllama => 'Local (Ollama)';

  @override
  String get localOllamaDesc =>
      'Run models on your own hardware. Full privacy, no API costs. Requires Ollama installed.';

  @override
  String get cloudProviderLabel => 'Cloud Provider';

  @override
  String get cloudProviderDesc =>
      'Use OpenAI, Anthropic, or other cloud APIs. Faster setup, requires an API key.';

  @override
  String get next => 'Next';

  @override
  String get ollamaConfiguration => 'Ollama Configuration';

  @override
  String get cloudApiConfiguration => 'Cloud API Configuration';

  @override
  String get ollamaConfigHint => 'Enter the URL where Ollama is running.';

  @override
  String get cloudConfigHint =>
      'Select your cloud provider and enter your API key.';

  @override
  String get ollamaUrl => 'Ollama URL';

  @override
  String get testConnection => 'Test Connection';

  @override
  String get testingConnection => 'Testing...';

  @override
  String get youreAllSet => 'You\'re All Set!';

  @override
  String get ollamaReadyMsg =>
      'Ollama is connected and ready. Cognithor will use your local models for planning and execution.';

  @override
  String cloudReadyMsg(String provider) {
    return '$provider is configured. Cognithor will use your cloud API for planning and execution.';
  }

  @override
  String get changeSettingsAnytime =>
      'You can change these settings at any time.';

  @override
  String get startUsingCognithor => 'Start Using Cognithor';

  @override
  String get ollamaNoModels =>
      'Connected to Ollama. No models installed yet — run \"ollama pull qwen3:8b\" to get started.';

  @override
  String ollamaModelsAvailable(int count) {
    return 'Connected to Ollama. $count model(s) available.';
  }

  @override
  String ollamaStatusError(int code) {
    return 'Ollama responded with status $code. Make sure the server is running.';
  }

  @override
  String get enterApiKey => 'Please enter an API key.';

  @override
  String apiKeyTooShort(String provider) {
    return 'That key looks too short. Double-check your $provider API key.';
  }

  @override
  String apiKeySaved(String provider) {
    return '$provider API key saved. You can change it later in Settings.';
  }

  @override
  String connectionFailed(String error) {
    return 'Connection failed: $error';
  }

  @override
  String get minimize => 'Minimize';

  @override
  String get shrink => 'Shrink';

  @override
  String get expandLabel => 'Expand';

  @override
  String get robotOffice => 'Robot Office';

  @override
  String get copy => 'Copy';

  @override
  String get share => 'Share';

  @override
  String get noLogEntries => 'No log entries';

  @override
  String get noPlanData => 'No plan data';

  @override
  String get noDagData => 'No DAG data';

  @override
  String get log => 'Log';

  @override
  String get fileReadError => 'File could not be read';

  @override
  String uploadError(String error) {
    return 'Upload error: $error';
  }

  @override
  String get toolSpecificTimeouts => 'Tool-Specific Timeouts';

  @override
  String get required => 'Required';

  @override
  String get stopLabel => 'Stop';

  @override
  String get resetLabel => 'Reset';

  @override
  String get exportLabel => 'Export';

  @override
  String get importLabel => 'Import';

  @override
  String get catAiEngine => 'AI Engine';

  @override
  String get catChannels => 'Channels';

  @override
  String get catKnowledge => 'Knowledge';

  @override
  String get catSecurity => 'Security';

  @override
  String get catSystem => 'System';

  @override
  String get saved => 'Saved';

  @override
  String get chatHistory => 'Chat History';

  @override
  String get newChat => 'New Chat';

  @override
  String get untitledChat => 'Untitled Chat';

  @override
  String get deleteChat => 'Delete Chat';

  @override
  String get confirmDeleteChat => 'Delete this conversation?';

  @override
  String messagesCount(String count) {
    return '$count messages';
  }

  @override
  String get justNow => 'Just now';

  @override
  String minutesAgo(String count) {
    return '$count min ago';
  }

  @override
  String hoursAgo(String count) {
    return '${count}h ago';
  }

  @override
  String daysAgo(String count) {
    return '${count}d ago';
  }

  @override
  String get folders => 'Folders';

  @override
  String get moveToFolder => 'Move to folder';

  @override
  String get newFolder => 'New Folder';

  @override
  String get folderName => 'Folder name';

  @override
  String get noFolder => 'Unfiled';

  @override
  String get renameChat => 'Rename';

  @override
  String get editTitle => 'Edit title';

  @override
  String sessionCount(int count) {
    return '$count chats';
  }

  @override
  String get idle => 'Idle';

  @override
  String get thinking => 'Thinking...';
}
