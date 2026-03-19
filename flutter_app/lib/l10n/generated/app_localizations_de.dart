// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for German (`de`).
class AppLocalizationsDe extends AppLocalizations {
  AppLocalizationsDe([String locale = 'de']) : super(locale);

  @override
  String get appTitle => 'Jarvis';

  @override
  String get chat => 'Chat';

  @override
  String get settings => 'Einstellungen';

  @override
  String get identity => 'Identitaet';

  @override
  String get workflows => 'Workflows';

  @override
  String get memory => 'Gedaechtnis';

  @override
  String get monitoring => 'Monitoring';

  @override
  String get skills => 'Skills';

  @override
  String get config => 'Konfiguration';

  @override
  String get sendMessage => 'Nachricht eingeben...';

  @override
  String get send => 'Senden';

  @override
  String get cancel => 'Abbrechen';

  @override
  String get approve => 'Genehmigen';

  @override
  String get reject => 'Ablehnen';

  @override
  String get retry => 'Erneut versuchen';

  @override
  String get close => 'Schliessen';

  @override
  String get save => 'Speichern';

  @override
  String get delete => 'Loeschen';

  @override
  String get loading => 'Laden...';

  @override
  String get connecting => 'Verbinde...';

  @override
  String get approvalTitle => 'Genehmigung erforderlich';

  @override
  String approvalBody(String tool) {
    return 'Das Tool $tool moechte ausfuehren:';
  }

  @override
  String approvalReason(String reason) {
    return 'Grund: $reason';
  }

  @override
  String get statusThinking => 'Denkt nach...';

  @override
  String get statusExecuting => 'Fuehrt aus...';

  @override
  String get statusFinishing => 'Fertigstellung...';

  @override
  String get voiceMessage => 'Sprachnachricht';

  @override
  String fileUpload(String name) {
    return 'Datei: $name';
  }

  @override
  String get connectionError => 'Backend nicht erreichbar';

  @override
  String connectionErrorDetail(String url) {
    return 'Pruefe ob das Jarvis-Backend unter $url laeuft';
  }

  @override
  String get authFailed => 'Authentifizierung fehlgeschlagen';

  @override
  String get tokenExpired => 'Sitzung abgelaufen. Verbinde neu...';

  @override
  String get serverUrl => 'Server-URL';

  @override
  String get serverUrlHint => 'http://localhost:8741';

  @override
  String version(String version) {
    return 'Version $version';
  }

  @override
  String get errorGeneric => 'Etwas ist schiefgelaufen';

  @override
  String get errorNetwork => 'Netzwerkfehler. Pruefe deine Verbindung.';

  @override
  String get errorTimeout => 'Zeitlimit ueberschritten';

  @override
  String get errorUnauthorized => 'Nicht autorisiert. Bitte neu verbinden.';

  @override
  String get errorServerDown => 'Backend nicht erreichbar';

  @override
  String get identityNotAvailable => 'Identitaetsschicht nicht verfuegbar';

  @override
  String get identityInstallHint =>
      'Installiere mit: pip install cognithor[identity]';

  @override
  String get identityEnergy => 'Energie';

  @override
  String get identityInteractions => 'Interaktionen';

  @override
  String get identityMemories => 'Erinnerungen';

  @override
  String get identityCharacterStrength => 'Charakterstaerke';

  @override
  String get identityFrozen => 'Eingefroren';

  @override
  String get identityActive => 'Aktiv';

  @override
  String get identityDream => 'Traumzyklus';

  @override
  String get identityFreeze => 'Einfrieren';

  @override
  String get identityUnfreeze => 'Auftauen';

  @override
  String get identityReset => 'Soft Reset';

  @override
  String get identityResetConfirm =>
      'Identitaet zuruecksetzen? Erinnerungen gehen verloren.';

  @override
  String get pipelinePlan => 'Planung';

  @override
  String get pipelineGate => 'Gatekeeper';

  @override
  String get pipelineExecute => 'Ausfuehrung';

  @override
  String get pipelineReplan => 'Neuplanung';

  @override
  String get pipelineComplete => 'Abgeschlossen';

  @override
  String get canvasTitle => 'Canvas';

  @override
  String get canvasClose => 'Canvas schliessen';

  @override
  String get models => 'Modelle';

  @override
  String get channels => 'Kanaele';

  @override
  String get security => 'Sicherheit';

  @override
  String get reload => 'Neu laden';

  @override
  String get reloading => 'Wird neu geladen...';

  @override
  String get configSaved => 'Konfiguration neu geladen';

  @override
  String get configError => 'Konfigurationsfehler';

  @override
  String get uptime => 'Betriebszeit';

  @override
  String get activeSessions => 'Aktive Sitzungen';

  @override
  String get totalRequests => 'Anfragen gesamt';

  @override
  String get events => 'Ereignisse';

  @override
  String get noEvents => 'Keine Ereignisse aufgezeichnet';

  @override
  String get severity => 'Schweregrad';

  @override
  String get refreshing => 'Auto-Aktualisierung: 10s';

  @override
  String get noData => 'Keine Daten verfuegbar';

  @override
  String get notAvailable => 'Nicht verfuegbar';

  @override
  String get dashboard => 'Dashboard';

  @override
  String get systemOverview => 'Systemuebersicht';

  @override
  String get cpuUsage => 'CPU-Auslastung';

  @override
  String get memoryUsage => 'Speicherauslastung';

  @override
  String get responseTime => 'Antwortzeit';

  @override
  String get toolExecutions => 'Tool-Ausfuehrungen';

  @override
  String get successRate => 'Erfolgsrate';

  @override
  String get recentEvents => 'Letzte Ereignisse';

  @override
  String get lastUpdated => 'Zuletzt aktualisiert';

  @override
  String get systemHealth => 'Systemzustand';

  @override
  String get performance => 'Leistung';

  @override
  String get trends => 'Trends';

  @override
  String get marketplace => 'Marktplatz';

  @override
  String get featured => 'Empfohlen';

  @override
  String get trending => 'Im Trend';

  @override
  String get categories => 'Kategorien';

  @override
  String get searchSkills => 'Faehigkeiten suchen...';

  @override
  String get installed => 'Installiert';

  @override
  String get installSkill => 'Installieren';

  @override
  String get uninstallSkill => 'Deinstallieren';

  @override
  String get installing => 'Wird installiert...';

  @override
  String get skillDetails => 'Details';

  @override
  String get reviews => 'Bewertungen';

  @override
  String get noSkills => 'Keine Faehigkeiten gefunden';

  @override
  String get browseMarketplace => 'Marktplatz durchsuchen';

  @override
  String get verified => 'Verifiziert';

  @override
  String get downloads => 'Downloads';

  @override
  String get rating => 'Bewertung';

  @override
  String get memoryTitle => 'Gedaechtnis';

  @override
  String get knowledgeGraph => 'Wissensgraph';

  @override
  String get entities => 'Entitaeten';

  @override
  String get relations => 'Beziehungen';

  @override
  String get hygiene => 'Hygiene';

  @override
  String get quarantine => 'Quarantaene';

  @override
  String get scanMemory => 'Scannen';

  @override
  String get scanning => 'Wird gescannt...';

  @override
  String get explainability => 'Erklaerbarkeit';

  @override
  String get decisionTrails => 'Entscheidungspfade';

  @override
  String get lowTrust => 'Geringes Vertrauen';

  @override
  String get graphStats => 'Graph-Statistiken';

  @override
  String get noEntities => 'Keine Entitaeten';

  @override
  String get noTrails => 'Keine Pfade';

  @override
  String get scanComplete => 'Scan abgeschlossen';

  @override
  String get threats => 'Bedrohungen';

  @override
  String get threatRate => 'Bedrohungsrate';

  @override
  String get totalScans => 'Scans gesamt';

  @override
  String get integrity => 'Integritaet';

  @override
  String get securityTitle => 'Sicherheit';

  @override
  String get complianceTitle => 'Compliance';

  @override
  String get rolesTitle => 'Rollen';

  @override
  String get permissions => 'Berechtigungen';

  @override
  String get auditLog => 'Audit-Protokoll';

  @override
  String get redTeam => 'Red Team';

  @override
  String get scanStatus => 'Scan-Status';

  @override
  String get complianceReport => 'Compliance-Bericht';

  @override
  String get decisionsTitle => 'Entscheidungen';

  @override
  String get remediations => 'Massnahmen';

  @override
  String get openStatus => 'Offen';

  @override
  String get inProgressStatus => 'In Bearbeitung';

  @override
  String get resolvedStatus => 'Erledigt';

  @override
  String get overdueStatus => 'Ueberfaellig';

  @override
  String get approvalRate => 'Genehmigungsrate';

  @override
  String get flaggedCount => 'Markiert';

  @override
  String get transparency => 'Transparenz';

  @override
  String get euAiAct => 'EU AI Act';

  @override
  String get dsgvo => 'DSGVO';

  @override
  String get runScan => 'Scan starten';

  @override
  String get adminTitle => 'Verwaltung';

  @override
  String get agentsTitle => 'Agenten';

  @override
  String get modelsTitle => 'Modelle';

  @override
  String get systemTitle => 'System';

  @override
  String get workflowsTitle => 'Workflows';

  @override
  String get vaultTitle => 'Tresor';

  @override
  String get credentialsTitle => 'Zugangsdaten';

  @override
  String get bindingsTitle => 'Bindungen';

  @override
  String get connectorsTitle => 'Konnektoren';

  @override
  String get commandsTitle => 'Befehle';

  @override
  String get isolationTitle => 'Isolation';

  @override
  String get sandboxTitle => 'Sandbox';

  @override
  String get circlesTitle => 'Kreise';

  @override
  String get wizardsTitle => 'Assistenten';

  @override
  String get systemStatus => 'Systemstatus';

  @override
  String get shutdownServer => 'Server herunterfahren';

  @override
  String get shutdownConfirm =>
      'Bist du sicher, dass du den Server herunterfahren willst?';

  @override
  String get startComponent => 'Starten';

  @override
  String get stopComponent => 'Stoppen';

  @override
  String get selectTemplate => 'Vorlage auswaehlen';

  @override
  String get workflowStarted => 'Workflow gestartet';

  @override
  String get noWorkflows => 'Keine Workflows';

  @override
  String get templates => 'Vorlagen';

  @override
  String get running => 'Laeuft';

  @override
  String get vaultStats => 'Tresor-Statistiken';

  @override
  String get totalEntries => 'Eintraege gesamt';

  @override
  String get agentVaults => 'Agenten-Tresore';

  @override
  String get noVaults => 'Keine Tresore';

  @override
  String get availableModels => 'Verfuegbare Modelle';

  @override
  String get modelStats => 'Modell-Statistiken';

  @override
  String get providers => 'Anbieter';

  @override
  String get capabilities => 'Faehigkeiten';

  @override
  String get plannerModel => 'Planer';

  @override
  String get executorModel => 'Ausfuehrer';

  @override
  String get coderModel => 'Programmierer';

  @override
  String get embeddingModel => 'Embedding';

  @override
  String get configured => 'Konfiguriert';

  @override
  String get modelWarnings => 'Warnungen';

  @override
  String get identityDreamCycle => 'Traumzyklus';

  @override
  String get identityGenesisAnchors => 'Genesis-Anker';

  @override
  String get identityNoAnchors => 'Keine Genesis-Anker';

  @override
  String get identityPersonality => 'Persoenlichkeit';

  @override
  String get identityCognitive => 'Kognitiver Zustand';

  @override
  String get identityEmotional => 'Emotionaler Zustand';

  @override
  String get identitySomatic => 'Somatischer Zustand';

  @override
  String get identityNarrative => 'Narrativ';

  @override
  String get identityExistential => 'Existenziell';

  @override
  String get identityPredictive => 'Praediktiv';

  @override
  String get identityEpistemic => 'Epistemisch';

  @override
  String get identityBiases => 'Aktive Verzerrungen';

  @override
  String get search => 'Suchen';

  @override
  String get filter => 'Filtern';

  @override
  String get sortBy => 'Sortieren nach';

  @override
  String get refresh => 'Aktualisieren';

  @override
  String get export => 'Exportieren';

  @override
  String get viewAll => 'Alle anzeigen';

  @override
  String get details => 'Details';

  @override
  String get back => 'Zurueck';

  @override
  String get confirm => 'Bestaetigen';

  @override
  String get actions => 'Aktionen';

  @override
  String get statusLabel => 'Status';

  @override
  String get enabled => 'Aktiviert';

  @override
  String get disabled => 'Deaktiviert';

  @override
  String get total => 'Gesamt';

  @override
  String get count => 'Anzahl';

  @override
  String get rate => 'Rate';

  @override
  String get average => 'Durchschnitt';

  @override
  String get duration => 'Dauer';

  @override
  String get timestamp => 'Zeitstempel';

  @override
  String get severityLabel => 'Schweregrad';

  @override
  String get critical => 'Kritisch';

  @override
  String get errorLabel => 'Fehler';

  @override
  String get warningLabel => 'Warnung';

  @override
  String get infoLabel => 'Info';

  @override
  String get successLabel => 'Erfolg';

  @override
  String get unknownLabel => 'Unbekannt';

  @override
  String get notConfigured => 'Nicht konfiguriert';

  @override
  String get comingSoon => 'Kommt bald';

  @override
  String get beta => 'Beta';

  @override
  String get copyToClipboard => 'In Zwischenablage kopieren';

  @override
  String get copied => 'Kopiert!';

  @override
  String get chatSettings => 'Chat-Einstellungen';

  @override
  String get clearChat => 'Chat loeschen';

  @override
  String get voiceMode => 'Sprachmodus';

  @override
  String get fileUploadAction => 'Datei hochladen';

  @override
  String get planDetails => 'Plandetails';

  @override
  String get noMessages => 'Noch keine Nachrichten';

  @override
  String get typeMessage => 'Nachricht eingeben...';

  @override
  String get settingsTitle => 'Einstellungen';

  @override
  String get language => 'Sprache';

  @override
  String get theme => 'Design';

  @override
  String get about => 'Ueber';

  @override
  String get licenses => 'Lizenzen';

  @override
  String get clearCache => 'Cache leeren';

  @override
  String get adminConfigSubtitle => 'Konfiguration verwalten';

  @override
  String get adminAgentsSubtitle => 'Agenten & Profile';

  @override
  String get adminModelsSubtitle => 'LLM-Modelle';

  @override
  String get adminSecuritySubtitle => 'Sicherheit & Compliance';

  @override
  String get adminWorkflowsSubtitle => 'Automatisierungen';

  @override
  String get adminMemorySubtitle => 'Wissensgraph';

  @override
  String get adminVaultSubtitle => 'Geheimnisse & Schluessel';

  @override
  String get adminSystemSubtitle => 'Systemstatus';

  @override
  String get dashboardRefreshing => 'Auto-Aktualisierung: 15s';

  @override
  String get backendVersion => 'Backend-Version';

  @override
  String get modelInfo => 'Modell-Info';

  @override
  String get confidence => 'Konfidenz';

  @override
  String get rolesAccess => 'Rollen & Zugriff';

  @override
  String get loadMore => 'Mehr laden';

  @override
  String get actor => 'Akteur';

  @override
  String get noAuditEntries => 'Keine Audit-Eintraege';

  @override
  String get allSeverities => 'Alle Schweregrade';

  @override
  String get allActions => 'Alle Aktionen';

  @override
  String get scanNotAvailable => 'Scan nicht verfuegbar';

  @override
  String get lastScan => 'Letzter Scan';

  @override
  String get scanResults => 'Scan-Ergebnisse';

  @override
  String get compliant => 'Konform';

  @override
  String get nonCompliant => 'Nicht konform';

  @override
  String get model => 'Modell';

  @override
  String get temperature => 'Temperatur';

  @override
  String get priority => 'Prioritaet';

  @override
  String get allowedTools => 'Erlaubte Tools';

  @override
  String get blockedTools => 'Blockierte Tools';

  @override
  String get noAgents => 'Keine Agenten konfiguriert';

  @override
  String get description => 'Beschreibung';

  @override
  String get provider => 'Anbieter';

  @override
  String get noModels => 'Keine Modelle verfuegbar';

  @override
  String get owner => 'Besitzer';

  @override
  String get llmBackend => 'LLM-Backend';

  @override
  String get components => 'Komponenten';

  @override
  String get dangerZone => 'Gefahrenzone';

  @override
  String get reloadConfig => 'Konfiguration neu laden';

  @override
  String get runtimeInfo => 'Laufzeitinfo';

  @override
  String get startWorkflow => 'Workflow starten';

  @override
  String get noCategories => 'Keine Kategorien';

  @override
  String templateCount(String count) {
    return '$count Vorlagen';
  }

  @override
  String get entityTypes => 'Entitaetstypen';

  @override
  String get activeTrails => 'Aktive Pfade';

  @override
  String get completedTrails => 'Abgeschlossen';

  @override
  String get lastAccessed => 'Letzter Zugriff';

  @override
  String get author => 'Autor';

  @override
  String get noQuarantine => 'Keine isolierten Eintraege';

  @override
  String get totalVaults => 'Tresore gesamt';

  @override
  String get scanNow => 'Jetzt scannen';

  @override
  String get startConversation => 'Starte eine Unterhaltung';

  @override
  String get attachFile => 'Datei anhaengen';

  @override
  String get voiceModeHint => 'Sprachmodus kommt bald';

  @override
  String get canvasLabel => 'Canvas';

  @override
  String get configGeneral => 'Allgemein';

  @override
  String get configLanguage => 'Sprache';

  @override
  String get configProviders => 'Anbieter';

  @override
  String get configModels => 'Modelle';

  @override
  String get configPlanner => 'Planer';

  @override
  String get configExecutor => 'Ausfuehrer';

  @override
  String get configMemory => 'Gedaechtnis';

  @override
  String get configChannels => 'Kanaele';

  @override
  String get configSecurity => 'Sicherheit';

  @override
  String get configWeb => 'Web';

  @override
  String get configMcp => 'MCP';

  @override
  String get configCron => 'Cron';

  @override
  String get configDatabase => 'Datenbank';

  @override
  String get configLogging => 'Protokollierung';

  @override
  String get configPrompts => 'Prompts';

  @override
  String get configAgents => 'Agenten';

  @override
  String get configBindings => 'Bindungen';

  @override
  String get configSystem => 'System';

  @override
  String get ownerName => 'Besitzername';

  @override
  String get operationMode => 'Betriebsmodus';

  @override
  String get costTracking => 'Kostenverfolgung';

  @override
  String get dailyBudget => 'Tagesbudget';

  @override
  String get monthlyBudget => 'Monatsbudget';

  @override
  String get apiKey => 'API-Schluessel';

  @override
  String get baseUrl => 'Basis-URL';

  @override
  String get maxTokens => 'Max Tokens';

  @override
  String get timeout => 'Zeitlimit';

  @override
  String get keepAlive => 'Keep Alive';

  @override
  String get contextWindow => 'Kontextfenster';

  @override
  String get vramGb => 'VRAM (GB)';

  @override
  String get topP => 'Top P';

  @override
  String get maxIterations => 'Max Iterationen';

  @override
  String get escalationAfter => 'Eskalation nach';

  @override
  String get responseBudget => 'Antwort-Token-Budget';

  @override
  String get policiesDir => 'Richtlinienverzeichnis';

  @override
  String get defaultRiskLevel => 'Standard-Risikostufe';

  @override
  String get maxBlockedRetries => 'Max blockierte Versuche';

  @override
  String get sandboxLevel => 'Sandbox-Stufe';

  @override
  String get maxMemoryMb => 'Max Speicher (MB)';

  @override
  String get maxCpuSeconds => 'Max CPU-Sekunden';

  @override
  String get allowedPaths => 'Erlaubte Pfade';

  @override
  String get networkAccess => 'Netzwerkzugriff';

  @override
  String get envVars => 'Umgebungsvariablen';

  @override
  String get defaultTimeout => 'Standard-Zeitlimit';

  @override
  String get maxOutputChars => 'Max Ausgabezeichen';

  @override
  String get maxRetries => 'Max Versuche';

  @override
  String get backoffDelay => 'Backoff-Verzoegerung';

  @override
  String get maxParallelTools => 'Max parallele Tools';

  @override
  String get chunkSize => 'Chunk-Groesse';

  @override
  String get chunkOverlap => 'Chunk-Ueberlappung';

  @override
  String get searchTopK => 'Suche Top K';

  @override
  String get searchWeights => 'Suchgewichte';

  @override
  String get vectorWeight => 'Vektor-Gewicht';

  @override
  String get bm25Weight => 'BM25-Gewicht';

  @override
  String get graphWeight => 'Graph-Gewicht';

  @override
  String get recencyHalfLife => 'Aktualitaets-Halbwertszeit';

  @override
  String get compactionThreshold => 'Kompaktierungsschwelle';

  @override
  String get compactionKeepLast => 'Kompaktierung letzte behalten';

  @override
  String get episodicRetention => 'Episodische Aufbewahrung';

  @override
  String get dynamicWeighting => 'Dynamische Gewichtung';

  @override
  String get voiceEnabled => 'Sprache aktiviert';

  @override
  String get ttsBackend => 'TTS-Backend';

  @override
  String get piperVoice => 'Piper-Stimme';

  @override
  String get piperLengthScale => 'Piper-Laengenskala';

  @override
  String get wakeWordEnabled => 'Aktivierungswort aktiviert';

  @override
  String get wakeWord => 'Aktivierungswort';

  @override
  String get wakeWordBackend => 'Aktivierungswort-Backend';

  @override
  String get talkMode => 'Sprechmodus';

  @override
  String get autoListen => 'Auto-Zuhoeren';

  @override
  String get blockedCommands => 'Blockierte Befehle';

  @override
  String get credentialPatterns => 'Zugangsmuster';

  @override
  String get maxSubAgentDepth => 'Max Sub-Agent-Tiefe';

  @override
  String get searchBackends => 'Such-Backends';

  @override
  String get domainFilters => 'Domain-Filter';

  @override
  String get blocklist => 'Sperrliste';

  @override
  String get allowlist => 'Erlaubnisliste';

  @override
  String get httpLimits => 'HTTP-Limits';

  @override
  String get maxFetchBytes => 'Max Abruf-Bytes';

  @override
  String get maxTextChars => 'Max Textzeichen';

  @override
  String get fetchTimeout => 'Abruf-Zeitlimit';

  @override
  String get searchTimeout => 'Such-Zeitlimit';

  @override
  String get maxSearchResults => 'Max Suchergebnisse';

  @override
  String get rateLimit => 'Ratenlimit';

  @override
  String get mcpServers => 'MCP-Server';

  @override
  String get a2aProtocol => 'A2A-Protokoll';

  @override
  String get remotes => 'Remotes';

  @override
  String get heartbeat => 'Heartbeat';

  @override
  String get intervalMinutes => 'Intervall (Minuten)';

  @override
  String get checklistFile => 'Checklisten-Datei';

  @override
  String get channel => 'Kanal';

  @override
  String get plugins => 'Plugins';

  @override
  String get skillsDir => 'Skills-Verzeichnis';

  @override
  String get autoUpdate => 'Auto-Update';

  @override
  String get cronJobs => 'Cron-Jobs';

  @override
  String get schedule => 'Zeitplan';

  @override
  String get command => 'Befehl';

  @override
  String get databaseBackend => 'Datenbank-Backend';

  @override
  String get encryption => 'Verschluesselung';

  @override
  String get pgHost => 'Host';

  @override
  String get pgPort => 'Port';

  @override
  String get pgDbName => 'Datenbankname';

  @override
  String get pgUser => 'Benutzer';

  @override
  String get pgPassword => 'Passwort';

  @override
  String get pgPoolMin => 'Pool Min';

  @override
  String get pgPoolMax => 'Pool Max';

  @override
  String get logLevel => 'Log-Stufe';

  @override
  String get jsonLogs => 'JSON-Logs';

  @override
  String get consoleOutput => 'Konsolenausgabe';

  @override
  String get systemPrompt => 'System-Prompt';

  @override
  String get replanPrompt => 'Replan-Prompt';

  @override
  String get escalationPrompt => 'Eskalations-Prompt';

  @override
  String get policyYaml => 'Richtlinien-YAML';

  @override
  String get heartbeatMd => 'Heartbeat-Checkliste';

  @override
  String get personalityPrompt => 'Persoenlichkeits-Prompt';

  @override
  String get promptEvolution => 'Prompt-Evolution';

  @override
  String get resetToDefault => 'Auf Standard zuruecksetzen';

  @override
  String get triggerPatterns => 'Trigger-Muster';

  @override
  String get channelFilter => 'Kanal-Filter';

  @override
  String get pattern => 'Muster';

  @override
  String get targetAgent => 'Ziel-Agent';

  @override
  String get restartBackend => 'Backend neustarten';

  @override
  String get exportConfig => 'Konfiguration exportieren';

  @override
  String get importConfig => 'Konfiguration importieren';

  @override
  String get factoryReset => 'Werkseinstellungen';

  @override
  String get factoryResetConfirm =>
      'Alle Einstellungen auf Werkseinstellungen zuruecksetzen. Fortfahren?';

  @override
  String get configurationSaved => 'Konfiguration gespeichert';

  @override
  String get saveHadErrors => 'Speichern hatte Fehler';

  @override
  String get unsavedChanges => 'Ungespeicherte Aenderungen';

  @override
  String get discard => 'Verwerfen';

  @override
  String get saving => 'Wird gespeichert...';

  @override
  String get voiceOff => 'Aus';

  @override
  String get voiceListening => 'Hoere zu...';

  @override
  String get voiceSpeakNow => 'Jetzt sprechen';

  @override
  String get voiceProcessing => 'Verarbeite...';

  @override
  String get voiceSpeaking => 'Spricht...';

  @override
  String get observe => 'Beobachten';

  @override
  String get agentLog => 'Agent-Protokoll';

  @override
  String get kanban => 'Kanban';

  @override
  String get dag => 'DAG';

  @override
  String get plan => 'Plan';

  @override
  String get toDo => 'Zu erledigen';

  @override
  String get inProgress => 'In Bearbeitung';

  @override
  String get verifying => 'Pruefen';

  @override
  String get done => 'Erledigt';

  @override
  String get searchConfigPages => 'Konfigurationsseiten suchen...';

  @override
  String get noMatchingPages => 'Keine passenden Seiten';

  @override
  String get knowledgeGraphTitle => 'Wissensgraph';

  @override
  String get searchEntities => 'Entitaeten suchen...';

  @override
  String get allTypes => 'Alle Typen';

  @override
  String get entityDetail => 'Entitaets-Detail';

  @override
  String get attributes => 'Attribute';

  @override
  String get instances => 'Instanzen';

  @override
  String get dagRuns => 'DAG-Laeufe';

  @override
  String get noInstances => 'Keine Instanzen';

  @override
  String get noDagRuns => 'Keine DAG-Laeufe';

  @override
  String get addCredential => 'Zugangsdaten hinzufuegen';

  @override
  String get service => 'Dienst';

  @override
  String get key => 'Schluessel';

  @override
  String get value => 'Wert';

  @override
  String get noCredentials => 'Keine Zugangsdaten';

  @override
  String get lightMode => 'Heller Modus';

  @override
  String get darkMode => 'Dunkler Modus';

  @override
  String get globalSearch => 'Suche (Strg+K)';

  @override
  String get configPageGeneral => 'Allgemein';

  @override
  String get configPageLanguage => 'Sprache';

  @override
  String get configPageProviders => 'Anbieter';

  @override
  String get configPageModels => 'Modelle';

  @override
  String get configPagePlanner => 'Planer';

  @override
  String get configPageExecutor => 'Ausfuehrer';

  @override
  String get configPageMemory => 'Gedaechtnis';

  @override
  String get configPageChannels => 'Kanaele';

  @override
  String get configPageSecurity => 'Sicherheit';

  @override
  String get configPageWeb => 'Web';

  @override
  String get configPageMcp => 'MCP';

  @override
  String get configPageCron => 'Zeitplanung';

  @override
  String get configPageDatabase => 'Datenbank';

  @override
  String get configPageLogging => 'Protokollierung';

  @override
  String get configPagePrompts => 'Prompts';

  @override
  String get configPageAgents => 'Agenten';

  @override
  String get configPageBindings => 'Bindungen';

  @override
  String get configPageSystem => 'System';

  @override
  String get configTitle => 'Konfiguration';

  @override
  String get reloadFromBackend => 'Konfiguration vom Backend neu laden';

  @override
  String get saveCtrlS => 'Speichern (Strg+S)';

  @override
  String savedWithErrors(String sections) {
    return 'Mit Fehlern gespeichert in: $sections';
  }

  @override
  String get saveFailed => 'Speichern fehlgeschlagen';

  @override
  String get fieldOwnerName => 'Besitzername';

  @override
  String get fieldOperationMode => 'Betriebsmodus';

  @override
  String get fieldCostTracking => 'Kostenverfolgung';

  @override
  String get fieldDailyBudget => 'Tagesbudget (USD)';

  @override
  String get fieldMonthlyBudget => 'Monatsbudget (USD)';

  @override
  String get fieldLlmBackend => 'LLM-Backend';

  @override
  String get fieldPrimaryProvider => 'Primaerer LLM-Anbieter';

  @override
  String get fieldApiKey => 'API-Schluessel';

  @override
  String get fieldBaseUrl => 'Basis-URL';

  @override
  String get fieldModelName => 'Modellname';

  @override
  String get fieldContextWindow => 'Kontextfenster';

  @override
  String get fieldTemperature => 'Temperatur';

  @override
  String get fieldMaxIterations => 'Max Iterationen';

  @override
  String get fieldEnabled => 'Aktiviert';

  @override
  String get fieldPort => 'Port';

  @override
  String get fieldHost => 'Host';

  @override
  String get fieldPassword => 'Passwort';

  @override
  String get fieldUser => 'Benutzer';

  @override
  String get fieldTimeout => 'Zeitlimit';

  @override
  String get fieldLevel => 'Stufe';

  @override
  String get sectionSearchBackends => 'Such-Backends';

  @override
  String get sectionDomainFilters => 'Domain-Filter';

  @override
  String get sectionFetchLimits => 'Abruf-Limits';

  @override
  String get sectionSearchLimits => 'Such-Limits';

  @override
  String get sectionHttpLimits => 'HTTP-Anfrage-Limits';

  @override
  String get sectionVoice => 'Sprache';

  @override
  String get sectionHeartbeat => 'Heartbeat';

  @override
  String get sectionPlugins => 'Plugins';

  @override
  String get sectionCronJobs => 'Zeitgesteuerte Aufgaben';

  @override
  String get sectionPromptEvolution => 'Prompt-Evolution';

  @override
  String get addItem => 'Hinzufuegen';

  @override
  String get removeItem => 'Entfernen';

  @override
  String get translatePrompts => 'Prompts ueber Ollama uebersetzen';

  @override
  String get translating => 'Wird uebersetzt...';

  @override
  String get promptsTranslated => 'Prompts uebersetzt';

  @override
  String get copiedToClipboard => 'Konfiguration in Zwischenablage kopiert';

  @override
  String get configImported => 'Konfiguration importiert';

  @override
  String get restartInitiated => 'Neustart eingeleitet';

  @override
  String get factoryResetComplete => 'Werkseinstellungen wiederhergestellt';

  @override
  String get factoryResetConfirmMsg =>
      'Alle Einstellungen auf Werkseinstellungen zuruecksetzen. Fortfahren?';

  @override
  String get languageEnglish => 'Englisch';

  @override
  String get languageGerman => 'Deutsch';

  @override
  String get languageChinese => 'Chinesisch';

  @override
  String get languageArabic => 'Arabisch';

  @override
  String get uiAndPromptLanguage => 'Oberflaeche und Prompt-Sprache';

  @override
  String get learningTitle => 'Lernen';

  @override
  String get knowledgeGaps => 'Wissensluecken';

  @override
  String get explorationQueue => 'Erkundungswarteschlange';

  @override
  String get filesProcessed => 'Dateien verarbeitet';

  @override
  String get entitiesCreated => 'Entitaeten erstellt';

  @override
  String get confidenceUpdates => 'Konfidenz-Updates';

  @override
  String get openGaps => 'Offene Luecken';

  @override
  String get importance => 'Wichtigkeit';

  @override
  String get curiosity => 'Neugier';

  @override
  String get explore => 'Erkunden';

  @override
  String get dismiss => 'Verwerfen';

  @override
  String get noGaps => 'Keine Wissensluecken erkannt';

  @override
  String get noTasks => 'Keine Erkundungsaufgaben';

  @override
  String get confidenceHistory => 'Konfidenz-Verlauf';

  @override
  String get feedback => 'Feedback';

  @override
  String get positive => 'Positiv';

  @override
  String get negative => 'Negativ';

  @override
  String get correction => 'Korrektur';

  @override
  String get adminLearningSubtitle => 'Aktives Lernen & Neugier';

  @override
  String get watchDirectories => 'Ueberwachte Verzeichnisse';

  @override
  String get directoryExists => 'Verzeichnis vorhanden';

  @override
  String get directoryMissing => 'Verzeichnis nicht gefunden';

  @override
  String get qaKnowledgeBase => 'Wissen';

  @override
  String get lineage => 'Herkunft';

  @override
  String get question => 'Frage';

  @override
  String get answer => 'Antwort';

  @override
  String get topic => 'Thema';

  @override
  String get addQA => 'Wissen hinzufuegen';

  @override
  String get verify => 'Verifizieren';

  @override
  String get source => 'Quelle';

  @override
  String get noQAPairs => 'Keine Wissenseintraege';

  @override
  String get noLineage => 'Keine Herkunftsdaten';

  @override
  String get entityLineage => 'Entitaets-Herkunft';

  @override
  String get recentChanges => 'Letzte Aenderungen';

  @override
  String get created => 'Erstellt';

  @override
  String get updated => 'Aktualisiert';

  @override
  String get decayed => 'Verfallen';

  @override
  String get runExploration => 'Erkundung starten';

  @override
  String get explorationComplete => 'Erkundung abgeschlossen';

  @override
  String get activityChart => 'Aktivitaet';

  @override
  String get stopped => 'Gestoppt';

  @override
  String get requestsOverTime => 'Anfragen im Zeitverlauf';

  @override
  String get teachCognithor => 'Cognithor beibringen';

  @override
  String get uploadFile => 'Datei hochladen';

  @override
  String get learnFromUrl => 'Von Website lernen';

  @override
  String get learnFromYoutube => 'Von Video lernen';

  @override
  String get dropFilesHere => 'Dateien hier ablegen oder durchsuchen';

  @override
  String get learningHistory => 'Lernverlauf';

  @override
  String chunksLearned(String count) {
    return '$count Abschnitte gelernt';
  }

  @override
  String get processingContent => 'Inhalt wird verarbeitet...';

  @override
  String get learnSuccess => 'Erfolgreich gelernt!';

  @override
  String get learnFailed => 'Lernen fehlgeschlagen';

  @override
  String get enterUrl => 'Website-URL eingeben...';

  @override
  String get enterYoutubeUrl => 'YouTube-URL eingeben...';

  @override
  String get adminTeachSubtitle => 'Dateien, URLs, Videos hochladen';

  @override
  String get newSkill => 'Neuer Skill';

  @override
  String get editSkill => 'Skill bearbeiten';

  @override
  String get createSkill => 'Skill erstellen';

  @override
  String get deleteSkill => 'Skill loeschen';

  @override
  String get skillName => 'Name';

  @override
  String get skillBody => 'Skill-Inhalt (Markdown)';

  @override
  String get triggerKeywords => 'Trigger-Schluesselwoerter';

  @override
  String get requiredTools => 'Benoetigte Tools';

  @override
  String get modelPreference => 'Modell-Praeferenz';

  @override
  String get skillSaved => 'Skill erfolgreich gespeichert';

  @override
  String get skillCreated => 'Skill erfolgreich erstellt';

  @override
  String get skillDeleted => 'Skill geloescht';

  @override
  String get confirmDeleteSkill =>
      'Bist du sicher, dass du diesen Skill loeschen moechtest? Dies kann nicht rueckgaengig gemacht werden.';

  @override
  String get discardChanges => 'Aenderungen verwerfen?';

  @override
  String get discardChangesBody =>
      'Du hast ungespeicherte Aenderungen. Verwerfen?';

  @override
  String get totalUses => 'Gesamtnutzungen';

  @override
  String get lastUsed => 'Zuletzt verwendet';

  @override
  String get commaSeparated => 'Komma-getrennt';

  @override
  String get skillBodyHint => 'Skill-Anweisungen in Markdown schreiben...';

  @override
  String get metadata => 'Metadaten';

  @override
  String get statistics => 'Statistiken';

  @override
  String get builtInSkill => 'Eingebauter Skill (nur lesen)';

  @override
  String get exportSkillMd => 'Als SKILL.md exportieren';

  @override
  String get skillExported => 'Skill in Zwischenablage exportiert';

  @override
  String get general => 'Allgemein';

  @override
  String get productivity => 'Produktivitaet';

  @override
  String get research => 'Recherche';

  @override
  String get analysis => 'Analyse';

  @override
  String get development => 'Entwicklung';

  @override
  String get automation => 'Automatisierung';

  @override
  String get newAgent => 'Neuer Agent';

  @override
  String get editAgent => 'Agent bearbeiten';

  @override
  String get deleteAgent => 'Agent loeschen';

  @override
  String get confirmDeleteAgent =>
      'Bist du sicher, dass du diesen Agenten loeschen moechtest? Dies kann nicht rueckgaengig gemacht werden.';

  @override
  String get agentCreated => 'Agent erfolgreich erstellt';

  @override
  String get agentSaved => 'Agent erfolgreich gespeichert';

  @override
  String get agentDeleted => 'Agent geloescht';

  @override
  String get displayName => 'Anzeigename';

  @override
  String get preferredModel => 'Bevorzugtes Modell';

  @override
  String get sandboxTimeout => 'Sandbox Timeout (s)';

  @override
  String get sandboxNetwork => 'Sandbox Netzwerk';

  @override
  String get canDelegateTo => 'Kann delegieren an';

  @override
  String get cannotDeleteDefault =>
      'Standard-Agent kann nicht geloescht werden';
}
