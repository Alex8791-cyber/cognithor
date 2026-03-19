import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/intl.dart' as intl;

import 'app_localizations_ar.dart';
import 'app_localizations_de.dart';
import 'app_localizations_en.dart';
import 'app_localizations_zh.dart';

// ignore_for_file: type=lint

/// Callers can lookup localized strings with an instance of AppLocalizations
/// returned by `AppLocalizations.of(context)`.
///
/// Applications need to include `AppLocalizations.delegate()` in their app's
/// `localizationDelegates` list, and the locales they support in the app's
/// `supportedLocales` list. For example:
///
/// ```dart
/// import 'generated/app_localizations.dart';
///
/// return MaterialApp(
///   localizationsDelegates: AppLocalizations.localizationsDelegates,
///   supportedLocales: AppLocalizations.supportedLocales,
///   home: MyApplicationHome(),
/// );
/// ```
///
/// ## Update pubspec.yaml
///
/// Please make sure to update your pubspec.yaml to include the following
/// packages:
///
/// ```yaml
/// dependencies:
///   # Internationalization support.
///   flutter_localizations:
///     sdk: flutter
///   intl: any # Use the pinned version from flutter_localizations
///
///   # Rest of dependencies
/// ```
///
/// ## iOS Applications
///
/// iOS applications define key application metadata, including supported
/// locales, in an Info.plist file that is built into the application bundle.
/// To configure the locales supported by your app, you’ll need to edit this
/// file.
///
/// First, open your project’s ios/Runner.xcworkspace Xcode workspace file.
/// Then, in the Project Navigator, open the Info.plist file under the Runner
/// project’s Runner folder.
///
/// Next, select the Information Property List item, select Add Item from the
/// Editor menu, then select Localizations from the pop-up menu.
///
/// Select and expand the newly-created Localizations item then, for each
/// locale your application supports, add a new item and select the locale
/// you wish to add from the pop-up menu in the Value field. This list should
/// be consistent with the languages listed in the AppLocalizations.supportedLocales
/// property.
abstract class AppLocalizations {
  AppLocalizations(String locale)
    : localeName = intl.Intl.canonicalizedLocale(locale.toString());

  final String localeName;

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations)!;
  }

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  /// A list of this localizations delegate along with the default localizations
  /// delegates.
  ///
  /// Returns a list of localizations delegates containing this delegate along with
  /// GlobalMaterialLocalizations.delegate, GlobalCupertinoLocalizations.delegate,
  /// and GlobalWidgetsLocalizations.delegate.
  ///
  /// Additional delegates can be added by appending to this list in
  /// MaterialApp. This list does not have to be used at all if a custom list
  /// of delegates is preferred or required.
  static const List<LocalizationsDelegate<dynamic>> localizationsDelegates =
      <LocalizationsDelegate<dynamic>>[
        delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
      ];

  /// A list of this localizations delegate's supported locales.
  static const List<Locale> supportedLocales = <Locale>[
    Locale('ar'),
    Locale('de'),
    Locale('en'),
    Locale('zh'),
  ];

  /// No description provided for @appTitle.
  ///
  /// In en, this message translates to:
  /// **'Jarvis'**
  String get appTitle;

  /// No description provided for @chat.
  ///
  /// In en, this message translates to:
  /// **'Chat'**
  String get chat;

  /// No description provided for @settings.
  ///
  /// In en, this message translates to:
  /// **'Settings'**
  String get settings;

  /// No description provided for @identity.
  ///
  /// In en, this message translates to:
  /// **'Identity'**
  String get identity;

  /// No description provided for @workflows.
  ///
  /// In en, this message translates to:
  /// **'Workflows'**
  String get workflows;

  /// No description provided for @memory.
  ///
  /// In en, this message translates to:
  /// **'Memory'**
  String get memory;

  /// No description provided for @monitoring.
  ///
  /// In en, this message translates to:
  /// **'Monitoring'**
  String get monitoring;

  /// No description provided for @skills.
  ///
  /// In en, this message translates to:
  /// **'Skills'**
  String get skills;

  /// No description provided for @config.
  ///
  /// In en, this message translates to:
  /// **'Configuration'**
  String get config;

  /// No description provided for @sendMessage.
  ///
  /// In en, this message translates to:
  /// **'Send a message...'**
  String get sendMessage;

  /// No description provided for @send.
  ///
  /// In en, this message translates to:
  /// **'Send'**
  String get send;

  /// No description provided for @cancel.
  ///
  /// In en, this message translates to:
  /// **'Cancel'**
  String get cancel;

  /// No description provided for @approve.
  ///
  /// In en, this message translates to:
  /// **'Approve'**
  String get approve;

  /// No description provided for @reject.
  ///
  /// In en, this message translates to:
  /// **'Reject'**
  String get reject;

  /// No description provided for @retry.
  ///
  /// In en, this message translates to:
  /// **'Retry'**
  String get retry;

  /// No description provided for @close.
  ///
  /// In en, this message translates to:
  /// **'Close'**
  String get close;

  /// No description provided for @save.
  ///
  /// In en, this message translates to:
  /// **'Save'**
  String get save;

  /// No description provided for @delete.
  ///
  /// In en, this message translates to:
  /// **'Delete'**
  String get delete;

  /// No description provided for @loading.
  ///
  /// In en, this message translates to:
  /// **'Loading...'**
  String get loading;

  /// No description provided for @connecting.
  ///
  /// In en, this message translates to:
  /// **'Connecting...'**
  String get connecting;

  /// No description provided for @approvalTitle.
  ///
  /// In en, this message translates to:
  /// **'Approval Required'**
  String get approvalTitle;

  /// No description provided for @approvalBody.
  ///
  /// In en, this message translates to:
  /// **'The tool {tool} wants to execute:'**
  String approvalBody(String tool);

  /// No description provided for @approvalReason.
  ///
  /// In en, this message translates to:
  /// **'Reason: {reason}'**
  String approvalReason(String reason);

  /// No description provided for @statusThinking.
  ///
  /// In en, this message translates to:
  /// **'Thinking...'**
  String get statusThinking;

  /// No description provided for @statusExecuting.
  ///
  /// In en, this message translates to:
  /// **'Executing...'**
  String get statusExecuting;

  /// No description provided for @statusFinishing.
  ///
  /// In en, this message translates to:
  /// **'Finishing...'**
  String get statusFinishing;

  /// No description provided for @voiceMessage.
  ///
  /// In en, this message translates to:
  /// **'Voice message'**
  String get voiceMessage;

  /// No description provided for @fileUpload.
  ///
  /// In en, this message translates to:
  /// **'File: {name}'**
  String fileUpload(String name);

  /// No description provided for @connectionError.
  ///
  /// In en, this message translates to:
  /// **'Cannot reach backend'**
  String get connectionError;

  /// No description provided for @connectionErrorDetail.
  ///
  /// In en, this message translates to:
  /// **'Check that the Jarvis backend is running at {url}'**
  String connectionErrorDetail(String url);

  /// No description provided for @authFailed.
  ///
  /// In en, this message translates to:
  /// **'Authentication failed'**
  String get authFailed;

  /// No description provided for @tokenExpired.
  ///
  /// In en, this message translates to:
  /// **'Session expired. Reconnecting...'**
  String get tokenExpired;

  /// No description provided for @serverUrl.
  ///
  /// In en, this message translates to:
  /// **'Server URL'**
  String get serverUrl;

  /// No description provided for @serverUrlHint.
  ///
  /// In en, this message translates to:
  /// **'http://localhost:8741'**
  String get serverUrlHint;

  /// No description provided for @version.
  ///
  /// In en, this message translates to:
  /// **'Version {version}'**
  String version(String version);

  /// No description provided for @errorGeneric.
  ///
  /// In en, this message translates to:
  /// **'Something went wrong'**
  String get errorGeneric;

  /// No description provided for @errorNetwork.
  ///
  /// In en, this message translates to:
  /// **'Network error. Check your connection.'**
  String get errorNetwork;

  /// No description provided for @errorTimeout.
  ///
  /// In en, this message translates to:
  /// **'Request timed out'**
  String get errorTimeout;

  /// No description provided for @errorUnauthorized.
  ///
  /// In en, this message translates to:
  /// **'Unauthorized. Please reconnect.'**
  String get errorUnauthorized;

  /// No description provided for @errorServerDown.
  ///
  /// In en, this message translates to:
  /// **'Backend is not reachable'**
  String get errorServerDown;

  /// No description provided for @identityNotAvailable.
  ///
  /// In en, this message translates to:
  /// **'Identity Layer not available'**
  String get identityNotAvailable;

  /// No description provided for @identityInstallHint.
  ///
  /// In en, this message translates to:
  /// **'Install with: pip install cognithor[identity]'**
  String get identityInstallHint;

  /// No description provided for @identityEnergy.
  ///
  /// In en, this message translates to:
  /// **'Energy'**
  String get identityEnergy;

  /// No description provided for @identityInteractions.
  ///
  /// In en, this message translates to:
  /// **'Interactions'**
  String get identityInteractions;

  /// No description provided for @identityMemories.
  ///
  /// In en, this message translates to:
  /// **'Memories'**
  String get identityMemories;

  /// No description provided for @identityCharacterStrength.
  ///
  /// In en, this message translates to:
  /// **'Character Strength'**
  String get identityCharacterStrength;

  /// No description provided for @identityFrozen.
  ///
  /// In en, this message translates to:
  /// **'Frozen'**
  String get identityFrozen;

  /// No description provided for @identityActive.
  ///
  /// In en, this message translates to:
  /// **'Active'**
  String get identityActive;

  /// No description provided for @identityDream.
  ///
  /// In en, this message translates to:
  /// **'Dream Cycle'**
  String get identityDream;

  /// No description provided for @identityFreeze.
  ///
  /// In en, this message translates to:
  /// **'Freeze'**
  String get identityFreeze;

  /// No description provided for @identityUnfreeze.
  ///
  /// In en, this message translates to:
  /// **'Unfreeze'**
  String get identityUnfreeze;

  /// No description provided for @identityReset.
  ///
  /// In en, this message translates to:
  /// **'Soft Reset'**
  String get identityReset;

  /// No description provided for @identityResetConfirm.
  ///
  /// In en, this message translates to:
  /// **'Reset identity? Memories will be lost.'**
  String get identityResetConfirm;

  /// No description provided for @pipelinePlan.
  ///
  /// In en, this message translates to:
  /// **'Planning'**
  String get pipelinePlan;

  /// No description provided for @pipelineGate.
  ///
  /// In en, this message translates to:
  /// **'Gatekeeper'**
  String get pipelineGate;

  /// No description provided for @pipelineExecute.
  ///
  /// In en, this message translates to:
  /// **'Executing'**
  String get pipelineExecute;

  /// No description provided for @pipelineReplan.
  ///
  /// In en, this message translates to:
  /// **'Replanning'**
  String get pipelineReplan;

  /// No description provided for @pipelineComplete.
  ///
  /// In en, this message translates to:
  /// **'Complete'**
  String get pipelineComplete;

  /// No description provided for @canvasTitle.
  ///
  /// In en, this message translates to:
  /// **'Canvas'**
  String get canvasTitle;

  /// No description provided for @canvasClose.
  ///
  /// In en, this message translates to:
  /// **'Close canvas'**
  String get canvasClose;

  /// No description provided for @models.
  ///
  /// In en, this message translates to:
  /// **'Models'**
  String get models;

  /// No description provided for @channels.
  ///
  /// In en, this message translates to:
  /// **'Channels'**
  String get channels;

  /// No description provided for @security.
  ///
  /// In en, this message translates to:
  /// **'Security'**
  String get security;

  /// No description provided for @reload.
  ///
  /// In en, this message translates to:
  /// **'Reload'**
  String get reload;

  /// No description provided for @reloading.
  ///
  /// In en, this message translates to:
  /// **'Reloading...'**
  String get reloading;

  /// No description provided for @configSaved.
  ///
  /// In en, this message translates to:
  /// **'Configuration reloaded'**
  String get configSaved;

  /// No description provided for @configError.
  ///
  /// In en, this message translates to:
  /// **'Configuration error'**
  String get configError;

  /// No description provided for @uptime.
  ///
  /// In en, this message translates to:
  /// **'Uptime'**
  String get uptime;

  /// No description provided for @activeSessions.
  ///
  /// In en, this message translates to:
  /// **'Active Sessions'**
  String get activeSessions;

  /// No description provided for @totalRequests.
  ///
  /// In en, this message translates to:
  /// **'Total Requests'**
  String get totalRequests;

  /// No description provided for @events.
  ///
  /// In en, this message translates to:
  /// **'Events'**
  String get events;

  /// No description provided for @noEvents.
  ///
  /// In en, this message translates to:
  /// **'No events recorded'**
  String get noEvents;

  /// No description provided for @severity.
  ///
  /// In en, this message translates to:
  /// **'Severity'**
  String get severity;

  /// No description provided for @refreshing.
  ///
  /// In en, this message translates to:
  /// **'Auto-refresh: 10s'**
  String get refreshing;

  /// No description provided for @noData.
  ///
  /// In en, this message translates to:
  /// **'No data available'**
  String get noData;

  /// No description provided for @notAvailable.
  ///
  /// In en, this message translates to:
  /// **'Not available'**
  String get notAvailable;

  /// No description provided for @dashboard.
  ///
  /// In en, this message translates to:
  /// **'Dashboard'**
  String get dashboard;

  /// No description provided for @systemOverview.
  ///
  /// In en, this message translates to:
  /// **'System Overview'**
  String get systemOverview;

  /// No description provided for @cpuUsage.
  ///
  /// In en, this message translates to:
  /// **'CPU Usage'**
  String get cpuUsage;

  /// No description provided for @memoryUsage.
  ///
  /// In en, this message translates to:
  /// **'Memory Usage'**
  String get memoryUsage;

  /// No description provided for @responseTime.
  ///
  /// In en, this message translates to:
  /// **'Response Time'**
  String get responseTime;

  /// No description provided for @toolExecutions.
  ///
  /// In en, this message translates to:
  /// **'Tool Executions'**
  String get toolExecutions;

  /// No description provided for @successRate.
  ///
  /// In en, this message translates to:
  /// **'Success Rate'**
  String get successRate;

  /// No description provided for @recentEvents.
  ///
  /// In en, this message translates to:
  /// **'Recent Events'**
  String get recentEvents;

  /// No description provided for @lastUpdated.
  ///
  /// In en, this message translates to:
  /// **'Last Updated'**
  String get lastUpdated;

  /// No description provided for @systemHealth.
  ///
  /// In en, this message translates to:
  /// **'System Health'**
  String get systemHealth;

  /// No description provided for @performance.
  ///
  /// In en, this message translates to:
  /// **'Performance'**
  String get performance;

  /// No description provided for @trends.
  ///
  /// In en, this message translates to:
  /// **'Trends'**
  String get trends;

  /// No description provided for @marketplace.
  ///
  /// In en, this message translates to:
  /// **'Marketplace'**
  String get marketplace;

  /// No description provided for @featured.
  ///
  /// In en, this message translates to:
  /// **'Featured'**
  String get featured;

  /// No description provided for @trending.
  ///
  /// In en, this message translates to:
  /// **'Trending'**
  String get trending;

  /// No description provided for @categories.
  ///
  /// In en, this message translates to:
  /// **'Categories'**
  String get categories;

  /// No description provided for @searchSkills.
  ///
  /// In en, this message translates to:
  /// **'Search skills...'**
  String get searchSkills;

  /// No description provided for @installed.
  ///
  /// In en, this message translates to:
  /// **'Installed'**
  String get installed;

  /// No description provided for @installSkill.
  ///
  /// In en, this message translates to:
  /// **'Install'**
  String get installSkill;

  /// No description provided for @uninstallSkill.
  ///
  /// In en, this message translates to:
  /// **'Uninstall'**
  String get uninstallSkill;

  /// No description provided for @installing.
  ///
  /// In en, this message translates to:
  /// **'Installing...'**
  String get installing;

  /// No description provided for @skillDetails.
  ///
  /// In en, this message translates to:
  /// **'Skill Details'**
  String get skillDetails;

  /// No description provided for @reviews.
  ///
  /// In en, this message translates to:
  /// **'Reviews'**
  String get reviews;

  /// No description provided for @noSkills.
  ///
  /// In en, this message translates to:
  /// **'No skills found'**
  String get noSkills;

  /// No description provided for @browseMarketplace.
  ///
  /// In en, this message translates to:
  /// **'Browse Marketplace'**
  String get browseMarketplace;

  /// No description provided for @verified.
  ///
  /// In en, this message translates to:
  /// **'Verified'**
  String get verified;

  /// No description provided for @downloads.
  ///
  /// In en, this message translates to:
  /// **'Downloads'**
  String get downloads;

  /// No description provided for @rating.
  ///
  /// In en, this message translates to:
  /// **'Rating'**
  String get rating;

  /// No description provided for @memoryTitle.
  ///
  /// In en, this message translates to:
  /// **'Memory'**
  String get memoryTitle;

  /// No description provided for @knowledgeGraph.
  ///
  /// In en, this message translates to:
  /// **'Knowledge Graph'**
  String get knowledgeGraph;

  /// No description provided for @entities.
  ///
  /// In en, this message translates to:
  /// **'Entities'**
  String get entities;

  /// No description provided for @relations.
  ///
  /// In en, this message translates to:
  /// **'Relations'**
  String get relations;

  /// No description provided for @hygiene.
  ///
  /// In en, this message translates to:
  /// **'Hygiene'**
  String get hygiene;

  /// No description provided for @quarantine.
  ///
  /// In en, this message translates to:
  /// **'Quarantine'**
  String get quarantine;

  /// No description provided for @scanMemory.
  ///
  /// In en, this message translates to:
  /// **'Scan'**
  String get scanMemory;

  /// No description provided for @scanning.
  ///
  /// In en, this message translates to:
  /// **'Scanning...'**
  String get scanning;

  /// No description provided for @explainability.
  ///
  /// In en, this message translates to:
  /// **'Explainability'**
  String get explainability;

  /// No description provided for @decisionTrails.
  ///
  /// In en, this message translates to:
  /// **'Decision Trails'**
  String get decisionTrails;

  /// No description provided for @lowTrust.
  ///
  /// In en, this message translates to:
  /// **'Low Trust'**
  String get lowTrust;

  /// No description provided for @graphStats.
  ///
  /// In en, this message translates to:
  /// **'Graph Statistics'**
  String get graphStats;

  /// No description provided for @noEntities.
  ///
  /// In en, this message translates to:
  /// **'No entities'**
  String get noEntities;

  /// No description provided for @noTrails.
  ///
  /// In en, this message translates to:
  /// **'No trails'**
  String get noTrails;

  /// No description provided for @scanComplete.
  ///
  /// In en, this message translates to:
  /// **'Scan complete'**
  String get scanComplete;

  /// No description provided for @threats.
  ///
  /// In en, this message translates to:
  /// **'Threats'**
  String get threats;

  /// No description provided for @threatRate.
  ///
  /// In en, this message translates to:
  /// **'Threat Rate'**
  String get threatRate;

  /// No description provided for @totalScans.
  ///
  /// In en, this message translates to:
  /// **'Total Scans'**
  String get totalScans;

  /// No description provided for @integrity.
  ///
  /// In en, this message translates to:
  /// **'Integrity'**
  String get integrity;

  /// No description provided for @securityTitle.
  ///
  /// In en, this message translates to:
  /// **'Security'**
  String get securityTitle;

  /// No description provided for @complianceTitle.
  ///
  /// In en, this message translates to:
  /// **'Compliance'**
  String get complianceTitle;

  /// No description provided for @rolesTitle.
  ///
  /// In en, this message translates to:
  /// **'Roles'**
  String get rolesTitle;

  /// No description provided for @permissions.
  ///
  /// In en, this message translates to:
  /// **'Permissions'**
  String get permissions;

  /// No description provided for @auditLog.
  ///
  /// In en, this message translates to:
  /// **'Audit Log'**
  String get auditLog;

  /// No description provided for @redTeam.
  ///
  /// In en, this message translates to:
  /// **'Red Team'**
  String get redTeam;

  /// No description provided for @scanStatus.
  ///
  /// In en, this message translates to:
  /// **'Scan Status'**
  String get scanStatus;

  /// No description provided for @complianceReport.
  ///
  /// In en, this message translates to:
  /// **'Compliance Report'**
  String get complianceReport;

  /// No description provided for @decisionsTitle.
  ///
  /// In en, this message translates to:
  /// **'Decisions'**
  String get decisionsTitle;

  /// No description provided for @remediations.
  ///
  /// In en, this message translates to:
  /// **'Remediations'**
  String get remediations;

  /// No description provided for @openStatus.
  ///
  /// In en, this message translates to:
  /// **'Open'**
  String get openStatus;

  /// No description provided for @inProgressStatus.
  ///
  /// In en, this message translates to:
  /// **'In Progress'**
  String get inProgressStatus;

  /// No description provided for @resolvedStatus.
  ///
  /// In en, this message translates to:
  /// **'Resolved'**
  String get resolvedStatus;

  /// No description provided for @overdueStatus.
  ///
  /// In en, this message translates to:
  /// **'Overdue'**
  String get overdueStatus;

  /// No description provided for @approvalRate.
  ///
  /// In en, this message translates to:
  /// **'Approval Rate'**
  String get approvalRate;

  /// No description provided for @flaggedCount.
  ///
  /// In en, this message translates to:
  /// **'Flagged'**
  String get flaggedCount;

  /// No description provided for @transparency.
  ///
  /// In en, this message translates to:
  /// **'Transparency'**
  String get transparency;

  /// No description provided for @euAiAct.
  ///
  /// In en, this message translates to:
  /// **'EU AI Act'**
  String get euAiAct;

  /// No description provided for @dsgvo.
  ///
  /// In en, this message translates to:
  /// **'GDPR'**
  String get dsgvo;

  /// No description provided for @runScan.
  ///
  /// In en, this message translates to:
  /// **'Run Scan'**
  String get runScan;

  /// No description provided for @adminTitle.
  ///
  /// In en, this message translates to:
  /// **'Administration'**
  String get adminTitle;

  /// No description provided for @agentsTitle.
  ///
  /// In en, this message translates to:
  /// **'Agents'**
  String get agentsTitle;

  /// No description provided for @modelsTitle.
  ///
  /// In en, this message translates to:
  /// **'Models'**
  String get modelsTitle;

  /// No description provided for @systemTitle.
  ///
  /// In en, this message translates to:
  /// **'System'**
  String get systemTitle;

  /// No description provided for @workflowsTitle.
  ///
  /// In en, this message translates to:
  /// **'Workflows'**
  String get workflowsTitle;

  /// No description provided for @vaultTitle.
  ///
  /// In en, this message translates to:
  /// **'Vault'**
  String get vaultTitle;

  /// No description provided for @credentialsTitle.
  ///
  /// In en, this message translates to:
  /// **'Credentials'**
  String get credentialsTitle;

  /// No description provided for @bindingsTitle.
  ///
  /// In en, this message translates to:
  /// **'Bindings'**
  String get bindingsTitle;

  /// No description provided for @connectorsTitle.
  ///
  /// In en, this message translates to:
  /// **'Connectors'**
  String get connectorsTitle;

  /// No description provided for @commandsTitle.
  ///
  /// In en, this message translates to:
  /// **'Commands'**
  String get commandsTitle;

  /// No description provided for @isolationTitle.
  ///
  /// In en, this message translates to:
  /// **'Isolation'**
  String get isolationTitle;

  /// No description provided for @sandboxTitle.
  ///
  /// In en, this message translates to:
  /// **'Sandbox'**
  String get sandboxTitle;

  /// No description provided for @circlesTitle.
  ///
  /// In en, this message translates to:
  /// **'Circles'**
  String get circlesTitle;

  /// No description provided for @wizardsTitle.
  ///
  /// In en, this message translates to:
  /// **'Wizards'**
  String get wizardsTitle;

  /// No description provided for @systemStatus.
  ///
  /// In en, this message translates to:
  /// **'System Status'**
  String get systemStatus;

  /// No description provided for @shutdownServer.
  ///
  /// In en, this message translates to:
  /// **'Shutdown Server'**
  String get shutdownServer;

  /// No description provided for @shutdownConfirm.
  ///
  /// In en, this message translates to:
  /// **'Are you sure you want to shut down the server?'**
  String get shutdownConfirm;

  /// No description provided for @startComponent.
  ///
  /// In en, this message translates to:
  /// **'Start'**
  String get startComponent;

  /// No description provided for @stopComponent.
  ///
  /// In en, this message translates to:
  /// **'Stop'**
  String get stopComponent;

  /// No description provided for @selectTemplate.
  ///
  /// In en, this message translates to:
  /// **'Select Template'**
  String get selectTemplate;

  /// No description provided for @workflowStarted.
  ///
  /// In en, this message translates to:
  /// **'Workflow started'**
  String get workflowStarted;

  /// No description provided for @noWorkflows.
  ///
  /// In en, this message translates to:
  /// **'No workflows'**
  String get noWorkflows;

  /// No description provided for @templates.
  ///
  /// In en, this message translates to:
  /// **'Templates'**
  String get templates;

  /// No description provided for @running.
  ///
  /// In en, this message translates to:
  /// **'Running'**
  String get running;

  /// No description provided for @vaultStats.
  ///
  /// In en, this message translates to:
  /// **'Vault Statistics'**
  String get vaultStats;

  /// No description provided for @totalEntries.
  ///
  /// In en, this message translates to:
  /// **'Total Entries'**
  String get totalEntries;

  /// No description provided for @agentVaults.
  ///
  /// In en, this message translates to:
  /// **'Agent Vaults'**
  String get agentVaults;

  /// No description provided for @noVaults.
  ///
  /// In en, this message translates to:
  /// **'No vaults'**
  String get noVaults;

  /// No description provided for @availableModels.
  ///
  /// In en, this message translates to:
  /// **'Available Models'**
  String get availableModels;

  /// No description provided for @modelStats.
  ///
  /// In en, this message translates to:
  /// **'Model Statistics'**
  String get modelStats;

  /// No description provided for @providers.
  ///
  /// In en, this message translates to:
  /// **'Providers'**
  String get providers;

  /// No description provided for @capabilities.
  ///
  /// In en, this message translates to:
  /// **'Capabilities'**
  String get capabilities;

  /// No description provided for @plannerModel.
  ///
  /// In en, this message translates to:
  /// **'Planner'**
  String get plannerModel;

  /// No description provided for @executorModel.
  ///
  /// In en, this message translates to:
  /// **'Executor'**
  String get executorModel;

  /// No description provided for @coderModel.
  ///
  /// In en, this message translates to:
  /// **'Coder'**
  String get coderModel;

  /// No description provided for @embeddingModel.
  ///
  /// In en, this message translates to:
  /// **'Embedding'**
  String get embeddingModel;

  /// No description provided for @configured.
  ///
  /// In en, this message translates to:
  /// **'Configured'**
  String get configured;

  /// No description provided for @modelWarnings.
  ///
  /// In en, this message translates to:
  /// **'Warnings'**
  String get modelWarnings;

  /// No description provided for @identityDreamCycle.
  ///
  /// In en, this message translates to:
  /// **'Dream Cycle'**
  String get identityDreamCycle;

  /// No description provided for @identityGenesisAnchors.
  ///
  /// In en, this message translates to:
  /// **'Genesis Anchors'**
  String get identityGenesisAnchors;

  /// No description provided for @identityNoAnchors.
  ///
  /// In en, this message translates to:
  /// **'No genesis anchors'**
  String get identityNoAnchors;

  /// No description provided for @identityPersonality.
  ///
  /// In en, this message translates to:
  /// **'Personality'**
  String get identityPersonality;

  /// No description provided for @identityCognitive.
  ///
  /// In en, this message translates to:
  /// **'Cognitive State'**
  String get identityCognitive;

  /// No description provided for @identityEmotional.
  ///
  /// In en, this message translates to:
  /// **'Emotional State'**
  String get identityEmotional;

  /// No description provided for @identitySomatic.
  ///
  /// In en, this message translates to:
  /// **'Somatic State'**
  String get identitySomatic;

  /// No description provided for @identityNarrative.
  ///
  /// In en, this message translates to:
  /// **'Narrative'**
  String get identityNarrative;

  /// No description provided for @identityExistential.
  ///
  /// In en, this message translates to:
  /// **'Existential'**
  String get identityExistential;

  /// No description provided for @identityPredictive.
  ///
  /// In en, this message translates to:
  /// **'Predictive'**
  String get identityPredictive;

  /// No description provided for @identityEpistemic.
  ///
  /// In en, this message translates to:
  /// **'Epistemic'**
  String get identityEpistemic;

  /// No description provided for @identityBiases.
  ///
  /// In en, this message translates to:
  /// **'Active Biases'**
  String get identityBiases;

  /// No description provided for @search.
  ///
  /// In en, this message translates to:
  /// **'Search'**
  String get search;

  /// No description provided for @filter.
  ///
  /// In en, this message translates to:
  /// **'Filter'**
  String get filter;

  /// No description provided for @sortBy.
  ///
  /// In en, this message translates to:
  /// **'Sort by'**
  String get sortBy;

  /// No description provided for @refresh.
  ///
  /// In en, this message translates to:
  /// **'Refresh'**
  String get refresh;

  /// No description provided for @export.
  ///
  /// In en, this message translates to:
  /// **'Export'**
  String get export;

  /// No description provided for @viewAll.
  ///
  /// In en, this message translates to:
  /// **'View All'**
  String get viewAll;

  /// No description provided for @details.
  ///
  /// In en, this message translates to:
  /// **'Details'**
  String get details;

  /// No description provided for @back.
  ///
  /// In en, this message translates to:
  /// **'Back'**
  String get back;

  /// No description provided for @confirm.
  ///
  /// In en, this message translates to:
  /// **'Confirm'**
  String get confirm;

  /// No description provided for @actions.
  ///
  /// In en, this message translates to:
  /// **'Actions'**
  String get actions;

  /// No description provided for @statusLabel.
  ///
  /// In en, this message translates to:
  /// **'Status'**
  String get statusLabel;

  /// No description provided for @enabled.
  ///
  /// In en, this message translates to:
  /// **'Enabled'**
  String get enabled;

  /// No description provided for @disabled.
  ///
  /// In en, this message translates to:
  /// **'Disabled'**
  String get disabled;

  /// No description provided for @total.
  ///
  /// In en, this message translates to:
  /// **'Total'**
  String get total;

  /// No description provided for @count.
  ///
  /// In en, this message translates to:
  /// **'Count'**
  String get count;

  /// No description provided for @rate.
  ///
  /// In en, this message translates to:
  /// **'Rate'**
  String get rate;

  /// No description provided for @average.
  ///
  /// In en, this message translates to:
  /// **'Average'**
  String get average;

  /// No description provided for @duration.
  ///
  /// In en, this message translates to:
  /// **'Duration'**
  String get duration;

  /// No description provided for @timestamp.
  ///
  /// In en, this message translates to:
  /// **'Timestamp'**
  String get timestamp;

  /// No description provided for @severityLabel.
  ///
  /// In en, this message translates to:
  /// **'Severity'**
  String get severityLabel;

  /// No description provided for @critical.
  ///
  /// In en, this message translates to:
  /// **'Critical'**
  String get critical;

  /// No description provided for @errorLabel.
  ///
  /// In en, this message translates to:
  /// **'Error'**
  String get errorLabel;

  /// No description provided for @warningLabel.
  ///
  /// In en, this message translates to:
  /// **'Warning'**
  String get warningLabel;

  /// No description provided for @infoLabel.
  ///
  /// In en, this message translates to:
  /// **'Info'**
  String get infoLabel;

  /// No description provided for @successLabel.
  ///
  /// In en, this message translates to:
  /// **'Success'**
  String get successLabel;

  /// No description provided for @unknownLabel.
  ///
  /// In en, this message translates to:
  /// **'Unknown'**
  String get unknownLabel;

  /// No description provided for @notConfigured.
  ///
  /// In en, this message translates to:
  /// **'Not configured'**
  String get notConfigured;

  /// No description provided for @comingSoon.
  ///
  /// In en, this message translates to:
  /// **'Coming Soon'**
  String get comingSoon;

  /// No description provided for @beta.
  ///
  /// In en, this message translates to:
  /// **'Beta'**
  String get beta;

  /// No description provided for @copyToClipboard.
  ///
  /// In en, this message translates to:
  /// **'Copy to clipboard'**
  String get copyToClipboard;

  /// No description provided for @copied.
  ///
  /// In en, this message translates to:
  /// **'Copied!'**
  String get copied;

  /// No description provided for @chatSettings.
  ///
  /// In en, this message translates to:
  /// **'Chat Settings'**
  String get chatSettings;

  /// No description provided for @clearChat.
  ///
  /// In en, this message translates to:
  /// **'Clear Chat'**
  String get clearChat;

  /// No description provided for @voiceMode.
  ///
  /// In en, this message translates to:
  /// **'Voice Mode'**
  String get voiceMode;

  /// No description provided for @fileUploadAction.
  ///
  /// In en, this message translates to:
  /// **'Upload File'**
  String get fileUploadAction;

  /// No description provided for @planDetails.
  ///
  /// In en, this message translates to:
  /// **'Plan Details'**
  String get planDetails;

  /// No description provided for @noMessages.
  ///
  /// In en, this message translates to:
  /// **'No messages yet'**
  String get noMessages;

  /// No description provided for @typeMessage.
  ///
  /// In en, this message translates to:
  /// **'Type a message...'**
  String get typeMessage;

  /// No description provided for @settingsTitle.
  ///
  /// In en, this message translates to:
  /// **'Settings'**
  String get settingsTitle;

  /// No description provided for @language.
  ///
  /// In en, this message translates to:
  /// **'Language'**
  String get language;

  /// No description provided for @theme.
  ///
  /// In en, this message translates to:
  /// **'Theme'**
  String get theme;

  /// No description provided for @about.
  ///
  /// In en, this message translates to:
  /// **'About'**
  String get about;

  /// No description provided for @licenses.
  ///
  /// In en, this message translates to:
  /// **'Licenses'**
  String get licenses;

  /// No description provided for @clearCache.
  ///
  /// In en, this message translates to:
  /// **'Clear Cache'**
  String get clearCache;

  /// No description provided for @adminConfigSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Manage configuration'**
  String get adminConfigSubtitle;

  /// No description provided for @adminAgentsSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Agents & profiles'**
  String get adminAgentsSubtitle;

  /// No description provided for @adminModelsSubtitle.
  ///
  /// In en, this message translates to:
  /// **'LLM models'**
  String get adminModelsSubtitle;

  /// No description provided for @adminSecuritySubtitle.
  ///
  /// In en, this message translates to:
  /// **'Security & compliance'**
  String get adminSecuritySubtitle;

  /// No description provided for @adminWorkflowsSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Automations'**
  String get adminWorkflowsSubtitle;

  /// No description provided for @adminMemorySubtitle.
  ///
  /// In en, this message translates to:
  /// **'Knowledge graph'**
  String get adminMemorySubtitle;

  /// No description provided for @adminVaultSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Secrets & keys'**
  String get adminVaultSubtitle;

  /// No description provided for @adminSystemSubtitle.
  ///
  /// In en, this message translates to:
  /// **'System status'**
  String get adminSystemSubtitle;

  /// No description provided for @dashboardRefreshing.
  ///
  /// In en, this message translates to:
  /// **'Auto-refresh: 15s'**
  String get dashboardRefreshing;

  /// No description provided for @backendVersion.
  ///
  /// In en, this message translates to:
  /// **'Backend Version'**
  String get backendVersion;

  /// No description provided for @modelInfo.
  ///
  /// In en, this message translates to:
  /// **'Model Info'**
  String get modelInfo;

  /// No description provided for @confidence.
  ///
  /// In en, this message translates to:
  /// **'Confidence'**
  String get confidence;

  /// No description provided for @rolesAccess.
  ///
  /// In en, this message translates to:
  /// **'Roles & Access'**
  String get rolesAccess;

  /// No description provided for @loadMore.
  ///
  /// In en, this message translates to:
  /// **'Load More'**
  String get loadMore;

  /// No description provided for @actor.
  ///
  /// In en, this message translates to:
  /// **'Actor'**
  String get actor;

  /// No description provided for @noAuditEntries.
  ///
  /// In en, this message translates to:
  /// **'No audit entries'**
  String get noAuditEntries;

  /// No description provided for @allSeverities.
  ///
  /// In en, this message translates to:
  /// **'All Severities'**
  String get allSeverities;

  /// No description provided for @allActions.
  ///
  /// In en, this message translates to:
  /// **'All Actions'**
  String get allActions;

  /// No description provided for @scanNotAvailable.
  ///
  /// In en, this message translates to:
  /// **'Scan not available'**
  String get scanNotAvailable;

  /// No description provided for @lastScan.
  ///
  /// In en, this message translates to:
  /// **'Last Scan'**
  String get lastScan;

  /// No description provided for @scanResults.
  ///
  /// In en, this message translates to:
  /// **'Scan Results'**
  String get scanResults;

  /// No description provided for @compliant.
  ///
  /// In en, this message translates to:
  /// **'Compliant'**
  String get compliant;

  /// No description provided for @nonCompliant.
  ///
  /// In en, this message translates to:
  /// **'Non-Compliant'**
  String get nonCompliant;

  /// No description provided for @model.
  ///
  /// In en, this message translates to:
  /// **'Model'**
  String get model;

  /// No description provided for @temperature.
  ///
  /// In en, this message translates to:
  /// **'Temperature'**
  String get temperature;

  /// No description provided for @priority.
  ///
  /// In en, this message translates to:
  /// **'Priority'**
  String get priority;

  /// No description provided for @allowedTools.
  ///
  /// In en, this message translates to:
  /// **'Allowed Tools'**
  String get allowedTools;

  /// No description provided for @blockedTools.
  ///
  /// In en, this message translates to:
  /// **'Blocked Tools'**
  String get blockedTools;

  /// No description provided for @noAgents.
  ///
  /// In en, this message translates to:
  /// **'No agents configured'**
  String get noAgents;

  /// No description provided for @description.
  ///
  /// In en, this message translates to:
  /// **'Description'**
  String get description;

  /// No description provided for @provider.
  ///
  /// In en, this message translates to:
  /// **'Provider'**
  String get provider;

  /// No description provided for @noModels.
  ///
  /// In en, this message translates to:
  /// **'No models available'**
  String get noModels;

  /// No description provided for @owner.
  ///
  /// In en, this message translates to:
  /// **'Owner'**
  String get owner;

  /// No description provided for @llmBackend.
  ///
  /// In en, this message translates to:
  /// **'LLM Backend'**
  String get llmBackend;

  /// No description provided for @components.
  ///
  /// In en, this message translates to:
  /// **'Components'**
  String get components;

  /// No description provided for @dangerZone.
  ///
  /// In en, this message translates to:
  /// **'Danger Zone'**
  String get dangerZone;

  /// No description provided for @reloadConfig.
  ///
  /// In en, this message translates to:
  /// **'Reload Config'**
  String get reloadConfig;

  /// No description provided for @runtimeInfo.
  ///
  /// In en, this message translates to:
  /// **'Runtime Info'**
  String get runtimeInfo;

  /// No description provided for @startWorkflow.
  ///
  /// In en, this message translates to:
  /// **'Start Workflow'**
  String get startWorkflow;

  /// No description provided for @noCategories.
  ///
  /// In en, this message translates to:
  /// **'No categories'**
  String get noCategories;

  /// No description provided for @templateCount.
  ///
  /// In en, this message translates to:
  /// **'{count} templates'**
  String templateCount(String count);

  /// No description provided for @entityTypes.
  ///
  /// In en, this message translates to:
  /// **'Entity Types'**
  String get entityTypes;

  /// No description provided for @activeTrails.
  ///
  /// In en, this message translates to:
  /// **'Active Trails'**
  String get activeTrails;

  /// No description provided for @completedTrails.
  ///
  /// In en, this message translates to:
  /// **'Completed'**
  String get completedTrails;

  /// No description provided for @lastAccessed.
  ///
  /// In en, this message translates to:
  /// **'Last Accessed'**
  String get lastAccessed;

  /// No description provided for @author.
  ///
  /// In en, this message translates to:
  /// **'Author'**
  String get author;

  /// No description provided for @noQuarantine.
  ///
  /// In en, this message translates to:
  /// **'No quarantined items'**
  String get noQuarantine;

  /// No description provided for @totalVaults.
  ///
  /// In en, this message translates to:
  /// **'Total Vaults'**
  String get totalVaults;

  /// No description provided for @scanNow.
  ///
  /// In en, this message translates to:
  /// **'Scan Now'**
  String get scanNow;

  /// No description provided for @startConversation.
  ///
  /// In en, this message translates to:
  /// **'Start a conversation'**
  String get startConversation;

  /// No description provided for @attachFile.
  ///
  /// In en, this message translates to:
  /// **'Attach file'**
  String get attachFile;

  /// No description provided for @voiceModeHint.
  ///
  /// In en, this message translates to:
  /// **'Voice mode coming soon'**
  String get voiceModeHint;

  /// No description provided for @canvasLabel.
  ///
  /// In en, this message translates to:
  /// **'Canvas'**
  String get canvasLabel;

  /// No description provided for @configGeneral.
  ///
  /// In en, this message translates to:
  /// **'General'**
  String get configGeneral;

  /// No description provided for @configLanguage.
  ///
  /// In en, this message translates to:
  /// **'Language'**
  String get configLanguage;

  /// No description provided for @configProviders.
  ///
  /// In en, this message translates to:
  /// **'Providers'**
  String get configProviders;

  /// No description provided for @configModels.
  ///
  /// In en, this message translates to:
  /// **'Models'**
  String get configModels;

  /// No description provided for @configPlanner.
  ///
  /// In en, this message translates to:
  /// **'Planner'**
  String get configPlanner;

  /// No description provided for @configExecutor.
  ///
  /// In en, this message translates to:
  /// **'Executor'**
  String get configExecutor;

  /// No description provided for @configMemory.
  ///
  /// In en, this message translates to:
  /// **'Memory'**
  String get configMemory;

  /// No description provided for @configChannels.
  ///
  /// In en, this message translates to:
  /// **'Channels'**
  String get configChannels;

  /// No description provided for @configSecurity.
  ///
  /// In en, this message translates to:
  /// **'Security'**
  String get configSecurity;

  /// No description provided for @configWeb.
  ///
  /// In en, this message translates to:
  /// **'Web'**
  String get configWeb;

  /// No description provided for @configMcp.
  ///
  /// In en, this message translates to:
  /// **'MCP'**
  String get configMcp;

  /// No description provided for @configCron.
  ///
  /// In en, this message translates to:
  /// **'Cron'**
  String get configCron;

  /// No description provided for @configDatabase.
  ///
  /// In en, this message translates to:
  /// **'Database'**
  String get configDatabase;

  /// No description provided for @configLogging.
  ///
  /// In en, this message translates to:
  /// **'Logging'**
  String get configLogging;

  /// No description provided for @configPrompts.
  ///
  /// In en, this message translates to:
  /// **'Prompts'**
  String get configPrompts;

  /// No description provided for @configAgents.
  ///
  /// In en, this message translates to:
  /// **'Agents'**
  String get configAgents;

  /// No description provided for @configBindings.
  ///
  /// In en, this message translates to:
  /// **'Bindings'**
  String get configBindings;

  /// No description provided for @configSystem.
  ///
  /// In en, this message translates to:
  /// **'System'**
  String get configSystem;

  /// No description provided for @ownerName.
  ///
  /// In en, this message translates to:
  /// **'Owner Name'**
  String get ownerName;

  /// No description provided for @operationMode.
  ///
  /// In en, this message translates to:
  /// **'Operation Mode'**
  String get operationMode;

  /// No description provided for @costTracking.
  ///
  /// In en, this message translates to:
  /// **'Cost Tracking'**
  String get costTracking;

  /// No description provided for @dailyBudget.
  ///
  /// In en, this message translates to:
  /// **'Daily Budget'**
  String get dailyBudget;

  /// No description provided for @monthlyBudget.
  ///
  /// In en, this message translates to:
  /// **'Monthly Budget'**
  String get monthlyBudget;

  /// No description provided for @apiKey.
  ///
  /// In en, this message translates to:
  /// **'API Key'**
  String get apiKey;

  /// No description provided for @baseUrl.
  ///
  /// In en, this message translates to:
  /// **'Base URL'**
  String get baseUrl;

  /// No description provided for @maxTokens.
  ///
  /// In en, this message translates to:
  /// **'Max Tokens'**
  String get maxTokens;

  /// No description provided for @timeout.
  ///
  /// In en, this message translates to:
  /// **'Timeout'**
  String get timeout;

  /// No description provided for @keepAlive.
  ///
  /// In en, this message translates to:
  /// **'Keep Alive'**
  String get keepAlive;

  /// No description provided for @contextWindow.
  ///
  /// In en, this message translates to:
  /// **'Context Window'**
  String get contextWindow;

  /// No description provided for @vramGb.
  ///
  /// In en, this message translates to:
  /// **'VRAM (GB)'**
  String get vramGb;

  /// No description provided for @topP.
  ///
  /// In en, this message translates to:
  /// **'Top P'**
  String get topP;

  /// No description provided for @maxIterations.
  ///
  /// In en, this message translates to:
  /// **'Max Iterations'**
  String get maxIterations;

  /// No description provided for @escalationAfter.
  ///
  /// In en, this message translates to:
  /// **'Escalation After'**
  String get escalationAfter;

  /// No description provided for @responseBudget.
  ///
  /// In en, this message translates to:
  /// **'Response Token Budget'**
  String get responseBudget;

  /// No description provided for @policiesDir.
  ///
  /// In en, this message translates to:
  /// **'Policies Directory'**
  String get policiesDir;

  /// No description provided for @defaultRiskLevel.
  ///
  /// In en, this message translates to:
  /// **'Default Risk Level'**
  String get defaultRiskLevel;

  /// No description provided for @maxBlockedRetries.
  ///
  /// In en, this message translates to:
  /// **'Max Blocked Retries'**
  String get maxBlockedRetries;

  /// No description provided for @sandboxLevel.
  ///
  /// In en, this message translates to:
  /// **'Sandbox Level'**
  String get sandboxLevel;

  /// No description provided for @maxMemoryMb.
  ///
  /// In en, this message translates to:
  /// **'Max Memory (MB)'**
  String get maxMemoryMb;

  /// No description provided for @maxCpuSeconds.
  ///
  /// In en, this message translates to:
  /// **'Max CPU Seconds'**
  String get maxCpuSeconds;

  /// No description provided for @allowedPaths.
  ///
  /// In en, this message translates to:
  /// **'Allowed Paths'**
  String get allowedPaths;

  /// No description provided for @networkAccess.
  ///
  /// In en, this message translates to:
  /// **'Network Access'**
  String get networkAccess;

  /// No description provided for @envVars.
  ///
  /// In en, this message translates to:
  /// **'Environment Variables'**
  String get envVars;

  /// No description provided for @defaultTimeout.
  ///
  /// In en, this message translates to:
  /// **'Default Timeout'**
  String get defaultTimeout;

  /// No description provided for @maxOutputChars.
  ///
  /// In en, this message translates to:
  /// **'Max Output Chars'**
  String get maxOutputChars;

  /// No description provided for @maxRetries.
  ///
  /// In en, this message translates to:
  /// **'Max Retries'**
  String get maxRetries;

  /// No description provided for @backoffDelay.
  ///
  /// In en, this message translates to:
  /// **'Backoff Delay'**
  String get backoffDelay;

  /// No description provided for @maxParallelTools.
  ///
  /// In en, this message translates to:
  /// **'Max Parallel Tools'**
  String get maxParallelTools;

  /// No description provided for @chunkSize.
  ///
  /// In en, this message translates to:
  /// **'Chunk Size'**
  String get chunkSize;

  /// No description provided for @chunkOverlap.
  ///
  /// In en, this message translates to:
  /// **'Chunk Overlap'**
  String get chunkOverlap;

  /// No description provided for @searchTopK.
  ///
  /// In en, this message translates to:
  /// **'Search Top K'**
  String get searchTopK;

  /// No description provided for @searchWeights.
  ///
  /// In en, this message translates to:
  /// **'Search Weights'**
  String get searchWeights;

  /// No description provided for @vectorWeight.
  ///
  /// In en, this message translates to:
  /// **'Vector Weight'**
  String get vectorWeight;

  /// No description provided for @bm25Weight.
  ///
  /// In en, this message translates to:
  /// **'BM25 Weight'**
  String get bm25Weight;

  /// No description provided for @graphWeight.
  ///
  /// In en, this message translates to:
  /// **'Graph Weight'**
  String get graphWeight;

  /// No description provided for @recencyHalfLife.
  ///
  /// In en, this message translates to:
  /// **'Recency Half-Life'**
  String get recencyHalfLife;

  /// No description provided for @compactionThreshold.
  ///
  /// In en, this message translates to:
  /// **'Compaction Threshold'**
  String get compactionThreshold;

  /// No description provided for @compactionKeepLast.
  ///
  /// In en, this message translates to:
  /// **'Compaction Keep Last'**
  String get compactionKeepLast;

  /// No description provided for @episodicRetention.
  ///
  /// In en, this message translates to:
  /// **'Episodic Retention'**
  String get episodicRetention;

  /// No description provided for @dynamicWeighting.
  ///
  /// In en, this message translates to:
  /// **'Dynamic Weighting'**
  String get dynamicWeighting;

  /// No description provided for @voiceEnabled.
  ///
  /// In en, this message translates to:
  /// **'Voice Enabled'**
  String get voiceEnabled;

  /// No description provided for @ttsBackend.
  ///
  /// In en, this message translates to:
  /// **'TTS Backend'**
  String get ttsBackend;

  /// No description provided for @piperVoice.
  ///
  /// In en, this message translates to:
  /// **'Piper Voice'**
  String get piperVoice;

  /// No description provided for @piperLengthScale.
  ///
  /// In en, this message translates to:
  /// **'Piper Length Scale'**
  String get piperLengthScale;

  /// No description provided for @wakeWordEnabled.
  ///
  /// In en, this message translates to:
  /// **'Wake Word Enabled'**
  String get wakeWordEnabled;

  /// No description provided for @wakeWord.
  ///
  /// In en, this message translates to:
  /// **'Wake Word'**
  String get wakeWord;

  /// No description provided for @wakeWordBackend.
  ///
  /// In en, this message translates to:
  /// **'Wake Word Backend'**
  String get wakeWordBackend;

  /// No description provided for @talkMode.
  ///
  /// In en, this message translates to:
  /// **'Talk Mode'**
  String get talkMode;

  /// No description provided for @autoListen.
  ///
  /// In en, this message translates to:
  /// **'Auto-Listen'**
  String get autoListen;

  /// No description provided for @blockedCommands.
  ///
  /// In en, this message translates to:
  /// **'Blocked Commands'**
  String get blockedCommands;

  /// No description provided for @credentialPatterns.
  ///
  /// In en, this message translates to:
  /// **'Credential Patterns'**
  String get credentialPatterns;

  /// No description provided for @maxSubAgentDepth.
  ///
  /// In en, this message translates to:
  /// **'Max Sub-Agent Depth'**
  String get maxSubAgentDepth;

  /// No description provided for @searchBackends.
  ///
  /// In en, this message translates to:
  /// **'Search Backends'**
  String get searchBackends;

  /// No description provided for @domainFilters.
  ///
  /// In en, this message translates to:
  /// **'Domain Filters'**
  String get domainFilters;

  /// No description provided for @blocklist.
  ///
  /// In en, this message translates to:
  /// **'Blocklist'**
  String get blocklist;

  /// No description provided for @allowlist.
  ///
  /// In en, this message translates to:
  /// **'Allowlist'**
  String get allowlist;

  /// No description provided for @httpLimits.
  ///
  /// In en, this message translates to:
  /// **'HTTP Limits'**
  String get httpLimits;

  /// No description provided for @maxFetchBytes.
  ///
  /// In en, this message translates to:
  /// **'Max Fetch Bytes'**
  String get maxFetchBytes;

  /// No description provided for @maxTextChars.
  ///
  /// In en, this message translates to:
  /// **'Max Text Chars'**
  String get maxTextChars;

  /// No description provided for @fetchTimeout.
  ///
  /// In en, this message translates to:
  /// **'Fetch Timeout'**
  String get fetchTimeout;

  /// No description provided for @searchTimeout.
  ///
  /// In en, this message translates to:
  /// **'Search Timeout'**
  String get searchTimeout;

  /// No description provided for @maxSearchResults.
  ///
  /// In en, this message translates to:
  /// **'Max Search Results'**
  String get maxSearchResults;

  /// No description provided for @rateLimit.
  ///
  /// In en, this message translates to:
  /// **'Rate Limit'**
  String get rateLimit;

  /// No description provided for @mcpServers.
  ///
  /// In en, this message translates to:
  /// **'MCP Servers'**
  String get mcpServers;

  /// No description provided for @a2aProtocol.
  ///
  /// In en, this message translates to:
  /// **'A2A Protocol'**
  String get a2aProtocol;

  /// No description provided for @remotes.
  ///
  /// In en, this message translates to:
  /// **'Remotes'**
  String get remotes;

  /// No description provided for @heartbeat.
  ///
  /// In en, this message translates to:
  /// **'Heartbeat'**
  String get heartbeat;

  /// No description provided for @intervalMinutes.
  ///
  /// In en, this message translates to:
  /// **'Interval (minutes)'**
  String get intervalMinutes;

  /// No description provided for @checklistFile.
  ///
  /// In en, this message translates to:
  /// **'Checklist File'**
  String get checklistFile;

  /// No description provided for @channel.
  ///
  /// In en, this message translates to:
  /// **'Channel'**
  String get channel;

  /// No description provided for @plugins.
  ///
  /// In en, this message translates to:
  /// **'Plugins'**
  String get plugins;

  /// No description provided for @skillsDir.
  ///
  /// In en, this message translates to:
  /// **'Skills Directory'**
  String get skillsDir;

  /// No description provided for @autoUpdate.
  ///
  /// In en, this message translates to:
  /// **'Auto Update'**
  String get autoUpdate;

  /// No description provided for @cronJobs.
  ///
  /// In en, this message translates to:
  /// **'Cron Jobs'**
  String get cronJobs;

  /// No description provided for @schedule.
  ///
  /// In en, this message translates to:
  /// **'Schedule'**
  String get schedule;

  /// No description provided for @command.
  ///
  /// In en, this message translates to:
  /// **'Command'**
  String get command;

  /// No description provided for @databaseBackend.
  ///
  /// In en, this message translates to:
  /// **'Database Backend'**
  String get databaseBackend;

  /// No description provided for @encryption.
  ///
  /// In en, this message translates to:
  /// **'Encryption'**
  String get encryption;

  /// No description provided for @pgHost.
  ///
  /// In en, this message translates to:
  /// **'Host'**
  String get pgHost;

  /// No description provided for @pgPort.
  ///
  /// In en, this message translates to:
  /// **'Port'**
  String get pgPort;

  /// No description provided for @pgDbName.
  ///
  /// In en, this message translates to:
  /// **'Database Name'**
  String get pgDbName;

  /// No description provided for @pgUser.
  ///
  /// In en, this message translates to:
  /// **'User'**
  String get pgUser;

  /// No description provided for @pgPassword.
  ///
  /// In en, this message translates to:
  /// **'Password'**
  String get pgPassword;

  /// No description provided for @pgPoolMin.
  ///
  /// In en, this message translates to:
  /// **'Pool Min'**
  String get pgPoolMin;

  /// No description provided for @pgPoolMax.
  ///
  /// In en, this message translates to:
  /// **'Pool Max'**
  String get pgPoolMax;

  /// No description provided for @logLevel.
  ///
  /// In en, this message translates to:
  /// **'Log Level'**
  String get logLevel;

  /// No description provided for @jsonLogs.
  ///
  /// In en, this message translates to:
  /// **'JSON Logs'**
  String get jsonLogs;

  /// No description provided for @consoleOutput.
  ///
  /// In en, this message translates to:
  /// **'Console Output'**
  String get consoleOutput;

  /// No description provided for @systemPrompt.
  ///
  /// In en, this message translates to:
  /// **'System Prompt'**
  String get systemPrompt;

  /// No description provided for @replanPrompt.
  ///
  /// In en, this message translates to:
  /// **'Replan Prompt'**
  String get replanPrompt;

  /// No description provided for @escalationPrompt.
  ///
  /// In en, this message translates to:
  /// **'Escalation Prompt'**
  String get escalationPrompt;

  /// No description provided for @policyYaml.
  ///
  /// In en, this message translates to:
  /// **'Policy YAML'**
  String get policyYaml;

  /// No description provided for @heartbeatMd.
  ///
  /// In en, this message translates to:
  /// **'Heartbeat Checklist'**
  String get heartbeatMd;

  /// No description provided for @personalityPrompt.
  ///
  /// In en, this message translates to:
  /// **'Personality Prompt'**
  String get personalityPrompt;

  /// No description provided for @promptEvolution.
  ///
  /// In en, this message translates to:
  /// **'Prompt Evolution'**
  String get promptEvolution;

  /// No description provided for @resetToDefault.
  ///
  /// In en, this message translates to:
  /// **'Reset to Default'**
  String get resetToDefault;

  /// No description provided for @triggerPatterns.
  ///
  /// In en, this message translates to:
  /// **'Trigger Patterns'**
  String get triggerPatterns;

  /// No description provided for @channelFilter.
  ///
  /// In en, this message translates to:
  /// **'Channel Filter'**
  String get channelFilter;

  /// No description provided for @pattern.
  ///
  /// In en, this message translates to:
  /// **'Pattern'**
  String get pattern;

  /// No description provided for @targetAgent.
  ///
  /// In en, this message translates to:
  /// **'Target Agent'**
  String get targetAgent;

  /// No description provided for @restartBackend.
  ///
  /// In en, this message translates to:
  /// **'Restart Backend'**
  String get restartBackend;

  /// No description provided for @exportConfig.
  ///
  /// In en, this message translates to:
  /// **'Export Configuration'**
  String get exportConfig;

  /// No description provided for @importConfig.
  ///
  /// In en, this message translates to:
  /// **'Import Configuration'**
  String get importConfig;

  /// No description provided for @factoryReset.
  ///
  /// In en, this message translates to:
  /// **'Factory Reset'**
  String get factoryReset;

  /// No description provided for @factoryResetConfirm.
  ///
  /// In en, this message translates to:
  /// **'This will reset ALL configuration to factory defaults. Continue?'**
  String get factoryResetConfirm;

  /// No description provided for @configurationSaved.
  ///
  /// In en, this message translates to:
  /// **'Configuration saved'**
  String get configurationSaved;

  /// No description provided for @saveHadErrors.
  ///
  /// In en, this message translates to:
  /// **'Save had errors'**
  String get saveHadErrors;

  /// No description provided for @unsavedChanges.
  ///
  /// In en, this message translates to:
  /// **'Unsaved changes'**
  String get unsavedChanges;

  /// No description provided for @discard.
  ///
  /// In en, this message translates to:
  /// **'Discard'**
  String get discard;

  /// No description provided for @saving.
  ///
  /// In en, this message translates to:
  /// **'Saving...'**
  String get saving;

  /// No description provided for @voiceOff.
  ///
  /// In en, this message translates to:
  /// **'Off'**
  String get voiceOff;

  /// No description provided for @voiceListening.
  ///
  /// In en, this message translates to:
  /// **'Listening...'**
  String get voiceListening;

  /// No description provided for @voiceSpeakNow.
  ///
  /// In en, this message translates to:
  /// **'Speak now'**
  String get voiceSpeakNow;

  /// No description provided for @voiceProcessing.
  ///
  /// In en, this message translates to:
  /// **'Processing...'**
  String get voiceProcessing;

  /// No description provided for @voiceSpeaking.
  ///
  /// In en, this message translates to:
  /// **'Speaking...'**
  String get voiceSpeaking;

  /// No description provided for @observe.
  ///
  /// In en, this message translates to:
  /// **'Observe'**
  String get observe;

  /// No description provided for @agentLog.
  ///
  /// In en, this message translates to:
  /// **'Agent Log'**
  String get agentLog;

  /// No description provided for @kanban.
  ///
  /// In en, this message translates to:
  /// **'Kanban'**
  String get kanban;

  /// No description provided for @dag.
  ///
  /// In en, this message translates to:
  /// **'DAG'**
  String get dag;

  /// No description provided for @plan.
  ///
  /// In en, this message translates to:
  /// **'Plan'**
  String get plan;

  /// No description provided for @toDo.
  ///
  /// In en, this message translates to:
  /// **'To Do'**
  String get toDo;

  /// No description provided for @inProgress.
  ///
  /// In en, this message translates to:
  /// **'In Progress'**
  String get inProgress;

  /// No description provided for @verifying.
  ///
  /// In en, this message translates to:
  /// **'Verifying'**
  String get verifying;

  /// No description provided for @done.
  ///
  /// In en, this message translates to:
  /// **'Done'**
  String get done;

  /// No description provided for @searchConfigPages.
  ///
  /// In en, this message translates to:
  /// **'Search config pages...'**
  String get searchConfigPages;

  /// No description provided for @noMatchingPages.
  ///
  /// In en, this message translates to:
  /// **'No matching pages'**
  String get noMatchingPages;

  /// No description provided for @knowledgeGraphTitle.
  ///
  /// In en, this message translates to:
  /// **'Knowledge Graph'**
  String get knowledgeGraphTitle;

  /// No description provided for @searchEntities.
  ///
  /// In en, this message translates to:
  /// **'Search entities...'**
  String get searchEntities;

  /// No description provided for @allTypes.
  ///
  /// In en, this message translates to:
  /// **'All Types'**
  String get allTypes;

  /// No description provided for @entityDetail.
  ///
  /// In en, this message translates to:
  /// **'Entity Detail'**
  String get entityDetail;

  /// No description provided for @attributes.
  ///
  /// In en, this message translates to:
  /// **'Attributes'**
  String get attributes;

  /// No description provided for @instances.
  ///
  /// In en, this message translates to:
  /// **'Instances'**
  String get instances;

  /// No description provided for @dagRuns.
  ///
  /// In en, this message translates to:
  /// **'DAG Runs'**
  String get dagRuns;

  /// No description provided for @noInstances.
  ///
  /// In en, this message translates to:
  /// **'No instances'**
  String get noInstances;

  /// No description provided for @noDagRuns.
  ///
  /// In en, this message translates to:
  /// **'No DAG runs'**
  String get noDagRuns;

  /// No description provided for @addCredential.
  ///
  /// In en, this message translates to:
  /// **'Add Credential'**
  String get addCredential;

  /// No description provided for @service.
  ///
  /// In en, this message translates to:
  /// **'Service'**
  String get service;

  /// No description provided for @key.
  ///
  /// In en, this message translates to:
  /// **'Key'**
  String get key;

  /// No description provided for @value.
  ///
  /// In en, this message translates to:
  /// **'Value'**
  String get value;

  /// No description provided for @noCredentials.
  ///
  /// In en, this message translates to:
  /// **'No credentials'**
  String get noCredentials;

  /// No description provided for @lightMode.
  ///
  /// In en, this message translates to:
  /// **'Light Mode'**
  String get lightMode;

  /// No description provided for @darkMode.
  ///
  /// In en, this message translates to:
  /// **'Dark Mode'**
  String get darkMode;

  /// No description provided for @globalSearch.
  ///
  /// In en, this message translates to:
  /// **'Search (Ctrl+K)'**
  String get globalSearch;

  /// No description provided for @configPageGeneral.
  ///
  /// In en, this message translates to:
  /// **'General'**
  String get configPageGeneral;

  /// No description provided for @configPageLanguage.
  ///
  /// In en, this message translates to:
  /// **'Language'**
  String get configPageLanguage;

  /// No description provided for @configPageProviders.
  ///
  /// In en, this message translates to:
  /// **'Providers'**
  String get configPageProviders;

  /// No description provided for @configPageModels.
  ///
  /// In en, this message translates to:
  /// **'Models'**
  String get configPageModels;

  /// No description provided for @configPagePlanner.
  ///
  /// In en, this message translates to:
  /// **'Planner'**
  String get configPagePlanner;

  /// No description provided for @configPageExecutor.
  ///
  /// In en, this message translates to:
  /// **'Executor'**
  String get configPageExecutor;

  /// No description provided for @configPageMemory.
  ///
  /// In en, this message translates to:
  /// **'Memory'**
  String get configPageMemory;

  /// No description provided for @configPageChannels.
  ///
  /// In en, this message translates to:
  /// **'Channels'**
  String get configPageChannels;

  /// No description provided for @configPageSecurity.
  ///
  /// In en, this message translates to:
  /// **'Security'**
  String get configPageSecurity;

  /// No description provided for @configPageWeb.
  ///
  /// In en, this message translates to:
  /// **'Web'**
  String get configPageWeb;

  /// No description provided for @configPageMcp.
  ///
  /// In en, this message translates to:
  /// **'MCP'**
  String get configPageMcp;

  /// No description provided for @configPageCron.
  ///
  /// In en, this message translates to:
  /// **'Cron'**
  String get configPageCron;

  /// No description provided for @configPageDatabase.
  ///
  /// In en, this message translates to:
  /// **'Database'**
  String get configPageDatabase;

  /// No description provided for @configPageLogging.
  ///
  /// In en, this message translates to:
  /// **'Logging'**
  String get configPageLogging;

  /// No description provided for @configPagePrompts.
  ///
  /// In en, this message translates to:
  /// **'Prompts'**
  String get configPagePrompts;

  /// No description provided for @configPageAgents.
  ///
  /// In en, this message translates to:
  /// **'Agents'**
  String get configPageAgents;

  /// No description provided for @configPageBindings.
  ///
  /// In en, this message translates to:
  /// **'Bindings'**
  String get configPageBindings;

  /// No description provided for @configPageSystem.
  ///
  /// In en, this message translates to:
  /// **'System'**
  String get configPageSystem;

  /// No description provided for @configTitle.
  ///
  /// In en, this message translates to:
  /// **'Configuration'**
  String get configTitle;

  /// No description provided for @reloadFromBackend.
  ///
  /// In en, this message translates to:
  /// **'Reload config from backend'**
  String get reloadFromBackend;

  /// No description provided for @saveCtrlS.
  ///
  /// In en, this message translates to:
  /// **'Save (Ctrl+S)'**
  String get saveCtrlS;

  /// No description provided for @savedWithErrors.
  ///
  /// In en, this message translates to:
  /// **'Saved with errors in: {sections}'**
  String savedWithErrors(String sections);

  /// No description provided for @saveFailed.
  ///
  /// In en, this message translates to:
  /// **'Save failed'**
  String get saveFailed;

  /// No description provided for @fieldOwnerName.
  ///
  /// In en, this message translates to:
  /// **'Owner Name'**
  String get fieldOwnerName;

  /// No description provided for @fieldOperationMode.
  ///
  /// In en, this message translates to:
  /// **'Operation Mode'**
  String get fieldOperationMode;

  /// No description provided for @fieldCostTracking.
  ///
  /// In en, this message translates to:
  /// **'Cost Tracking'**
  String get fieldCostTracking;

  /// No description provided for @fieldDailyBudget.
  ///
  /// In en, this message translates to:
  /// **'Daily Budget (USD)'**
  String get fieldDailyBudget;

  /// No description provided for @fieldMonthlyBudget.
  ///
  /// In en, this message translates to:
  /// **'Monthly Budget (USD)'**
  String get fieldMonthlyBudget;

  /// No description provided for @fieldLlmBackend.
  ///
  /// In en, this message translates to:
  /// **'LLM Backend'**
  String get fieldLlmBackend;

  /// No description provided for @fieldPrimaryProvider.
  ///
  /// In en, this message translates to:
  /// **'Primary LLM provider'**
  String get fieldPrimaryProvider;

  /// No description provided for @fieldApiKey.
  ///
  /// In en, this message translates to:
  /// **'API Key'**
  String get fieldApiKey;

  /// No description provided for @fieldBaseUrl.
  ///
  /// In en, this message translates to:
  /// **'Base URL'**
  String get fieldBaseUrl;

  /// No description provided for @fieldModelName.
  ///
  /// In en, this message translates to:
  /// **'Model Name'**
  String get fieldModelName;

  /// No description provided for @fieldContextWindow.
  ///
  /// In en, this message translates to:
  /// **'Context Window'**
  String get fieldContextWindow;

  /// No description provided for @fieldTemperature.
  ///
  /// In en, this message translates to:
  /// **'Temperature'**
  String get fieldTemperature;

  /// No description provided for @fieldMaxIterations.
  ///
  /// In en, this message translates to:
  /// **'Max Iterations'**
  String get fieldMaxIterations;

  /// No description provided for @fieldEnabled.
  ///
  /// In en, this message translates to:
  /// **'Enabled'**
  String get fieldEnabled;

  /// No description provided for @fieldPort.
  ///
  /// In en, this message translates to:
  /// **'Port'**
  String get fieldPort;

  /// No description provided for @fieldHost.
  ///
  /// In en, this message translates to:
  /// **'Host'**
  String get fieldHost;

  /// No description provided for @fieldPassword.
  ///
  /// In en, this message translates to:
  /// **'Password'**
  String get fieldPassword;

  /// No description provided for @fieldUser.
  ///
  /// In en, this message translates to:
  /// **'User'**
  String get fieldUser;

  /// No description provided for @fieldTimeout.
  ///
  /// In en, this message translates to:
  /// **'Timeout'**
  String get fieldTimeout;

  /// No description provided for @fieldLevel.
  ///
  /// In en, this message translates to:
  /// **'Level'**
  String get fieldLevel;

  /// No description provided for @sectionSearchBackends.
  ///
  /// In en, this message translates to:
  /// **'Search Backends'**
  String get sectionSearchBackends;

  /// No description provided for @sectionDomainFilters.
  ///
  /// In en, this message translates to:
  /// **'Domain Filters'**
  String get sectionDomainFilters;

  /// No description provided for @sectionFetchLimits.
  ///
  /// In en, this message translates to:
  /// **'Fetch Limits'**
  String get sectionFetchLimits;

  /// No description provided for @sectionSearchLimits.
  ///
  /// In en, this message translates to:
  /// **'Search Limits'**
  String get sectionSearchLimits;

  /// No description provided for @sectionHttpLimits.
  ///
  /// In en, this message translates to:
  /// **'HTTP Request Limits'**
  String get sectionHttpLimits;

  /// No description provided for @sectionVoice.
  ///
  /// In en, this message translates to:
  /// **'Voice'**
  String get sectionVoice;

  /// No description provided for @sectionHeartbeat.
  ///
  /// In en, this message translates to:
  /// **'Heartbeat'**
  String get sectionHeartbeat;

  /// No description provided for @sectionPlugins.
  ///
  /// In en, this message translates to:
  /// **'Plugins'**
  String get sectionPlugins;

  /// No description provided for @sectionCronJobs.
  ///
  /// In en, this message translates to:
  /// **'Cron Jobs'**
  String get sectionCronJobs;

  /// No description provided for @sectionPromptEvolution.
  ///
  /// In en, this message translates to:
  /// **'Prompt Evolution'**
  String get sectionPromptEvolution;

  /// No description provided for @addItem.
  ///
  /// In en, this message translates to:
  /// **'Add'**
  String get addItem;

  /// No description provided for @removeItem.
  ///
  /// In en, this message translates to:
  /// **'Remove'**
  String get removeItem;

  /// No description provided for @translatePrompts.
  ///
  /// In en, this message translates to:
  /// **'Translate Prompts via Ollama'**
  String get translatePrompts;

  /// No description provided for @translating.
  ///
  /// In en, this message translates to:
  /// **'Translating...'**
  String get translating;

  /// No description provided for @promptsTranslated.
  ///
  /// In en, this message translates to:
  /// **'Prompts translated'**
  String get promptsTranslated;

  /// No description provided for @copiedToClipboard.
  ///
  /// In en, this message translates to:
  /// **'Config copied to clipboard'**
  String get copiedToClipboard;

  /// No description provided for @configImported.
  ///
  /// In en, this message translates to:
  /// **'Config imported'**
  String get configImported;

  /// No description provided for @restartInitiated.
  ///
  /// In en, this message translates to:
  /// **'Restart initiated'**
  String get restartInitiated;

  /// No description provided for @factoryResetComplete.
  ///
  /// In en, this message translates to:
  /// **'Factory reset complete'**
  String get factoryResetComplete;

  /// No description provided for @factoryResetConfirmMsg.
  ///
  /// In en, this message translates to:
  /// **'This will reset ALL configuration to factory defaults. Continue?'**
  String get factoryResetConfirmMsg;

  /// No description provided for @languageEnglish.
  ///
  /// In en, this message translates to:
  /// **'English'**
  String get languageEnglish;

  /// No description provided for @languageGerman.
  ///
  /// In en, this message translates to:
  /// **'Deutsch'**
  String get languageGerman;

  /// No description provided for @languageChinese.
  ///
  /// In en, this message translates to:
  /// **'中文'**
  String get languageChinese;

  /// No description provided for @languageArabic.
  ///
  /// In en, this message translates to:
  /// **'العربية'**
  String get languageArabic;

  /// No description provided for @uiAndPromptLanguage.
  ///
  /// In en, this message translates to:
  /// **'UI and prompt language'**
  String get uiAndPromptLanguage;

  /// No description provided for @learningTitle.
  ///
  /// In en, this message translates to:
  /// **'Learning'**
  String get learningTitle;

  /// No description provided for @knowledgeGaps.
  ///
  /// In en, this message translates to:
  /// **'Knowledge Gaps'**
  String get knowledgeGaps;

  /// No description provided for @explorationQueue.
  ///
  /// In en, this message translates to:
  /// **'Exploration Queue'**
  String get explorationQueue;

  /// No description provided for @filesProcessed.
  ///
  /// In en, this message translates to:
  /// **'Files Processed'**
  String get filesProcessed;

  /// No description provided for @entitiesCreated.
  ///
  /// In en, this message translates to:
  /// **'Entities Created'**
  String get entitiesCreated;

  /// No description provided for @confidenceUpdates.
  ///
  /// In en, this message translates to:
  /// **'Confidence Updates'**
  String get confidenceUpdates;

  /// No description provided for @openGaps.
  ///
  /// In en, this message translates to:
  /// **'Open Gaps'**
  String get openGaps;

  /// No description provided for @importance.
  ///
  /// In en, this message translates to:
  /// **'Importance'**
  String get importance;

  /// No description provided for @curiosity.
  ///
  /// In en, this message translates to:
  /// **'Curiosity'**
  String get curiosity;

  /// No description provided for @explore.
  ///
  /// In en, this message translates to:
  /// **'Explore'**
  String get explore;

  /// No description provided for @dismiss.
  ///
  /// In en, this message translates to:
  /// **'Dismiss'**
  String get dismiss;

  /// No description provided for @noGaps.
  ///
  /// In en, this message translates to:
  /// **'No knowledge gaps detected'**
  String get noGaps;

  /// No description provided for @noTasks.
  ///
  /// In en, this message translates to:
  /// **'No exploration tasks'**
  String get noTasks;

  /// No description provided for @confidenceHistory.
  ///
  /// In en, this message translates to:
  /// **'Confidence History'**
  String get confidenceHistory;

  /// No description provided for @feedback.
  ///
  /// In en, this message translates to:
  /// **'Feedback'**
  String get feedback;

  /// No description provided for @positive.
  ///
  /// In en, this message translates to:
  /// **'Positive'**
  String get positive;

  /// No description provided for @negative.
  ///
  /// In en, this message translates to:
  /// **'Negative'**
  String get negative;

  /// No description provided for @correction.
  ///
  /// In en, this message translates to:
  /// **'Correction'**
  String get correction;

  /// No description provided for @adminLearningSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Active learning & curiosity'**
  String get adminLearningSubtitle;

  /// No description provided for @watchDirectories.
  ///
  /// In en, this message translates to:
  /// **'Watch Directories'**
  String get watchDirectories;

  /// No description provided for @directoryExists.
  ///
  /// In en, this message translates to:
  /// **'Directory exists'**
  String get directoryExists;

  /// No description provided for @directoryMissing.
  ///
  /// In en, this message translates to:
  /// **'Directory not found'**
  String get directoryMissing;

  /// No description provided for @qaKnowledgeBase.
  ///
  /// In en, this message translates to:
  /// **'Q&A'**
  String get qaKnowledgeBase;

  /// No description provided for @lineage.
  ///
  /// In en, this message translates to:
  /// **'Lineage'**
  String get lineage;

  /// No description provided for @question.
  ///
  /// In en, this message translates to:
  /// **'Question'**
  String get question;

  /// No description provided for @answer.
  ///
  /// In en, this message translates to:
  /// **'Answer'**
  String get answer;

  /// No description provided for @topic.
  ///
  /// In en, this message translates to:
  /// **'Topic'**
  String get topic;

  /// No description provided for @addQA.
  ///
  /// In en, this message translates to:
  /// **'Add Q&A'**
  String get addQA;

  /// No description provided for @verify.
  ///
  /// In en, this message translates to:
  /// **'Verify'**
  String get verify;

  /// No description provided for @source.
  ///
  /// In en, this message translates to:
  /// **'Source'**
  String get source;

  /// No description provided for @noQAPairs.
  ///
  /// In en, this message translates to:
  /// **'No knowledge entries'**
  String get noQAPairs;

  /// No description provided for @noLineage.
  ///
  /// In en, this message translates to:
  /// **'No lineage data'**
  String get noLineage;

  /// No description provided for @entityLineage.
  ///
  /// In en, this message translates to:
  /// **'Entity Lineage'**
  String get entityLineage;

  /// No description provided for @recentChanges.
  ///
  /// In en, this message translates to:
  /// **'Recent Changes'**
  String get recentChanges;

  /// No description provided for @created.
  ///
  /// In en, this message translates to:
  /// **'Created'**
  String get created;

  /// No description provided for @updated.
  ///
  /// In en, this message translates to:
  /// **'Updated'**
  String get updated;

  /// No description provided for @decayed.
  ///
  /// In en, this message translates to:
  /// **'Decayed'**
  String get decayed;

  /// No description provided for @runExploration.
  ///
  /// In en, this message translates to:
  /// **'Run Exploration'**
  String get runExploration;

  /// No description provided for @explorationComplete.
  ///
  /// In en, this message translates to:
  /// **'Exploration complete'**
  String get explorationComplete;

  /// No description provided for @activityChart.
  ///
  /// In en, this message translates to:
  /// **'Activity'**
  String get activityChart;

  /// No description provided for @stopped.
  ///
  /// In en, this message translates to:
  /// **'Stopped'**
  String get stopped;

  /// No description provided for @requestsOverTime.
  ///
  /// In en, this message translates to:
  /// **'Requests over time'**
  String get requestsOverTime;

  /// No description provided for @teachCognithor.
  ///
  /// In en, this message translates to:
  /// **'Teach Cognithor'**
  String get teachCognithor;

  /// No description provided for @uploadFile.
  ///
  /// In en, this message translates to:
  /// **'Upload File'**
  String get uploadFile;

  /// No description provided for @learnFromUrl.
  ///
  /// In en, this message translates to:
  /// **'Learn from Website'**
  String get learnFromUrl;

  /// No description provided for @learnFromYoutube.
  ///
  /// In en, this message translates to:
  /// **'Learn from Video'**
  String get learnFromYoutube;

  /// No description provided for @dropFilesHere.
  ///
  /// In en, this message translates to:
  /// **'Drop files here or click to browse'**
  String get dropFilesHere;

  /// No description provided for @learningHistory.
  ///
  /// In en, this message translates to:
  /// **'Learning History'**
  String get learningHistory;

  /// No description provided for @chunksLearned.
  ///
  /// In en, this message translates to:
  /// **'{count} chunks learned'**
  String chunksLearned(String count);

  /// No description provided for @processingContent.
  ///
  /// In en, this message translates to:
  /// **'Processing content...'**
  String get processingContent;

  /// No description provided for @learnSuccess.
  ///
  /// In en, this message translates to:
  /// **'Successfully learned!'**
  String get learnSuccess;

  /// No description provided for @learnFailed.
  ///
  /// In en, this message translates to:
  /// **'Learning failed'**
  String get learnFailed;

  /// No description provided for @enterUrl.
  ///
  /// In en, this message translates to:
  /// **'Enter website URL...'**
  String get enterUrl;

  /// No description provided for @enterYoutubeUrl.
  ///
  /// In en, this message translates to:
  /// **'Enter YouTube URL...'**
  String get enterYoutubeUrl;

  /// No description provided for @adminTeachSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Upload files, URLs, videos'**
  String get adminTeachSubtitle;

  /// No description provided for @newSkill.
  ///
  /// In en, this message translates to:
  /// **'New Skill'**
  String get newSkill;

  /// No description provided for @editSkill.
  ///
  /// In en, this message translates to:
  /// **'Edit Skill'**
  String get editSkill;

  /// No description provided for @createSkill.
  ///
  /// In en, this message translates to:
  /// **'Create Skill'**
  String get createSkill;

  /// No description provided for @deleteSkill.
  ///
  /// In en, this message translates to:
  /// **'Delete Skill'**
  String get deleteSkill;

  /// No description provided for @skillName.
  ///
  /// In en, this message translates to:
  /// **'Name'**
  String get skillName;

  /// No description provided for @skillBody.
  ///
  /// In en, this message translates to:
  /// **'Skill Body (Markdown)'**
  String get skillBody;

  /// No description provided for @triggerKeywords.
  ///
  /// In en, this message translates to:
  /// **'Trigger Keywords'**
  String get triggerKeywords;

  /// No description provided for @requiredTools.
  ///
  /// In en, this message translates to:
  /// **'Required Tools'**
  String get requiredTools;

  /// No description provided for @modelPreference.
  ///
  /// In en, this message translates to:
  /// **'Model Preference'**
  String get modelPreference;

  /// No description provided for @skillSaved.
  ///
  /// In en, this message translates to:
  /// **'Skill saved successfully'**
  String get skillSaved;

  /// No description provided for @skillCreated.
  ///
  /// In en, this message translates to:
  /// **'Skill created successfully'**
  String get skillCreated;

  /// No description provided for @skillDeleted.
  ///
  /// In en, this message translates to:
  /// **'Skill deleted'**
  String get skillDeleted;

  /// No description provided for @confirmDeleteSkill.
  ///
  /// In en, this message translates to:
  /// **'Are you sure you want to delete this skill? This cannot be undone.'**
  String get confirmDeleteSkill;

  /// No description provided for @discardChanges.
  ///
  /// In en, this message translates to:
  /// **'Discard changes?'**
  String get discardChanges;

  /// No description provided for @discardChangesBody.
  ///
  /// In en, this message translates to:
  /// **'You have unsaved changes. Discard them?'**
  String get discardChangesBody;

  /// No description provided for @totalUses.
  ///
  /// In en, this message translates to:
  /// **'Total Uses'**
  String get totalUses;

  /// No description provided for @lastUsed.
  ///
  /// In en, this message translates to:
  /// **'Last Used'**
  String get lastUsed;

  /// No description provided for @commaSeparated.
  ///
  /// In en, this message translates to:
  /// **'Comma-separated'**
  String get commaSeparated;

  /// No description provided for @skillBodyHint.
  ///
  /// In en, this message translates to:
  /// **'Write skill instructions in Markdown...'**
  String get skillBodyHint;

  /// No description provided for @metadata.
  ///
  /// In en, this message translates to:
  /// **'Metadata'**
  String get metadata;

  /// No description provided for @statistics.
  ///
  /// In en, this message translates to:
  /// **'Statistics'**
  String get statistics;

  /// No description provided for @builtInSkill.
  ///
  /// In en, this message translates to:
  /// **'Built-in skill (read-only)'**
  String get builtInSkill;

  /// No description provided for @exportSkillMd.
  ///
  /// In en, this message translates to:
  /// **'Export as SKILL.md'**
  String get exportSkillMd;

  /// No description provided for @skillExported.
  ///
  /// In en, this message translates to:
  /// **'Skill exported to clipboard'**
  String get skillExported;

  /// No description provided for @general.
  ///
  /// In en, this message translates to:
  /// **'General'**
  String get general;

  /// No description provided for @productivity.
  ///
  /// In en, this message translates to:
  /// **'Productivity'**
  String get productivity;

  /// No description provided for @research.
  ///
  /// In en, this message translates to:
  /// **'Research'**
  String get research;

  /// No description provided for @analysis.
  ///
  /// In en, this message translates to:
  /// **'Analysis'**
  String get analysis;

  /// No description provided for @development.
  ///
  /// In en, this message translates to:
  /// **'Development'**
  String get development;

  /// No description provided for @automation.
  ///
  /// In en, this message translates to:
  /// **'Automation'**
  String get automation;

  /// No description provided for @newAgent.
  ///
  /// In en, this message translates to:
  /// **'New Agent'**
  String get newAgent;

  /// No description provided for @editAgent.
  ///
  /// In en, this message translates to:
  /// **'Edit Agent'**
  String get editAgent;

  /// No description provided for @deleteAgent.
  ///
  /// In en, this message translates to:
  /// **'Delete Agent'**
  String get deleteAgent;

  /// No description provided for @confirmDeleteAgent.
  ///
  /// In en, this message translates to:
  /// **'Are you sure you want to delete this agent? This cannot be undone.'**
  String get confirmDeleteAgent;

  /// No description provided for @agentCreated.
  ///
  /// In en, this message translates to:
  /// **'Agent created successfully'**
  String get agentCreated;

  /// No description provided for @agentSaved.
  ///
  /// In en, this message translates to:
  /// **'Agent saved successfully'**
  String get agentSaved;

  /// No description provided for @agentDeleted.
  ///
  /// In en, this message translates to:
  /// **'Agent deleted'**
  String get agentDeleted;

  /// No description provided for @displayName.
  ///
  /// In en, this message translates to:
  /// **'Display Name'**
  String get displayName;

  /// No description provided for @preferredModel.
  ///
  /// In en, this message translates to:
  /// **'Preferred Model'**
  String get preferredModel;

  /// No description provided for @sandboxTimeout.
  ///
  /// In en, this message translates to:
  /// **'Sandbox Timeout (s)'**
  String get sandboxTimeout;

  /// No description provided for @sandboxNetwork.
  ///
  /// In en, this message translates to:
  /// **'Sandbox Network'**
  String get sandboxNetwork;

  /// No description provided for @canDelegateTo.
  ///
  /// In en, this message translates to:
  /// **'Can Delegate To'**
  String get canDelegateTo;

  /// No description provided for @cannotDeleteDefault.
  ///
  /// In en, this message translates to:
  /// **'Cannot delete the default agent'**
  String get cannotDeleteDefault;
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  Future<AppLocalizations> load(Locale locale) {
    return SynchronousFuture<AppLocalizations>(lookupAppLocalizations(locale));
  }

  @override
  bool isSupported(Locale locale) =>
      <String>['ar', 'de', 'en', 'zh'].contains(locale.languageCode);

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}

AppLocalizations lookupAppLocalizations(Locale locale) {
  // Lookup logic when only language code is specified.
  switch (locale.languageCode) {
    case 'ar':
      return AppLocalizationsAr();
    case 'de':
      return AppLocalizationsDe();
    case 'en':
      return AppLocalizationsEn();
    case 'zh':
      return AppLocalizationsZh();
  }

  throw FlutterError(
    'AppLocalizations.delegate failed to load unsupported locale "$locale". This is likely '
    'an issue with the localizations generation tool. Please file an issue '
    'on GitHub with a reproducible sample app and the gen-l10n configuration '
    'that was used.',
  );
}
