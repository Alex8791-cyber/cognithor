// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Arabic (`ar`).
class AppLocalizationsAr extends AppLocalizations {
  AppLocalizationsAr([String locale = 'ar']) : super(locale);

  @override
  String get appTitle => 'جارفيس';

  @override
  String get chat => 'محادثة';

  @override
  String get settings => 'الإعدادات';

  @override
  String get identity => 'الهوية';

  @override
  String get workflows => 'سير العمل';

  @override
  String get memory => 'الذاكرة';

  @override
  String get monitoring => 'المراقبة';

  @override
  String get skills => 'المهارات';

  @override
  String get config => 'التكوين';

  @override
  String get sendMessage => 'اكتب رسالة...';

  @override
  String get send => 'إرسال';

  @override
  String get cancel => 'إلغاء';

  @override
  String get approve => 'موافقة';

  @override
  String get reject => 'رفض';

  @override
  String get retry => 'إعادة المحاولة';

  @override
  String get close => 'إغلاق';

  @override
  String get save => 'حفظ';

  @override
  String get delete => 'حذف';

  @override
  String get loading => 'جاري التحميل...';

  @override
  String get connecting => 'جاري الاتصال...';

  @override
  String get approvalTitle => 'مطلوب موافقة';

  @override
  String approvalBody(String tool) {
    return 'الأداة $tool تريد تنفيذ:';
  }

  @override
  String approvalReason(String reason) {
    return 'السبب: $reason';
  }

  @override
  String get statusThinking => 'يفكر...';

  @override
  String get statusExecuting => 'ينفذ...';

  @override
  String get statusFinishing => 'ينهي...';

  @override
  String get voiceMessage => 'رسالة صوتية';

  @override
  String fileUpload(String name) {
    return 'ملف: $name';
  }

  @override
  String get connectionError => 'لا يمكن الوصول إلى الخادم';

  @override
  String connectionErrorDetail(String url) {
    return 'تأكد من أن خادم جارفيس يعمل على $url';
  }

  @override
  String get authFailed => 'فشل المصادقة';

  @override
  String get tokenExpired => 'انتهت الجلسة. جاري إعادة الاتصال...';

  @override
  String get serverUrl => 'عنوان الخادم';

  @override
  String get serverUrlHint => 'http://localhost:8741';

  @override
  String version(String version) {
    return 'الإصدار $version';
  }

  @override
  String get errorGeneric => 'حدث خطأ ما';

  @override
  String get errorNetwork => 'خطأ في الشبكة. تحقق من اتصالك.';

  @override
  String get errorTimeout => 'انتهت مهلة الطلب';

  @override
  String get errorUnauthorized => 'غير مصرح. يرجى إعادة الاتصال.';

  @override
  String get errorServerDown => 'الخادم غير متاح';

  @override
  String get identityNotAvailable => 'طبقة الهوية غير متوفرة';

  @override
  String get identityInstallHint =>
      'قم بالتثبيت: pip install cognithor[identity]';

  @override
  String get identityEnergy => 'الطاقة';

  @override
  String get identityInteractions => 'التفاعلات';

  @override
  String get identityMemories => 'الذكريات';

  @override
  String get identityCharacterStrength => 'قوة الشخصية';

  @override
  String get identityFrozen => 'مجمد';

  @override
  String get identityActive => 'نشط';

  @override
  String get identityDream => 'دورة الحلم';

  @override
  String get identityFreeze => 'تجميد';

  @override
  String get identityUnfreeze => 'إلغاء التجميد';

  @override
  String get identityReset => 'إعادة ضبط';

  @override
  String get identityResetConfirm => 'إعادة ضبط الهوية؟ سيتم فقدان الذكريات.';

  @override
  String get pipelinePlan => 'التخطيط';

  @override
  String get pipelineGate => 'حارس البوابة';

  @override
  String get pipelineExecute => 'التنفيذ';

  @override
  String get pipelineReplan => 'إعادة التخطيط';

  @override
  String get pipelineComplete => 'مكتمل';

  @override
  String get canvasTitle => 'اللوحة';

  @override
  String get canvasClose => 'إغلاق اللوحة';

  @override
  String get models => 'النماذج';

  @override
  String get channels => 'القنوات';

  @override
  String get security => 'الأمان';

  @override
  String get reload => 'إعادة تحميل';

  @override
  String get reloading => 'جاري إعادة التحميل...';

  @override
  String get configSaved => 'تم إعادة تحميل التكوين';

  @override
  String get configError => 'خطأ في التكوين';

  @override
  String get uptime => 'وقت التشغيل';

  @override
  String get activeSessions => 'الجلسات النشطة';

  @override
  String get totalRequests => 'إجمالي الطلبات';

  @override
  String get events => 'الأحداث';

  @override
  String get noEvents => 'لا توجد أحداث مسجلة';

  @override
  String get severity => 'الخطورة';

  @override
  String get refreshing => 'تحديث تلقائي: 10 ثوانٍ';

  @override
  String get noData => 'لا توجد بيانات';

  @override
  String get notAvailable => 'غير متوفر';

  @override
  String get dashboard => 'لوحة المعلومات';

  @override
  String get systemOverview => 'نظرة عامة على النظام';

  @override
  String get cpuUsage => 'استخدام المعالج';

  @override
  String get memoryUsage => 'استخدام الذاكرة';

  @override
  String get responseTime => 'وقت الاستجابة';

  @override
  String get toolExecutions => 'تنفيذ الأدوات';

  @override
  String get successRate => 'معدل النجاح';

  @override
  String get recentEvents => 'الأحداث الأخيرة';

  @override
  String get lastUpdated => 'آخر تحديث';

  @override
  String get systemHealth => 'صحة النظام';

  @override
  String get performance => 'الأداء';

  @override
  String get trends => 'الاتجاهات';

  @override
  String get marketplace => 'السوق';

  @override
  String get featured => 'مميز';

  @override
  String get trending => 'رائج';

  @override
  String get categories => 'الفئات';

  @override
  String get searchSkills => 'البحث عن مهارات...';

  @override
  String get installed => 'مثبت';

  @override
  String get installSkill => 'تثبيت';

  @override
  String get uninstallSkill => 'إلغاء التثبيت';

  @override
  String get installing => 'جارٍ التثبيت...';

  @override
  String get skillDetails => 'تفاصيل المهارة';

  @override
  String get reviews => 'التقييمات';

  @override
  String get noSkills => 'لم يتم العثور على مهارات';

  @override
  String get browseMarketplace => 'تصفح السوق';

  @override
  String get verified => 'موثق';

  @override
  String get downloads => 'التنزيلات';

  @override
  String get rating => 'التقييم';

  @override
  String get memoryTitle => 'الذاكرة';

  @override
  String get knowledgeGraph => 'رسم المعرفة';

  @override
  String get entities => 'الكيانات';

  @override
  String get relations => 'العلاقات';

  @override
  String get hygiene => 'النظافة';

  @override
  String get quarantine => 'الحجر';

  @override
  String get scanMemory => 'فحص';

  @override
  String get scanning => 'جارٍ الفحص...';

  @override
  String get explainability => 'قابلية التفسير';

  @override
  String get decisionTrails => 'مسارات القرار';

  @override
  String get lowTrust => 'ثقة منخفضة';

  @override
  String get graphStats => 'إحصائيات الرسم';

  @override
  String get noEntities => 'لا توجد كيانات';

  @override
  String get noTrails => 'لا توجد مسارات';

  @override
  String get scanComplete => 'اكتمل الفحص';

  @override
  String get threats => 'التهديدات';

  @override
  String get threatRate => 'معدل التهديدات';

  @override
  String get totalScans => 'إجمالي عمليات الفحص';

  @override
  String get integrity => 'النزاهة';

  @override
  String get securityTitle => 'الأمان';

  @override
  String get complianceTitle => 'الامتثال';

  @override
  String get rolesTitle => 'الأدوار';

  @override
  String get permissions => 'الصلاحيات';

  @override
  String get auditLog => 'سجل التدقيق';

  @override
  String get redTeam => 'الفريق الأحمر';

  @override
  String get scanStatus => 'حالة الفحص';

  @override
  String get complianceReport => 'تقرير الامتثال';

  @override
  String get decisionsTitle => 'القرارات';

  @override
  String get remediations => 'الإصلاحات';

  @override
  String get openStatus => 'مفتوح';

  @override
  String get inProgressStatus => 'قيد التنفيذ';

  @override
  String get resolvedStatus => 'تم الحل';

  @override
  String get overdueStatus => 'متأخر';

  @override
  String get approvalRate => 'معدل الموافقة';

  @override
  String get flaggedCount => 'مُعلَّم';

  @override
  String get transparency => 'الشفافية';

  @override
  String get euAiAct => 'قانون الذكاء الاصطناعي الأوروبي';

  @override
  String get dsgvo => 'اللائحة العامة لحماية البيانات';

  @override
  String get runScan => 'تشغيل الفحص';

  @override
  String get adminTitle => 'الإدارة';

  @override
  String get agentsTitle => 'الوكلاء';

  @override
  String get modelsTitle => 'النماذج';

  @override
  String get systemTitle => 'النظام';

  @override
  String get workflowsTitle => 'سير العمل';

  @override
  String get vaultTitle => 'الخزنة';

  @override
  String get credentialsTitle => 'بيانات الاعتماد';

  @override
  String get bindingsTitle => 'الارتباطات';

  @override
  String get connectorsTitle => 'الموصلات';

  @override
  String get commandsTitle => 'الأوامر';

  @override
  String get isolationTitle => 'العزل';

  @override
  String get sandboxTitle => 'البيئة المعزولة';

  @override
  String get circlesTitle => 'الدوائر';

  @override
  String get wizardsTitle => 'المعالجات';

  @override
  String get systemStatus => 'حالة النظام';

  @override
  String get shutdownServer => 'إيقاف الخادم';

  @override
  String get shutdownConfirm => 'هل أنت متأكد من إيقاف الخادم؟';

  @override
  String get startComponent => 'تشغيل';

  @override
  String get stopComponent => 'إيقاف';

  @override
  String get selectTemplate => 'اختر قالبًا';

  @override
  String get workflowStarted => 'بدأ سير العمل';

  @override
  String get noWorkflows => 'لا توجد سير عمل';

  @override
  String get templates => 'القوالب';

  @override
  String get running => 'قيد التشغيل';

  @override
  String get vaultStats => 'إحصائيات الخزنة';

  @override
  String get totalEntries => 'إجمالي الإدخالات';

  @override
  String get agentVaults => 'خزائن الوكلاء';

  @override
  String get noVaults => 'لا توجد خزائن';

  @override
  String get availableModels => 'النماذج المتاحة';

  @override
  String get modelStats => 'إحصائيات النماذج';

  @override
  String get providers => 'مزودو الخدمة';

  @override
  String get capabilities => 'القدرات';

  @override
  String get plannerModel => 'المخطط';

  @override
  String get executorModel => 'المنفذ';

  @override
  String get coderModel => 'المبرمج';

  @override
  String get embeddingModel => 'التضمين';

  @override
  String get configured => 'مكوَّن';

  @override
  String get modelWarnings => 'تحذيرات';

  @override
  String get identityDreamCycle => 'دورة الحلم';

  @override
  String get identityGenesisAnchors => 'مراسي التكوين';

  @override
  String get identityNoAnchors => 'لا توجد مراسي تكوين';

  @override
  String get identityPersonality => 'الشخصية';

  @override
  String get identityCognitive => 'الحالة الإدراكية';

  @override
  String get identityEmotional => 'الحالة العاطفية';

  @override
  String get identitySomatic => 'الحالة الجسدية';

  @override
  String get identityNarrative => 'السرد';

  @override
  String get identityExistential => 'الوجودية';

  @override
  String get identityPredictive => 'التنبؤية';

  @override
  String get identityEpistemic => 'المعرفية';

  @override
  String get identityBiases => 'التحيزات النشطة';

  @override
  String get search => 'بحث';

  @override
  String get filter => 'تصفية';

  @override
  String get sortBy => 'ترتيب حسب';

  @override
  String get refresh => 'تحديث';

  @override
  String get export => 'تصدير';

  @override
  String get viewAll => 'عرض الكل';

  @override
  String get details => 'التفاصيل';

  @override
  String get back => 'رجوع';

  @override
  String get confirm => 'تأكيد';

  @override
  String get actions => 'الإجراءات';

  @override
  String get statusLabel => 'الحالة';

  @override
  String get enabled => 'مفعّل';

  @override
  String get disabled => 'معطّل';

  @override
  String get total => 'الإجمالي';

  @override
  String get count => 'العدد';

  @override
  String get rate => 'المعدل';

  @override
  String get average => 'المتوسط';

  @override
  String get duration => 'المدة';

  @override
  String get timestamp => 'الطابع الزمني';

  @override
  String get severityLabel => 'مستوى الخطورة';

  @override
  String get critical => 'حرج';

  @override
  String get errorLabel => 'خطأ';

  @override
  String get warningLabel => 'تحذير';

  @override
  String get infoLabel => 'معلومات';

  @override
  String get successLabel => 'نجاح';

  @override
  String get unknownLabel => 'غير معروف';

  @override
  String get notConfigured => 'غير مكوَّن';

  @override
  String get comingSoon => 'قريبًا';

  @override
  String get beta => 'تجريبي';

  @override
  String get copyToClipboard => 'نسخ إلى الحافظة';

  @override
  String get copied => 'تم النسخ!';

  @override
  String get chatSettings => 'إعدادات المحادثة';

  @override
  String get clearChat => 'مسح المحادثة';

  @override
  String get voiceMode => 'وضع الصوت';

  @override
  String get fileUploadAction => 'رفع ملف';

  @override
  String get planDetails => 'تفاصيل الخطة';

  @override
  String get noMessages => 'لا توجد رسائل بعد';

  @override
  String get typeMessage => 'اكتب رسالة...';

  @override
  String get settingsTitle => 'الإعدادات';

  @override
  String get language => 'اللغة';

  @override
  String get theme => 'المظهر';

  @override
  String get about => 'حول';

  @override
  String get licenses => 'التراخيص';

  @override
  String get clearCache => 'مسح ذاكرة التخزين المؤقت';

  @override
  String get adminConfigSubtitle => 'إدارة التكوين';

  @override
  String get adminAgentsSubtitle => 'الوكلاء والملفات الشخصية';

  @override
  String get adminModelsSubtitle => 'نماذج LLM';

  @override
  String get adminSecuritySubtitle => 'الأمان والامتثال';

  @override
  String get adminWorkflowsSubtitle => 'الأتمتة';

  @override
  String get adminMemorySubtitle => 'رسم المعرفة';

  @override
  String get adminVaultSubtitle => 'الأسرار والمفاتيح';

  @override
  String get adminSystemSubtitle => 'حالة النظام';

  @override
  String get dashboardRefreshing => 'تحديث تلقائي: 15 ثانية';

  @override
  String get backendVersion => 'إصدار الخادم';

  @override
  String get modelInfo => 'معلومات النموذج';

  @override
  String get confidence => 'الثقة';

  @override
  String get rolesAccess => 'الأدوار والصلاحيات';

  @override
  String get loadMore => 'تحميل المزيد';

  @override
  String get actor => 'المنفذ';

  @override
  String get noAuditEntries => 'لا توجد سجلات تدقيق';

  @override
  String get allSeverities => 'جميع مستويات الخطورة';

  @override
  String get allActions => 'جميع الإجراءات';

  @override
  String get scanNotAvailable => 'الفحص غير متاح';

  @override
  String get lastScan => 'آخر فحص';

  @override
  String get scanResults => 'نتائج الفحص';

  @override
  String get compliant => 'متوافق';

  @override
  String get nonCompliant => 'غير متوافق';

  @override
  String get model => 'النموذج';

  @override
  String get temperature => 'درجة الحرارة';

  @override
  String get priority => 'الأولوية';

  @override
  String get allowedTools => 'الأدوات المسموحة';

  @override
  String get blockedTools => 'الأدوات المحظورة';

  @override
  String get noAgents => 'لا يوجد وكلاء مكوَّنين';

  @override
  String get description => 'الوصف';

  @override
  String get provider => 'المزوّد';

  @override
  String get noModels => 'لا توجد نماذج متاحة';

  @override
  String get owner => 'المالك';

  @override
  String get llmBackend => 'الواجهة الخلفية لـ LLM';

  @override
  String get components => 'المكونات';

  @override
  String get dangerZone => 'منطقة الخطر';

  @override
  String get reloadConfig => 'إعادة تحميل التكوين';

  @override
  String get runtimeInfo => 'وقت التشغيل';

  @override
  String get startWorkflow => 'بدء سير العمل';

  @override
  String get noCategories => 'لا توجد فئات';

  @override
  String templateCount(String count) {
    return '$count قوالب';
  }

  @override
  String get entityTypes => 'أنواع الكيانات';

  @override
  String get activeTrails => 'المسارات النشطة';

  @override
  String get completedTrails => 'مكتملة';

  @override
  String get lastAccessed => 'آخر وصول';

  @override
  String get author => 'المؤلف';

  @override
  String get noQuarantine => 'لا توجد عناصر محجورة';

  @override
  String get totalVaults => 'إجمالي الخزائن';

  @override
  String get scanNow => 'فحص الآن';

  @override
  String get startConversation => 'ابدأ محادثة';

  @override
  String get attachFile => 'إرفاق ملف';

  @override
  String get voiceModeHint => 'وضع الصوت قريبًا';

  @override
  String get canvasLabel => 'اللوحة';

  @override
  String get configGeneral => 'عام';

  @override
  String get configLanguage => 'اللغة';

  @override
  String get configProviders => 'المزودون';

  @override
  String get configModels => 'النماذج';

  @override
  String get configPlanner => 'المخطط';

  @override
  String get configExecutor => 'المنفذ';

  @override
  String get configMemory => 'الذاكرة';

  @override
  String get configChannels => 'القنوات';

  @override
  String get configSecurity => 'الأمان';

  @override
  String get configWeb => 'الويب';

  @override
  String get configMcp => 'MCP';

  @override
  String get configCron => 'المهام المجدولة';

  @override
  String get configDatabase => 'قاعدة البيانات';

  @override
  String get configLogging => 'السجلات';

  @override
  String get configPrompts => 'الأوامر';

  @override
  String get configAgents => 'الوكلاء';

  @override
  String get configBindings => 'الارتباطات';

  @override
  String get configSystem => 'النظام';

  @override
  String get ownerName => 'اسم المالك';

  @override
  String get operationMode => 'وضع التشغيل';

  @override
  String get costTracking => 'تتبع التكاليف';

  @override
  String get dailyBudget => 'الميزانية اليومية';

  @override
  String get monthlyBudget => 'الميزانية الشهرية';

  @override
  String get apiKey => 'مفتاح API';

  @override
  String get baseUrl => 'عنوان URL الأساسي';

  @override
  String get maxTokens => 'الحد الأقصى للرموز';

  @override
  String get timeout => 'المهلة';

  @override
  String get keepAlive => 'إبقاء الاتصال';

  @override
  String get contextWindow => 'نافذة السياق';

  @override
  String get vramGb => 'ذاكرة الفيديو (جيجابايت)';

  @override
  String get topP => 'Top P';

  @override
  String get maxIterations => 'الحد الأقصى للتكرارات';

  @override
  String get escalationAfter => 'التصعيد بعد';

  @override
  String get responseBudget => 'ميزانية رموز الاستجابة';

  @override
  String get policiesDir => 'مجلد السياسات';

  @override
  String get defaultRiskLevel => 'مستوى الخطر الافتراضي';

  @override
  String get maxBlockedRetries => 'الحد الأقصى لمحاولات الحظر';

  @override
  String get sandboxLevel => 'مستوى البيئة المعزولة';

  @override
  String get maxMemoryMb => 'الحد الأقصى للذاكرة (ميجابايت)';

  @override
  String get maxCpuSeconds => 'الحد الأقصى لثواني المعالج';

  @override
  String get allowedPaths => 'المسارات المسموحة';

  @override
  String get networkAccess => 'الوصول إلى الشبكة';

  @override
  String get envVars => 'متغيرات البيئة';

  @override
  String get defaultTimeout => 'المهلة الافتراضية';

  @override
  String get maxOutputChars => 'الحد الأقصى لأحرف الإخراج';

  @override
  String get maxRetries => 'الحد الأقصى لإعادة المحاولة';

  @override
  String get backoffDelay => 'تأخير التراجع';

  @override
  String get maxParallelTools => 'الحد الأقصى للأدوات المتوازية';

  @override
  String get chunkSize => 'حجم الجزء';

  @override
  String get chunkOverlap => 'تداخل الأجزاء';

  @override
  String get searchTopK => 'أفضل K نتائج بحث';

  @override
  String get searchWeights => 'أوزان البحث';

  @override
  String get vectorWeight => 'وزن المتجهات';

  @override
  String get bm25Weight => 'وزن BM25';

  @override
  String get graphWeight => 'وزن الرسم البياني';

  @override
  String get recencyHalfLife => 'عمر النصف للحداثة';

  @override
  String get compactionThreshold => 'حد الضغط';

  @override
  String get compactionKeepLast => 'الاحتفاظ بآخر عند الضغط';

  @override
  String get episodicRetention => 'الاحتفاظ بالأحداث';

  @override
  String get dynamicWeighting => 'الأوزان الديناميكية';

  @override
  String get voiceEnabled => 'الصوت مفعل';

  @override
  String get ttsBackend => 'محرك TTS';

  @override
  String get piperVoice => 'صوت Piper';

  @override
  String get piperLengthScale => 'مقياس طول Piper';

  @override
  String get wakeWordEnabled => 'كلمة التنبيه مفعلة';

  @override
  String get wakeWord => 'كلمة التنبيه';

  @override
  String get wakeWordBackend => 'محرك كلمة التنبيه';

  @override
  String get talkMode => 'وضع الحديث';

  @override
  String get autoListen => 'استماع تلقائي';

  @override
  String get blockedCommands => 'الأوامر المحظورة';

  @override
  String get credentialPatterns => 'أنماط بيانات الاعتماد';

  @override
  String get maxSubAgentDepth => 'الحد الأقصى لعمق الوكيل الفرعي';

  @override
  String get searchBackends => 'محركات البحث';

  @override
  String get domainFilters => 'فلاتر النطاقات';

  @override
  String get blocklist => 'القائمة السوداء';

  @override
  String get allowlist => 'القائمة البيضاء';

  @override
  String get httpLimits => 'حدود HTTP';

  @override
  String get maxFetchBytes => 'الحد الأقصى لبايتات الجلب';

  @override
  String get maxTextChars => 'الحد الأقصى لأحرف النص';

  @override
  String get fetchTimeout => 'مهلة الجلب';

  @override
  String get searchTimeout => 'مهلة البحث';

  @override
  String get maxSearchResults => 'الحد الأقصى لنتائج البحث';

  @override
  String get rateLimit => 'حد المعدل';

  @override
  String get mcpServers => 'خوادم MCP';

  @override
  String get a2aProtocol => 'بروتوكول A2A';

  @override
  String get remotes => 'الأجهزة البعيدة';

  @override
  String get heartbeat => 'نبض القلب';

  @override
  String get intervalMinutes => 'الفاصل الزمني (دقائق)';

  @override
  String get checklistFile => 'ملف قائمة التحقق';

  @override
  String get channel => 'القناة';

  @override
  String get plugins => 'الإضافات';

  @override
  String get skillsDir => 'مجلد المهارات';

  @override
  String get autoUpdate => 'تحديث تلقائي';

  @override
  String get cronJobs => 'المهام المجدولة';

  @override
  String get schedule => 'الجدول';

  @override
  String get command => 'الأمر';

  @override
  String get databaseBackend => 'محرك قاعدة البيانات';

  @override
  String get encryption => 'التشفير';

  @override
  String get pgHost => 'المضيف';

  @override
  String get pgPort => 'المنفذ';

  @override
  String get pgDbName => 'اسم قاعدة البيانات';

  @override
  String get pgUser => 'المستخدم';

  @override
  String get pgPassword => 'كلمة المرور';

  @override
  String get pgPoolMin => 'الحد الأدنى للاتصالات';

  @override
  String get pgPoolMax => 'الحد الأقصى للاتصالات';

  @override
  String get logLevel => 'مستوى السجل';

  @override
  String get jsonLogs => 'سجلات JSON';

  @override
  String get consoleOutput => 'إخراج وحدة التحكم';

  @override
  String get systemPrompt => 'أمر النظام';

  @override
  String get replanPrompt => 'أمر إعادة التخطيط';

  @override
  String get escalationPrompt => 'أمر التصعيد';

  @override
  String get policyYaml => 'ملف سياسة YAML';

  @override
  String get heartbeatMd => 'قائمة تحقق النبض';

  @override
  String get personalityPrompt => 'أمر الشخصية';

  @override
  String get promptEvolution => 'تطوير الأوامر';

  @override
  String get resetToDefault => 'إعادة تعيين الافتراضي';

  @override
  String get triggerPatterns => 'أنماط التشغيل';

  @override
  String get channelFilter => 'فلتر القناة';

  @override
  String get pattern => 'النمط';

  @override
  String get targetAgent => 'الوكيل المستهدف';

  @override
  String get restartBackend => 'إعادة تشغيل الخادم';

  @override
  String get exportConfig => 'تصدير التكوين';

  @override
  String get importConfig => 'استيراد التكوين';

  @override
  String get factoryReset => 'إعادة ضبط المصنع';

  @override
  String get factoryResetConfirm =>
      'سيتم إعادة تعيين جميع الإعدادات إلى الافتراضي. هل تريد المتابعة؟';

  @override
  String get configurationSaved => 'تم حفظ التكوين';

  @override
  String get saveHadErrors => 'حدثت أخطاء أثناء الحفظ';

  @override
  String get unsavedChanges => 'تغييرات غير محفوظة';

  @override
  String get discard => 'تجاهل';

  @override
  String get saving => 'جارِ الحفظ...';

  @override
  String get voiceOff => 'إيقاف';

  @override
  String get voiceListening => 'جارِ الاستماع...';

  @override
  String get voiceSpeakNow => 'تحدث الآن';

  @override
  String get voiceProcessing => 'جارِ المعالجة...';

  @override
  String get voiceSpeaking => 'جارِ التحدث...';

  @override
  String get observe => 'مراقبة';

  @override
  String get agentLog => 'سجل الوكيل';

  @override
  String get kanban => 'كانبان';

  @override
  String get dag => 'DAG';

  @override
  String get plan => 'الخطة';

  @override
  String get toDo => 'قيد الانتظار';

  @override
  String get inProgress => 'قيد التنفيذ';

  @override
  String get verifying => 'جارِ التحقق';

  @override
  String get done => 'مكتمل';

  @override
  String get searchConfigPages => 'بحث في صفحات التكوين...';

  @override
  String get noMatchingPages => 'لا توجد صفحات مطابقة';

  @override
  String get knowledgeGraphTitle => 'رسم المعرفة';

  @override
  String get searchEntities => 'بحث في الكيانات...';

  @override
  String get allTypes => 'جميع الأنواع';

  @override
  String get entityDetail => 'تفاصيل الكيان';

  @override
  String get attributes => 'الخصائص';

  @override
  String get instances => 'النسخ';

  @override
  String get dagRuns => 'تشغيلات DAG';

  @override
  String get noInstances => 'لا توجد نسخ';

  @override
  String get noDagRuns => 'لا توجد تشغيلات DAG';

  @override
  String get addCredential => 'إضافة بيانات اعتماد';

  @override
  String get service => 'الخدمة';

  @override
  String get key => 'المفتاح';

  @override
  String get value => 'القيمة';

  @override
  String get noCredentials => 'لا توجد بيانات اعتماد';

  @override
  String get deleteCredential => 'حذف بيانات الاعتماد';

  @override
  String get lightMode => 'الوضع الفاتح';

  @override
  String get darkMode => 'الوضع الداكن';

  @override
  String get globalSearch => 'بحث (Ctrl+K)';

  @override
  String get configPageGeneral => 'عام';

  @override
  String get configPageLanguage => 'اللغة';

  @override
  String get configPageProviders => 'المزودون';

  @override
  String get configPageModels => 'النماذج';

  @override
  String get configPagePlanner => 'المخطط';

  @override
  String get configPageExecutor => 'المنفذ';

  @override
  String get configPageMemory => 'الذاكرة';

  @override
  String get configPageChannels => 'القنوات';

  @override
  String get configPageSecurity => 'الأمان';

  @override
  String get configPageWeb => 'الويب';

  @override
  String get configPageMcp => 'MCP';

  @override
  String get configPageCron => 'المهام المجدولة';

  @override
  String get configPageDatabase => 'قاعدة البيانات';

  @override
  String get configPageLogging => 'السجلات';

  @override
  String get configPagePrompts => 'الأوامر';

  @override
  String get configPageAgents => 'الوكلاء';

  @override
  String get configPageBindings => 'الارتباطات';

  @override
  String get configPageSystem => 'النظام';

  @override
  String get configTitle => 'التكوين';

  @override
  String get reloadFromBackend => 'إعادة تحميل التكوين من الخادم';

  @override
  String get saveCtrlS => 'حفظ (Ctrl+S)';

  @override
  String savedWithErrors(String sections) {
    return 'تم الحفظ مع أخطاء في: $sections';
  }

  @override
  String get saveFailed => 'فشل الحفظ';

  @override
  String get fieldOwnerName => 'اسم المالك';

  @override
  String get fieldOperationMode => 'وضع التشغيل';

  @override
  String get fieldCostTracking => 'تتبع التكاليف';

  @override
  String get fieldDailyBudget => 'الميزانية اليومية (دولار)';

  @override
  String get fieldMonthlyBudget => 'الميزانية الشهرية (دولار)';

  @override
  String get fieldLlmBackend => 'محرك LLM';

  @override
  String get fieldPrimaryProvider => 'مزود LLM الرئيسي';

  @override
  String get fieldApiKey => 'مفتاح API';

  @override
  String get fieldBaseUrl => 'عنوان URL الأساسي';

  @override
  String get fieldModelName => 'اسم النموذج';

  @override
  String get fieldContextWindow => 'نافذة السياق';

  @override
  String get fieldTemperature => 'درجة الحرارة';

  @override
  String get fieldMaxIterations => 'الحد الأقصى للتكرارات';

  @override
  String get fieldEnabled => 'مفعل';

  @override
  String get fieldPort => 'المنفذ';

  @override
  String get fieldHost => 'المضيف';

  @override
  String get fieldPassword => 'كلمة المرور';

  @override
  String get fieldUser => 'المستخدم';

  @override
  String get fieldTimeout => 'المهلة';

  @override
  String get fieldLevel => 'المستوى';

  @override
  String get sectionSearchBackends => 'محركات البحث';

  @override
  String get sectionDomainFilters => 'فلاتر النطاقات';

  @override
  String get sectionFetchLimits => 'حدود الجلب';

  @override
  String get sectionSearchLimits => 'حدود البحث';

  @override
  String get sectionHttpLimits => 'حدود طلبات HTTP';

  @override
  String get sectionVoice => 'الصوت';

  @override
  String get sectionHeartbeat => 'نبض القلب';

  @override
  String get sectionPlugins => 'الإضافات';

  @override
  String get sectionCronJobs => 'المهام المجدولة';

  @override
  String get sectionPromptEvolution => 'تطوير الأوامر';

  @override
  String get addItem => 'إضافة';

  @override
  String get removeItem => 'إزالة';

  @override
  String get translatePrompts => 'ترجمة الأوامر عبر Ollama';

  @override
  String get translating => 'جارِ الترجمة...';

  @override
  String get promptsTranslated => 'تمت ترجمة الأوامر';

  @override
  String get copiedToClipboard => 'تم نسخ التكوين إلى الحافظة';

  @override
  String get configImported => 'تم استيراد التكوين';

  @override
  String get restartInitiated => 'تم بدء إعادة التشغيل';

  @override
  String get factoryResetComplete => 'تمت إعادة ضبط المصنع';

  @override
  String get factoryResetConfirmMsg =>
      'سيتم إعادة تعيين جميع الإعدادات. هل تريد المتابعة؟';

  @override
  String get languageEnglish => 'الإنجليزية';

  @override
  String get languageGerman => 'الألمانية';

  @override
  String get languageChinese => 'الصينية';

  @override
  String get languageArabic => 'العربية';

  @override
  String get uiAndPromptLanguage => 'لغة الواجهة والأوامر';

  @override
  String get learningTitle => 'التعلم';

  @override
  String get knowledgeGaps => 'فجوات المعرفة';

  @override
  String get explorationQueue => 'قائمة الاستكشاف';

  @override
  String get filesProcessed => 'الملفات المعالجة';

  @override
  String get entitiesCreated => 'الكيانات المنشأة';

  @override
  String get confidenceUpdates => 'تحديثات الثقة';

  @override
  String get openGaps => 'الفجوات المفتوحة';

  @override
  String get importance => 'الأهمية';

  @override
  String get curiosity => 'الفضول';

  @override
  String get explore => 'استكشاف';

  @override
  String get dismiss => 'تجاهل';

  @override
  String get noGaps => 'لم يتم اكتشاف فجوات معرفية';

  @override
  String get noTasks => 'لا توجد مهام استكشاف';

  @override
  String get confidenceHistory => 'سجل الثقة';

  @override
  String get feedback => 'ملاحظات';

  @override
  String get positive => 'إيجابي';

  @override
  String get negative => 'سلبي';

  @override
  String get correction => 'تصحيح';

  @override
  String get adminLearningSubtitle => 'التعلم النشط والفضول';

  @override
  String get watchDirectories => 'مجلدات المراقبة';

  @override
  String get directoryExists => 'المجلد موجود';

  @override
  String get directoryMissing => 'المجلد غير موجود';

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
  String get activityChart => 'النشاط';

  @override
  String get stopped => 'متوقف';

  @override
  String get requestsOverTime => 'الطلبات بمرور الوقت';

  @override
  String get teachCognithor => 'تعليم Cognithor';

  @override
  String get uploadFile => 'رفع ملف';

  @override
  String get learnFromUrl => 'التعلم من موقع ويب';

  @override
  String get learnFromYoutube => 'التعلم من فيديو';

  @override
  String get dropFilesHere => 'اسحب الملفات هنا أو انقر للتصفح';

  @override
  String get learningHistory => 'سجل التعلم';

  @override
  String chunksLearned(String count) {
    return 'تم تعلم $count أجزاء';
  }

  @override
  String get processingContent => 'جارٍ معالجة المحتوى...';

  @override
  String get learnSuccess => 'تم التعلم بنجاح!';

  @override
  String get learnFailed => 'فشل التعلم';

  @override
  String get enterUrl => 'أدخل عنوان URL للموقع...';

  @override
  String get enterYoutubeUrl => 'أدخل عنوان YouTube...';

  @override
  String get adminTeachSubtitle => 'رفع الملفات وعناوين URL والفيديو';

  @override
  String get newSkill => 'مهارة جديدة';

  @override
  String get editSkill => 'تعديل المهارة';

  @override
  String get createSkill => 'إنشاء مهارة';

  @override
  String get deleteSkill => 'حذف المهارة';

  @override
  String get skillName => 'الاسم';

  @override
  String get skillBody => 'محتوى المهارة (Markdown)';

  @override
  String get triggerKeywords => 'كلمات التشغيل';

  @override
  String get requiredTools => 'الأدوات المطلوبة';

  @override
  String get modelPreference => 'تفضيل النموذج';

  @override
  String get skillSaved => 'تم حفظ المهارة بنجاح';

  @override
  String get skillCreated => 'تم إنشاء المهارة بنجاح';

  @override
  String get skillDeleted => 'تم حذف المهارة';

  @override
  String get confirmDeleteSkill =>
      'هل أنت متأكد من حذف هذه المهارة؟ لا يمكن التراجع.';

  @override
  String get discardChanges => 'تجاهل التغييرات؟';

  @override
  String get discardChangesBody => 'لديك تغييرات غير محفوظة. تجاهلها؟';

  @override
  String get totalUses => 'إجمالي الاستخدامات';

  @override
  String get lastUsed => 'آخر استخدام';

  @override
  String get commaSeparated => 'مفصولة بفاصلة';

  @override
  String get skillBodyHint => 'اكتب تعليمات المهارة بصيغة Markdown...';

  @override
  String get metadata => 'البيانات الوصفية';

  @override
  String get statistics => 'الإحصائيات';

  @override
  String get builtInSkill => 'مهارة مدمجة (للقراءة فقط)';

  @override
  String get exportSkillMd => 'تصدير كـ SKILL.md';

  @override
  String get skillExported => 'تم تصدير المهارة إلى الحافظة';

  @override
  String get general => 'عام';

  @override
  String get productivity => 'إنتاجية';

  @override
  String get research => 'بحث';

  @override
  String get analysis => 'تحليل';

  @override
  String get development => 'تطوير';

  @override
  String get automation => 'أتمتة';

  @override
  String get newAgent => 'وكيل جديد';

  @override
  String get editAgent => 'تعديل الوكيل';

  @override
  String get deleteAgent => 'حذف الوكيل';

  @override
  String get confirmDeleteAgent =>
      'هل أنت متأكد من حذف هذا الوكيل؟ لا يمكن التراجع عن هذا.';

  @override
  String get agentCreated => 'تم إنشاء الوكيل بنجاح';

  @override
  String get agentSaved => 'تم حفظ الوكيل بنجاح';

  @override
  String get agentDeleted => 'تم حذف الوكيل';

  @override
  String get displayName => 'اسم العرض';

  @override
  String get preferredModel => 'النموذج المفضل';

  @override
  String get sandboxTimeout => 'مهلة الحماية (ث)';

  @override
  String get sandboxNetwork => 'شبكة الحماية';

  @override
  String get canDelegateTo => 'يمكن التفويض إلى';

  @override
  String get cannotDeleteDefault => 'لا يمكن حذف الوكيل الافتراضي';

  @override
  String get robotOfficePipMode => 'Robot Office في وضع صورة داخل صورة';

  @override
  String get fullscreen => 'ملء الشاشة';

  @override
  String get pipLabel => 'صورة مصغرة';

  @override
  String taskCount(int count) {
    return '$count مهام';
  }

  @override
  String get hackerMode => 'وضع القرصنة';

  @override
  String get entityVisualization => 'تصور الكيانات';

  @override
  String get manageSecrets => 'إدارة الأسرار';

  @override
  String get channelToggles => 'مفاتيح القنوات';

  @override
  String get channelSettings => 'إعدادات القنوات';

  @override
  String get tapToSelect => 'اضغط للاختيار...';

  @override
  String get selectModel => 'اختر نموذجًا';

  @override
  String get searchModels => 'البحث عن نماذج...';

  @override
  String get remove => 'إزالة';

  @override
  String get stopBackend => 'إيقاف الخادم';

  @override
  String get stopBackendDescription =>
      'إيقاف خادم Jarvis. ستحتاج إلى إعادة تشغيله يدويًا.';

  @override
  String get stopBackendConfirmBody =>
      'سيؤدي هذا إلى إيقاف عملية خادم Jarvis. ستحتاج إلى إعادة تشغيلها يدويًا من سطر الأوامر.';

  @override
  String get backendStopped => 'تم إيقاف الخادم. يرجى إعادة التشغيل يدويًا.';

  @override
  String get downloadConfigDesc => 'تنزيل التكوين الحالي بصيغة JSON';

  @override
  String get loadConfigDesc => 'تحميل التكوين من ملف JSON';

  @override
  String get resetAllDesc =>
      'إعادة تعيين جميع الإعدادات إلى الافتراضي. لا يمكن التراجع عن هذا.';

  @override
  String get factoryResetNotImpl =>
      'لم يتم تنفيذ إعادة ضبط المصنع بعد في الخادم. لإعادة التعيين يدويًا، احذف config.yaml وأعد تشغيل Jarvis.';

  @override
  String get ok => 'موافق';

  @override
  String get wizardSubtitle => 'مساعدك الشخصي بالذكاء الاصطناعي';

  @override
  String get chooseLlmProvider => 'اختر مزود LLM الخاص بك';

  @override
  String get localOllama => 'محلي (Ollama)';

  @override
  String get localOllamaDesc =>
      'تشغيل النماذج على أجهزتك. خصوصية كاملة، بدون تكاليف API. يتطلب تثبيت Ollama.';

  @override
  String get cloudProviderLabel => 'مزود سحابي';

  @override
  String get cloudProviderDesc =>
      'استخدام OpenAI أو Anthropic أو واجهات API سحابية أخرى. إعداد أسرع، يتطلب مفتاح API.';

  @override
  String get next => 'التالي';

  @override
  String get ollamaConfiguration => 'تكوين Ollama';

  @override
  String get cloudApiConfiguration => 'تكوين API السحابي';

  @override
  String get ollamaConfigHint => 'أدخل عنوان URL حيث يعمل Ollama.';

  @override
  String get cloudConfigHint => 'اختر مزود السحابة وأدخل مفتاح API الخاص بك.';

  @override
  String get ollamaUrl => 'عنوان Ollama';

  @override
  String get testConnection => 'اختبار الاتصال';

  @override
  String get testingConnection => 'جارٍ الاختبار...';

  @override
  String get youreAllSet => 'كل شيء جاهز!';

  @override
  String get ollamaReadyMsg =>
      'Ollama متصل وجاهز. سيستخدم Cognithor نماذجك المحلية للتخطيط والتنفيذ.';

  @override
  String cloudReadyMsg(String provider) {
    return 'تم تكوين $provider. سيستخدم Cognithor واجهة API السحابية للتخطيط والتنفيذ.';
  }

  @override
  String get changeSettingsAnytime => 'يمكنك تغيير هذه الإعدادات في أي وقت.';

  @override
  String get startUsingCognithor => 'ابدأ باستخدام Cognithor';

  @override
  String get ollamaNoModels =>
      'متصل بـ Ollama. لم يتم تثبيت نماذج بعد — شغّل \"ollama pull qwen3:8b\" للبدء.';

  @override
  String ollamaModelsAvailable(int count) {
    return 'متصل بـ Ollama. $count نموذج(نماذج) متاحة.';
  }

  @override
  String ollamaStatusError(int code) {
    return 'استجاب Ollama بالحالة $code. تأكد من أن الخادم يعمل.';
  }

  @override
  String get enterApiKey => 'يرجى إدخال مفتاح API.';

  @override
  String apiKeyTooShort(String provider) {
    return 'يبدو أن المفتاح قصير جدًا. تحقق من مفتاح API الخاص بـ $provider.';
  }

  @override
  String apiKeySaved(String provider) {
    return 'تم حفظ مفتاح API لـ $provider. يمكنك تغييره لاحقًا في الإعدادات.';
  }

  @override
  String connectionFailed(String error) {
    return 'فشل الاتصال: $error';
  }

  @override
  String get minimize => 'تصغير';

  @override
  String get shrink => 'تقليص';

  @override
  String get expandLabel => 'توسيع';

  @override
  String get robotOffice => 'Robot Office';

  @override
  String get copy => 'نسخ';

  @override
  String get share => 'مشاركة';

  @override
  String get noLogEntries => 'لا توجد سجلات';

  @override
  String get noPlanData => 'لا توجد بيانات خطة';

  @override
  String get noDagData => 'لا توجد بيانات DAG';

  @override
  String get log => 'السجل';

  @override
  String get fileReadError => 'تعذر قراءة الملف';

  @override
  String uploadError(String error) {
    return 'خطأ في الرفع: $error';
  }

  @override
  String get toolSpecificTimeouts => 'مهل الأدوات المحددة';

  @override
  String get required => 'مطلوب';

  @override
  String get stopLabel => 'إيقاف';

  @override
  String get resetLabel => 'إعادة تعيين';

  @override
  String get exportLabel => 'تصدير';

  @override
  String get importLabel => 'استيراد';

  @override
  String get catAiEngine => 'محرك الذكاء الاصطناعي';

  @override
  String get catChannels => 'القنوات';

  @override
  String get catKnowledge => 'المعرفة';

  @override
  String get catSecurity => 'الأمان';

  @override
  String get catSystem => 'النظام';

  @override
  String get saved => 'تم الحفظ';

  @override
  String get chatHistory => 'سجل المحادثات';

  @override
  String get newChat => 'محادثة جديدة';

  @override
  String get untitledChat => 'محادثة بدون عنوان';

  @override
  String get deleteChat => 'حذف المحادثة';

  @override
  String get confirmDeleteChat => 'حذف هذه المحادثة؟';

  @override
  String messagesCount(String count) {
    return '$count رسائل';
  }

  @override
  String get justNow => 'الآن';

  @override
  String minutesAgo(String count) {
    return 'منذ $count دقيقة';
  }

  @override
  String hoursAgo(String count) {
    return 'منذ $count ساعة';
  }

  @override
  String daysAgo(String count) {
    return 'منذ $count يوم';
  }
}
