// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Chinese (`zh`).
class AppLocalizationsZh extends AppLocalizations {
  AppLocalizationsZh([String locale = 'zh']) : super(locale);

  @override
  String get appTitle => 'Jarvis';

  @override
  String get chat => '聊天';

  @override
  String get settings => '设置';

  @override
  String get identity => '身份';

  @override
  String get workflows => '工作流';

  @override
  String get memory => '记忆';

  @override
  String get monitoring => '监控';

  @override
  String get skills => '技能';

  @override
  String get config => '配置';

  @override
  String get sendMessage => '输入消息...';

  @override
  String get send => '发送';

  @override
  String get cancel => '取消';

  @override
  String get approve => '批准';

  @override
  String get reject => '拒绝';

  @override
  String get retry => '重试';

  @override
  String get close => '关闭';

  @override
  String get save => '保存';

  @override
  String get delete => '删除';

  @override
  String get loading => '加载中...';

  @override
  String get connecting => '连接中...';

  @override
  String get approvalTitle => '需要审批';

  @override
  String approvalBody(String tool) {
    return '工具 $tool 要执行：';
  }

  @override
  String approvalReason(String reason) {
    return '原因：$reason';
  }

  @override
  String get statusThinking => '思考中...';

  @override
  String get statusExecuting => '执行中...';

  @override
  String get statusFinishing => '完成中...';

  @override
  String get voiceMessage => '语音消息';

  @override
  String fileUpload(String name) {
    return '文件：$name';
  }

  @override
  String get connectionError => '无法连接到后端';

  @override
  String connectionErrorDetail(String url) {
    return '请确认 Jarvis 后端正在 $url 运行';
  }

  @override
  String get authFailed => '认证失败';

  @override
  String get tokenExpired => '会话已过期，正在重新连接...';

  @override
  String get serverUrl => '服务器地址';

  @override
  String get serverUrlHint => 'http://localhost:8741';

  @override
  String version(String version) {
    return '版本 $version';
  }

  @override
  String get errorGeneric => '出了点问题';

  @override
  String get errorNetwork => '网络错误，请检查连接。';

  @override
  String get errorTimeout => '请求超时';

  @override
  String get errorUnauthorized => '未授权，请重新连接。';

  @override
  String get errorServerDown => '后端不可用';

  @override
  String get identityNotAvailable => '身份层不可用';

  @override
  String get identityInstallHint => '安装：pip install cognithor[identity]';

  @override
  String get identityEnergy => '能量';

  @override
  String get identityInteractions => '交互次数';

  @override
  String get identityMemories => '记忆';

  @override
  String get identityCharacterStrength => '性格强度';

  @override
  String get identityFrozen => '已冻结';

  @override
  String get identityActive => '活跃';

  @override
  String get identityDream => '梦境循环';

  @override
  String get identityFreeze => '冻结';

  @override
  String get identityUnfreeze => '解冻';

  @override
  String get identityReset => '软重置';

  @override
  String get identityResetConfirm => '重置身份？记忆将丢失。';

  @override
  String get pipelinePlan => '规划';

  @override
  String get pipelineGate => '门控';

  @override
  String get pipelineExecute => '执行';

  @override
  String get pipelineReplan => '重新规划';

  @override
  String get pipelineComplete => '完成';

  @override
  String get canvasTitle => '画布';

  @override
  String get canvasClose => '关闭画布';

  @override
  String get models => '模型';

  @override
  String get channels => '频道';

  @override
  String get security => '安全';

  @override
  String get reload => '重新加载';

  @override
  String get reloading => '正在重新加载...';

  @override
  String get configSaved => '配置已重新加载';

  @override
  String get configError => '配置错误';

  @override
  String get uptime => '运行时间';

  @override
  String get activeSessions => '活跃会话';

  @override
  String get totalRequests => '总请求数';

  @override
  String get events => '事件';

  @override
  String get noEvents => '暂无事件记录';

  @override
  String get severity => '严重程度';

  @override
  String get refreshing => '自动刷新：10秒';

  @override
  String get noData => '暂无数据';

  @override
  String get notAvailable => '不可用';

  @override
  String get dashboard => '仪表盘';

  @override
  String get systemOverview => '系统概览';

  @override
  String get cpuUsage => 'CPU使用率';

  @override
  String get memoryUsage => '内存使用率';

  @override
  String get responseTime => '响应时间';

  @override
  String get toolExecutions => '工具执行次数';

  @override
  String get successRate => '成功率';

  @override
  String get recentEvents => '最近事件';

  @override
  String get lastUpdated => '最后更新';

  @override
  String get systemHealth => '系统健康';

  @override
  String get performance => '性能';

  @override
  String get trends => '趋势';

  @override
  String get marketplace => '市场';

  @override
  String get featured => '精选';

  @override
  String get trending => '热门';

  @override
  String get categories => '分类';

  @override
  String get searchSkills => '搜索技能...';

  @override
  String get installed => '已安装';

  @override
  String get installSkill => '安装';

  @override
  String get uninstallSkill => '卸载';

  @override
  String get installing => '安装中...';

  @override
  String get skillDetails => '技能详情';

  @override
  String get reviews => '评价';

  @override
  String get noSkills => '未找到技能';

  @override
  String get browseMarketplace => '浏览市场';

  @override
  String get verified => '已验证';

  @override
  String get downloads => '下载量';

  @override
  String get rating => '评分';

  @override
  String get memoryTitle => '记忆';

  @override
  String get knowledgeGraph => '知识图谱';

  @override
  String get entities => '实体';

  @override
  String get relations => '关系';

  @override
  String get hygiene => '卫生检查';

  @override
  String get quarantine => '隔离';

  @override
  String get scanMemory => '扫描';

  @override
  String get scanning => '扫描中...';

  @override
  String get explainability => '可解释性';

  @override
  String get decisionTrails => '决策路径';

  @override
  String get lowTrust => '低信任';

  @override
  String get graphStats => '图谱统计';

  @override
  String get noEntities => '暂无实体';

  @override
  String get noTrails => '暂无路径';

  @override
  String get scanComplete => '扫描完成';

  @override
  String get threats => '威胁';

  @override
  String get threatRate => '威胁率';

  @override
  String get totalScans => '总扫描次数';

  @override
  String get integrity => '完整性';

  @override
  String get securityTitle => '安全';

  @override
  String get complianceTitle => '合规';

  @override
  String get rolesTitle => '角色';

  @override
  String get permissions => '权限';

  @override
  String get auditLog => '审计日志';

  @override
  String get redTeam => '红队测试';

  @override
  String get scanStatus => '扫描状态';

  @override
  String get complianceReport => '合规报告';

  @override
  String get decisionsTitle => '决策';

  @override
  String get remediations => '修复措施';

  @override
  String get openStatus => '待处理';

  @override
  String get inProgressStatus => '进行中';

  @override
  String get resolvedStatus => '已解决';

  @override
  String get overdueStatus => '已逾期';

  @override
  String get approvalRate => '批准率';

  @override
  String get flaggedCount => '已标记';

  @override
  String get transparency => '透明度';

  @override
  String get euAiAct => '欧盟AI法案';

  @override
  String get dsgvo => '通用数据保护条例';

  @override
  String get runScan => '开始扫描';

  @override
  String get adminTitle => '管理';

  @override
  String get agentsTitle => '代理';

  @override
  String get modelsTitle => '模型';

  @override
  String get systemTitle => '系统';

  @override
  String get workflowsTitle => '工作流';

  @override
  String get vaultTitle => '保险库';

  @override
  String get credentialsTitle => '凭据';

  @override
  String get bindingsTitle => '绑定';

  @override
  String get connectorsTitle => '连接器';

  @override
  String get commandsTitle => '命令';

  @override
  String get isolationTitle => '隔离';

  @override
  String get sandboxTitle => '沙箱';

  @override
  String get circlesTitle => '圈子';

  @override
  String get wizardsTitle => '向导';

  @override
  String get systemStatus => '系统状态';

  @override
  String get shutdownServer => '关闭服务器';

  @override
  String get shutdownConfirm => '确定要关闭服务器吗？';

  @override
  String get startComponent => '启动';

  @override
  String get stopComponent => '停止';

  @override
  String get selectTemplate => '选择模板';

  @override
  String get workflowStarted => '工作流已启动';

  @override
  String get noWorkflows => '暂无工作流';

  @override
  String get templates => '模板';

  @override
  String get running => '运行中';

  @override
  String get vaultStats => '保险库统计';

  @override
  String get totalEntries => '总条目';

  @override
  String get agentVaults => '代理保险库';

  @override
  String get noVaults => '暂无保险库';

  @override
  String get availableModels => '可用模型';

  @override
  String get modelStats => '模型统计';

  @override
  String get providers => '提供商';

  @override
  String get capabilities => '能力';

  @override
  String get plannerModel => '规划器';

  @override
  String get executorModel => '执行器';

  @override
  String get coderModel => '编码器';

  @override
  String get embeddingModel => '嵌入';

  @override
  String get configured => '已配置';

  @override
  String get modelWarnings => '警告';

  @override
  String get identityDreamCycle => '梦境循环';

  @override
  String get identityGenesisAnchors => '创世锚点';

  @override
  String get identityNoAnchors => '暂无创世锚点';

  @override
  String get identityPersonality => '个性';

  @override
  String get identityCognitive => '认知状态';

  @override
  String get identityEmotional => '情绪状态';

  @override
  String get identitySomatic => '躯体状态';

  @override
  String get identityNarrative => '叙事';

  @override
  String get identityExistential => '存在性';

  @override
  String get identityPredictive => '预测性';

  @override
  String get identityEpistemic => '认识论';

  @override
  String get identityBiases => '活跃偏见';

  @override
  String get search => '搜索';

  @override
  String get filter => '筛选';

  @override
  String get sortBy => '排序方式';

  @override
  String get refresh => '刷新';

  @override
  String get export => '导出';

  @override
  String get viewAll => '查看全部';

  @override
  String get details => '详情';

  @override
  String get back => '返回';

  @override
  String get confirm => '确认';

  @override
  String get actions => '操作';

  @override
  String get statusLabel => '状态';

  @override
  String get enabled => '已启用';

  @override
  String get disabled => '已禁用';

  @override
  String get total => '总计';

  @override
  String get count => '数量';

  @override
  String get rate => '比率';

  @override
  String get average => '平均';

  @override
  String get duration => '持续时间';

  @override
  String get timestamp => '时间戳';

  @override
  String get severityLabel => '严重程度';

  @override
  String get critical => '严重';

  @override
  String get errorLabel => '错误';

  @override
  String get warningLabel => '警告';

  @override
  String get infoLabel => '信息';

  @override
  String get successLabel => '成功';

  @override
  String get unknownLabel => '未知';

  @override
  String get notConfigured => '未配置';

  @override
  String get comingSoon => '即将推出';

  @override
  String get beta => '测试版';

  @override
  String get copyToClipboard => '复制到剪贴板';

  @override
  String get copied => '已复制！';

  @override
  String get chatSettings => '聊天设置';

  @override
  String get clearChat => '清除聊天';

  @override
  String get voiceMode => '语音模式';

  @override
  String get fileUploadAction => '上传文件';

  @override
  String get planDetails => '计划详情';

  @override
  String get noMessages => '暂无消息';

  @override
  String get typeMessage => '输入消息...';

  @override
  String get settingsTitle => '设置';

  @override
  String get language => '语言';

  @override
  String get theme => '主题';

  @override
  String get about => '关于';

  @override
  String get licenses => '许可证';

  @override
  String get clearCache => '清除缓存';

  @override
  String get adminConfigSubtitle => '管理配置';

  @override
  String get adminAgentsSubtitle => '代理与配置文件';

  @override
  String get adminModelsSubtitle => 'LLM模型';

  @override
  String get adminSecuritySubtitle => '安全与合规';

  @override
  String get adminWorkflowsSubtitle => '自动化';

  @override
  String get adminMemorySubtitle => '知识图谱';

  @override
  String get adminVaultSubtitle => '密钥与凭证';

  @override
  String get adminSystemSubtitle => '系统状态';

  @override
  String get dashboardRefreshing => '自动刷新：15秒';

  @override
  String get backendVersion => '后端版本';

  @override
  String get modelInfo => '模型信息';

  @override
  String get confidence => '置信度';

  @override
  String get rolesAccess => '角色与权限';

  @override
  String get loadMore => '加载更多';

  @override
  String get actor => '执行者';

  @override
  String get noAuditEntries => '暂无审计记录';

  @override
  String get allSeverities => '所有级别';

  @override
  String get allActions => '所有操作';

  @override
  String get scanNotAvailable => '扫描不可用';

  @override
  String get lastScan => '最后扫描';

  @override
  String get scanResults => '扫描结果';

  @override
  String get compliant => '合规';

  @override
  String get nonCompliant => '不合规';

  @override
  String get model => '模型';

  @override
  String get temperature => '温度';

  @override
  String get priority => '优先级';

  @override
  String get allowedTools => '允许的工具';

  @override
  String get blockedTools => '已屏蔽的工具';

  @override
  String get noAgents => '未配置代理';

  @override
  String get description => '描述';

  @override
  String get provider => '提供商';

  @override
  String get noModels => '暂无可用模型';

  @override
  String get owner => '所有者';

  @override
  String get llmBackend => 'LLM后端';

  @override
  String get components => '组件';

  @override
  String get dangerZone => '危险区域';

  @override
  String get reloadConfig => '重载配置';

  @override
  String get runtimeInfo => '运行时';

  @override
  String get startWorkflow => '启动工作流';

  @override
  String get noCategories => '暂无分类';

  @override
  String templateCount(String count) {
    return '$count 个模板';
  }

  @override
  String get entityTypes => '实体类型';

  @override
  String get activeTrails => '活跃路径';

  @override
  String get completedTrails => '已完成';

  @override
  String get lastAccessed => '最后访问';

  @override
  String get author => '作者';

  @override
  String get noQuarantine => '暂无隔离项';

  @override
  String get totalVaults => '保险库总数';

  @override
  String get scanNow => '立即扫描';

  @override
  String get startConversation => '开始对话';

  @override
  String get attachFile => '附加文件';

  @override
  String get voiceModeHint => '语音模式即将推出';

  @override
  String get canvasLabel => '画布';

  @override
  String get configGeneral => '通用';

  @override
  String get configLanguage => '语言';

  @override
  String get configProviders => '提供商';

  @override
  String get configModels => '模型';

  @override
  String get configPlanner => '规划器';

  @override
  String get configExecutor => '执行器';

  @override
  String get configMemory => '记忆';

  @override
  String get configChannels => '频道';

  @override
  String get configSecurity => '安全';

  @override
  String get configWeb => '网络';

  @override
  String get configMcp => 'MCP';

  @override
  String get configCron => '定时任务';

  @override
  String get configDatabase => '数据库';

  @override
  String get configLogging => '日志';

  @override
  String get configPrompts => '提示词';

  @override
  String get configAgents => '代理';

  @override
  String get configBindings => '绑定';

  @override
  String get configSystem => '系统';

  @override
  String get ownerName => '所有者';

  @override
  String get operationMode => '运行模式';

  @override
  String get costTracking => '成本追踪';

  @override
  String get dailyBudget => '每日预算';

  @override
  String get monthlyBudget => '每月预算';

  @override
  String get apiKey => 'API密钥';

  @override
  String get baseUrl => '基础URL';

  @override
  String get maxTokens => '最大令牌数';

  @override
  String get timeout => '超时';

  @override
  String get keepAlive => '保持连接';

  @override
  String get contextWindow => '上下文窗口';

  @override
  String get vramGb => '显存(GB)';

  @override
  String get topP => 'Top P';

  @override
  String get maxIterations => '最大迭代次数';

  @override
  String get escalationAfter => '升级阈值';

  @override
  String get responseBudget => '响应令牌预算';

  @override
  String get policiesDir => '策略目录';

  @override
  String get defaultRiskLevel => '默认风险等级';

  @override
  String get maxBlockedRetries => '最大阻止重试';

  @override
  String get sandboxLevel => '沙箱级别';

  @override
  String get maxMemoryMb => '最大内存(MB)';

  @override
  String get maxCpuSeconds => '最大CPU秒';

  @override
  String get allowedPaths => '允许路径';

  @override
  String get networkAccess => '网络访问';

  @override
  String get envVars => '环境变量';

  @override
  String get defaultTimeout => '默认超时';

  @override
  String get maxOutputChars => '最大输出字符';

  @override
  String get maxRetries => '最大重试';

  @override
  String get backoffDelay => '退避延迟';

  @override
  String get maxParallelTools => '最大并行工具';

  @override
  String get chunkSize => '块大小';

  @override
  String get chunkOverlap => '块重叠';

  @override
  String get searchTopK => '搜索Top K';

  @override
  String get searchWeights => '搜索权重';

  @override
  String get vectorWeight => '向量权重';

  @override
  String get bm25Weight => 'BM25权重';

  @override
  String get graphWeight => '图权重';

  @override
  String get recencyHalfLife => '时效半衰期';

  @override
  String get compactionThreshold => '压缩阈值';

  @override
  String get compactionKeepLast => '压缩保留数';

  @override
  String get episodicRetention => '情景保留';

  @override
  String get dynamicWeighting => '动态加权';

  @override
  String get voiceEnabled => '语音已启用';

  @override
  String get ttsBackend => 'TTS后端';

  @override
  String get piperVoice => 'Piper语音';

  @override
  String get piperLengthScale => 'Piper语速比例';

  @override
  String get wakeWordEnabled => '唤醒词已启用';

  @override
  String get wakeWord => '唤醒词';

  @override
  String get wakeWordBackend => '唤醒词后端';

  @override
  String get talkMode => '对话模式';

  @override
  String get autoListen => '自动收听';

  @override
  String get blockedCommands => '阻止的命令';

  @override
  String get credentialPatterns => '凭证模式';

  @override
  String get maxSubAgentDepth => '最大子代理深度';

  @override
  String get searchBackends => '搜索后端';

  @override
  String get domainFilters => '域名过滤';

  @override
  String get blocklist => '黑名单';

  @override
  String get allowlist => '白名单';

  @override
  String get httpLimits => 'HTTP限制';

  @override
  String get maxFetchBytes => '最大获取字节';

  @override
  String get maxTextChars => '最大文本字符';

  @override
  String get fetchTimeout => '获取超时';

  @override
  String get searchTimeout => '搜索超时';

  @override
  String get maxSearchResults => '最大搜索结果数';

  @override
  String get rateLimit => '速率限制';

  @override
  String get mcpServers => 'MCP服务器';

  @override
  String get a2aProtocol => 'A2A协议';

  @override
  String get remotes => '远程节点';

  @override
  String get heartbeat => '心跳';

  @override
  String get intervalMinutes => '间隔(分钟)';

  @override
  String get checklistFile => '检查清单文件';

  @override
  String get channel => '频道';

  @override
  String get plugins => '插件';

  @override
  String get skillsDir => '技能目录';

  @override
  String get autoUpdate => '自动更新';

  @override
  String get cronJobs => '定时任务';

  @override
  String get schedule => '计划';

  @override
  String get command => '命令';

  @override
  String get databaseBackend => '数据库后端';

  @override
  String get encryption => '加密';

  @override
  String get pgHost => '主机';

  @override
  String get pgPort => '端口';

  @override
  String get pgDbName => '数据库名称';

  @override
  String get pgUser => '用户';

  @override
  String get pgPassword => '密码';

  @override
  String get pgPoolMin => '连接池最小值';

  @override
  String get pgPoolMax => '连接池最大值';

  @override
  String get logLevel => '日志级别';

  @override
  String get jsonLogs => 'JSON日志';

  @override
  String get consoleOutput => '控制台输出';

  @override
  String get systemPrompt => '系统提示词';

  @override
  String get replanPrompt => '重规划提示词';

  @override
  String get escalationPrompt => '升级提示词';

  @override
  String get policyYaml => '策略YAML';

  @override
  String get heartbeatMd => '心跳检查清单';

  @override
  String get personalityPrompt => '个性提示词';

  @override
  String get promptEvolution => '提示词进化';

  @override
  String get resetToDefault => '恢复默认';

  @override
  String get triggerPatterns => '触发模式';

  @override
  String get channelFilter => '频道过滤';

  @override
  String get pattern => '模式';

  @override
  String get targetAgent => '目标代理';

  @override
  String get restartBackend => '重启后端';

  @override
  String get exportConfig => '导出配置';

  @override
  String get importConfig => '导入配置';

  @override
  String get factoryReset => '恢复出厂设置';

  @override
  String get factoryResetConfirm => '将所有设置恢复为出厂默认值。继续?';

  @override
  String get configurationSaved => '配置已保存';

  @override
  String get saveHadErrors => '保存出错';

  @override
  String get unsavedChanges => '未保存的更改';

  @override
  String get discard => '放弃';

  @override
  String get saving => '保存中...';

  @override
  String get voiceOff => '关闭';

  @override
  String get voiceListening => '聆听中...';

  @override
  String get voiceSpeakNow => '请说话';

  @override
  String get voiceProcessing => '处理中...';

  @override
  String get voiceSpeaking => '播放中...';

  @override
  String get observe => '观察';

  @override
  String get agentLog => '代理日志';

  @override
  String get kanban => '看板';

  @override
  String get dag => 'DAG';

  @override
  String get plan => '计划';

  @override
  String get toDo => '待办';

  @override
  String get inProgress => '进行中';

  @override
  String get verifying => '验证中';

  @override
  String get done => '完成';

  @override
  String get searchConfigPages => '搜索配置页面...';

  @override
  String get noMatchingPages => '没有匹配的页面';

  @override
  String get knowledgeGraphTitle => '知识图谱';

  @override
  String get searchEntities => '搜索实体...';

  @override
  String get allTypes => '所有类型';

  @override
  String get entityDetail => '实体详情';

  @override
  String get attributes => '属性';

  @override
  String get instances => '实例';

  @override
  String get dagRuns => 'DAG运行';

  @override
  String get noInstances => '暂无实例';

  @override
  String get noDagRuns => '暂无DAG运行';

  @override
  String get addCredential => '添加凭证';

  @override
  String get service => '服务';

  @override
  String get key => '密钥';

  @override
  String get value => '值';

  @override
  String get noCredentials => '没有凭证';

  @override
  String get lightMode => '浅色模式';

  @override
  String get darkMode => '深色模式';

  @override
  String get globalSearch => '搜索 (Ctrl+K)';

  @override
  String get configPageGeneral => '通用';

  @override
  String get configPageLanguage => '语言';

  @override
  String get configPageProviders => '提供商';

  @override
  String get configPageModels => '模型';

  @override
  String get configPagePlanner => '规划器';

  @override
  String get configPageExecutor => '执行器';

  @override
  String get configPageMemory => '记忆';

  @override
  String get configPageChannels => '频道';

  @override
  String get configPageSecurity => '安全';

  @override
  String get configPageWeb => '网络';

  @override
  String get configPageMcp => 'MCP';

  @override
  String get configPageCron => '定时任务';

  @override
  String get configPageDatabase => '数据库';

  @override
  String get configPageLogging => '日志';

  @override
  String get configPagePrompts => '提示词';

  @override
  String get configPageAgents => '代理';

  @override
  String get configPageBindings => '绑定';

  @override
  String get configPageSystem => '系统';

  @override
  String get configTitle => '配置';

  @override
  String get reloadFromBackend => '从后端重新加载配置';

  @override
  String get saveCtrlS => '保存 (Ctrl+S)';

  @override
  String savedWithErrors(String sections) {
    return '保存时出现错误: $sections';
  }

  @override
  String get saveFailed => '保存失败';

  @override
  String get fieldOwnerName => '所有者名称';

  @override
  String get fieldOperationMode => '运行模式';

  @override
  String get fieldCostTracking => '成本追踪';

  @override
  String get fieldDailyBudget => '每日预算 (USD)';

  @override
  String get fieldMonthlyBudget => '每月预算 (USD)';

  @override
  String get fieldLlmBackend => 'LLM后端';

  @override
  String get fieldPrimaryProvider => '主要LLM提供商';

  @override
  String get fieldApiKey => 'API密钥';

  @override
  String get fieldBaseUrl => '基础URL';

  @override
  String get fieldModelName => '模型名称';

  @override
  String get fieldContextWindow => '上下文窗口';

  @override
  String get fieldTemperature => '温度';

  @override
  String get fieldMaxIterations => '最大迭代次数';

  @override
  String get fieldEnabled => '已启用';

  @override
  String get fieldPort => '端口';

  @override
  String get fieldHost => '主机';

  @override
  String get fieldPassword => '密码';

  @override
  String get fieldUser => '用户';

  @override
  String get fieldTimeout => '超时';

  @override
  String get fieldLevel => '级别';

  @override
  String get sectionSearchBackends => '搜索后端';

  @override
  String get sectionDomainFilters => '域名过滤';

  @override
  String get sectionFetchLimits => '获取限制';

  @override
  String get sectionSearchLimits => '搜索限制';

  @override
  String get sectionHttpLimits => 'HTTP请求限制';

  @override
  String get sectionVoice => '语音';

  @override
  String get sectionHeartbeat => '心跳';

  @override
  String get sectionPlugins => '插件';

  @override
  String get sectionCronJobs => '定时任务';

  @override
  String get sectionPromptEvolution => '提示词进化';

  @override
  String get addItem => '添加';

  @override
  String get removeItem => '移除';

  @override
  String get translatePrompts => '通过Ollama翻译提示词';

  @override
  String get translating => '翻译中...';

  @override
  String get promptsTranslated => '提示词已翻译';

  @override
  String get copiedToClipboard => '配置已复制到剪贴板';

  @override
  String get configImported => '配置已导入';

  @override
  String get restartInitiated => '正在重启';

  @override
  String get factoryResetComplete => '已恢复出厂设置';

  @override
  String get factoryResetConfirmMsg => '将所有设置恢复为出厂默认值。继续?';

  @override
  String get languageEnglish => '英语';

  @override
  String get languageGerman => '德语';

  @override
  String get languageChinese => '中文';

  @override
  String get languageArabic => '阿拉伯语';

  @override
  String get uiAndPromptLanguage => '界面和提示词语言';

  @override
  String get learningTitle => '学习';

  @override
  String get knowledgeGaps => '知识缺口';

  @override
  String get explorationQueue => '探索队列';

  @override
  String get filesProcessed => '已处理文件';

  @override
  String get entitiesCreated => '已创建实体';

  @override
  String get confidenceUpdates => '置信度更新';

  @override
  String get openGaps => '未解决缺口';

  @override
  String get importance => '重要性';

  @override
  String get curiosity => '好奇度';

  @override
  String get explore => '探索';

  @override
  String get dismiss => '忽略';

  @override
  String get noGaps => '未发现知识缺口';

  @override
  String get noTasks => '无探索任务';

  @override
  String get confidenceHistory => '置信度历史';

  @override
  String get feedback => '反馈';

  @override
  String get positive => '正面';

  @override
  String get negative => '负面';

  @override
  String get correction => '纠正';

  @override
  String get adminLearningSubtitle => '主动学习与好奇心';

  @override
  String get watchDirectories => '监控目录';

  @override
  String get directoryExists => '目录存在';

  @override
  String get directoryMissing => '目录未找到';

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
  String get activityChart => '活动图表';

  @override
  String get stopped => '已停止';

  @override
  String get requestsOverTime => '请求趋势';

  @override
  String get teachCognithor => '教导 Cognithor';

  @override
  String get uploadFile => '上传文件';

  @override
  String get learnFromUrl => '从网站学习';

  @override
  String get learnFromYoutube => '从视频学习';

  @override
  String get dropFilesHere => '将文件拖放到此处或点击浏览';

  @override
  String get learningHistory => '学习历史';

  @override
  String chunksLearned(String count) {
    return '已学习 $count 个片段';
  }

  @override
  String get processingContent => '正在处理内容...';

  @override
  String get learnSuccess => '学习成功！';

  @override
  String get learnFailed => '学习失败';

  @override
  String get enterUrl => '输入网站URL...';

  @override
  String get enterYoutubeUrl => '输入YouTube URL...';

  @override
  String get adminTeachSubtitle => '上传文件、URL、视频';
}
