// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Chinese (`zh`).
class AppLocalizationsZh extends AppLocalizations {
  AppLocalizationsZh([String locale = 'zh']) : super(locale);

  @override
  String get appTitle => 'Cognithor';

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
    return '请确认 Cognithor 后端正在 $url 运行';
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
  String get sendingApproval => '正在发送审批...';

  @override
  String get sendingRejection => '正在发送拒绝...';

  @override
  String get actionApproved => '✓ 操作已批准';

  @override
  String get actionRejected => '✓ 操作已拒绝';

  @override
  String errorWithDetail(String detail) {
    return '错误：$detail';
  }

  @override
  String get noBackendConnection => '无法连接到后端。请检查您的连接。';

  @override
  String get approvalSendFailed => '无法发送审批（连接丢失）。请重试。';

  @override
  String get connectionLost => '与服务器的连接丢失';

  @override
  String get connectionRestoring => '正在恢复连接...';

  @override
  String get connectNow => '立即连接';

  @override
  String get recheck => '重新检查';

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
  String get deleteCredential => '删除凭证';

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
  String get configPageTools => '工具';

  @override
  String get configPageAudit => '审计';

  @override
  String get auditChainIntegrity => '链完整性';

  @override
  String get auditVerifyChain => '验证链';

  @override
  String get auditTimestamps => '时间戳';

  @override
  String get auditGdprExport => 'GDPR导出';

  @override
  String get auditExport => '导出审计';

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
  String get qaKnowledgeBase => '问答';

  @override
  String get lineage => '溯源';

  @override
  String get question => '问题';

  @override
  String get answer => '回答';

  @override
  String get topic => '主题';

  @override
  String get addQA => '添加问答';

  @override
  String get verify => '验证';

  @override
  String get source => '来源';

  @override
  String get noQAPairs => '暂无知识条目';

  @override
  String get noLineage => '暂无溯源数据';

  @override
  String get entityLineage => '实体溯源';

  @override
  String get recentChanges => '最近更改';

  @override
  String get created => '创建时间';

  @override
  String get updated => '更新时间';

  @override
  String get decayed => '已衰减';

  @override
  String get runExploration => '运行探索';

  @override
  String get explorationComplete => '探索完成';

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

  @override
  String get newSkill => '新技能';

  @override
  String get editSkill => '编辑技能';

  @override
  String get createSkill => '创建技能';

  @override
  String get deleteSkill => '删除技能';

  @override
  String get skillName => '名称';

  @override
  String get skillBody => '技能内容 (Markdown)';

  @override
  String get triggerKeywords => '触发关键词';

  @override
  String get requiredTools => '所需工具';

  @override
  String get modelPreference => '模型偏好';

  @override
  String get skillSaved => '技能保存成功';

  @override
  String get skillCreated => '技能创建成功';

  @override
  String get skillDeleted => '技能已删除';

  @override
  String get confirmDeleteSkill => '确定要删除此技能吗？此操作不可撤销。';

  @override
  String get discardChanges => '放弃更改？';

  @override
  String get discardChangesBody => '你有未保存的更改。放弃吗？';

  @override
  String get totalUses => '总使用次数';

  @override
  String get lastUsed => '最后使用';

  @override
  String get commaSeparated => '逗号分隔';

  @override
  String get skillBodyHint => '用Markdown编写技能说明...';

  @override
  String get metadata => '元数据';

  @override
  String get statistics => '统计';

  @override
  String get builtInSkill => '内置技能（只读）';

  @override
  String get exportSkillMd => '导出为 SKILL.md';

  @override
  String get skillExported => '技能已导出到剪贴板';

  @override
  String get general => '通用';

  @override
  String get productivity => '生产力';

  @override
  String get research => '研究';

  @override
  String get analysis => '分析';

  @override
  String get development => '开发';

  @override
  String get automation => '自动化';

  @override
  String get newAgent => '新建代理';

  @override
  String get editAgent => '编辑代理';

  @override
  String get deleteAgent => '删除代理';

  @override
  String get confirmDeleteAgent => '确定要删除此代理吗？此操作无法撤销。';

  @override
  String get agentCreated => '代理创建成功';

  @override
  String get agentSaved => '代理保存成功';

  @override
  String get agentDeleted => '代理已删除';

  @override
  String get displayName => '显示名称';

  @override
  String get preferredModel => '首选模型';

  @override
  String get sandboxTimeout => '沙箱超时 (秒)';

  @override
  String get sandboxNetwork => '沙箱网络';

  @override
  String get canDelegateTo => '可委托给';

  @override
  String get cannotDeleteDefault => '无法删除默认代理';

  @override
  String get robotOfficePipMode => 'Robot Office 处于画中画模式';

  @override
  String get fullscreen => '全屏';

  @override
  String get pipLabel => '画中画';

  @override
  String taskCount(int count) {
    return '$count 个任务';
  }

  @override
  String get hackerMode => '黑客模式';

  @override
  String get entityVisualization => '实体可视化';

  @override
  String get manageSecrets => '管理密钥';

  @override
  String get channelToggles => '频道开关';

  @override
  String get channelSettings => '频道设置';

  @override
  String get tapToSelect => '点击选择...';

  @override
  String get selectModel => '选择模型';

  @override
  String get searchModels => '搜索模型...';

  @override
  String get remove => '移除';

  @override
  String get stopBackend => '停止后端';

  @override
  String get stopBackendDescription => '停止 Cognithor 后端。你需要手动重新启动。';

  @override
  String get stopBackendConfirmBody => '这将停止 Cognithor 后端进程。你需要通过命令行手动重新启动。';

  @override
  String get backendStopped => '后端已停止。请手动重新启动。';

  @override
  String get downloadConfigDesc => '以 JSON 格式下载当前配置';

  @override
  String get loadConfigDesc => '从 JSON 文件加载配置';

  @override
  String get resetAllDesc => '将所有设置恢复为默认值。此操作不可撤销。';

  @override
  String get factoryResetNotImpl =>
      '后端尚未实现恢复出厂设置功能。要手动重置，请删除 config.yaml 并重新启动 Cognithor。';

  @override
  String get ok => '确定';

  @override
  String get wizardSubtitle => '你的个人 AI 助手';

  @override
  String get chooseLlmProvider => '选择你的 LLM 提供商';

  @override
  String get localOllama => '本地 (Ollama)';

  @override
  String get localOllamaDesc => '在自己的硬件上运行模型。完全隐私，无 API 费用。需要安装 Ollama。';

  @override
  String get cloudProviderLabel => '云提供商';

  @override
  String get cloudProviderDesc =>
      '使用 OpenAI、Anthropic 或其他云 API。更快的设置，需要 API 密钥。';

  @override
  String get next => '下一步';

  @override
  String get ollamaConfiguration => 'Ollama 配置';

  @override
  String get cloudApiConfiguration => '云 API 配置';

  @override
  String get ollamaConfigHint => '输入 Ollama 运行的 URL。';

  @override
  String get cloudConfigHint => '选择你的云提供商并输入 API 密钥。';

  @override
  String get ollamaUrl => 'Ollama URL';

  @override
  String get testConnection => '测试连接';

  @override
  String get testingConnection => '测试中...';

  @override
  String get youreAllSet => '一切就绪！';

  @override
  String get ollamaReadyMsg => 'Ollama 已连接并准备就绪。Cognithor 将使用你的本地模型进行规划和执行。';

  @override
  String cloudReadyMsg(String provider) {
    return '$provider 已配置。Cognithor 将使用你的云 API 进行规划和执行。';
  }

  @override
  String get changeSettingsAnytime => '你可以随时更改这些设置。';

  @override
  String get startUsingCognithor => '开始使用 Cognithor';

  @override
  String get ollamaNoModels =>
      '已连接到 Ollama。尚未安装模型 - 运行 \"ollama pull qwen3:8b\" 开始使用。';

  @override
  String ollamaModelsAvailable(int count) {
    return '已连接到 Ollama。$count 个模型可用。';
  }

  @override
  String ollamaStatusError(int code) {
    return 'Ollama 响应状态 $code。请确保服务器正在运行。';
  }

  @override
  String get enterApiKey => '请输入 API 密钥。';

  @override
  String apiKeyTooShort(String provider) {
    return '密钥看起来太短了。请仔细检查你的 $provider API 密钥。';
  }

  @override
  String apiKeySaved(String provider) {
    return '$provider API 密钥已保存。你可以稍后在设置中更改。';
  }

  @override
  String connectionFailed(String error) {
    return '连接失败：$error';
  }

  @override
  String get minimize => '最小化';

  @override
  String get shrink => '缩小';

  @override
  String get expandLabel => '展开';

  @override
  String get robotOffice => '机器人办公室';

  @override
  String get copy => '复制';

  @override
  String get share => '分享';

  @override
  String get noLogEntries => '暂无日志记录';

  @override
  String get noPlanData => '暂无计划数据';

  @override
  String get noDagData => '暂无 DAG 数据';

  @override
  String get log => '日志';

  @override
  String get fileReadError => '文件无法读取';

  @override
  String uploadError(String error) {
    return '上传错误：$error';
  }

  @override
  String get toolSpecificTimeouts => '工具专用超时';

  @override
  String get required => '必填';

  @override
  String get stopLabel => '停止';

  @override
  String get resetLabel => '重置';

  @override
  String get exportLabel => '导出';

  @override
  String get importLabel => '导入';

  @override
  String get catAiEngine => 'AI 引擎';

  @override
  String get catChannels => '频道';

  @override
  String get catKnowledge => '知识';

  @override
  String get catSecurity => '安全';

  @override
  String get catSystem => '系统';

  @override
  String get saved => '已保存';

  @override
  String get chatHistory => '聊天历史';

  @override
  String get newChat => '新聊天';

  @override
  String get untitledChat => '未命名聊天';

  @override
  String get deleteChat => '删除聊天';

  @override
  String get confirmDeleteChat => '删除这个对话？';

  @override
  String messagesCount(String count) {
    return '$count 条消息';
  }

  @override
  String get justNow => '刚刚';

  @override
  String minutesAgo(String count) {
    return '$count 分钟前';
  }

  @override
  String hoursAgo(String count) {
    return '$count 小时前';
  }

  @override
  String daysAgo(String count) {
    return '$count 天前';
  }

  @override
  String get folders => '文件夹';

  @override
  String get moveToFolder => '移动到文件夹';

  @override
  String get newFolder => '新建文件夹';

  @override
  String get folderName => '文件夹名称';

  @override
  String get noFolder => '未分类';

  @override
  String get renameChat => '重命名';

  @override
  String get editTitle => '编辑标题';

  @override
  String sessionCount(int count) {
    return '$count 个聊天';
  }

  @override
  String get idle => '空闲';

  @override
  String get thinking => '思考中...';

  @override
  String get chooseBackend => '选择 LLM 后端';

  @override
  String get claudeSubscription => 'Claude 订阅';

  @override
  String get claudeSubscriptionDesc => '使用你的 Claude Pro/Max -- 无需 API 密钥';

  @override
  String get ollamaLocal => 'Ollama (本地)';

  @override
  String get ollamaLocalDesc => '免费，使用你的 GPU -- 无需联网';

  @override
  String get openaiApi => 'OpenAI API';

  @override
  String get anthropicApi => 'Anthropic API';

  @override
  String get connected => '已连接';

  @override
  String get notInstalled => '未安装';

  @override
  String get noKey => '无 API 密钥';

  @override
  String get keyConfigured => '密钥已配置';

  @override
  String get recommended => '推荐';

  @override
  String get switchBackend => '切换后端';

  @override
  String get restartRequired => '需要重启才能完全生效';

  @override
  String get installClaude => '安装 Claude Code';

  @override
  String get toolsComputerUseLabel => '计算机控制';

  @override
  String get toolsComputerUseDesc => '启用计算机控制（鼠标、键盘、截图）';

  @override
  String get toolsDesktopLabel => '桌面工具';

  @override
  String get toolsDesktopDesc => '访问剪贴板和桌面截图';

  @override
  String get toolsSectionDesktop => '桌面和自动化';

  @override
  String get toolsWarning => '更改需要重新启动';

  @override
  String get configPageSystemProfile => '系统配置';

  @override
  String get systemTier => '系统等级';

  @override
  String get systemRecommendedMode => '推荐模式';

  @override
  String get systemRescan => '重新扫描';

  @override
  String get configPageBudget => '预算';

  @override
  String get configPageEvolution => '进化引擎';

  @override
  String get kanbanNewTask => '新任务';

  @override
  String get kanbanMyTasks => '我的任务';

  @override
  String get kanbanLivePipeline => '实时管道';

  @override
  String get kanbanDescription => '描述';

  @override
  String get kanbanResult => '结果';

  @override
  String kanbanSubtasks(int count) {
    return '子任务 ($count)';
  }

  @override
  String get kanbanHistory => '历史';

  @override
  String get kanbanNoHistory => '暂无状态变更。';

  @override
  String get kanbanMetadata => '元数据';

  @override
  String kanbanSource(String source) {
    return '来源：$source';
  }

  @override
  String kanbanCreated(String date) {
    return '创建：$date';
  }

  @override
  String kanbanUpdated(String date) {
    return '更新：$date';
  }

  @override
  String kanbanCompleted(String date) {
    return '完成：$date';
  }

  @override
  String get kanbanEditTask => '编辑任务';

  @override
  String get kanbanSave => '保存';

  @override
  String get kanbanCreate => '创建';

  @override
  String get deviceSettings => '设备设置';

  @override
  String get refreshSensors => '刷新传感器';

  @override
  String get documentsTitle => '文档和模板';

  @override
  String get connectedDevices => '已连接设备';

  @override
  String get connectedDevicesSubtitle => '管理已配对的移动设备';

  @override
  String get noDevicesPaired => '尚未配对任何设备';

  @override
  String get noDevicesHint => '通过扫描二维码配对移动设备';

  @override
  String get pairNewDevice => '配对新设备';

  @override
  String get deviceName => '设备名称';

  @override
  String get deviceId => '设备 ID';

  @override
  String get pairedAt => '配对时间';

  @override
  String get expiresAt => '过期时间';

  @override
  String get revokeDevice => '撤销';

  @override
  String get revokeDeviceConfirm => '撤销此设备？需要重新配对。';

  @override
  String get deviceRevoked => '设备已撤销';

  @override
  String get pairingQrTitle => '扫描二维码';

  @override
  String get pairingQrHint => '在手机上打开 Cognithor 应用并扫描此二维码';

  @override
  String get pairingSuccess => '设备配对成功';

  @override
  String get scanQrCode => '扫描二维码';

  @override
  String get scanQrHint => '将摄像头对准配对二维码';

  @override
  String get qrScanError => '无法读取二维码';

  @override
  String get networkSettings => '网络与连接';

  @override
  String get networkSettingsSubtitle => '管理网络接口和 API 绑定';

  @override
  String get detectedInterfaces => '检测到的接口';

  @override
  String get noInterfacesDetected => '未检测到网络接口';

  @override
  String get enabledEndpoints => '已启用的端点';

  @override
  String get autoDetect => '自动检测受信任接口';

  @override
  String get autoDetectHint => '自动启用 Tailscale、ZeroTier 和其他 VPN 接口';

  @override
  String get interfaceLoopback => '回环';

  @override
  String get interfaceLan => '局域网';

  @override
  String get interfaceTailscale => 'Tailscale';

  @override
  String get interfaceZerotier => 'ZeroTier';

  @override
  String get interfaceWireguard => 'WireGuard';

  @override
  String get interfaceCloudflare => 'Cloudflare';

  @override
  String get interfaceUnknown => '未知';

  @override
  String get bindHost => '绑定主机';

  @override
  String get trusted => '受信任';

  @override
  String get untrusted => '不受信任';

  @override
  String get vaultActive => '保险库激活';

  @override
  String get vaultActiveDesc => '启用/禁用知识保险库';

  @override
  String get vaultEncryption => '文件加密';

  @override
  String get vaultEncryptionDesc => '使用 AES-256 加密保险库 .md 文件。';

  @override
  String get vaultEncryptOn =>
      '最高安全性：保险库文件已加密。数据库 + 记忆 + 保险库 = 全部受保护。Obsidian 无法读取这些文件。';

  @override
  String get vaultEncryptOff =>
      '兼容 Obsidian：保险库文件为明文。数据库和记忆文件仍然加密。要完全保护保险库文件：启用 BitLocker（Windows）或 LUKS（Linux）。';

  @override
  String get vaultAlwaysEncrypted => '始终加密（与此开关无关）：';

  @override
  String get vaultAlwaysEncryptedList =>
      '  - 33 个 SQLite 数据库（SQLCipher / AES-256）\n  - CORE.md（代理个性）\n  - 情景记忆（.md）\n  - 学习过程（.md）\n  - 学习计划（.json）\n  - 凭证（Fernet / PBKDF2）\n  - 密钥：OS 密钥环（不在磁盘上）';

  @override
  String get vaultAutoSave => '自动保存研究';

  @override
  String get vaultAutoSaveDesc => '自动将网络研究保存到保险库';

  @override
  String get kanbanBacklog => '待办';

  @override
  String get kanbanInProgress => '进行中';

  @override
  String get kanbanReview => '审核';

  @override
  String get kanbanDone => '完成';

  @override
  String get kanbanBlocked => '已阻塞';

  @override
  String get kanbanArchive => '归档';

  @override
  String get kanbanSettings => '看板设置';

  @override
  String get taskSources => '任务来源';

  @override
  String get fromChat => '来自聊天';

  @override
  String get fromChatDesc => '自动检测对话中的任务创建';

  @override
  String get fromCron => '来自定时任务';

  @override
  String get fromCronDesc => '从定时任务结果创建任务';

  @override
  String get fromEvolution => '来自进化引擎';

  @override
  String get fromEvolutionDesc => '检测到的改进机会的任务';

  @override
  String get fromAgents => '来自代理';

  @override
  String get fromAgentsDesc => '允许代理在执行期间创建任务';

  @override
  String get guards => '限制';

  @override
  String get maxAutoTasks => '每次会话最大自动任务数';

  @override
  String get maxSubtaskDepth => '最大子任务深度';

  @override
  String get defaults => '默认值';

  @override
  String get defaultPriority => '默认优先级';

  @override
  String get defaultAgent => '默认代理';

  @override
  String get autoArchiveDays => '自动归档天数';

  @override
  String get priorityLow => '低（仅索引）';

  @override
  String get priorityNormal => '正常';

  @override
  String get priorityHigh => '高（优先学习）';

  @override
  String get deepLearningQueued => '深度学习已排队';

  @override
  String get deepLearningSkipped => '仅索引';

  @override
  String get learnPriority => '优先级';

  @override
  String get deepLearningQueue => '深度学习队列';

  @override
  String get operationModeDesc =>
      'Auto：根据API密钥自动检测。Offline：仅本地Ollama，无网络工具。Online：云端LLM+网络搜索。Hybrid：Ollama+云端备用。';

  @override
  String get copyQrPayload => '复制QR数据';

  @override
  String get copyToken => '复制令牌';

  @override
  String get newLearningGoal => '新学习目标';

  @override
  String get create => '创建';

  @override
  String get evolutionEngine => '进化引擎';

  @override
  String get searchLabel => '搜索';

  @override
  String get hideRobotOffice => '隐藏办公室';

  @override
  String get socialListening => '社交监听';

  @override
  String get productName => '产品名称';

  @override
  String get productDescription => '产品描述';

  @override
  String get replyTone => '回复语气';

  @override
  String get subreddits => '子版块';

  @override
  String get subredditsHint => '逗号分隔，不含r/';

  @override
  String get minIntentScore => '最低意图分数';

  @override
  String get scanInterval => '扫描间隔（分钟）';

  @override
  String get autoScan => '自动扫描';

  @override
  String get autoPost => '自动发布';

  @override
  String get autoPostHint => '需要在浏览器中一次性登录Reddit';

  @override
  String get socialSetupRequired => '请设置产品名称和至少一个子版块以开始搜索潜在客户。';

  @override
  String get redditLeads => '潜在客户';

  @override
  String get noLeadsFound => '尚未发现潜在客户';

  @override
  String get noLeadsHint => '在设置中配置产品和子版块，然后扫描Reddit';

  @override
  String leadScore(int score) {
    return '分数: $score';
  }

  @override
  String get leadNew => '新';

  @override
  String get leadReviewed => '已审核';

  @override
  String get leadReplied => '已回复';

  @override
  String get leadArchived => '已归档';

  @override
  String get draftReply => '回复草稿';

  @override
  String get editReply => '编辑回复';

  @override
  String get postReply => '发布回复';

  @override
  String get copyReply => '复制回复';

  @override
  String get openOnReddit => '在Reddit上打开';

  @override
  String get markReviewed => '标记为已审核';

  @override
  String get archiveLead => '归档';

  @override
  String get intentScore => '意图分数';

  @override
  String get scoreReason => '原因';

  @override
  String get leadStats => '潜在客户统计';

  @override
  String get filterAll => '全部';

  @override
  String get processQueue => '处理队列';

  @override
  String get wizardComplete => '队列完成';

  @override
  String wizardSummary(int replied, int skipped, int archived) {
    return '$replied已回复, $skipped已跳过, $archived已归档';
  }

  @override
  String get improve => '改进';

  @override
  String get variants => '变体';

  @override
  String get useTemplate => '使用模板';

  @override
  String get skipLead => '跳过';

  @override
  String get noTemplates => '尚无保存的模板';

  @override
  String get feedbackTitle => '这条回复表现如何?';

  @override
  String get feedbackConverted => '转化（用户试用了产品）';

  @override
  String get feedbackConversation => '产生了对话';

  @override
  String get feedbackIgnored => '被忽略（无反应）';

  @override
  String get feedbackNegative => '负面（被踩）';

  @override
  String get feedbackDeleted => '被版主删除';

  @override
  String get engagementScore => '参与度';

  @override
  String get discoverSubreddits => '发现子版块';

  @override
  String get discovering => '发现中...';

  @override
  String get pendingReview => '待审核';

  @override
  String get approveTask => '批准';

  @override
  String get rejectTask => '拒绝';

  @override
  String get scheduled => '计划任务';

  @override
  String get noScheduledTasks => '没有计划任务';

  @override
  String get scheduledTasksHint => '在管理中配置定时任务以安排周期性任务。';

  @override
  String get activeJobs => '活跃';

  @override
  String get pausedJobs => '已暂停';

  @override
  String get nextRun => '下次';

  @override
  String get navTraces => '追踪';

  @override
  String get traceStatusRunning => '运行中';

  @override
  String get traceStatusCompleted => '已完成';

  @override
  String get traceStatusFailed => '失败';

  @override
  String get traceFilterAll => '全部';

  @override
  String get traceFilterRunning => '运行中';

  @override
  String get traceFilterCompleted => '已完成';

  @override
  String get traceFilterFailed => '失败';

  @override
  String get traceEmpty => '暂无追踪记录';

  @override
  String get traceNotFound => '追踪未找到 — 可能已轮换。';
}
