# Changelog

## Unreleased

## [0.3.0] - 2026-08-06

### Added

- 可选的 Automotive Profile，支持经过 Schema 校验的意图识别、实体抽取和槽位状态管理。
- 时区感知的出发时间规范化、槽位补全、冲突确认、用户纠错、取消和安全降级。
- 基于 FastAPI 的本地 Mock Vehicle API、类型化客户端、车辆权限策略和六步行程充电流程。
- 确定性的路线与充电站测试数据、线程安全车辆状态、幂等空调控制和仅测试环境可用的故障注入。
- 共享 `ActionRegistry`、执行前 `PlanValidator`、输出契约和受限顺序 `PlanExecutor`。
- 只允许引用已声明依赖的安全 `$ref` 前序结果解析，并在必需步骤失败时立即终止。
- 默认规则 Planner，以及可选的受限 Hybrid Planner；LLM 计划校验失败时自动回退规则模板。
- 中文本地 RAG：安全 Manifest 加载、标题感知分块、jieba 预分词、SQLite FTS5/BM25、增量原子索引和 Citation。
- RAG 离线评测脚本，覆盖 Recall@K、MRR 和无答案拒答率。

### Changed

- OpenAI 与 Anthropic Provider 统一支持规范化的强制工具选择。
- Agent 可按 `coding` 或 `automotive` Profile 装配能力，并共享同一套工具权限边界。

### Quality

- 扩充至 206 个通过的离线测试，分支覆盖率达到 86.47%。
- CI 增加汽车意图、车辆 API、Planner、RAG 回归测试和 wheel 安装验证。

本项目遵循语义化版本。

## [0.2.0] - 2026-08-02

### Added

- 统一的 `allow / ask / deny` 写操作权限决策和兼容旧配置的迁移。
- Shell `low / medium / high / critical` 风险分级与执行前展示。
- `MINICODER_READ_ONLY` 和 `--read-only` 全局只读模式。
- 模型级上下文窗口、输出预留和模型感知的本地 token 估算。
- 显式 Provider capabilities，包括工具、流式、流式 usage 和并行工具能力。
- JSON/Markdown 原子会话导出和 `/export` 命令。
- 离线黑盒 E2E、85% 分支覆盖率门禁及 Windows/Linux CI 矩阵。

### Changed

- 上下文压缩改为完整 Turn 原子分组，不拆分 user、assistant、tool calls 和 results。
- Provider 重试区分认证、限流、超时和服务端错误，并遵守 `Retry-After`。
- 会话升级至 Schema v2，使用原子替换、有效备份和损坏恢复。

### Security

- 所有文件工具默认限制在工作区，拒绝 `..`、绝对路径和符号链接逃逸。
- 文件写入、编辑、Shell 和子 Agent 共享同一个 `WorkspacePolicy`。

## [0.1.0]

- 初始教学版本：Agent Loop、双 Provider、核心工具与基础 REPL。
