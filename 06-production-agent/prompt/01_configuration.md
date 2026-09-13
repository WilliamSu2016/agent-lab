现在完成 Production Agent 实验：

Configuration & Secrets。

重构现有 Agent 项目。

要求：

1. 所有 API keys 从环境变量读取。
2. 不允许 secrets 出现在 source code。
3. 不允许 secrets 出现在 Git。
4. 提供 .env.example。
5. .env 必须加入 .gitignore。
6. Model name 必须可配置。
7. Agent limits 必须可配置。
8. Timeout 必须可配置。
9. Max iterations 必须可配置。
10. Retry policy 必须可配置。
11. Environment 至少区分：
    development
    test
    production

创建：

config/
.env.example

docs/01-CONFIGURATION.md

添加 automated test：

确保应用在缺少 required secret 时能够明确失败。

不要把真实 secret 写入任何文件。
