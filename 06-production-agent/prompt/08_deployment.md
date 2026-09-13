现在实现 Production Agent 实验 ：

Deployment。

将当前 LangGraph Agent 做成可部署应用。

建立：

langgraph.json

并定义：

graph entrypoint
dependencies
environment variables

要求：

Development：

local LangGraph server

Production：

persistent storage
production checkpointer
production environment variables

API 必须支持：

POST /runs
GET /runs/{id}
GET /runs/{id}/state
POST /runs/{id}/resume

要求：

1. Authentication
2. Request validation
3. Rate limiting
4. Timeout
5. Error handling
6. Request ID
7. Health check
8. Readiness check
9. Graceful shutdown

创建：

deployment/

Dockerfile
docker-compose.yml
langgraph.json

docs/08-DEPLOYMENT.md

创建：

/health
/ready

两个 endpoint。

不要把 secrets 写入 Docker image。
