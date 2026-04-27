# 公网部署说明

## A. 为什么需要后端部署

智谱 API Key 不能写在前端 JavaScript 中，否则任何访问者都可以在浏览器开发者工具中看到密钥。本项目采用 Flask 后端作为安全代理：

浏览器页面 → Flask `/recommend` 或 `/api/recommend` → 智谱 API → Flask → 浏览器

这样 API Key 只保存在服务器环境变量里。

## B. 推荐部署方式

### 方式 1：Render

1. 将项目上传到 GitHub。
2. 在 Render 创建 Web Service。
3. Runtime 选择 Python。
4. Build Command：

```bash
pip install -r requirements.txt
```

5. Start Command：

```bash
gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
```

6. 添加环境变量：

```text
ZHIPU_API_KEY=你的智谱APIKey
SECRET_KEY=随机长字符串
ZHIPU_MODEL=glm-4-flash-250414
ENABLE_WEB_FALLBACK=true
ADMIN_UPLOAD_TOKEN=你的上传口令
```

### 方式 2：Railway

Railway 可直接识别 `Procfile`：

```text
web: gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
```

你只需要在 Variables 中配置环境变量。

### 方式 3：Docker VPS

```bash
docker build -t ldpe-glm-site .
docker run -d --name ldpe-glm-site \
  -p 80:5000 \
  -e ZHIPU_API_KEY=你的智谱APIKey \
  -e SECRET_KEY=随机长字符串 \
  -e ADMIN_UPLOAD_TOKEN=上传口令 \
  ldpe-glm-site
```

## C. 常见问题

### 1. 页面能打开，但 AI 推荐不生成

检查：

- 是否配置 `ZHIPU_API_KEY`
- API Key 是否有效
- 部署平台是否允许外网请求
- 日志里是否出现 HTTP 401、403 或超时

### 2. 本地结果一直触发联网

说明本地匹配没有达到双阈值。可检查：

- 数据库中是否有目标参数值
- 工况范围是否覆盖输入 T/P/剪切率
- 是否过度限定了牌号或黏度模型
- 可适当降低 `LOCAL_MATCH_SCORE_THRESHOLD` 或增加数据库样本

### 3. 上传数据库失败

公共部署默认要求 `ADMIN_UPLOAD_TOKEN`。请在环境变量中设置，并在上传页面输入同样口令。

### 4. 如何更新 Excel

有三种方式：

- 通过受保护的 `/upload` 页面上传；
- 在仓库中替换 Excel 后重新部署；
- 在服务器中直接替换项目目录下的 xlsx 文件并重启服务。
