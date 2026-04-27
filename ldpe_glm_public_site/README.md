# LDPE 熔体 Fluent 仿真参数智能推荐公共网站

这是基于你原有 Flask + Excel 参数库网站重构后的公网可部署版本。它面向 **LDPE 聚合物熔体本身** 的参数检索、推荐与解释，不直接绑定论文中的 M1–M4 模型页面，但可作为你的四模型 Fluent 对比研究的统一参数基础。

## 1. 核心能力

- **LDPE 参数数据库展示**：读取 `source_table` 与 `ldpe_param_table`，展示热物性、流变参数、温度依赖黏度模型系数及来源字段。
- **本地库优先推荐**：按温度、压力、剪切率、牌号、黏度模型、应用场景、试验方法、设备装置、通道形状、质量评分和置信等级计算相似度。
- **双阈值质量判定**：只有当 `样本数 >= LOCAL_MIN_SAMPLES` 且 `最高匹配分 >= LOCAL_MATCH_SCORE_THRESHOLD` 时，才判定为本地匹配充分。
- **智谱 GLM-4-Flash 解释**：后端安全调用智谱 API，API Key 不暴露到浏览器。
- **联网检索兜底**：本地样本不足时自动调用 `web_search`，网络结果与本地结果分区展示，仅作临时参考，禁止自动入库。
- **缺失字段静默过滤**：传给大模型前自动删除空值字段，避免模型围绕空字段生成冗余解释。
- **公共部署支持**：内置 `Procfile`、`Dockerfile`、`render.yaml`，可部署到 Render、Railway、Fly.io、自有服务器等。
- **API 接口**：提供 `/api/recommend`，方便后续被其他页面、脚本或 Fluent 参数预处理程序调用。

## 2. 目录结构

```text
ldpe_glm_public_site/
├── app.py                         # Flask 主程序：数据库读取、推荐、GLM 调用、联网兜底
├── wsgi.py                        # WSGI 入口
├── requirements.txt               # Python 依赖
├── Procfile                       # Heroku/Railway/Render 等平台启动文件
├── Dockerfile                     # Docker 部署文件
├── render.yaml                    # Render 一键部署配置
├── .env.example                   # 环境变量示例
├── LDPE_dedup_confidence_review.xlsx
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── database.html
│   ├── recommend.html
│   ├── upload.html
│   └── about.html
├── static/
│   └── style.css
└── docs/
    ├── DEPLOYMENT.md
    └── API_USAGE.md
```

## 3. 环境变量

部署公网时，至少需要配置：

```bash
ZHIPU_API_KEY=你的智谱APIKey
SECRET_KEY=一个足够长的随机字符串
ZHIPU_MODEL=glm-4-flash-250414
ENABLE_WEB_FALLBACK=true
```

建议同时配置上传保护：

```bash
ADMIN_UPLOAD_TOKEN=一个只有你知道的上传口令
```

可调节本地库判据：

```bash
LOCAL_MIN_SAMPLES=3
LOCAL_MATCH_SCORE_THRESHOLD=5.5
MAX_LOCAL_CONTEXT_ROWS=6
WEB_SEARCH_COUNT=5
```

## 4. 本地开发运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填入 ZHIPU_API_KEY
python app.py
```

访问：`http://127.0.0.1:5000`

> 注意：本地运行只是开发调试。公网部署请使用 Render/Railway/Docker 等方式，并在平台环境变量中填写 API Key。

## 5. Render 部署流程

1. 将本文件夹上传到 GitHub 仓库。
2. 登录 Render，选择 New Web Service。
3. 连接该仓库。
4. Build Command：`pip install -r requirements.txt`
5. Start Command：`gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
6. 在 Environment 中添加：
   - `ZHIPU_API_KEY`
   - `SECRET_KEY`
   - `ADMIN_UPLOAD_TOKEN`（可选但推荐）
7. 部署完成后访问 Render 分配的网址。

也可以直接使用 `render.yaml` 创建 Blueprint。

## 6. Railway 部署流程

1. 上传到 GitHub。
2. Railway 新建项目并选择该仓库。
3. Railway 会识别 `Procfile`。
4. 添加环境变量 `ZHIPU_API_KEY`、`SECRET_KEY`、`ADMIN_UPLOAD_TOKEN`。
5. 部署完成后生成公网域名。

## 7. Docker 部署

```bash
docker build -t ldpe-glm-site .
docker run -p 5000:5000 \
  -e ZHIPU_API_KEY=你的智谱APIKey \
  -e SECRET_KEY=please-change \
  -e ADMIN_UPLOAD_TOKEN=please-change \
  ldpe-glm-site
```

## 8. API 调用示例

```bash
curl -X POST https://你的域名/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "target": "mu_ref",
    "temperature": 190,
    "pressure": 1,
    "shear_rate": 100,
    "application_scene": "LDPE 熔体管道输送 Fluent 非等温流动",
    "channel_geometry": "圆管",
    "question": "请推荐该工况可用于 Fluent 的 LDPE 熔体黏度参数"
  }'
```

## 9. 数据库字段建议

你已经补充的字段均已被系统纳入大模型解释上下文：

- `application_scene`：应用场景
- `experimental_method`：试验方法
- `equipment_or_setup`：设备或装置
- `channel_geometry`：通道形状
- `channel_length`：通道长度
- `channel_diameter`：通道直径或间隙
- `ldpe_grade_detail`：LDPE 牌号或型号
- `main_result_summary`：文献主要结果简述
- `data_extraction_method`：参数提取方式
- `limitation_note`：数据局限性

系统在发送给大模型前会自动过滤空值字段，避免大模型为缺失数据生成解释。

## 10. 学术使用建议

在论文中建议将该系统定位为：

> 面向 LDPE 熔体 Fluent 数值模拟的参数数据库与智能推荐原型，用于统一热物性参数、流变参数与温度依赖黏度模型系数的整理、拟合、检索和解释，为不同物理简化层级模型的输入参数提供可追溯、可复现的依据。

网络检索结果只适合辅助发现潜在来源，不能直接替代人工文献核验。正式写入数据库前，应检查单位、温度范围、剪切率范围、牌号一致性、数据提取方式、版权和引用规范。

## v3 更新：严格工况覆盖判定

为避免“文本相似度/质量评分较高但温度、压力或剪切率已经超出数据库范围”的情况被误判为本地充分，AI 推荐页新增严格覆盖规则：

- `STRICT_RANGE_COVERAGE=true` 时，温度、压力、剪切率必须被本地库有效范围覆盖；
- 至少 `LOCAL_MIN_JOINT_COVERAGE` 条样本需要同时覆盖当前输入工况；
- 一旦温度、压力或剪切率超出本地数据库范围，即判定为“本地不足”，自动触发联网兜底；
- 联网结果仍然只作为临时参考，不自动入库。
