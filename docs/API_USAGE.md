# API 使用说明

## 1. 健康检查

```http
GET /api/health
```

返回示例：

```json
{
  "ok": true,
  "records": 128,
  "data_file": "LDPE_dedup_confidence_review.xlsx",
  "zhipu_model": "glm-4-flash-250414",
  "api_key_ready": true,
  "web_fallback_enabled": true
}
```

## 2. 参数推荐

```http
POST /api/recommend
Content-Type: application/json
```

请求体示例：

```json
{
  "target": "mu_ref",
  "temperature": 190,
  "pressure": 1.0,
  "shear_rate": 100,
  "grade": "",
  "vis_model": "",
  "application_scene": "LDPE 熔体管道输送 Fluent 非等温流动",
  "experimental_method": "毛细管流变或文献拟合",
  "equipment_or_setup": "圆管输送",
  "channel_geometry": "圆管",
  "channel_length": "1 m",
  "channel_diameter": "10 mm",
  "question": "请推荐可用于 Fluent 的 LDPE 熔体黏度参数，并说明是否需要联网补充。"
}
```

`target` 可选：

- `mu_ref`：参考黏度
- `rho`：密度
- `cp`：比热
- `k`：导热系数
- `A`：温度依赖黏度模型系数 A
- `E_mu`：黏流活化能
- `K`：幂律或 Carreau 稠度系数
- `m`：模型指数

返回结构核心字段：

```json
{
  "local": {
    "value": 1234.5,
    "value_text": "1234.5",
    "sufficient": true,
    "sample_count": 4,
    "max_score": 7.2,
    "explanation": "..."
  },
  "matched_rows": [
    {"参数编号": "PARAM_001", "温度下限/℃": 160, "匹配分": 7.2}
  ],
  "web": {
    "triggered": false,
    "explanation": "",
    "results": [],
    "policy": "网络结果仅为临时补充参考..."
  }
}
```

## 3. 结果判定逻辑

```text
本地充分 = 相似样本数量 >= LOCAL_MIN_SAMPLES
        且 最高匹配分 >= LOCAL_MATCH_SCORE_THRESHOLD
```

本地充分时：

- 只展示本地库推荐解释；
- 不触发联网检索；
- 适合用于 Fluent 初始参数选择或敏感性分析中心值。

本地不足时：

- 展示本地库不足原因；
- 调用智谱 web_search；
- 网络结果只作为临时参考，不自动入库。
