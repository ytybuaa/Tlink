# 知识管理系统 - 落地计划

## 一、架构总览

```
┌──────────────────────────────────────────┐
│              ChromaDB 单表                │
│  每个 chunk 冗余存储完整 metadata          │
│  entry_id / source / domain / author     │
│  title / time_start / tags               │
└──────────────┬───────────────────────────┘
               │
        ┌──────▼───────┐
        │   FastAPI     │
        │  CRUD + Search│
        └──────┬───────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  BGE-M3    LLM(MiMo)  Parser
  embedding  意图理解    文本解析
             精排/分类
```

**核心设计原则**：
- **ChromaDB 单表**，不引入 SQLite
- 每个 chunk 的 metadata 冗余存储 entry 级信息（source/domain/author/title/time/tags）
- "条目"概念通过 `entry_id` 实现：同 entry 的 chunks 共享 entry_id
- CRUD = 通过 entry_id 批量操作 chunks
- 三种检索模式：相关性 / 时间线 / 时间过滤

---

## 二、数据结构

### 单表设计：chunks 即一切

每个 chunk 是 ChromaDB 中的一条 document，metadata 承担结构化查询职责：

```python
# ChromaDB collection: "knowledge"
{
    "id": "zhijiang_0042",                    # chunk ID
    "document": "调研工作务求深实细准效 ...",    # 文本（可能含 LLM 增强）
    "embedding": [0.012, -0.034, ...],         # BGE-M3 向量
    "metadata": {
        "entry_id": "e3a1b2c4-...",           # 条目 ID（核心关联字段）
        "source": "之江新语",                  # 来源（书名）
        "domain": "政治",                      # 领域
        "author": "习近平",                    # 作者
        "title": "调研工作务求深实细准效",       # 条目标题（冗余）
        "topic": "调研方法",                   # 主题
        "tags": "调研,务实,基层",              # 标签（逗号分隔）
        "time_start": "2003-02-25",           # 时间 ISO
        "original": "调研工作务求..."          # 原始文本（未增强）
    }
}
```

### "条目"的实现

条目 = 共享同一 `entry_id` 的 chunks 集合。

```
entry_id: "e3a1b2c4"
  ├── chunk 0: "调研工作务求..."
  ├── chunk 1: "深入群众..."
  └── chunk 2: "做到听实话..."
```

查询某条目所有 chunks：
```python
collection.get(where={"entry_id": "e3a1b2c4"})
```

### metadata 字段说明

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| entry_id | str | 条目 ID（UUID） | `e3a1b2c4-...` |
| source | str | 来源（书名） | `之江新语` |
| domain | str | 领域（大类） | `政治` |
| author | str | 作者 | `习近平` |
| title | str | 条目标题 | `调研工作务求深实细准效` |
| topic | str | 主题（细粒度） | `调研方法` |
| tags | str | 标签（逗号分隔） | `调研,务实,基层` |
| time_start | str | 时间 ISO | `2003-02-25` |
| original | str | 原始文本 | 未 LLM 增强的原文 |

---

## 三、领域分类

### 预设领域

| 领域 | 说明 | 当前包含 |
|------|------|---------|
| 哲学 | 思想、伦理、人生观 | 论语 |
| 政治 | 治国、党建、政策 | 之江新语、政治的人生、毛主席语录 |
| 经济 | （预留） | — |
| 科学 | （预留） | — |
| 文学 | （预留） | — |

### 分类策略：手动优先，LLM 兜底

```python
def classify_domain(entry):
    if entry.domain:           # 用户指定了 → 直接用
        return entry.domain
    return llm_classify(entry) # LLM 自动判断
```

**LLM 分类 prompt**：
```
根据以下文本内容，判断属于哪个领域：
"哲学"、"政治"、"经济"、"科学"、"文学"、"历史"、"其他"

文本：{content[:200]}

只返回领域名称。
```

---

## 四、三种检索模式

### Mode 1: 相关性排序（默认）

用途：事实问答，如 "X是什么"、"孔子怎么看待仁"

```
用户问题
  → LLM 意图理解（领域过滤 + 问题改写）
  → ChromaDB 向量检索（带 where 过滤）
  → LLM 精排（direct=true/false + score）
  → 返回按相关性排序的结果
```

### Mode 2: 时间线排序

用途：叙事演变，如 "A如何变成B"、"调研思想的发展"

```
用户问题
  → 相关性召回（3x 候选）
  → 过滤有时间的条目
  → 按 time_start 升序排列
  → 返回时间线结果
```

适用：政治的人生（日记有日期）、之江新语（部分有日期）
不适用：论语、毛主席语录（无时间）→ 自动降级为相关性排序

### Mode 3: 时间过滤

用途：指定时段，如 "2003年说了什么"、"1994年1月发生了什么"

```
用户问题
  → LLM 提取时间范围
  → ChromaDB where 过滤 + 向量检索
  → LLM 精排
  → 返回结果
```

**LLM 时间提取 prompt**：
```
从用户问题中提取时间范围。
用户问题：{query}

返回JSON：
{"time_start": "YYYY-MM-DD", "time_end": "YYYY-MM-DD"}
无法提取则返回 {"time_start": null, "time_end": null}
```

---

## 五、CRUD 操作

### 创建条目

```python
def create_entry(source, domain, title, author, topic, tags, time_start, content):
    entry_id = str(uuid4())

    # 1. 分块
    chunks = chunk_text(content, source)

    # 2. 短文本 LLM 增强（< 100 字的 chunks）
    enriched = enrich_short_chunks(chunks)

    # 3. 生成 embedding
    embeddings = batch_get_embeddings([c['text'] for c in enriched])

    # 4. 入库（metadata 冗余存储 entry 信息）
    for i, chunk in enumerate(enriched):
        collection.add(
            ids=[chunk['id']],
            documents=[chunk['text']],
            embeddings=[embeddings[i]],
            metadatas=[{
                "entry_id": entry_id,
                "source": source,
                "domain": domain,
                "author": author,
                "title": title,
                "topic": topic,
                "tags": ",".join(tags),
                "time_start": time_start or "",
                "original": chunk.get('original_text', chunk['text'])
            }]
        )

    return entry_id
```

### 查询条目

```python
def get_entry(entry_id):
    """获取一个条目的所有 chunks"""
    results = collection.get(where={"entry_id": entry_id})
    if not results['ids']:
        return None
    meta = results['metadatas'][0]
    return {
        "entry_id": entry_id,
        "source": meta['source'],
        "domain": meta['domain'],
        "title": meta['title'],
        "author": meta['author'],
        "topic": meta['topic'],
        "tags": meta['tags'].split(','),
        "time_start": meta['time_start'],
        "content": "\n".join(results['documents']),
        "chunks": list(zip(results['ids'], results['documents']))
    }
```

### 更新条目（内容变更）

```python
def update_entry(entry_id, new_content, **kwargs):
    # 1. 删除旧 chunks
    old = collection.get(where={"entry_id": entry_id})
    collection.delete(ids=old['ids'])

    # 2. 重新分块 + embedding + 入库
    create_entry_from_existing(entry_id, new_content, **kwargs)
```

### 更新条目（仅 metadata）

```python
def update_entry_meta(entry_id, **kwargs):
    """更新标签、领域等 metadata（不需要重新 embed）"""
    results = collection.get(where={"entry_id": entry_id})
    for chunk_id, meta in zip(results['ids'], results['metadatas']):
        meta.update(kwargs)  # 如 {"tags": "新标签", "domain": "哲学"}
        collection.update(ids=[chunk_id], metadatas=[meta])
```

### 删除条目

```python
def delete_entry(entry_id):
    """级联删除：删所有 chunks"""
    results = collection.get(where={"entry_id": entry_id})
    if results['ids']:
        collection.delete(ids=results['ids'])
```

### 列表查询

```python
def list_entries(source=None, domain=None, author=None, time_start=None, time_end=None):
    """列出条目（去重：同 entry_id 只返回一条）"""
    # 构建 where 条件
    where = build_where(source, domain, author, time_start, time_end)

    # 查询所有匹配 chunks
    results = collection.get(where=where, include=["metadatas"])

    # 按 entry_id 去重，只保留每个 entry 的 metadata
    seen = {}
    for meta in results['metadatas']:
        eid = meta['entry_id']
        if eid not in seen:
            seen[eid] = {
                "entry_id": eid,
                "source": meta['source'],
                "domain": meta['domain'],
                "title": meta['title'],
                "author": meta['author'],
                "time_start": meta['time_start']
            }

    return list(seen.values())
```

---

## 六、API 设计

### 条目 CRUD

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /entries | 创建条目（自动分块+embedding） |
| GET | /entries | 列表（支持 source/domain/author/时间 过滤） |
| GET | /entries/{id} | 单条详情（含 chunks） |
| PUT | /entries/{id} | 更新（内容变则重新分块） |
| PATCH | /entries/{id} | 更新 metadata（不重新 embed） |
| DELETE | /entries/{id} | 删除（级联删所有 chunks） |
| POST | /entries/bulk | 批量导入 |

### 搜索

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /search | 统一搜索入口 |

请求体：
```json
{
    "query": "如何调研",
    "mode": "relevance",
    "filters": {
        "source": "之江新语",
        "domain": null,
        "author": null,
        "time_start": null,
        "time_end": null,
        "tags": null
    },
    "top_k": 5
}
```

mode 可选值：`relevance` / `chronological` / `time_filtered`

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /sources | 来源列表及条目数 |
| GET | /domains | 领域列表及条目数 |
| GET | /tags | 标签频率统计 |
| GET | /stats | 系统统计 |
| POST | /reindex | 重建向量库 |

---

## 七、文件结构

```
TLink/
  app/
    __init__.py
    main.py               # FastAPI 入口
    config.py             # 配置（路径、模型、API key）
    models.py             # Pydantic 请求/响应模型
    routers/
      __init__.py
      entries.py          # CRUD 接口
      search.py           # 搜索接口
      system.py           # 系统接口
    services/
      __init__.py
      entry_service.py    # 条目 CRUD 逻辑（操作 ChromaDB）
      search_service.py   # 三种检索逻辑
      chunker.py          # 分块
      embedder.py         # BGE-M3 embedding
      llm_service.py      # LLM 调用
      parsers/
        __init__.py
        base.py           # 解析器基类
        lunyu_parser.py   # 论语
        mao_parser.py     # 毛主席语录
        zhidu_parser.py   # 政治的人生
        zhijiang_parser.py # 之江新语
  migrate.py              # 数据迁移脚本
  chroma_db/              # ChromaDB 数据目录
  requirements.txt        # fastapi, uvicorn, pydantic
```

---

## 八、各书解析策略

| 书名 | 分块单位 | 时间提取 | 标题来源 |
|------|---------|---------|---------|
| 论语 | 一章一句（如 2.15） | 无 | 章节号 |
| 毛主席语录 | 一行一句 | 无 | 前20字 |
| 政治的人生 | 一天一段（如 1月2日） | 正则 → 1994-MM-DD | 日期 |
| 之江新语 | 一篇一文 | 部分有（二〇〇四年二月三日） | 首行标题 |

---

## 九、实施计划

### Phase 1：后端核心 demo

- [ ] 1.1 FastAPI 骨架 + ChromaDB 连接
- [ ] 1.2 重构现有代码为 service 模块（embedder/llm/chunker）
- [ ] 1.3 四个 parser + 迁移脚本 migrate.py
- [ ] 1.4 CRUD 接口（POST/GET/PUT/PATCH/DELETE /entries）
- [ ] 1.5 搜索接口（三种模式）
- [ ] 1.6 端到端测试

### Phase 2：完善后端

- [ ] 2.1 标签管理（LLM 提取 + API）
- [ ] 2.2 批量操作 + 重建索引
- [ ] 2.3 错误处理 + 输入校验
- [ ] 2.4 日志 + 性能监控

### Phase 3：前端界面

- [ ] 3.1 条目列表 + 详情页
- [ ] 3.2 搜索界面（模式选择器）
- [ ] 3.3 条目编辑器（新增/修改/删除）
- [ ] 3.4 时间线可视化
- [ ] 3.5 标签云 + 侧边栏过滤

---

## 十、关键技术风险

| 风险 | 影响 | 对策 |
|------|------|------|
| 更新 entry 内容需重新 embed | 慢 | 低频操作，可接受 |
| 批量更新 metadata | 慢 | ChromaDB update 逐条调用，5000 chunks 内可接受 |
| 列表去重查询 | 效率 | client-side 去重，数据量小没问题 |
| 论语 GBK 编码 | 解析失败 | parser 显式指定 encoding='gbk' |

---

## 十一、验证指标

| 指标 | 目标 |
|------|------|
| CRUD 响应时间 | < 500ms（含 embedding） |
| 搜索响应时间 | < 2s（含 LLM） |
| 相关性 Hit@5 | ≥ 95% |
| 时间线排序准确率 | 人工评估 |
