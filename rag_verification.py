#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG验证脚本 - 认知管理系统核心技术验证
使用 BGE-M3 embedding 模型进行语义检索验证
"""

import os
import re
import json
import time
import hashlib
import requests
import numpy as np
import chromadb
from typing import List, Dict
import chardet
from sentence_transformers import SentenceTransformer

# LLM 配置（小米 MiMo）
LLM_API_KEY = "tp-c3zwnoj1nx64wloqc42hbacmvcjdp93e64lufttkkqf7qkat"
LLM_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
LLM_MODEL = "mimo-v2.5"

# 分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 向量库路径
CHROMA_PATH = "./chroma_db"

# BGE-M3 模型配置
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def read_lunyu(file_path: str) -> str:
    """读取论语.txt（GBK编码）"""
    with open(file_path, 'rb') as f:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding']
        print(f"论语.txt 编码: {encoding}")

    with open(file_path, 'r', encoding='gbk') as f:
        return f.read()


def read_mao(file_path: str) -> str:
    """读取毛主席语录.txt"""
    with open(file_path, 'rb') as f:
        raw = f.read()
        result = chardet.detect(raw)
        encoding = result['encoding'] or 'utf-8'
        print(f"毛主席语录.txt 编码: {encoding}")

    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()


# 全局模型实例
_model = None

def get_model() -> SentenceTransformer:
    """获取 BGE-M3 模型（懒加载）"""
    global _model
    if _model is None:
        print(f"正在加载 {EMBEDDING_MODEL_NAME} 模型...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("模型加载完成")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """获取单条文本的向量"""
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def batch_get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """批量获取文本向量"""
    model = get_model()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, batch_size=batch_size)
        all_embeddings.extend(embeddings)
        print(f"  已完成 {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)


def read_zhijiang(file_path: str) -> str:
    """读取之江新语.txt（UTF-8编码）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_zhidu(file_path: str) -> str:
    """读取政治的人生.txt（UTF-8编码）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_epub(file_path: str) -> str:
    """读取 epub 文件中的文本内容"""
    book = epub.read_epub(file_path)
    texts = []

    for item in book.get_items():
        if item.get_type() == 9:  # ITEM_DOCUMENT
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            if text.strip() and len(text.strip()) > 50:  # 过滤太短的文本
                texts.append(text.strip())

    content = '\n\n'.join(texts)
    print(f"epub 文本总长度: {len(content)} 字符")

    if len(content) < 100:
        print("警告: epub 文件似乎是扫描版（图片），无法提取文本")
        return ""

    return content


def chunk_text(text: str, source_name: str) -> List[Dict]:
    """将文本切分为 chunks"""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP  # 每次前进的步长

    # 按行分割，每行作为一个候选 chunk
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    chunk_id = 0
    for line in lines:
        # 过滤纯标点/空白
        if len(line.replace('—', '').replace(' ', '').replace('。', '').replace('，', '')) < 3:
            continue

        if len(line) <= CHUNK_SIZE:
            chunks.append({
                'id': f"{source_name}_{chunk_id:04d}",
                'text': line,
                'source': source_name
            })
            chunk_id += 1
        else:
            # 长段落按固定步长切分
            for start in range(0, len(line), step):
                end = min(start + CHUNK_SIZE, len(line))
                chunk = line[start:end]
                chunks.append({
                    'id': f"{source_name}_{chunk_id:04d}",
                    'text': chunk,
                    'source': source_name
                })
                chunk_id += 1

    return chunks


def llm_extract_metadata_batch(chunks: List[Dict], batch_size: int = 20) -> List[Dict]:
    """批量用 LLM 提取元数据（人物、主题、关键词）"""
    enriched = []

    for i in range(0, len(chunks), batch_size):
        if i > 0:
            time.sleep(2)  # 批次间延迟，避免限流
        batch = chunks[i:i + batch_size]
        numbered_texts = "\n".join(
            f"[{j+1}] {c['text'][:150]}" for j, c in enumerate(batch)
        )

        prompt = f"""分析以下文本，为每条提取元数据。这些文本可能很短，你需要根据内容推断隐含信息。

{numbered_texts}

对每条文本，提取：
- people: 提到的或暗指的人物（如"文小姐"暗指丁玲）
- topic: 主题关键词（2-4字）
- keywords: 语义关键词，包括同义词、关联概念、背景信息。短文本要多提取，确保检索时能命中。

严格按以下 JSON 数组格式返回（不要其他内容）：
[
  {{"people": ["人名"], "topic": "主题", "keywords": ["关键词1", "关键词2"]}},
  {{"people": [], "topic": "主题", "keywords": []}}
]"""

        content = _llm_call(prompt, max_tokens=1500)
        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                metadata_list = json.loads(json_match.group())
                for j, c in enumerate(batch):
                    meta = metadata_list[j] if j < len(metadata_list) else {}
                    enriched.append({
                        **c,
                        'people': meta.get('people', []),
                        'topic': meta.get('topic', ''),
                        'keywords': meta.get('keywords', [])
                    })
                print(f"    元数据提取: {min(i + batch_size, len(chunks))}/{len(chunks)}")
                continue
        except Exception as e:
            print(f"    元数据解析失败: {e}")

        # fallback: 无元数据
        for c in batch:
            enriched.append({**c, 'people': [], 'topic': '', 'keywords': []})

    return enriched


SHORT_THRESHOLD = 100  # 短文本阈值（字符数）


def enrich_chunks(chunks: List[Dict]) -> List[Dict]:
    """只对短文本做 LLM 元数据增强，长文本直接跳过"""
    short_chunks = [c for c in chunks if len(c['text']) < SHORT_THRESHOLD]
    long_chunks = [c for c in chunks if len(c['text']) >= SHORT_THRESHOLD]

    print(f"  短文本（<{SHORT_THRESHOLD}字）: {len(short_chunks)} 条，需 LLM 增强")
    print(f"  长文本（>={SHORT_THRESHOLD}字）: {len(long_chunks)} 条，跳过")

    # 只对短文本调 LLM
    enriched_short = llm_extract_metadata_batch(short_chunks)

    # 合并，长文本不带元数据
    enriched = []
    short_idx = 0
    for c in chunks:
        if len(c['text']) < SHORT_THRESHOLD:
            enriched.append(enriched_short[short_idx])
            short_idx += 1
        else:
            enriched.append({**c, 'people': [], 'topic': '', 'keywords': []})

    # 生成增强文本（只有短文本会有元数据前缀）
    for c in enriched:
        parts = []
        if c.get('people'):
            parts.append(f"人物:{','.join(c['people'])}")
        if c.get('topic'):
            parts.append(f"主题:{c['topic']}")
        if c.get('keywords'):
            parts.append(f"关键词:{','.join(c['keywords'])}")

        meta_str = " ".join(parts)
        c['original_text'] = c['text']
        c['text'] = f"{meta_str} {c['text']}" if meta_str else c['text']

    return enriched


SOURCE_NAME_MAP = {
    "论语": "lunyu",
    "毛主席语录": "mao",
    "政治的人生": "zhidu",
    "之江新语": "zhijiang"
}


def get_file_hash(file_path: str) -> str:
    """计算文件内容的 MD5 哈希"""
    import hashlib
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def build_vector_store(chunks: List[Dict], force_rebuild: bool = False,
                       only_sources: list = None) -> Dict[str, chromadb.Collection]:
    """为每个语料源建立独立的向量库（支持增量更新）

    Args:
        chunks: 所有 chunks
        force_rebuild: 是否强制重建所有
        only_sources: 只重建指定来源（如 ["之江新语"]），None 表示检查所有
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 按来源分组
    sources = {}
    for chunk in chunks:
        src = chunk['source']
        if src not in sources:
            sources[src] = []
        sources[src].append(chunk)

    # 检查哪些来源需要重建
    hash_file = os.path.join(CHROMA_PATH, "source_hashes.json")
    if os.path.exists(hash_file) and not force_rebuild:
        with open(hash_file, 'r') as f:
            saved_hashes = json.load(f)
    else:
        saved_hashes = {}

    # 计算当前各来源的文本哈希
    current_hashes = {}
    for src, src_chunks in sources.items():
        content = "".join(c['text'] for c in src_chunks)
        current_hashes[src] = hashlib.md5(content.encode()).hexdigest()

    # 确定需要重建的来源
    rebuild_sources = set()
    for src in sources:
        if only_sources and src not in only_sources:
            continue
        if force_rebuild:
            rebuild_sources.add(src)
        elif src not in saved_hashes:
            rebuild_sources.add(src)
        elif saved_hashes[src] != current_hashes[src]:
            rebuild_sources.add(src)

    if rebuild_sources:
        print(f"  需要重建: {', '.join(rebuild_sources)}")
    else:
        print(f"  所有来源数据未变，跳过重建")

    # 只对需要重建的来源做 LLM 增强
    chunks_to_enrich = [c for c in chunks if c['source'] in rebuild_sources]
    if chunks_to_enrich:
        enriched = enrich_chunks(chunks_to_enrich)
        enriched_map = {c['id']: c for c in enriched}
    else:
        enriched_map = {}

    collections = {}
    for src, src_chunks in sources.items():
        safe_name = SOURCE_NAME_MAP.get(src, src)
        collection = client.get_or_create_collection(
            name=f"rag_{safe_name}",
            metadata={"hnsw:space": "cosine"}
        )

        if src in rebuild_sources:
            # 清空并重建
            if collection.count() > 0:
                existing = collection.get()
                if existing['ids']:
                    collection.delete(ids=existing['ids'])

            # 用增强后的文本生成 embedding
            enriched_chunks = [enriched_map.get(c['id'], c) for c in src_chunks]
            texts = [c['text'] for c in enriched_chunks]

            print(f"  正在为 {src} 生成 embeddings...")
            embeddings = batch_get_embeddings(texts)

            batch_size = 100
            for i in range(0, len(src_chunks), batch_size):
                batch = src_chunks[i:i + batch_size]
                batch_enriched = enriched_chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size].tolist()
                collection.add(
                    ids=[c['id'] for c in batch],
                    documents=[c['text'] for c in batch_enriched],
                    embeddings=batch_embeddings,
                    metadatas=[{
                        "source": c['source'],
                        "original": c.get('original_text', c['text'])
                    } for c in batch]
                )
            print(f"  {src}: {len(src_chunks)} chunks 已重建")
        else:
            print(f"  {src}: 未变化，跳过")

        collections[src] = collection

    # 保存哈希
    all_hashes = {**saved_hashes, **current_hashes}
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(hash_file, 'w') as f:
        json.dump(all_hashes, f)

    return collections


def _llm_call(prompt: str, max_tokens: int = 1000) -> str:
    """调用 LLM（小米 MiMo，关闭思考模式，含重试）"""
    for attempt in range(5):
        try:
            resp = requests.post(
                LLM_BASE_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {LLM_API_KEY}"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"}
                },
                timeout=60
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                time.sleep(1)  # 请求间隔 1 秒，避免限流
                return content
            if resp.status_code == 429:
                wait = (attempt + 1) * 5  # 指数退避: 5, 10, 15, 20, 25 秒
                print(f"  [429限流] 等待 {wait} 秒后重试 {attempt + 1}/5...")
                time.sleep(wait)
                continue
            print(f"  [LLM错误] {resp.status_code}: {resp.text[:200]}")
            return ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ProxyError) as e:
            wait = (attempt + 1) * 3
            if attempt < 4:
                print(f"  [网络错误] {type(e).__name__}，等待 {wait} 秒后重试 {attempt + 1}/5...")
                time.sleep(wait)
            else:
                print(f"  [网络错误] 已重试5次，放弃")
                return ""
    return ""


def llm_understand_query(query: str) -> Dict:
    """Step 1: LLM 理解意图，改写/拆解问题，提取领域过滤"""
    prompt = f"""你是一个检索助手。用户提了一个问题，你需要：

1. 判断这个问题属于哪个领域："论语"、"毛主席语录"、"政治的人生"、"之江新语"、或"跨领域"
2. 把问题改写成 1-3 个更适合向量检索的短句（去掉疑问词，保留核心概念）

领域判断规则：
- 只有明确提到某本书/某位作者时才指定对应领域（如"孔子"→论语，"毛主席"→毛主席语录，"习近平"→之江新语）
- 问抽象概念（如"德政"、"治国"、"仁义"、"学习"）→ 跨领域
- 问古今对比、异同、共同点 → 跨领域
- 不确定时 → 跨领域

用户问题：{query}

请严格按以下 JSON 格式返回（不要返回其他内容）：
{{
  "domain": "论语" 或 "毛主席语录" 或 "政治的人生" 或 "之江新语" 或 "跨领域",
  "queries": ["核心概念1", "核心概念2"]
}}"""

    content = _llm_call(prompt, max_tokens=200)
    try:
        json_match = re.search(r'\{[\s\S]*?\}', content)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except Exception as e:
        print(f"  意图解析失败: {e}")

    return {"domain": None, "queries": [query]}


def llm_rerank(query: str, results: List[Dict], top_n: int = 5) -> List[Dict]:
    """Step 3: LLM 精排（含事实约束），按相关性排序并剔除不直接相关的结果"""
    if len(results) <= 1:
        return results

    result_texts = "\n".join(
        f"[{i+1}] {r['text'][:200]}" for i, r in enumerate(results)
    )

    prompt = f"""你是一个检索结果排序专家。请严格评估每条结果是否直接回答问题。

问题：{query}

检索结果：
{result_texts}

严格评估规则（违反任何一条即 direct=false）：
1. 结果的核心主题必须与问题的核心主题完全一致，不能只是"沾边"
2. 共享部分词汇但具体对象不同 → direct=false
   - 例：问"核武器"，"核潜艇"是装备不是武器 → false
   - 例：问"战争"，"打仗"是战争但"打麻将"不是 → false
3. 表达了类似情感/态度但领域不同 → direct=false
4. 只有字面部分重叠，实际讨论不同事物 → direct=false

请对每条结果返回 JSON 数组：
[
  {{"id": 1, "score": 9, "direct": true, "reason": "原子弹是核武器"}},
  {{"id": 5, "score": 3, "direct": false, "reason": "核潜艇是装备不是核武器"}}
]
- score: 1-10 分
- direct: 是否直接回答问题（必须严格判断）
- reason: 一句话说明理由
- 最终只返回 direct=true 的结果，按 score 降序
- 如果没有 direct=true 的，返回空数组 []
只返回 JSON 数组，不要返回其他内容。"""

    content = _llm_call(prompt, max_tokens=300)
    try:
        json_match = re.search(r'\[[\s\S]*?\]', content)
        if json_match:
            scored = json.loads(json_match.group())
            # 只保留 direct=true 的结果
            direct_items = [item for item in scored if item.get('direct', False)]
            # 按 score 降序
            direct_items.sort(key=lambda x: x.get('score', 0), reverse=True)

            reranked = []
            for item in direct_items:
                idx = item['id']
                if 1 <= idx <= len(results):
                    reranked.append(results[idx - 1])

            if reranked:
                return reranked[:top_n]
    except Exception as e:
        print(f"  精排解析失败: {e}")

    return results[:top_n]


def assess_query_complexity(query: str) -> Dict[str, any]:
    """评估问题复杂度，返回复杂度分数和建议返回数量"""
    # 1. 问题长度
    length = len(query)

    # 2. 问题类型关键词
    definition_kw = ['是什么', '定义', '含义', '意思']  # 定义型问题 → 返回少量精确结果
    comparison_kw = ['区别', '差异', '比较', '异同', '共同']  # 比较型问题 → 返回多角度结果
    relationship_kw = ['关系', '如何', '怎样', '为什么']  # 关系型问题 → 需要多条互补信息
    open_kw = ['看法', '观点', '思考', '理念']  # 开放型问题 → 返回多角度结果

    type_score = 0
    query_type = "definition"

    if any(kw in query for kw in comparison_kw):
        type_score = 3
        query_type = "comparison"
    elif any(kw in query for kw in open_kw):
        type_score = 2
        query_type = "open"
    elif any(kw in query for kw in relationship_kw):
        type_score = 2
        query_type = "relationship"
    elif any(kw in query for kw in definition_kw):
        type_score = 1
        query_type = "definition"

    # 3. 概念密度（问号、顿号等分隔符数量）
    concept_density = query.count('？') + query.count('、') + query.count('和')

    # 综合复杂度分数 (1-5)
    complexity_score = min(5, max(1,
        (length / 20) +  # 长度因子
        type_score +      # 类型因子
        concept_density   # 概念密度
    ))

    # 根据复杂度建议返回数量
    if complexity_score <= 1.5:
        suggested_k = 3  # 简单定义题
    elif complexity_score <= 2.5:
        suggested_k = 5  # 一般问题
    elif complexity_score <= 3.5:
        suggested_k = 7  # 复杂关系题
    else:
        suggested_k = 10  # 开放式探索题

    return {
        'complexity_score': round(complexity_score, 2),
        'query_type': query_type,
        'suggested_k': suggested_k,
        'factors': {
            'length': length,
            'type_score': type_score,
            'concept_density': concept_density
        }
    }


def search(query: str, collections: Dict[str, chromadb.Collection],
           source_filter: str = None, top_k: int = None,
           use_llm: bool = True) -> List[Dict]:
    """语义检索（支持 LLM 意图理解 + 精排）"""
    # 动态确定返回数量
    if top_k is None:
        complexity = assess_query_complexity(query)
        top_k = complexity['suggested_k']
        print(f"  复杂度评估: {complexity['complexity_score']} ({complexity['query_type']}) → 返回 {top_k} 条")

    # Step 1: LLM 理解意图
    domain = source_filter
    queries = [query]
    if use_llm:
        intent = llm_understand_query(query)
        domain = intent.get('domain') or source_filter
        queries = intent.get('queries', [query])
        if domain and domain not in ("论语", "毛主席语录", "政治的人生", "之江新语"):
            domain = None  # 跨领域不做过滤
        print(f"  意图识别: domain={domain}, queries={queries}")

    # 确定要搜索的 collections
    if domain and domain in collections:
        search_collections = {domain: collections[domain]}
    else:
        search_collections = collections

    # Step 2: 对每个改写后的 query 做向量检索
    all_results = []
    for q in queries:
        query_embedding = get_embedding(q).tolist()
        for _, collection in search_collections.items():
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                all_results.append({
                    'id': results['ids'][0][i],
                    'text': meta.get('original', results['documents'][0][i]),
                    'source': meta['source'],
                    'distance': results['distances'][0][i]
                })

    # 去重（按 id）
    seen = set()
    deduped = []
    for r in all_results:
        if r['id'] not in seen:
            seen.add(r['id'])
            deduped.append(r)
    deduped.sort(key=lambda x: x['distance'])

    # 取 top-k * 2 给精排留余量
    candidates = deduped[:top_k * 2]

    # Step 3: LLM 精排
    if use_llm and len(candidates) > 1:
        print(f"  精排: {len(candidates)} 条候选 → {top_k} 条")
        return llm_rerank(query, candidates, top_n=top_k)

    return candidates[:top_k]


def evaluate_results(results_data: List[Dict]) -> Dict:
    """评估检索结果"""
    # 预期答案关键词映射（扩展版，允许多个可能的关键词）
    expected_answers = {
        "孔子如何看待学习与思考的关系？": ["学", "思", "习"],
        "孔子对'仁'的定义是什么？": ["仁"],
        "孔子如何看待君子和小人的区别？": ["君子", "小人"],
        "孔子对为政有什么看法？": ["政", "为政", "治"],
        "孔子如何看待礼和仁的关系？": ["礼", "仁"],
        "孔子关于治国的理念是什么？": ["治", "政", "国"],
        "孔子如何定义君子？": ["君子"],
        "毛主席对战争有什么看法？": ["战争", "打", "敌人", "军队"],
        "毛主席如何描述革命？": ["革命", "阶级", "斗争"],
        "毛主席对青年有什么期望？": ["青年", "朝气", "太阳"],
        "古代和现代领袖对治国的思考有什么共同点？": ["治", "政", "国", "民"],
        "习近平如何看待调研工作？": ["调研", "深", "实", "细"],
        "之江新语中如何论述务实作风？": ["务实", "实", "作风"],
        "习近平对领导干部提出了什么要求？": ["干部", "领导", "要求"],
    }

    hit3_count = 0
    hit5_count = 0
    mrr_sum = 0
    total = 0

    for item in results_data:
        question = item['question']
        results = item['results']

        if question not in expected_answers:
            continue

        total += 1
        keywords = expected_answers[question]
        found = False
        rank = -1

        for i, r in enumerate(results):
            text = r['text']
            # 检查是否包含任意关键词
            if any(kw in text for kw in keywords):
                if not found:
                    rank = i + 1
                    found = True

        if found:
            if rank <= 3:
                hit3_count += 1
            if rank <= 5:
                hit5_count += 1
            mrr_sum += 1.0 / rank

    metrics = {
        'total_questions': total,
        'hit@3': hit3_count / total if total > 0 else 0,
        'hit@5': hit5_count / total if total > 0 else 0,
        'mrr': mrr_sum / total if total > 0 else 0,
    }

    return metrics


def main():
    print("=" * 60)
    print("RAG验证脚本 - 认知管理系统核心技术验证")
    print("=" * 60)

    # Step 1: 读取语料
    print("\n[Step 1] 读取语料文件...")

    lunyu_content = read_lunyu("论语.txt")
    print(f"论语长度: {len(lunyu_content)} 字符")

    mao_content = read_mao("毛主席语录.txt")
    print(f"毛主席语录长度: {len(mao_content)} 字符")

    zhidu_content = ""
    zhidu_path = "政治的人生.txt"
    if os.path.exists(zhidu_path):
        zhidu_content = read_zhidu(zhidu_path)
        print(f"政治的人生长度: {len(zhidu_content)} 字符")

    zhijiang_content = ""
    zhijiang_path = "之江新语.txt"
    if os.path.exists(zhijiang_path):
        zhijiang_content = read_zhijiang(zhijiang_path)
        print(f"之江新语长度: {len(zhijiang_content)} 字符")

    # Step 2: 分块
    print("\n[Step 2] 分块处理...")
    all_chunks = []
    all_chunks.extend(chunk_text(lunyu_content, "论语"))
    all_chunks.extend(chunk_text(mao_content, "毛主席语录"))
    if zhidu_content:
        all_chunks.extend(chunk_text(zhidu_content, "政治的人生"))
    if zhijiang_content:
        all_chunks.extend(chunk_text(zhijiang_content, "之江新语"))
    print(f"总分块数量: {len(all_chunks)}")
    print(f"  论语: {len([c for c in all_chunks if c['source'] == '论语'])} chunks")
    print(f"  毛主席语录: {len([c for c in all_chunks if c['source'] == '毛主席语录'])} chunks")
    if zhidu_content:
        print(f"  政治的人生: {len([c for c in all_chunks if c['source'] == '政治的人生'])} chunks")
    if zhijiang_content:
        print(f"  之江新语: {len([c for c in all_chunks if c['source'] == '之江新语'])} chunks")

    # Step 3: 构建向量库
    print("\n[Step 3] 构建向量库（使用 chromadb 内置 embedding）...")
    collections = build_vector_store(all_chunks)

    # Step 4: 检索测试
    print("\n[Step 4] 检索测试...")

    # 测试问题（带领域过滤）
    test_questions = [
        # 论语方向 - 只搜论语
        {"q": "孔子如何看待学习与思考的关系？", "filter": "论语"},
        {"q": "孔子对'仁'的定义是什么？", "filter": "论语"},
        {"q": "孔子如何看待君子和小人的区别？", "filter": "论语"},
        {"q": "孔子对为政有什么看法？", "filter": "论语"},
        {"q": "孔子如何看待礼和仁的关系？", "filter": "论语"},
        {"q": "孔子关于治国的理念是什么？", "filter": "论语"},
        {"q": "孔子如何定义君子？", "filter": "论语"},
        # 毛主席语录方向 - 只搜毛主席语录
        {"q": "毛主席对战争有什么看法？", "filter": "毛主席语录"},
        {"q": "毛主席如何描述革命？", "filter": "毛主席语录"},
        {"q": "毛主席对青年有什么期望？", "filter": "毛主席语录"},
        # 之江新语方向 - 只搜之江新语
        {"q": "习近平如何看待调研工作？", "filter": "之江新语"},
        {"q": "之江新语中如何论述务实作风？", "filter": "之江新语"},
        {"q": "习近平对领导干部提出了什么要求？", "filter": "之江新语"},
        # 跨语料方向 - 搜所有
        {"q": "古代和现代领袖对治国的思考有什么共同点？", "filter": None},
    ]

    results_data = []

    for i, item in enumerate(test_questions):
        question = item['q']
        source_filter = item['filter']

        if i > 0:
            time.sleep(1)  # 避免 API 限流

        print(f"\n问题 {i + 1}/{len(test_questions)}: {question}")
        if source_filter:
            print(f"  领域过滤: {source_filter}")

        results = search(question, collections, source_filter=source_filter, top_k=5)

        print("Top-5 结果:")
        for j, r in enumerate(results):
            print(f"  {j + 1}. [距离: {r['distance']:.4f}] [{r['source']}] {r['text'][:60]}...")

        results_data.append({
            'question': question,
            'source_filter': source_filter,
            'results': results
        })

    # Step 5: 评估
    print("\n[Step 5] 评估结果...")

    metrics = evaluate_results(results_data)

    print("\n" + "=" * 60)
    print("评估指标:")
    print(f"  问题总数: {metrics['total_questions']}")
    print(f"  Hit@3: {metrics['hit@3']:.2%}")
    print(f"  Hit@5: {metrics['hit@5']:.2%}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print("=" * 60)

    # 保存结果
    output = {
        'metrics': metrics,
        'results': results_data
    }

    with open('rag_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n结果已保存到 rag_results.json")

    # 评估结论
    print("\n" + "=" * 60)
    if metrics['hit@5'] >= 0.8 and metrics['mrr'] >= 0.6:
        print("验证通过! 可以继续推进产品开发")
    elif metrics['hit@5'] >= 0.6:
        print("验证部分通过，需要调整分块策略或 embedding 模型")
    else:
        print("验证失败，需要重新评估技术路线")
    print("=" * 60)


if __name__ == "__main__":
    main()
