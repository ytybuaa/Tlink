#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联想式检索评估 - 评估检索结果的多样性、覆盖度、互补性
"""

import json
import time
import requests
from typing import List, Dict, Set


def load_results(file_path: str) -> Dict:
    """加载检索结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_keywords(text: str) -> Set[str]:
    """提取关键词（简单分词）"""
    # 移除标点和数字
    import re
    text = re.sub(r'[^一-龥]', '', text)
    # 简单按2-4字切分
    keywords = set()
    for length in [2, 3, 4]:
        for i in range(len(text) - length + 1):
            word = text[i:i + length]
            keywords.add(word)
    return keywords


def compute_coverage(results: List[Dict], topic_keywords: Set[str]) -> float:
    """计算主题覆盖度"""
    all_text = ' '.join([r['text'] for r in results])
    result_keywords = extract_keywords(all_text)

    if not topic_keywords:
        return 0.0

    covered = len(topic_keywords & result_keywords)
    return covered / len(topic_keywords)


def compute_diversity(results: List[Dict]) -> float:
    """计算信息多样性（结果之间关键词的差异度）"""
    if len(results) < 2:
        return 0.0

    # 提取每个结果的关键词
    all_keywords = []
    for r in results:
        keywords = extract_keywords(r['text'])
        all_keywords.append(keywords)

    # 计算两两之间的Jaccard距离
    total_distance = 0
    count = 0
    for i in range(len(all_keywords)):
        for j in range(i + 1, len(all_keywords)):
            intersection = len(all_keywords[i] & all_keywords[j])
            union = len(all_keywords[i] | all_keywords[j])
            if union > 0:
                jaccard_sim = intersection / union
                total_distance += (1 - jaccard_sim)
                count += 1

    return total_distance / count if count > 0 else 0.0


def compute_complementarity(results: List[Dict]) -> float:
    """计算互补性（结果是否提供不同角度的信息）"""
    if len(results) < 2:
        return 0.0

    # 提取每个结果的核心短语（前20字）
    phrases = []
    for r in results:
        text = r['text'].replace('——毛泽东', '').replace('——毛主席语录', '').strip()
        phrases.append(text[:30])

    # 检查短语之间的差异
    unique_phrases = set(phrases)
    return len(unique_phrases) / len(phrases)


DEEPSEEK_API_KEY = "tp-c3zwnoj1nx64wloqc42hbacmvcjdp93e64lufttkkqf7qkat"
DEEPSEEK_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


def llm_analyze_aspects(results: List[Dict], topic: str) -> Dict[str, List[str]]:
    """用 LLM 动态分析检索结果覆盖了哪些维度"""
    # 构造结果文本
    result_texts = "\n".join(
        f"[{i+1}] {r['text'][:150]}" for i, r in enumerate(results)
    )

    prompt = f"""你是一个文本分析专家。以下是针对问题「{topic}」的检索结果，请分析这些结果分别从哪些角度/维度回答了这个问题。

检索结果：
{result_texts}

要求：
1. 根据这些结果的实际内容，动态归纳出覆盖了哪些维度（不要用预设分类）
2. 每个维度给出简短名称（2-4字），并将结果编号归入对应维度
3. 一个结果可以属于多个维度

请严格按以下 JSON 格式返回（不要返回其他内容）：
{{
  "维度名称": [1, 3],
  "另一个维度": [2, 4, 5]
}}"""

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            },
            json={
                "model": "mimo-v2.5",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
                "thinking": {"type": "disabled"}
            },
            timeout=120
        )

        if resp.status_code != 200:
            print(f"    [API错误] status={resp.status_code}, body={resp.text[:200]}")

        if resp.status_code == 200:
            data = resp.json()
            print(f"    [API响应] {json.dumps(data, ensure_ascii=False)[:500]}")
            content = data["choices"][0]["message"]["content"]
            print(f"    [LLM原始输出] [{content[:200]}]")
            # 提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                raw = json.loads(json_match.group())
                # 将编号转回文本
                aspects = {}
                for dim, indices in raw.items():
                    aspects[dim] = [
                        results[i-1]['text'][:80]
                        for i in indices
                        if 1 <= i <= len(results)
                    ]
                return aspects
    except Exception as e:
        print(f"  LLM 分析失败: {e}")

    # fallback: 返回空
    return {"未分类": [r['text'][:80] for r in results]}


def compute_aspects(results: List[Dict], topic: str) -> Dict[str, List[str]]:
    """分析结果覆盖了哪些方面（LLM 动态分析）"""
    return llm_analyze_aspects(results, topic)


def evaluate联想检索(results_data: Dict) -> Dict:
    """评估联想式检索"""

    # 为每个测试问题定义主题关键词
    topic_keywords = {
        "孔子对'仁'的定义是什么？": {"仁", "爱人", "克己", "复礼", "恭宽信敏惠"},
        "孔子如何看待学习与思考的关系？": {"学", "思", "罔", "殆", "习"},
        "孔子如何看待君子和小人的区别？": {"君子", "小人", "义", "利", "和", "同"},
        "毛主席如何描述革命？": {"革命", "阶级", "暴动", "敌人", "朋友"},
        "毛主席对战争有什么看法？": {"战争", "打", "敌人", "军队", "战略", "战术"},
        "古代和现代领袖对治国的思考有什么共同点？": {"政", "治", "国", "民", "德"},
        "习近平如何看待调研工作？": {"调研", "深", "实", "细", "准", "效"},
        "之江新语中如何论述务实作风？": {"务实", "实", "作风", "求真"},
        "习近平对领导干部提出了什么要求？": {"干部", "领导", "要求", "作风"},
    }

    eval_results = []

    for item in results_data['results']:
        question = item['question']
        results = item['results']

        # 跳过没有预定义关键词的问题
        if question not in topic_keywords:
            continue

        # 计算各项指标
        coverage = compute_coverage(results, topic_keywords[question])
        diversity = compute_diversity(results)
        complementarity = compute_complementarity(results)
        aspects = compute_aspects(results, question)
        time.sleep(0.5)  # 避免 API 限流

        eval_results.append({
            'question': question,
            'coverage': coverage,
            'diversity': diversity,
            'complementarity': complementarity,
            'aspects': {k: len(v) for k, v in aspects.items()},
            'num_dimensions': len(aspects),
            'top_results': [r['text'][:60] for r in results[:3]]
        })

    # 计算平均指标
    avg_metrics = {
        'avg_coverage': sum(e['coverage'] for e in eval_results) / len(eval_results),
        'avg_diversity': sum(e['diversity'] for e in eval_results) / len(eval_results),
        'avg_complementarity': sum(e['complementarity'] for e in eval_results) / len(eval_results),
        'avg_dimensions': sum(e['num_dimensions'] for e in eval_results) / len(eval_results),
    }

    return {
        'detailed_results': eval_results,
        'summary': avg_metrics
    }


def print_report(eval_output: Dict):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("联想式检索评估报告")
    print("=" * 60)

    summary = eval_output['summary']
    print(f"\n【总体指标】")
    print(f"  主题覆盖度: {summary['avg_coverage']:.2%}")
    print(f"  信息多样性: {summary['avg_diversity']:.2%}")
    print(f"  互补性: {summary['avg_complementarity']:.2%}")
    print(f"  平均维度数: {summary['avg_dimensions']:.1f}")

    print(f"\n【详细分析】")
    for item in eval_output['detailed_results']:
        print(f"\n问题: {item['question']}")
        print(f"  覆盖度: {item['coverage']:.2%}")
        print(f"  多样性: {item['diversity']:.2%}")
        print(f"  互补性: {item['complementarity']:.2%}")
        print(f"  维度覆盖({item['num_dimensions']}个): {item['aspects']}")
        print(f"  Top-3 结果:")
        for i, text in enumerate(item['top_results']):
            print(f"    {i+1}. {text}...")

    print("\n" + "=" * 60)


def main():
    # 加载结果
    results_data = load_results('rag_results.json')

    # 评估
    eval_output = evaluate联想检索(results_data)

    # 打印报告
    print_report(eval_output)

    # 保存评估结果
    with open('rag_diversity_eval.json', 'w', encoding='utf-8') as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)

    print("\n评估结果已保存到 rag_diversity_eval.json")


if __name__ == "__main__":
    main()
