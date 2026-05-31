#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式 RAG 查询

用法:
  问题                    → 自动决定返回条数
  问题 /n                 → 返回 n 条结果（如 "孔子论仁 /10"）
  领域过滤 @论语          → 只搜论语
  领域过滤 @毛主席语录    → 只搜毛主席语录
  领域过滤 @之江新语      → 只搜之江新语
  领域过滤 @政治的人生    → 只搜政治的人生
"""

import os
import re
from rag_verification import (
    read_lunyu, read_mao, chunk_text, build_vector_store, search
)

CHROMA_PATH = "./chroma_db"


def parse_input(user_input: str):
    """解析用户输入，提取问题、返回数量、领域过滤"""
    text = user_input.strip()

    # 提取 /n 返回数量
    top_k = None
    k_match = re.search(r'/(\d+)\s*$', text)
    if k_match:
        top_k = int(k_match.group(1))
        text = text[:k_match.start()].strip()

    # 提取 @领域 过滤
    source_filter = None
    domain_match = re.search(r'@(论语|毛主席语录|之江新语|政治的人生)\s*$', text)
    if domain_match:
        source_filter = domain_match.group(1)
        text = text[:domain_match.start()].strip()

    return text, top_k, source_filter


def main():
    print("=" * 50)
    print("RAG 交互式查询")
    print("=" * 50)

    # 检查是否已有向量库
    if os.path.exists(CHROMA_PATH):
        print("检测到已有向量库，正在加载...")
        import chromadb
        from rag_verification import get_model

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = {
            "论语": client.get_collection("rag_lunyu"),
            "毛主席语录": client.get_collection("rag_mao"),
        }
        try:
            collections["政治的人生"] = client.get_collection("rag_zhidu")
        except Exception:
            pass
        try:
            collections["之江新语"] = client.get_collection("rag_zhijiang")
        except Exception:
            pass
        get_model()
        print(f"加载完成（{len(collections)} 个语料库）\n")
    else:
        print("首次运行，构建向量库...")
        from rag_verification import main as build_main
        build_main()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = {
            "论语": client.get_collection("rag_lunyu"),
            "毛主席语录": client.get_collection("rag_mao"),
        }
        try:
            collections["政治的人生"] = client.get_collection("rag_zhidu")
        except Exception:
            pass
        try:
            collections["之江新语"] = client.get_collection("rag_zhijiang")
        except Exception:
            pass
        print("构建完成\n")

    print("用法: 问题 /数量 @领域")
    print("示例: 孔子论仁 /5 @论语")
    print("      习近平谈创新 /3 @之江新语")
    print("      治国思想 /10  （跨所有语料）")
    print()

    while True:
        try:
            user_input = input("问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() == 'q':
            break

        query, top_k, source_filter = parse_input(user_input)
        if not query:
            continue

        print()
        if source_filter:
            print(f"  领域: {source_filter}")
        if top_k:
            print(f"  返回: {top_k} 条")

        results = search(query, collections, source_filter=source_filter,
                         top_k=top_k, use_llm=True)

        print(f"\n返回 {len(results)} 条结果:")
        print("-" * 50)
        for i, r in enumerate(results):
            print(f"{i+1}. [{r['source']}] {r['text']}")
            print()

    print("再见!")


if __name__ == "__main__":
    main()
