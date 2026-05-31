#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从之江新语PDF提取文本"""

import pdfplumber
import re


def clean_text(text: str) -> str:
    text = re.sub(r'第\s*\d+\s*页', '', text)
    text = re.sub(r'([一-鿿])\s+([一-鿿])', r'\1\2', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def main():
    pdf_path = "之江新语 (习近平) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    output_path = "之江新语.txt"

    print(f"正在提取 {pdf_path}...")
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"共 {total} 页")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 10:
                cleaned = clean_text(text)
                if cleaned:
                    all_text.append(cleaned)

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{total} 页")

    full_text = '\n\n'.join(all_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"\n完成! 共提取 {len(all_text)} 页有效文本")
    print(f"总字符数: {len(full_text)}")
    print(f"已保存到 {output_path}")


if __name__ == "__main__":
    main()
