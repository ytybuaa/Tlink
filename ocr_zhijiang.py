#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 OCR 从扫描版 PDF 提取文本"""

import pdfplumber
import easyocr
import re
from PIL import Image
import io
import numpy as np


def main():
    pdf_path = "之江新语 (习近平) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    output_path = "之江新语.txt"

    print("正在初始化 OCR 引擎...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    print("OCR 引擎就绪")

    print(f"正在处理 {pdf_path}...")
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"共 {total} 页")

        for i, page in enumerate(pdf.pages):
            # 先尝试直接提取文字
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                all_text.append(text.strip())
            else:
                # 转成图片做 OCR
                img = page.to_image(resolution=200)
                img_bytes = img.original.convert('RGB')
                img_array = np.array(img_bytes)

                results = reader.readtext(img_array, detail=0)
                page_text = '\n'.join(results)
                if len(page_text.strip()) > 20:
                    all_text.append(page_text.strip())

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{total} 页")

    # 清洗并保存
    full_text = '\n\n'.join(all_text)
    # 合并被空格分开的中文
    full_text = re.sub(r'([一-鿿])\s+([一-鿿])', r'\1\2', full_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"\n完成! 共处理 {total} 页")
    print(f"总字符数: {len(full_text)}")
    print(f"已保存到 {output_path}")


if __name__ == "__main__":
    main()
