#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cost_calculator.py — OpenAI Embedding 费用计算器
-----------------------------------------
计算 JSONL 文件被 OpenAI Embedding 模型处理后的费用。

用法示例：
    # 计算默认模型费用
    python cost_calculator.py --input chunks.jsonl
    
    # 指定模型
    python cost_calculator.py --input chunks.jsonl --model text-embedding-ada-002
    
    # 显示详细信息
    python cost_calculator.py --input chunks.jsonl --verbose
"""

import json
import argparse
from typing import Dict, List
import os
import sys

# --- Tokenizer ---
try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    print("请先安装 tiktoken： pip install tiktoken")
    sys.exit(1)

# OpenAI Embedding 模型定价（美元/1k tokens）
EMBEDDING_PRICING = {
    "text-embedding-ada-002": 0.0001,  # $0.0001 per 1k tokens
    "text-embedding-3-small": 0.00002,  # $0.00002 per 1k tokens  
    "text-embedding-3-large": 0.00013,  # $0.00013 per 1k tokens
}

# 模型别名
MODEL_ALIASES = {
    "ada-002": "text-embedding-ada-002",
    "ada": "text-embedding-ada-002",
    "3-small": "text-embedding-3-small",
    "small": "text-embedding-3-small",
    "3-large": "text-embedding-3-large", 
    "large": "text-embedding-3-large",
}

def count_tokens(text: str) -> int:
    """计算文本的 token 数量"""
    try:
        return len(ENC.encode(text))
    except Exception:
        # 粗略估算：英文约4字符=1token，中文约2字符=1token
        return len(text) // 3

def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行 JSON 解析错误: {e}")
                    continue
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取文件时发生异常: {e}")
        sys.exit(1)
    
    return chunks

def calculate_cost(chunks: List[Dict], model: str, verbose: bool = False) -> Dict:
    """计算 embedding 费用"""
    
    # 规范化模型名称
    model = MODEL_ALIASES.get(model, model)
    
    if model not in EMBEDDING_PRICING:
        print(f"错误：不支持的模型 '{model}'")
        print(f"支持的模型: {list(EMBEDDING_PRICING.keys())}")
        sys.exit(1)
    
    price_per_1k = EMBEDDING_PRICING[model]
    
    total_tokens = 0
    total_chunks = len(chunks)
    file_stats = {}
    kind_stats = {}
    
    print(f"正在计算 {total_chunks} 个文本块的 token 数量...")
    
    for i, chunk in enumerate(chunks):
        if verbose and (i + 1) % 1000 == 0:
            print(f"  处理进度: {i + 1}/{total_chunks}")
            
        # 从 chunk 中提取文本
        text = ""
        if "text" in chunk:
            text = chunk["text"]
        elif "content" in chunk:
            text = chunk["content"]
        else:
            print(f"警告：chunk {i+1} 中没有找到 'text' 或 'content' 字段")
            continue
        
        # 计算 tokens
        if "tokens" in chunk:
            # 如果已经有 token 计数，直接使用
            tokens = chunk["tokens"]
        else:
            # 重新计算
            tokens = count_tokens(text)
        
        total_tokens += tokens
        
        # 文件统计
        rel_path = chunk.get("rel_path", chunk.get("file", "unknown"))
        if rel_path not in file_stats:
            file_stats[rel_path] = {"tokens": 0, "chunks": 0}
        file_stats[rel_path]["tokens"] += tokens
        file_stats[rel_path]["chunks"] += 1
        
        # 类型统计
        kind = chunk.get("kind", "unknown")
        if kind not in kind_stats:
            kind_stats[kind] = {"tokens": 0, "chunks": 0}
        kind_stats[kind]["tokens"] += tokens
        kind_stats[kind]["chunks"] += 1
    
    # 计算费用
    cost_usd = (total_tokens / 1000) * price_per_1k
    
    return {
        "model": model,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "price_per_1k": price_per_1k,
        "cost_usd": cost_usd,
        "file_stats": file_stats,
        "kind_stats": kind_stats
    }

def format_number(num):
    """格式化数字显示"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)

def print_results(result: Dict, verbose: bool = False):
    """打印计算结果"""
    
    print("\n" + "="*60)
    print("OpenAI Embedding 费用计算结果")
    print("="*60)
    
    print(f"模型：{result['model']}")
    print(f"总文本块数：{result['total_chunks']:,}")
    print(f"总 Token 数：{result['total_tokens']:,} ({format_number(result['total_tokens'])})")
    print(f"定价：${result['price_per_1k']:.6f} / 1K tokens")
    print(f"总费用：${result['cost_usd']:.6f} USD")
    
    # 人民币估算（假设汇率7.2）
    cost_cny = result['cost_usd'] * 7.2
    print(f"人民币估算：¥{cost_cny:.4f} CNY (按汇率7.2计算)")
    
    # 平均费用
    avg_cost_per_chunk = result['cost_usd'] / result['total_chunks'] if result['total_chunks'] > 0 else 0
    avg_tokens_per_chunk = result['total_tokens'] / result['total_chunks'] if result['total_chunks'] > 0 else 0
    
    print(f"\n平均每个文本块：")
    print(f"  - Tokens: {avg_tokens_per_chunk:.1f}")
    print(f"  - 费用: ${avg_cost_per_chunk:.8f} USD")
    
    if verbose:
        print("\n" + "-"*40)
        print("按文件类型统计:")
        print("-"*40)
        
        for kind, stats in sorted(result['kind_stats'].items()):
            cost = (stats['tokens'] / 1000) * result['price_per_1k']
            print(f"{kind:>10}: {stats['chunks']:>6} chunks, {stats['tokens']:>10,} tokens, ${cost:>8.6f}")
        
        print("\n" + "-"*40)
        print("Top 10 文件 (按 token 数):")
        print("-"*40)
        
        top_files = sorted(result['file_stats'].items(), 
                          key=lambda x: x[1]['tokens'], reverse=True)[:10]
        
        for file_path, stats in top_files:
            cost = (stats['tokens'] / 1000) * result['price_per_1k']
            # 截断过长的路径
            display_path = file_path if len(file_path) <= 50 else "..." + file_path[-47:]
            print(f"{display_path:>50}: {stats['chunks']:>4} chunks, {stats['tokens']:>8,} tokens, ${cost:>8.6f}")

def parse_args():
    parser = argparse.ArgumentParser(description="计算 OpenAI Embedding 费用")
    parser.add_argument("--input", "-i", required=True, help="输入的 JSONL 文件路径")
    parser.add_argument("--model", "-m", default="text-embedding-3-small", 
                       help=f"Embedding 模型 (默认: text-embedding-3-small)\n"
                            f"支持的模型: {list(EMBEDDING_PRICING.keys())}\n"
                            f"别名: {list(MODEL_ALIASES.keys())}")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细统计信息")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"正在读取文件: {args.input}")
    chunks = load_jsonl(args.input)
    
    if not chunks:
        print("错误：没有找到有效的文本块")
        sys.exit(1)
    
    print(f"成功加载 {len(chunks)} 个文本块")
    
    result = calculate_cost(chunks, args.model, args.verbose)
    print_results(result, args.verbose)

if __name__ == "__main__":
    main()