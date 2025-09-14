#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk.py — 切块（Chunker）
-----------------------------------------
特性：
1) 语义优先切块（Semantic Chunking）：
   - 代码文件：按函数/类/模块头部启发式分段，尽量与语义边界对齐；
   - 文档/配置：按标题/段落/行块分段。
2) Token-aware 控制：使用 tiktoken 计算 token，保证每块 ≤ max_tokens，超限再二次切分；
3) Overlap（重叠）：相邻块保留 overlap tokens，防止语义断裂；
4) 自动跳过二进制与超大文件；可配置排除目录/文件；
5) 输出 JSONL（每行一个 chunk，含文件路径、chunk_id、类型、token 数、文本内容）。

用法示例：
    # 处理当前目录（默认）
    python chunk.py --output chunks.jsonl
    
    # 处理指定目录
    python chunk.py --repo ./your_repo --output chunks.jsonl
    
    # 完整参数示例
    python chunk.py --repo ./your_repo \
                    --output chunks.jsonl \
                    --max-tokens 500 \
                    --overlap 50 \
                    --max-file-size 5242880 \
                    --exclude-dirs .git node_modules .venv dist build \
                    --no-file-type-limit

智能识别类型包括：
- 扩展名识别：.py/.js/.java 等 → code，.md/.txt → doc，.json/.yml → config
- 无后缀文件：Jenkinsfile/Dockerfile → code/config，README/LICENSE → doc
- 内容特征：pipeline{} → Jenkinsfile(code)，apiVersion: → K8s(config)
- 通用文本：可打印字符 ≥95% → code（默认）

随后可将 chunks.jsonl 送入 embedding + 索引流程。
"""

import os
import re
import sys
import json
import argparse
from typing import List, Dict, Iterable, Optional

# --- Tokenizer ---
try:
    import tiktoken
except ImportError:
    print("请先安装 tiktoken： pip install tiktoken", file=sys.stderr)
    sys.exit(1)

# ---------------- 配置默认项 ----------------

# 代码/文档/配置扩展名（可根据需要扩充）
CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".cpp", ".c", ".h",
    ".go", ".rs", ".php", ".rb", ".kt", ".swift", ".m", ".mm", ".scala"
}
DOC_EXT = {
    ".md", ".txt", ".rst", ".adoc", ".tex"
}
CONFIG_EXT = {
    ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".env", ".log",
    ".xml", ".plist", ".properties"
}

# 无后缀文件名映射（精确匹配文件名）
NO_EXT_FILES = {
    # DevOps & CI/CD
    "Jenkinsfile": "code",
    "Dockerfile": "config", 
    "Containerfile": "config",
    "Makefile": "code",
    "makefile": "code",
    "Rakefile": "code",
    "Vagrantfile": "config",
    "Brewfile": "config",
    "Pipfile": "config",
    "Gemfile": "config",
    "Procfile": "config",
    
    # K8s & 容器编排
    "docker-compose": "config",
    "compose": "config",
    
    # 配置文件
    "LICENSE": "doc",
    "CHANGELOG": "doc", 
    "README": "doc",
    "INSTALL": "doc",
    "COPYING": "doc",
    "AUTHORS": "doc",
    "CONTRIBUTORS": "doc",
    "NOTICE": "doc",
    "VERSION": "config",
    "MANIFEST": "config",
    ".gitignore": "config",
    ".dockerignore": "config",
    ".eslintrc": "config",
    ".babelrc": "config",
    ".editorconfig": "config",
}

# 内容特征识别模式（正则表达式）
CONTENT_PATTERNS = {
    "code": [
        # Jenkinsfile 特征
        re.compile(r"pipeline\s*\{|node\s*\(|stage\s*\(", re.IGNORECASE),
        # Dockerfile 特征  
        re.compile(r"FROM\s+[\w/:.-]+|RUN\s+|COPY\s+|ADD\s+|WORKDIR\s+|CMD\s+|ENTRYPOINT\s+", re.IGNORECASE),
        # Makefile 特征
        re.compile(r"^\w+\s*:.*$", re.MULTILINE),
        # Shell script 特征
        re.compile(r"#!/bin/(bash|sh)|export\s+\w+="),
    ],
    "config": [
        # K8s YAML 特征
        re.compile(r"apiVersion\s*:|kind\s*:|metadata\s*:|spec\s*:", re.IGNORECASE),
        # Docker Compose 特征
        re.compile(r"version\s*:\s*[\"']?\d|services\s*:|networks\s*:|volumes\s*:", re.IGNORECASE),
        # Ansible 特征
        re.compile(r"hosts\s*:|tasks\s*:|vars\s*:|roles\s*:", re.IGNORECASE),
        # 通用配置特征
        re.compile(r"^\s*[\w.-]+\s*[:=]\s*.*$", re.MULTILINE),
    ]
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".svn", ".hg", ".idea", ".vscode", ".venv",
    "node_modules", "dist", "build", "target", "out", "__pycache__"
}

DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP = 50
DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# ------------------------------------------------
# 基础工具
# ------------------------------------------------

def get_encoder(model_name: str = "text-embedding-3-large"):
    """获取 tiktoken 编码器；不可用时回退到 cl100k_base。"""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

ENC = get_encoder()

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def is_probably_binary(sample: bytes, threshold: float = 0.90) -> bool:
    """根据可打印字符占比与是否含有 NUL 字节判断是否为二进制。"""
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    printable = sum(chr(b).isprintable() or chr(b).isspace() for b in sample)
    return (printable / len(sample)) < threshold

def safe_read_text(path: str, max_file_size: int) -> Optional[str]:
    """尽量安全地读取文本文件；过大/疑似二进制则返回 None。"""
    try:
        size = os.path.getsize(path)
        if size > max_file_size:
            return None
        with open(path, "rb") as f:
            head = f.read(8192)
            if is_probably_binary(head):
                return None
        # 以 utf-8 为主；无法解码时忽略错误
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None

def normalize_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")

# ------------------------------------------------
# 切块核心：token-based with overlap
# ------------------------------------------------

def chunk_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    """将文本按 token 大小切块，并添加 overlap 重叠。"""
    tokens = ENC.encode(text)
    n = len(tokens)
    chunks: List[str] = []
    if n == 0:
        return chunks
    start = 0
    step = max(max_tokens - overlap, 1)
    while start < n:
        end = min(start + max_tokens, n)
        chunk_tokens = tokens[start:end]
        chunks.append(ENC.decode(chunk_tokens))
        start += step
    return chunks

# ------------------------------------------------
# 语义切块（启发式）
# ------------------------------------------------

# 代码：函数/类/模块头部识别（跨多语言的启发式正则）
RE_FUNC_HEADERS = [
    # Python
    re.compile(r"^\s*(def|class)\s+\w+", re.MULTILINE),
    # JS/TS
    re.compile(r"^\s*(export\s+)?(async\s+)?function\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*\(.*?\)\s*=>\s*\{?", re.MULTILINE),
    re.compile(r"^\s*(class)\s+\w+", re.MULTILINE),
    # Java/C#/Kotlin/Swift/Go/C/C++（近似）
    re.compile(r"^\s*(public|private|protected)?\s*(static\s+)?(final\s+)?(class|interface)\s+\w+", re.MULTILINE),
    re.compile(r"^\s*[\w:<>\*\[\]\s]+?\s+\w+\s*\([^;]*\)\s*\{", re.MULTILINE),  # func sig + {
    # Rust
    re.compile(r"^\s*fn\s+\w+\s*\(", re.MULTILINE),
]

RE_MD_TITLE = re.compile(r"^\s{0,3}#{1,6}\s+.+$", re.MULTILINE)

def split_code_semantic(text: str) -> List[str]:
    """
    代码语义切块：按可能的函数/类头部将文本分段（启发式），
    如无匹配则返回整段；之后由 token 控制二次切分。
    """
    # 收集所有匹配位置
    cut_points = set()
    for pat in RE_FUNC_HEADERS:
        for m in pat.finditer(text):
            cut_points.add(m.start())
    if not cut_points:
        return [text]
    # 加入文本起点
    cut_list = sorted({0, *cut_points})
    blocks: List[str] = []
    for i in range(len(cut_list)):
        start = cut_list[i]
        end = cut_list[i + 1] if i + 1 < len(cut_list) else len(text)
        block = text[start:end].strip("\n")
        if block:
            blocks.append(block)
    return blocks

def split_doc_semantic(text: str) -> List[str]:
    """
    文档语义切块：优先按 Markdown 标题分段，否则按空行段落。
    """
    titles = list(RE_MD_TITLE.finditer(text))
    if titles:
        # 标题分段
        idxs = [0] + [m.start() for m in titles] + [len(text)]
        idxs = sorted(set(idxs))
        chunks: List[str] = []
        for i in range(len(idxs) - 1):
            seg = text[idxs[i]:idxs[i+1]].strip("\n")
            if seg.strip():
                chunks.append(seg)
        return chunks
    # 段落
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]

def split_config_lines(text: str, approx_line_tokens: int = 16, target_tokens: int = DEFAULT_MAX_TOKENS) -> List[str]:
    """
    配置/日志：按行聚合成接近 target_tokens 的块（粗略），再交给 token 控制精修。
    """
    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []
    budget_lines = max(target_tokens // max(approx_line_tokens, 1), 16)  # 至少 16 行
    for ln in lines:
        buf.append(ln)
        if len(buf) >= budget_lines:
            block = "\n".join(buf).strip("\n")
            if block:
                blocks.append(block)
            buf = []
    if buf:
        block = "\n".join(buf).strip("\n")
        if block:
            blocks.append(block)
    return blocks or [text]

def classify_kind(path: str, text: str, no_file_type_limit: bool = False) -> str:
    """
    智能类型判定：code / doc / config
    1. 先检查扩展名
    2. 再检查无后缀文件名
    3. 然后基于内容特征识别
    4. 最后根据可打印性判定（默认为 code）
    """
    # 1. 扩展名识别
    ext = os.path.splitext(path)[1].lower()
    if ext in CODE_EXT:
        return "code"
    if ext in DOC_EXT:
        return "doc"
    if ext in CONFIG_EXT:
        return "config"
    
    # 2. 无后缀文件名识别
    filename = os.path.basename(path)
    if filename in NO_EXT_FILES:
        return NO_EXT_FILES[filename]
    
    # 3. 基于内容特征识别
    for pattern_type, patterns in CONTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return pattern_type
    
    # 4. 通用可打印性判定
    total = len(text)
    visible = sum(1 for c in text if c.isprintable() or c.isspace())
    is_printable = total > 0 and (visible / total) >= 0.95
    
    if no_file_type_limit and is_printable:
        # 启用 no_file_type_limit 时，默认当作代码处理
        return "code"
    elif is_printable:
        # 原有逻辑：可打印文本默认为代码
        return "code"
    
    return "skip"

def semantic_then_token_chunks(kind: str, text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    先做语义切块，再进行 token-aware 限制与二次切分，最后统一加 overlap。
    """
    if kind == "code":
        first = split_code_semantic(text)
    elif kind == "doc":
        first = split_doc_semantic(text)
    elif kind == "config":
        first = split_config_lines(text)
    else:
        return []

    final_chunks: List[str] = []
    for seg in first:
        if not seg.strip():
            continue
        if count_tokens(seg) <= max_tokens:
            final_chunks.append(seg)
        else:
            final_chunks.extend(chunk_by_tokens(seg, max_tokens=max_tokens, overlap=overlap))
    return final_chunks

# ------------------------------------------------
# 仓库遍历与输出
# ------------------------------------------------

def iter_files(repo_dir: str,
               exclude_dirs: Iterable[str],
               exclude_globs: Iterable[str],
               max_file_size: int) -> Iterable[str]:
    """遍历仓库，过滤目录与不可解析的文件。"""
    repo_dir = os.path.abspath(repo_dir)
    for root, dirs, files in os.walk(repo_dir):
        # 目录过滤
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for name in files:
            path = os.path.join(root, name)
            # 文件名/路径 glob 过滤
            skip = False
            for pat in exclude_globs:
                # 简易包含判断（可改 fnmatch）
                if pat and pat in path:
                    skip = True
                    break
            if skip:
                continue
            # 大小与二进制判断 & 读取
            text = safe_read_text(path, max_file_size=max_file_size)
            if text is None:
                continue
            yield path, text

def chunk_repo(repo_dir: str,
               max_tokens: int,
               overlap: int,
               exclude_dirs: Iterable[str],
               exclude_globs: Iterable[str],
               max_file_size: int,
               no_file_type_limit: bool = False) -> Iterable[Dict]:
    repo_dir = os.path.abspath(repo_dir)
    for path, text in iter_files(repo_dir, exclude_dirs, exclude_globs, max_file_size):
        kind = classify_kind(path, text, no_file_type_limit)
        if kind == "skip":
            continue
        blocks = semantic_then_token_chunks(kind, text, max_tokens, overlap)
        rel = os.path.relpath(path, repo_dir)
        rel = normalize_path(rel)
        for i, block in enumerate(blocks):
            yield {
                "repo": normalize_path(repo_dir),
                "file": normalize_path(path),
                "rel_path": rel,
                "kind": kind,
                "chunk_id": i,
                "tokens": count_tokens(block),
                "text": block
            }

# ------------------------------------------------
# CLI
# ------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="语义 + token-aware 代码仓库切块器（Chunker）")
    p.add_argument("--repo", default=".", help="仓库根目录（本地路径），默认为当前目录")
    p.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"每块最大 tokens（默认 {DEFAULT_MAX_TOKENS}）")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help=f"相邻块重叠 tokens（默认 {DEFAULT_OVERLAP}）")
    p.add_argument("--max-file-size", type=int, default=DEFAULT_MAX_FILE_SIZE, help=f"单文件最大字节数（默认 {DEFAULT_MAX_FILE_SIZE}）")
    p.add_argument("--exclude-dirs", nargs="*", default=list(DEFAULT_EXCLUDE_DIRS), help="需要排除的目录名列表（精确匹配）")
    p.add_argument("--exclude-globs", nargs="*", default=[], help="需要排除的路径片段（简单包含匹配）")
    p.add_argument("--no-file-type-limit", action="store_true", help="不限制文件类型，处理所有可读文本文件")
    return p.parse_args()

def main():
    args = parse_args()
    repo = args.repo
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in chunk_repo(
            repo_dir=repo,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            exclude_dirs=set(args.exclude_dirs),
            exclude_globs=args.exclude_globs,
            max_file_size=args.max_file_size,
            no_file_type_limit=args.no_file_type_limit,
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ 切块完成：{total} 个 chunks 已写入 {out_path}")

if __name__ == "__main__":
    main()
