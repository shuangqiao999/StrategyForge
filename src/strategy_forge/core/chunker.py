"""文本分块器 — 智能分层：Markdown 结构感知 + 递归中文标点分割"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ChunkResult:
    """单个分块结果"""

    index: int
    content: str
    token_estimate: int


class TextChunker:
    """智能分层文本分块器，支持 Markdown 结构感知和中文标点递归分割。

    Usage:
        chunker = TextChunker(strategy="paragraph", max_chunk_size=1536)
        chunks = chunker.chunk(text)                        # plain text
        chunks = chunker.chunk(text, file_type=".md")       # markdown
    """

    # 递归分割优先级：段落 → 换行 → 中文句子结束符 → 中文分隔符 → 英文标点 → 空格 → 字符
    _CN_SEPARATORS = [
        "\n\n",
        "\n",
        "\u3002",  # 。
        "\uff01",  # ！
        "\uff1f",  # ？
        "\uff1b",  # ；
        "\uff0c",  # ，
        "\u3001",  # 、
        "\uff1a",  # ：
        ".", "!", "?", ";",
        " ",
        "",
    ]

    # Markdown 结构保护：不切断这些块
    _MD_CODE_FENCE = re.compile(r"^(```|~~~)")
    _MD_TABLE_ROW = re.compile(r"^\s*\|.*\|$")
    _MD_TABLE_SEP = re.compile(r"^\s*\|[-: ]+\|$")
    _MD_HEADING = re.compile(r"^(#{1,6})\s+")

    def __init__(
        self,
        strategy: str = "paragraph",
        chunk_size: int = 500,
        overlap: int = 50,
        max_chunk_size: int = 1536,
        min_chunk_size: int = 50,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = max(1, chunk_size)
        self.overlap = min(overlap, max_chunk_size - 1) if overlap > 0 else 0
        self.max_chunk_size = max(32, max_chunk_size)
        self.min_chunk_size = max(1, min_chunk_size)

    def chunk(self, text: str, file_type: str = ".txt") -> list[ChunkResult]:
        if self.strategy == "fixed":
            return self._chunk_fixed(text)

        if file_type in (".md", ".markdown"):
            chunks = self._chunk_markdown(text)
        else:
            chunks = self._chunk_recursive(text)

        chunks = self._merge_small_chunks(chunks)
        if self.overlap > 0:
            chunks = self._add_overlap(chunks)
        return chunks

    # ------------------------------------------------------------------
    # fixed
    # ------------------------------------------------------------------

    def _chunk_fixed(self, text: str) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        step = max(1, self.chunk_size - self.overlap)
        start = 0
        i = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(ChunkResult(
                    index=i,
                    content=chunk_text,
                    token_estimate=_estimate_tokens(chunk_text),
                ))
                i += 1
            start += step
        return chunks

    # ------------------------------------------------------------------
    # recursive (plain text)
    # ------------------------------------------------------------------

    def _chunk_recursive(self, text: str) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        i = 0
        for piece in self._split_recursive(text, self._CN_SEPARATORS):
            piece = piece.strip()
            if piece:
                chunks.append(ChunkResult(
                    index=i,
                    content=piece,
                    token_estimate=_estimate_tokens(piece),
                ))
                i += 1
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        if len(text) <= self.max_chunk_size:
            return [text]

        for sep in separators:
            if not sep:
                continue
            idx = text.rfind(sep, 0, self.max_chunk_size)
            if idx <= 0:
                continue
            split_pos = idx + len(sep)
            before = text[:split_pos]
            after = text[split_pos:]
            result: list[str] = []
            result.extend(self._split_recursive(before, separators))
            result.extend(self._split_recursive(after, separators))
            return result

        result: list[str] = []
        pos = 0
        while pos < len(text):
            result.append(text[pos:pos + self.max_chunk_size])
            pos += self.max_chunk_size
        return result

    # ------------------------------------------------------------------
    # markdown structure-aware
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str) -> list[ChunkResult]:
        """Markdown 结构感知分块：按标题切分 → 保护代码块/表格 → 内部递归分割。"""
        sections = self._split_by_headings(text)
        chunks: list[ChunkResult] = []
        i = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.max_chunk_size:
                chunks.append(ChunkResult(
                    index=i,
                    content=section,
                    token_estimate=_estimate_tokens(section),
                ))
                i += 1
            else:
                subs = self._chunk_section_with_protected_blocks(section)
                for sub in subs:
                    sub = sub.strip()
                    if sub:
                        chunks.append(ChunkResult(
                            index=i,
                            content=sub,
                            token_estimate=_estimate_tokens(sub),
                        ))
                        i += 1
        return chunks

    def _split_by_headings(self, text: str) -> list[str]:
        """按 H1/H2 标题分割文档。H3+ 子标题保持在所属节内。"""
        lines = text.split("\n")
        sections: list[list[str]] = []
        current: list[str] = []
        in_code = False

        for line in lines:
            stripped = line.rstrip("\r")

            if self._MD_CODE_FENCE.match(stripped):
                in_code = not in_code
                current.append(line)
                continue

            if in_code:
                current.append(line)
                continue

            m = self._MD_HEADING.match(stripped)
            if m:
                level = len(m.group(1))
                if level <= 2:
                    if current:
                        sections.append(current)
                        current = []
                current.append(line)
                continue

            current.append(line)

        if current:
            sections.append(current)

        return ["\n".join(sec).strip() for sec in sections if sec]

    def _chunk_section_with_protected_blocks(self, section: str) -> list[str]:
        """对长 section 进行递归分割，但保护代码块和表格不被切断。"""
        lines = section.split("\n")
        result: list[str] = []
        buffer: list[str] = []
        buf_len = 0
        in_code = False

        for line in lines:
            stripped = line.rstrip("\r")

            # 代码块保护
            if self._MD_CODE_FENCE.match(stripped):
                if in_code:
                    buffer.append(stripped)
                    buf_len += len(stripped)
                    in_code = False
                    # 完整代码块结束 — 保护它不被后续合并
                    if buffer and buf_len >= self.max_chunk_size:
                        result.extend(self._finalize_buffer(buffer))
                        buffer, buf_len = [], 0
                    continue
                else:
                    # flush buffer before code block
                    if buffer:
                        result.extend(self._finalize_buffer(buffer))
                        buffer, buf_len = [], 0
                    in_code = True
                    buffer.append(stripped)
                    buf_len += len(stripped)
                    continue

            if in_code:
                buffer.append(stripped)
                buf_len += len(stripped)
                continue

            # 表格行保护
            if self._MD_TABLE_ROW.match(stripped) or self._MD_TABLE_SEP.match(stripped):
                buffer.append(stripped)
                buf_len += len(stripped)
                continue

            # 普通行：检查是否需要 flush
            line_len = len(stripped)
            if buf_len + line_len > self.max_chunk_size and buffer:
                result.extend(self._finalize_buffer(buffer))
                buffer, buf_len = [], 0

            buffer.append(stripped)
            buf_len += line_len

        if buffer:
            result.extend(self._finalize_buffer(buffer))

        return result

    def _finalize_buffer(self, lines: list[str]) -> list[str]:
        """将缓冲区内容输出为块。如果超限则递归分割，否则保持整体。"""
        text = "\n".join(lines)
        if len(text) <= self.max_chunk_size:
            return [text]
        return self._split_recursive(text, self._CN_SEPARATORS)

    # ------------------------------------------------------------------
    # merge & overlap
    # ------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
        if len(chunks) <= 1:
            return chunks

        merged: list[ChunkResult] = []
        for c in chunks:
            if len(c.content) < self.min_chunk_size and merged:
                prev = merged[-1]
                if len(prev.content) + len(c.content) + 2 <= self.max_chunk_size:
                    new_content = prev.content + "\n\n" + c.content
                    merged[-1] = ChunkResult(
                        index=prev.index,
                        content=new_content,
                        token_estimate=_estimate_tokens(new_content),
                    )
                    continue
            merged.append(c)

        for idx, c in enumerate(merged):
            c.index = idx
        return merged

    def _add_overlap(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        result: list[ChunkResult] = [chunks[0]]
        for i in range(1, len(chunks)):
            c = chunks[i]
            prev = result[-1]
            prev_tail = prev.content
            ov = min(self.overlap, len(prev_tail))
            if len(c.content) + ov <= self.max_chunk_size:
                overlap_text = prev_tail[-ov:]
                new_content = overlap_text + "\n" + c.content
                result.append(ChunkResult(
                    index=i,
                    content=new_content,
                    token_estimate=_estimate_tokens(new_content),
                ))
            else:
                result.append(c)

        for idx, c in enumerate(result):
            c.index = idx
        return result


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """估算 token 数量（中文字符 ≈ 1 token，英文约 3 字符/token）。"""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return chinese_chars + (other_chars // 3)
