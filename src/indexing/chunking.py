"""Split Python source into searchable chunks with exact character offsets."""

import ast
import re
from collections.abc import Callable
from pathlib import Path


class Chunker:
    """Chunk Python source while preserving original source coordinates."""

    @staticmethod
    def _validate_size(max_size: int) -> None:
        """Reject chunk limits incompatible with the project requirements."""
        if not 1 <= max_size <= 2000:
            raise ValueError("max_size must be between 1 and 2000")

    @staticmethod
    def _locator(text: str) -> Callable[[int, int], int]:
        """Build a UTF-8-byte-to-character position converter for a source."""
        starts = [0]
        for match in re.finditer(r"\r\n|\r|\n", text):
            starts.append(match.end())

        def locate(line_number: int, byte_column: int) -> int:
            """Convert an AST line/byte column to a character index."""
            if not 1 <= line_number <= len(starts):
                raise ValueError("Invalid AST line number")
            start = starts[line_number - 1]
            end = (
                starts[line_number] if line_number < len(starts)
                else len(text)
            )
            encoded_line = text[start:end].encode("utf-8")
            if not 0 <= byte_column <= len(encoded_line):
                raise ValueError("Invalid AST byte column")
            try:
                return start + len(encoded_line[:byte_column].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError(
                    "AST column splits a UTF-8 character"
                ) from error

        return locate

    @staticmethod
    def get_index(text: str, line_start: int, col_offset: int) -> int:
        """Convert an AST line and UTF-8 byte column to a character index."""
        return Chunker._locator(text)(line_start, col_offset)

    @staticmethod
    def _comment_start(
        lines: list[str], start_line: int, min_line: int
    ) -> int:
        """Find a preceding comment block using already-split source lines."""
        current_line = start_line
        while current_line > min_line:
            if not lines[current_line - 2].lstrip().startswith("#"):
                break
            current_line -= 1
        return current_line

    @staticmethod
    def find_comment_start(
        text: str, start_line: int, min_line: int = 1
    ) -> int:
        """Find contiguous comment-only lines before a statement."""
        return Chunker._comment_start(
            re.split(r"\r\n|\r|\n", text), start_line, min_line
        )

    @staticmethod
    def find_gaps(
        text: str, spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Find source regions not covered by non-overlapping spans."""
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in sorted(spans):
            if not 0 <= start < end <= len(text) or start < cursor:
                raise ValueError("Invalid or overlapping source spans")
            if cursor < start:
                gaps.append((cursor, start))
            cursor = end
        if cursor < len(text):
            gaps.append((cursor, len(text)))
        return gaps

    @staticmethod
    def process_gaps(
        text: str, gaps: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Keep non-whitespace gaps, including standalone comments."""
        return [(start, end) for start, end in gaps if text[start:end].strip()]

    @staticmethod
    def combine_spans(
        spans: list[tuple[int, int]],
        comment_gaps: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Order AST spans and retained gaps by source position."""
        return sorted(spans + comment_gaps)

    @staticmethod
    def split_by_lines(
        text: str, start: int, end: int, max_size: int = 2000
    ) -> list[tuple[int, int]]:
        """Split an original source range at line endings where possible."""
        Chunker._validate_size(max_size)
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid source span")
        chunks: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            limit = min(cursor + max_size, end)
            if limit == end:
                chunks.append((cursor, end))
                break
            split_at = None
            for match in re.compile(r"\r\n|\r|\n").finditer(
                text, cursor, min(limit + 1, end)
            ):
                if match.end() <= limit:
                    split_at = match.end()
            if split_at is None:
                split_at = limit
                if (
                    split_at > cursor + 1
                    and text[split_at - 1:split_at + 1] == "\r\n"
                ):
                    split_at -= 1
            chunks.append((cursor, split_at))
            cursor = split_at
        return chunks

    @staticmethod
    def merge_spans(
        text: str, spans: list[tuple[int, int]], max_size: int = 2000
    ) -> list[tuple[int, int]]:
        """Group adjacent AST pieces while skipping empty-only chunks."""
        Chunker._validate_size(max_size)
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if not 0 <= start < end <= len(text) or end - start > max_size:
                raise ValueError("Invalid or oversized source span")
            if not text[start:end].strip():
                continue
            if merged and start < merged[-1][1]:
                raise ValueError("Overlapping source spans")
            if merged and end - merged[-1][0] <= max_size:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _node_span(
        node: ast.AST, locate: Callable[[int, int], int]
    ) -> tuple[int, int] | None:
        """Find a statement's original range, including its indentation."""
        if isinstance(node, ast.match_case):
            first = getattr(node.pattern, "lineno", None)
            body = node.body
            last = body[-1] if body else None
            last_line = getattr(last, "end_lineno", None)
            last_col = getattr(last, "end_col_offset", None)
        else:
            first = getattr(node, "lineno", None)
            last_line = getattr(node, "end_lineno", None)
            last_col = getattr(node, "end_col_offset", None)
        if first is None or last_line is None or last_col is None:
            return None
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and node.decorator_list:
            first = min(
                first, *(decorator.lineno for decorator in node.decorator_list)
            )
        return locate(first, 0), locate(last_line, last_col)

    def _split_ast(
        self,
        text: str,
        node: ast.AST,
        start: int,
        end: int,
        max_size: int,
        locate: Callable[[int, int], int],
    ) -> list[tuple[int, int]]:
        """Split an AST node while preserving gaps between children."""
        if start == end:
            return []
        if end - start <= max_size:
            return [(start, end)] if text[start:end].strip() else []
        children: list[tuple[int, int, ast.AST]] = []
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            members = getattr(node, field, None)
            if not isinstance(members, list):
                continue
            for child in members:
                if not isinstance(
                    child, (ast.stmt, ast.ExceptHandler, ast.match_case)
                ):
                    continue
                position = self._node_span(child, locate)
                if position is None:
                    continue
                child_start, child_end = position
                child_start = max(start, child_start)
                child_end = min(end, child_end)
                if child_start < child_end:
                    children.append((child_start, child_end, child))
        if not children:
            return self.merge_spans(
                text, self.split_by_lines(text, start, end, max_size), max_size
            )
        pieces: list[tuple[int, int]] = []
        cursor = start
        for child_start, child_end, child in sorted(
            children, key=lambda item: (item[0], item[1])
        ):
            if child_end <= cursor:
                continue
            child_start = max(child_start, cursor)
            if cursor < child_start:
                pieces.extend(
                    self.split_by_lines(text, cursor, child_start, max_size)
                )
            if child_end - child_start <= max_size:
                pieces.append((child_start, child_end))
            else:
                pieces.extend(
                    self._split_ast(
                        text, child, child_start, child_end, max_size, locate
                    )
                )
            cursor = child_end
        if cursor < end:
            pieces.extend(self.split_by_lines(text, cursor, end, max_size))
        if isinstance(node, ast.Module):
            return [span for span in pieces if text[span[0]:span[1]].strip()]
        return self.merge_spans(text, pieces, max_size)

    def split_by_ast(
        self,
        text: str,
        node: ast.AST,
        start: int,
        end: int,
        max_size: int = 2000,
    ) -> list[tuple[int, int]]:
        """Return size-bounded original-source ranges for an AST node."""
        self._validate_size(max_size)
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid source span")
        return self._split_ast(
            text, node, start, end, max_size, self._locator(text)
        )

    def parse_text(
        self, text: str, max_size: int = 2000
    ) -> list[tuple[int, int]]:
        """Split Python text, using line fallback for invalid syntax."""
        self._validate_size(max_size)
        if not text.strip():
            return []
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return self.merge_spans(
                text,
                self.split_by_lines(text, 0, len(text), max_size),
                max_size
            )
        locate = self._locator(text)
        spans: list[tuple[int, int]] = []
        previous_end_line = 0
        lines = re.split(r"\r\n|\r|\n", text)
        for node in tree.body:
            position = self._node_span(node, locate)
            if position is None:
                continue
            start, end = position
            first_line = getattr(node, "lineno", 1)
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.decorator_list:
                first_line = min(
                    first_line,
                    *(decorator.lineno for decorator in node.decorator_list)
                )
            comment_line = self._comment_start(
                lines, first_line, previous_end_line + 1
            )
            if comment_line < first_line:
                start = locate(comment_line, 0)
            spans.extend(
                self._split_ast(text, node, start, end, max_size, locate)
            )
            end_line = getattr(node, "end_lineno", None)
            if isinstance(end_line, int):
                previous_end_line = end_line
        gaps = self.process_gaps(text, self.find_gaps(text, spans))
        additional_spans: list[tuple[int, int]] = []
        for start, end in gaps:
            additional_spans.extend(
                self.merge_spans(
                    text,
                    self.split_by_lines(text, start, end, max_size),
                    max_size
                )
            )
        return self.combine_spans(spans, additional_spans)

    def parse_file(
        self, path: str | Path, max_size: int = 2000
    ) -> list[tuple[int, int]]:
        """Read a Python file without changing its line endings."""
        with Path(path).open("r", encoding="utf-8", newline="") as file:
            return self.parse_text(file.read(), max_size)
