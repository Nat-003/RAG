import ast
import re
from pathlib import Path


class Chunker:
    """Find original character ranges of top-level Python AST statements."""

    def parse_file(self, path: str | Path) -> list[tuple[int, int]]:
        """Return (start, end) ranges for top-level statements.

        Ranges use Python string character indices: source_code[start:end].
        Leading comment blocks and decorators are part of the statement span.
        """
        file_path = Path(path)

        # newline="" retains CRLF/CR exactly as stored on disk.
        with file_path.open("r", encoding="utf-8", newline="") as file:
            source_code = file.read()

        tree = ast.parse(source_code, filename=file_path.name)
        lines = re.split(r"\r\n|\r|\n", source_code)
        spans: list[tuple[int, int]] = []

        for i, node in enumerate(tree.body):
            # Start at the statement itself by default.
            start_line = node.lineno
            start_index = self.get_index(
                source_code, start_line, node.col_offset
            )

            # Decorated definitions start at their first '@', not at 'def'.
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.decorator_list:
                first_decorator = node.decorator_list[0]
                start_line = first_decorator.lineno
                at_column = lines[start_line - 1].find("@")
                if at_column == -1:
                    raise ValueError("Decorator line has no '@' marker")
                start_index = (
                    self.get_index(source_code, start_line, 0) + at_column
                )

            # Don't let preceding comments overlap the previous AST node.
            min_line = 1
            if i > 0:
                previous_node = tree.body[i - 1]
                if previous_node.end_lineno is not None:
                    min_line = previous_node.end_lineno + 1

            comment_start_line = self.find_comment_start(
                source_code, start_line, min_line
            )
            if comment_start_line < start_line:
                start_index = self.get_index(
                    source_code, comment_start_line, 0
                )

            end_index = self.get_index(
                source_code, node.end_lineno, node.end_col_offset
            )

            # Task A: remember the source range instead of only printing it.
            spans.append((start_index, end_index))

        return spans

    @staticmethod
    def find_comment_start(
        text: str, start_line: int, min_line: int = 1
    ) -> int:
        """Include contiguous preceding comment-only physical lines."""
        lines = re.split(r"\r\n|\r|\n", text)
        current_line = start_line

        while current_line > min_line:
            previous_line = lines[current_line - 2]
            if not previous_line.lstrip().startswith("#"):
                break
            current_line -= 1

        return current_line

    @staticmethod
    def get_index(text: str, line_start: int, col_offset: int) -> int:
        """Convert an AST line and UTF-8 byte column to a character index."""
        content: list[str] = []
        start = 0
        for newline in re.finditer(r"\r\n|\r|\n", text):
            content.append(text[start:newline.end()])
            start = newline.end()
        if start < len(text):
            content.append(text[start:])

        current_line = content[line_start - 1]
        preceding_lines_len = sum(
            len(line) for line in content[:line_start - 1]
        )
        byte_prefix = current_line.encode("utf-8")[:col_offset]
        character_offset = len(byte_prefix.decode("utf-8"))
        return preceding_lines_len + character_offset

    @staticmethod
    def find_gaps(
        text: str,
        spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Return all character ranges not covered by AST spans."""
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in sorted(spans):
            if not (0 <= start <= end <= len(text)):
                raise ValueError(
                    f"Invalid span ({start}, {end}) for text length {len(text)}"
                )
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(text):
            gaps.append((cursor, len(text)))
        return gaps
