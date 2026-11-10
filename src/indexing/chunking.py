import ast
import re
from pathlib import Path


class Chunker:
    def parse_file(self, path: str | Path) -> None:
        file_path = Path(path)
        source_code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=file_path.name)
        for i, node in enumerate(tree.body):
            print(f"Statement {i + 1}: {type(node).__name__} (line {node.lineno})")
            start_index = self.get_index(source_code, node.lineno, node.col_offset)
            end_index = self.get_index(source_code, node.end_lineno, node.end_col_offset)
            text = source_code[start_index:end_index]
            print(text)

    @staticmethod
    def get_index(text: str, line_start: int, col_offset: int) -> int:
        """Convert an AST line and UTF-8 byte column into a character index."""
        content: list[str] = []
        start = 0
        for newline in re.finditer(r"\r\n|\r|\n", text):
            content.append(text[start:newline.end()])
            start = newline.end()
        if start < len(text):
            content.append(text[start:])
        current_line = content[line_start - 1]
        preceding_lines_len = sum(len(line) for line in content[:line_start - 1])
        line_bytes = current_line.encode("utf-8")
        byte_prefix = line_bytes[:col_offset]
        character_prefix = byte_prefix.decode("utf-8")
        character_offset = len(character_prefix)
        return preceding_lines_len + character_offset