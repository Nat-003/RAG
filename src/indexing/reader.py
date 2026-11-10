from pathlib import Path
from typing import Literal
import os
from collections.abc import Iterator
from src.models import SourceFile


class Reader:
    """Read source files from a repository."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the reader with the project root."""
        self.project_root = project_root.resolve()

    def read_file(self, path: Path) -> SourceFile:
        """Read a source file without modifying its contents."""
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not supported: {path}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {path}")
        path = path.resolve()
        relative_path = path.relative_to(self.project_root)
        file_type: Literal["python", "markdown", "text"]
        match path.suffix.lower():
            case ".py":
                file_type = "python"
            case ".md":
                file_type = "markdown"
            case ".txt":
                file_type = "text"
            case _:
                raise ValueError(f"Unsupported file type: {path}")
        with path.open("r", encoding="utf-8", newline="") as file:
            content = file.read()
        if not content:
            raise ValueError(f"Empty file: {path}")
        if "\x00" in content:
            raise ValueError(f"Binary-like file: {path}")
        return SourceFile(
            file_path=relative_path.as_posix(),
            text=content,
            file_type=file_type,
        )

    def read_directory(self, corpus_root: Path, failed_read: dict[Path, str]) -> Iterator[SourceFile]:
        """Yield eligible source files recursively."""
        def record_error(oserror: OSError) -> None:
            if oserror.filename is not None:
                failed_read[Path(oserror.filename)] = str(oserror)
            else:
                failed_read[corpus_root] = str(oserror)
        corpus_root = corpus_root.resolve()
        if not corpus_root.is_dir():
            raise NotADirectoryError(
                f"Invalid corpus directory: {corpus_root}"
            )
        corpus_root.relative_to(self.project_root)
        excluded_directories = {
            ".git",
            ".venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
        }
        allowed_extensions = {".py", ".md", ".txt"}
        for current_dir, directories, filenames in os.walk(
            corpus_root,
            topdown=True,
            followlinks=False,
            onerror=record_error,
        ):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in excluded_directories
                and not (Path(current_dir) / directory).is_symlink()
            )
            for filename in sorted(filenames):
                path = Path(current_dir) / filename
                if path.suffix.lower() not in allowed_extensions:
                    continue
                if path.is_symlink():
                    continue
                try:
                    source = self.read_file(path)
                except UnicodeDecodeError as e:
                    failed_read[path] = str(e)
                    continue
                except ValueError as e:
                    failed_read[path] = str(e)
                    continue
                except OSError as e:
                    failed_read[path] = str(e)
                    continue
                yield source