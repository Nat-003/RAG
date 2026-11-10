from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError

def load_json(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load a JSON file and validate it against a Pydantic model."""
    file_path = Path(path)
    try:
        content = file_path.read_text(encoding="utf-8")
        return model.model_validate_json(content)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        ) from exc
    except ValidationError as exc:
        raise ValueError(
            f"Invalid JSON data in {file_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Failed to read JSON file {file_path}: {exc}"
        ) from exc


def save_json(path: str | Path, data: BaseModel) -> None:
    """Serialize a Pydantic model and save it to a JSON file."""
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        content = data.model_dump_json(indent=2)
        file_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise OSError(
            f"Failed to save JSON file {file_path}: {exc}"
        ) from exc
