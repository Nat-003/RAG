from fire import Fire
from . import cli
from src.dataset_io import load_json
from src.models import RagDataset
from typing import TypeVar
from pydantic import BaseModel

if __name__ == "__main__":
    T = TypeVar("T", bound=BaseModel)
    # Fire(cli)
    data = load_json("src/test.json", RagDataset)
    print(data)