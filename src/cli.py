
def index(max_chunk_size: int = 2000) -> None:
    print(f"Indexing started with max_chunk_size: {max_chunk_size}")

def search(query: str, k: int = 5) -> None:
    print(f"Searching for {query}\nTop k: {k}")

def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    print(f"Dataset path: {dataset_path}\nTop K: {k}\n save_directory: {save_directory}")

def answer(query: str, k: int = 5) -> None:
    print(f"answer for query: {query}\nTop K: {k}")

def answer_dataset(student_search_results_path: str, save_directory: str) -> None:
    print(f"student result path: {student_search_results_path}\nSave directory: {save_directory}")

def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    print(f"student result path: {student_search_results_path}\nDataset path: {dataset_path}")
    
