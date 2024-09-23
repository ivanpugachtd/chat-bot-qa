from typing import Optional, Self
from pathlib import Path

class BaseLoader:
    def get_data(self) -> str:
        raise NotImplementedError("Method load not implemented")
  
  
class FileDataLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._data: Optional[str] = None
        self._extension = ".txt"
        self._encoding = "utf-8"

    def _validate_file_path(self) -> None:
        if not self.file_path.endswith(self._extension):
            raise ValueError(
                f"File {self.file_path} does not have a valid extension, please use {self._extension}"
            )
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist")

    def _load(self) -> Self:
        with open(self.file_path, "r", encoding=self._encoding) as file:
            self._data = file.read()
        return self

    def get_data(self) -> str | None:
        self._validate_file_path()
        if self._data is None:
            self._load()

        return self._data
