from typing import Protocol


class IDataProvider(Protocol):
    def get_data(self) -> None: ...
