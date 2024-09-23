from typing import List
from nltk.tokenize import sent_tokenize
from nltk.tokenize import BlanklineTokenizer


class Splitter:
    def __init__(self, data: str) -> None:
        self.data = data
        self._splitted_data = None

    def _split(self) -> None:
        raise NotImplementedError("Method split not implemented")

    def __getitem__(self, index: int) -> str:
        if self._splitted_data is None:
            self._split()
        if not self._splitted_data:
            raise ValueError("No data found")
        if not 0 <= index < len(self._splitted_data):
            raise IndexError("Index out of range")

        return self._splitted_data[index]

    def get_splitted_data(self) -> List[str]:
        if self._splitted_data is None:
            self._split()

        if not self._splitted_data:
            raise ValueError("No data found")

        return self._splitted_data


class SentencesSplitter(Splitter):
    def __init__(self, data: str) -> None:
        super().__init__(data)
        self._language = "english"

    def _split(self) -> None:
        self._splitted_data = sent_tokenize(self.data, self._language)
        print(f"Found sentences: {len(self._splitted_data)}")


class ParagraphSplitter(Splitter):
    def __init__(self, data: str) -> None:
        super().__init__(data)

    def _split(self) -> None:
        self._splitted_data = BlanklineTokenizer().tokenize(self.data)
        print(f"Found paragraphs: {len(self._splitted_data)}")