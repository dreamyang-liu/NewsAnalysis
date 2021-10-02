from enum import Enum

__all__ = ["SystemModuleType"]


class SystemModuleType(Enum):
    QA = 1 # Question Answering
    TS = 2 # Text Summarization
    FD = 3 # Fruad Dection
    SA = 4 # Sentiment Analysis