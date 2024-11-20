from typing import List
import numpy as np
from core.document import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """基于BM25算法的检索器"""

    def __init__(self):
        self.documents: List[Document] = []
        self.bm25 = None
        self.tokenized_docs = None