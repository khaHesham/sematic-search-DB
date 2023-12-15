from typing import Dict, List, Annotated
import numpy as np
from index import Index

class VecDB:
    def __init__(self, index: Index, file_path: str = "out/saved_db.csv", new_db: bool = True) -> None:
        self.file_path = file_path
        self.index = index
        
        if new_db:
            with open(self.file_path, "w") as fout:
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
        data = np.array([row["embed"] for row in rows])
        self._build_index(data)

    def retrive(self, query: Annotated[List[float], 70], top_k: int = 5):
        self.index.load()
        
        q = np.array(query)
        return list(self.index.search(q, top_k))

    def _build_index(self, data):
        self.index.train(data)
        self.index.save()
        
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
