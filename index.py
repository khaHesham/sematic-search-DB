import numpy as np
from abc import ABC, abstractmethod
from typing import Any

class Index(ABC):    
    '''Abstract Index Class.
    '''
        
    @abstractmethod
    def train(self,  data: np.ndarray) -> Any:
        pass
    
    @abstractmethod
    def search(self, q: np.ndarray, top_k) -> np.ndarray:
        pass
    
    @abstractmethod
    def save(self, filename):
        pass
        
    @abstractmethod   
    def load(self, filename):
        pass
                