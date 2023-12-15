import numpy as np
from scipy.cluster.vq import kmeans2

from index import Index
from typing import Dict, List, Any, Tuple

class IndexPQ(Index):
    '''PQ Index.

    Attributes:
        D: Dimensionality of the vectors.
        M: Number of subspaces.
        K: Number of clusters within each subspace.
        d: Dimensionality of each subspace (d = D / M).
        
        centroids: ndarray of shape (M, K, d)
            The learned centroids of each subspace during training
        pqcodes: ndarray of shape (N, M)
            The assigned PQ codes for each vector in the training data
    '''
    
    def __init__(self, D: int, M: int, nbits: int) -> None:
        '''Initialize PQ Index.

        Args:
            D: Dimensionality of the vectors.
            M: Number of subspaces.
            nbits: Number of bits to represent centroids, K = 2^nbits
        '''
        self.M = M
        self.D = D
        self.K = 2**nbits
        self.d = D//self.M
        
        self.index_file = 'out/PQ_index.centroids'
        self.pqcodes_file = 'out/PQ_index.codes'
        self.is_loaded = False
        
    def train(self,  data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Trains the PQ index on the given data

        Args:
            data: ndarray of shape (N, D)
                An array of input vectors to train the index

        Returns: 
            centroids: ndarray of shape (M, K, d)
                    The learned centroids which define the clusters of all subspaces
            pqcodes: ndarray of shape(N, M)
                    The PQ codes assined to each vector of the training data
        '''
        M, K, D, d, N = self.M, self.K, self.D, self.d, data.shape[0]
        
        # norms = np.linalg.norm(data, axis=1, keepdims=True)
        # data = data / norms
        
        centroids = np.zeros((M, K, d)) 
        labels = np.zeros((M, N), dtype=np.uint32) # the dtype should be modified according to max number of centroids
        
        for m in range(M):
            centroids[m, :, :], labels[m, :] = self._train_subspace(data[: , m*d: (m+1)*d])
        
        self.centroids = centroids
        self.pqcodes = labels.T
        
        return self.pqcodes
        
    
    def search(self, q: np.ndarray, top_k: int=10, pqcodes=None) -> np.ndarray:
        '''Search the index for a given query to find top K

        Args:
            q: ndarray of shape (D,)
                The query vector
            top_k: The number of the required nearest vectors to the query
            pqcodes: ndarray of shape (*,M)
                The candidate list of encoded vectors to search for the query
                If none, an exhaustive is done on the stored pqcodes

        Returns: 
            ndarray of shape (top_k,)
            The positions of the top_k nearest neighbors to the query       
        '''
        M, K, D, d= self.M, self.K, self.D, self.d
          
        # If no candidates are given do an exuastive search on all the data 
        # if not pqcodes:
            # pqcodes = np.loadtxt(self.pqcodes_file, dtype=np.uint32)
          
        # q = q / np.linalg.norm(q)
        q = q.reshape(M, d)
        
        # The distances matrix of shape(M, K) 
        # It holds the distances between each query subvector and all the centroids of this subspace
        
        # distances = np.linalg.norm(self.centroids - q[:, np.newaxis, :], axis=2)
        # scores = np.linalg.norm(distances[np.arange(M), pqcodes], axis=1)
        
        distances = np.zeros((M, K))
        for m in range(M):
            distances[m] = np.linalg.norm(self.centroids[m] - q[m], axis=1)**2
            
        N, _ = pqcodes.shape
        scores = np.zeros(N)
        for m in range(M):
            scores += distances[m, pqcodes[:, m]]
        
        # argpartition has a complexity of O(N+klogK) thus it's better than argsort that has complexiy of O(nlogn)
        # where K here is top_k and n is the length of scores
        return np.argpartition(scores, top_k)[:top_k]
        

    def _train_subspace(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Runs the K-means clustering algorithm on a single subspace

        Args:
            data: ndarray of shape (N, d)
                An array of vectors within a single subspace

        Returns: 
            centroids: ndarray of shape (K, d)
                    The centroids which define the clusters of the subspace
            labels: ndarray of shape(N,)
                    The clusters assigned to each data vector 
                    (i.e., labels[i] is the id of the closest centroid to the ith observation)
        '''
        centroids, labels = kmeans2(data, self.K, minit='points', iter = 128)
        return centroids, labels
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        # norms = np.linalg.norm(data, axis=1, keepdims=True)
        # data = data / norms
        
        N, _ = data.shape
        labels = np.zeros((self.M, N), dtype=np.uint32) 
        
        for m in range(self.M):
            labels[m], _ = vq(data[: , m*d: (m+1)*d], self.centroids[m])

        return labels.T
    
    def save(self, filename=None):
        '''Saves the index into disk

        Args:
            filename: the name of index file
        '''
        if filename is None:
            filename = self.index_file
        # savetxt requires only 1D or 2D arrays so the array must be reshaped before saving
        np.savetxt(filename, self.centroids.reshape(self.M*self.K, self.d))
        np.savetxt(self.pqcodes_file, self.pqcodes, fmt="%d")
        
    
    def load(self, filename=None):
        '''Loades the index into memory

        Args:
            filename: the name of index file
        '''
        if filename is None:
            filename = self.index_file
            
        if not self.is_loaded:
            print("loading IVF index")  
            self.is_loaded = True
            self.centroids = np.loadtxt(filename).reshape((self.M, self.K, self.d))
                