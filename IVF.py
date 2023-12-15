'''
IVF implementation
Authors: Khaled Hesham, Kirollos Samy
Date : 12/10/2024
'''
import numpy as np
from index import Index
from indexpq import IndexPQ
from scipy.cluster.vq import kmeans2

class IVF_PQ(Index):

    def __init__(self, D: int, nbits: int, nprob: int, pq_index: IndexPQ) -> None:
        '''Initialize IVF_PQ.

        Args:
            D: Dimensionality of the vectors.
            M: Number of subspaces.
            nbits: Number of bits to represent centroids, K = 2^nbits
        '''
        self.D = D
        self.K = 2**nbits 
        self.nprob = nprob          # Number of clusters to be retrieved during search
        self.pq_index = pq_index
        
        self.index_file = 'out/IVF_index.centroids'
        self.is_loaded = False
        
            
    def train(self, data: np.ndarray):
        print("Training IVF")
        self.centroids, labels = kmeans2(data, self.K, minit='points', iter = 128)

        print("Training PQ")
        pqcodes = self.pq_index.train(data)

        print("clustering vectors") 
        for clusterID in range(self.K):
            vectorIDs, = np.where(labels==clusterID)
            cluster = np.column_stack((pqcodes[vectorIDs], vectorIDs))
            np.savetxt(f"out/clusters/{clusterID}.cluster", cluster, fmt="%d")
                    
    def predict(self, data: np.ndarray):
        labels, _ = vq(data, self.centroids)
        
        pqcodes = self.pq_index.predict(data)
        
        print("clustering new vectors") 
        for clusterID in range(self.K):
            vectorIDs, = np.where(labels==clusterID)
            cluster = np.column_stack((pqcodes[vectorIDs], vectorIDs))
            
            #append clusters to to clusters files
            pre_cluster = np.loadtxt(f"out/clusters/{clusterID}.cluster", dtype=int)
            post_cluster = np.vstack((pre_cluster, cluster))
            np.savetxt(f"out/clusters/{clusterID}.cluster", post_cluster, fmt="%d")
        
    
    def search(self, q: np.ndarray, top_k: int) -> np.ndarray:  
        print("searching IVF")
        q = q.reshape((70,))
        
        distances = self._cosine_similarity(self.centroids, q)
        nearest_clusters = np.argsort(distances)[-self.nprob:]
            
        candidates = np.empty((0, self.pq_index.M + 1))
        for cluster in nearest_clusters:
            loaded_cluster = np.loadtxt(f"out/clusters/{cluster}.cluster", dtype=int)
            candidates = np.append(candidates, loaded_cluster, axis = 0)
            
            
        candidates = candidates.astype(int)
        vectorIDs = candidates[:,-1]
        pqcodes = candidates[:,:-1] 
            
        print("searching PQ index")
        top_indices = self.pq_index.search(q, top_k, pqcodes)
        
        return vectorIDs[top_indices]       
        
        
    def save(self, filename=None):
        if filename is None:
            filename = self.index_file
            
        print("saving IVF index")         
        np.savetxt(filename, self.centroids)
        
        print("Saving PQ index")
        self.pq_index.save()
        
        # for clusterID in range(self.K):
            # save cluster to disk with its Id as name
            # np.savetxt(f"out/clusters/{clusterID}.cluster", self.clusters[clusterID], fmt="%d")

           
    def load(self, filename=None):
        if filename is None:
            filename = self.index_file
            
        if not self.is_loaded:
            print("loading IVF index")  
            self.is_loaded = True
            self.centroids = np.loadtxt(filename)
            
        self.pq_index.load()
            
    
    def _cosine_similarity(self, X, y):
        dot_product = X @ y.T 
        norm_vec1 = np.linalg.norm(X, axis=1)
        norm_vec2 = np.linalg.norm(y)
        norm = norm_vec1 * norm_vec2
        distances = dot_product / norm
        
        return distances