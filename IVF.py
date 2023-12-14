'''
IVF implementation
Authors: Khaled Hesham, Kirollos Samy
Date : 12/10/2024
'''
import numpy as np
from index import Index
from indexpq import IndexPQ
from scipy.cluster.vq import kmeans2
import os


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

        self.clusters = []          # Vectors assigned to each cluster
        self.metadata = None        # the number of vectors within each cluster
        
        self.clusters_file = 'out/clusters_all_in_one'
        self.pq_index_file = 'out/centroids'
        
        self.is_loaded_PQ = False
        self.is_loaded_IVF = False
        
            
    def train(self, data: np.ndarray):
        print("Training IVF")
        #TODO In case of more than 1M, we need to train on 1M, then predict the other vectors
        self.centroids, labels = kmeans2(data, self.K, minit='points', iter = 128)

        print("Training PQ")
        _, pqcodes = self.pq_index.train(data)
        print("Saving PQ")
        self.pq_index.save(self.pq_index_file)

        clusters = [None] * self.K

        print("clustering vectors") 
        for clusterID in range(self.K):
            vectorIDs, = np.where(labels==clusterID)
            clusters[clusterID] = np.column_stack((pqcodes[vectorIDs], vectorIDs))
            
        self.clusters = clusters
        
    
    def search(self, q: np.ndarray, top_k: int) -> np.ndarray:  
        print("searching IVF")
        # distances = np.linalg.norm(self.centroids - q, axis=1)  
        # nearest_clusters = np.argpartition(distances, self.nprob)[ :self.nprob]
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
        
        if not self.is_loaded_PQ:
            print("loading PQ index")
            self.is_loaded_PQ = True
            self.pq_index.load(self.pq_index_file)
            
        print("searching PQ index")
        top_indices = self.pq_index.search(q, top_k, pqcodes)
        
        return vectorIDs[top_indices]       
        
        
    def save(self, filename):
        # save clusters file
        print("saving IVF index")
        
        for clusterID in range(self.K):
            # save cluster to disk with its Id as name
            np.savetxt(f"out/clusters/{clusterID}.cluster", self.clusters[clusterID], fmt="%d")
            
        np.savetxt(filename, self.centroids)

           
    def load(self, filename):
        if not self.is_loaded_IVF:
            print("loading IVF index")  
            self.is_loaded_IVF = True
            self.centroids = np.loadtxt(filename)
    
    def _cosine_similarity(self, X, y):
        dot_product = X @ y.T 
        norm_vec1 = np.linalg.norm(X, axis=1)
        norm_vec2 = np.linalg.norm(y)
        norm = norm_vec1 * norm_vec2
        distances = dot_product / norm
        
        return distances