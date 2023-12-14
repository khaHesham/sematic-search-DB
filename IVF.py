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

        self.clusters = []          # Vectors assigned to each cluster
        self.metadata = None        # the number of vectors within each cluster
        
        self.clusters_file = 'out/clusters'
        self.pq_index_file = 'out/centroids'

            
    def train(self, data: np.ndarray):
        print("Training IVF")
        #TODO In case of more than 1M, we need to train on 1M, then predict the other vectors
        self.centroids, labels = kmeans2(data, self.K, minit='points', iter = 128)

        '''
        Steps:
            1- Assign each vector to its cluster
            2- store cluster assignments in self.cluster_assignments
            3- store vectors assigned to each cluster in disk 
            4- store centroids in disk
            
        '''
        print("Training PQ")
        _, pqcodes = self.pq_index.train(data)
        print("Saving PQ")
        self.pq_index.save(self.pq_index_file)

        clusters = [None] * self.K


        print("clustering vectors") 
        for clusterID in range(self.K):
            vectorIDs, = np.where(labels==clusterID)
            clusters[clusterID] = np.column_stack((pqcodes[vectorIDs], vectorIDs))
            
        self.metadata = np.array([len(clusters[i]) for i in range(self.K)])
        self.clusters = np.concatenate(clusters)
        
        
    
    def search(self, q: np.ndarray, top_k: int) -> np.ndarray:  
        print("searching IVF")
        # distances = np.linalg.norm(self.centroids - q, axis=1)  
        # nearest_clusters = np.argpartition(distances, self.nprob)[ :self.nprob]
        q = q.reshape((70,))
        
        dot_product = self.centroids @ q.T # (K, M) * (M, 1) -> (K, 1)
        norm_vec1 = np.linalg.norm(self.centroids, axis=1) # (K, 1)
        norm_vec2 = np.linalg.norm(q)
        norm = norm_vec1 * norm_vec2
        distances = dot_product / norm
        
        nearest_clusters = np.argsort(distances)[-self.nprob:]
            
        candidates = np.empty((0, self.pq_index.M + 1))
        for cluster in nearest_clusters:
            skip_rows, max_rows = self._magic_seek(cluster, self.metadata)
            loaded_cluster = np.loadtxt(self.clusters_file, skiprows=skip_rows, max_rows=max_rows, dtype=int)
            candidates = np.append(candidates,loaded_cluster,axis = 0)
            
            
        candidates = candidates.astype(int)
        vectorIDs = candidates[:,-1]
        pqcodes = candidates[:,:-1] 
        
        print("loading PQ index")
        self.pq_index.load(self.pq_index_file)
        print("searching PQ index")
        top_indices = self.pq_index.search(q, top_k, pqcodes)
        
        return vectorIDs[top_indices]       
        
        
    def save(self, filename):
        # save clusters file
        print("saving IVF index")
        np.savetxt(self.clusters_file, self.clusters, fmt="%d")

        # save index file (centroids + metadata)
        index_data = np.column_stack((self.centroids, self.metadata))  
        np.savetxt(filename, index_data)

           
    def load(self, filename):
        print("loading IVF index")
        index_data = np.loadtxt(filename)
        self.metadata = index_data[:,-1].astype(int)
        self.centroids = index_data[:,:-1]
        

        
        # skip_rows, max_rows = magic_seek(3, metadata)
        # loaded_cluster = np.loadtxt("file.kiro", skiprows=skip_rows, max_rows=max_rows, dtype=int)
        
    def _magic_seek(self, cluster_id, metadata):
        skip_rows = np.sum(metadata[:cluster_id])
        max_rows = metadata[cluster_id]
        return skip_rows, max_rows
    