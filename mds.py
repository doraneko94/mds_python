import numpy as np

class MDS:

    n_components = 2
    dissimilarity = "euclidean"
    embedding_ = None
    dissimilarity_matrix_ = None

    def __init__(self, n_components:int=2, dissimilarity="euclidean"):

        self.n_components = n_components
        if not dissimilarity in ["euclidean", "cosine", "precomputed"]:
            raise ValueError()
        self.dissimilarity = dissimilarity


    def fit(self, X:np.ndarray):
        
        n = X.shape[0]

        if self.dissimilarity == "euclidean":
            tmp = np.tile(X, (n, 1, 1))
            self.dissimilarity_matrix_ = np.sqrt(np.square(tmp - tmp.transpose((1, 0, 2))).sum(axis=2))
        elif self.dissimilarity == "cosine":
            tmp = np.dot(X, X.T)
            norm = np.sqrt(np.square(X).sum(axis=1))
            self.dissimilarity_matrix_ = 1 - tmp / norm / norm.reshape(-1, 1)
        else:
            self.dissimilarity_matrix_ = X
        
        h = np.eye(n) - np.ones((n, n)) / n
        k = -0.5 * np.dot(np.dot(h, self.dissimilarity_matrix_), h)
        evl, evc = np.linalg.eigh(k)
        evl_asort = evl.argsort()
        eids = evl_asort[::-1][:self.n_components]
        lamb = np.diag(evl[eids])
        v = evc[:, eids]
        self.embedding_ = np.dot(np.sqrt(lamb), v.T).T

    def fit_transform(self, X:np.ndarray):

        self.fit(X)
        return self.embedding_

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris["data"]
    target = iris["target"]
    name = iris["target_names"]
    
    mds = MDS()
    x = mds.fit_transform(data)
    
    x_se = x[target==0]
    x_ve = x[target==1]
    x_vi = x[target==2]

    plt.scatter(x_se[:, 0], x_se[:, 1], label=name[0])
    plt.scatter(x_ve[:, 0], x_ve[:, 1], label=name[1])
    plt.scatter(x_vi[:, 0], x_vi[:, 1], label=name[2])
    plt.legend()
    plt.show()