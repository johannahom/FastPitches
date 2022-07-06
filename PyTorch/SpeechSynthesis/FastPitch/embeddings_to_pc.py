import sys
import numpy as np

from sklearn.decomposition import PCA
import pickle

np.set_printoptions(suppress=True)

# create and save PCA

embeddings = np.loadtxt(sys.argv[1])
pca = PCA(whiten=False)
pca.fit(embeddings)

# save
with open('embedding_pca.pkl','wb') as f:
    pickle.dump(pca,f)

