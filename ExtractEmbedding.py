import numpy as np
import os
from sklearn.decomposition import PCA

embedding_folder = './embeddings/'
all_embeddings = []

for f in os.listdir(embedding_folder):
    if f.endswith('.npy'):
        emb = np.load(os.path.join(embedding_folder, f))
        # Flatten time dimension if present, e.g. (time_steps, 128) -> (time_steps*128)
        emb_flat = emb.flatten()
        all_embeddings.append(emb_flat)

all_embeddings = np.stack(all_embeddings)

# Fit PCA to reduce dimensionality to 16
pca = PCA(n_components=16)
pca.fit(all_embeddings)

# Save PCA for later use
import joblib
joblib.dump(pca, 'pca_vggish_to_16d.pkl')
