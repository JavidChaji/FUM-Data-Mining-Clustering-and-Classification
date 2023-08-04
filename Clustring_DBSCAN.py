from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def dbscan(vectors):
    # for i in range(len(text_array)):
    #     if not pd.isna(text_array.loc[i]):
    #         text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).replace("', '", " ")
    #     else:
    #         text_array.loc[i] = ""
    # model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # embeddings = model.encode(text_array)
    # print(embeddings)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vectors)

    X_normalized = normalize(X_scaled)

    X_normalized = pd.DataFrame(X_normalized)

    pca = PCA()
    X_principal = pca.fit_transform(X_normalized)
    print(X_principal)

    clustering = DBSCAN(eps=0.67, min_samples=3).fit(X_principal)
    print(clustering.labels_)
    return vectors


tokens_with_lemmatizing_csv = pd.read_csv('./three_tokens_with_lemmatizing.csv')

extracted_vectors = np.load('./extracted_vectors.npy')

print(extracted_vectors)

extracted_vectors = dbscan(extracted_vectors)

