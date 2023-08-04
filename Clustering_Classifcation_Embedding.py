from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def sentence_transformer(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).replace("', '", " ")
        else:
            text_array.loc[i] = ""
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    embeddings = model.encode(text_array)
    print(embeddings)
    return embeddings


tokens_with_lemmatizing_csv = pd.read_csv('./three_tokens_with_lemmatizing.csv')

extracted_vectors = sentence_transformer(tokens_with_lemmatizing_csv['Outcome_Description'])

np.save('./extracted_vectors.npy', extracted_vectors)

# extracted_key_tokens_dataframe = pd.DataFrame(extracted_vectors)
# extracted_key_tokens_dataframe.to_csv('./five_vectors.csv', index=False)