import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def read_data():
    all_vectors = np.load("all_vectors.npy")
    all_names = np.load("all_names.npy")
    return all_vectors, all_names


def find_similar_characters(query_embedding, all_embeddings, n_results=5):
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_indices = np.argsort(similarities)[-2::-1][:n_results]
    return top_indices


all_vectors, all_names = read_data()

_, fcol2, _ = st.columns(3)

scol1, scol2 = st.columns(2)

change = scol1.button("Start / Change")
find_similar = scol2.button("Find similar images")

if change:
    random_name = all_names[np.random.randint(len(all_names))]
    fcol2.image(Image.open("./images/" + random_name))
    st.session_state["disp_img"] = random_name
    st.write(st.session_state["disp_img"])

if find_similar:
    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    idx = int(np.argwhere(all_names == st.session_state["disp_img"]))
    target_vector = all_vectors[idx]
    fcol2.image(Image.open("./images/" + st.session_state["disp_img"]))
    top5 = find_similar_characters(target_vector, all_vectors)
    col_1.image(Image.open("./images/" + all_names[top5][0]))
    col_2.image(Image.open("./images/" + all_names[top5][1]))
    col_3.image(Image.open("./images/" + all_names[top5][2]))
    col_4.image(Image.open("./images/" + all_names[top5][3]))
    col_5.image(Image.open("./images/" + all_names[top5][4]))
