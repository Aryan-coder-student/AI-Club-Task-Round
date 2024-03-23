import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("models/model_KNN.pkl", "rb") as f:
    knn_model = pickle.load(f)

st.set_page_config(
    page_title="Task",
    page_icon="👋",
)


st.markdown(
    """
    <h2><u>Predict the iris</u></h2>
    """,
    unsafe_allow_html=True,
)
def main():
    st.title("Iris Flower Predictor")

    st.sidebar.header("Input Features")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)



    # Display the input values
    st.write("### Input Features")
    st.write("- Sepal Length (cm):", sepal_length)
    st.write("- Sepal Width (cm):", sepal_width)
    st.write("- Petal Length (cm):", petal_length)
    st.write("- Petal Width (cm):", petal_width)

    data = [sepal_length, sepal_width, petal_length, petal_width]
    pred = knn_model.predict([data])
    if pred[0] == 0:
        st.markdown("#### Predicated Type is Setosa")
    elif pred[0] == 1:
        st.markdown("#### Predicated Type is Versicolor")
    elif pred[0] == 2:
        st.markdown("#### Predicated Type is Virginica")
    else:
        st.markdow("#### Not Found")
if __name__ == "__main__":
    main()
