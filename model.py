import numpy as np
import pandas as pd
import tensorflow as tf


def load_data(file_path):
    # Load the book ratings dataset
    df = pd.read_csv(file_path, sep=";", names=["User-ID", "ISBN", "Book-Rating"])
    return df


def build_model(num_users, num_books, embedding_dim):
    # Add an input layer for the user IDs with a shape of (1,), which means that it accepts a single integer as input
    user_input = tf.keras.layers.Input(shape=(1,))
    # Create an embedding layer for the user id
    user_embedding = tf.keras.layers.Embedding(
        num_users, embedding_dim, name="user_embedding"
    )(user_input)

    # Represent the embedding in 1-dimensional array tensor
    user_embedding = tf.keras.layers.Flatten()(user_embedding)
    # Add an input layer for the book IDs with a shape of (1,), which means that it accepts a single integer as input
    book_input = tf.keras.layers.Input(shape=(1,))
    # Create an embedding layer for the book id
    book_embedding = tf.keras.layers.Embedding(
        num_books, embedding_dim, name="book_embedding"
    )(book_input)
    # Represent the embedding in 1-dimensional array tensor
    book_embedding = tf.keras.layers.Flatten()(book_embedding)
    # Calculates the dot product of the embeddings to produce a single output value representing the predicted rating of the user for the book
    dot = tf.keras.layers.Dot(axes=1)([user_embedding, book_embedding])
    # Create the model structure for the input and output
    model = tf.keras.Model(inputs=[user_input, book_input], outputs=dot)
    # Compile the model with Adam optimizer and the mean squared error loss function.
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def recommend_books(model, user_id, book_titles, num_recommendations=10):
    # Get the user embedding weight to calculate the accuracy score of a prediction
    user_embedding = model.get_layer(name="user_embedding").get_weights()[0][user_id]
    # Calculate the accuracy score of a user for an item
    scores = np.dot(
        user_embedding, model.get_layer(name="book_embedding").get_weights()[0].T
    )
    # Sort the score descendingly and return the position of each score
    book_indexes = np.argsort(scores)[::-1]

    # Display the recommended books
    for i in range(num_recommendations):
        book_id = book_indexes[i]
        book_title = book_titles[book_id]
        score = scores[book_id]
        print("Recommended book {}: {} with score {}".format(i + 1, book_title, score))


if __name__ == "__main__":
    # Load the book ratings dataset
    df = load_data("Data/BX-Book-Ratings.csv")
    # Get the unique user id
    num_users = df["User-ID"].nunique()
    # Get the unique book_id
    num_books = df["ISBN"].nunique()

    # Dimensionality of the embedding space used to represent users and books
    embedding_dim = 10

    model = build_model(num_users, num_books, embedding_dim)
    model.fit(
        [df["User-ID"].values, df["ISBN"].values], df["Book-Rating"].values, epochs=10
    )

    book_titles = np.loadtxt(
        "Data/BX-Books.csv", dtype=str, delimiter=";", usecols=(1,), unpack=True
    )

    recommend_books(model, user_id=0, book_titles=book_titles)
