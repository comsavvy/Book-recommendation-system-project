import numpy as np
import pandas as pd
import tensorflow as tf


def load_data(file_path):
    df = pd.read_csv(file_path, sep=";", names=["user_id", "book_id", "rating"])
    return df


def build_model(num_users, num_books, embedding_dim):
    user_input = tf.keras.layers.Input(shape=(1,))
    user_embedding = tf.keras.layers.Embedding(
        num_users, embedding_dim, name="user_embedding"
    )(user_input)
    user_embedding = tf.keras.layers.Flatten()(user_embedding)
    book_input = tf.keras.layers.Input(shape=(1,))
    book_embedding = tf.keras.layers.Embedding(
        num_books, embedding_dim, name="book_embedding"
    )(book_input)
    book_embedding = tf.keras.layers.Flatten()(book_embedding)
    dot = tf.keras.layers.Dot(axes=1)([user_embedding, book_embedding])
    model = tf.keras.Model(inputs=[user_input, book_input], outputs=dot)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
