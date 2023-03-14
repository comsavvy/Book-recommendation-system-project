import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf


class RecommendationSystem:
    def __init__(self, ratings, books, users) -> None:
        self.ratings = ratings
        self.books = books
        self.users = users

    def build_model(self, embedding_dim: int = 10):
        # The total number of the unique users and books
        num_users = self.users["User-ID"].nunique()
        num_books = self.books["ISBN"].nunique()

        # Add an input layer for the user IDs with a shape of (1,), which means that it accepts a single integer as input
        user_input = tf.keras.layers.Input(shape=(1,))
        # Create an embedding layer for the user id
        user_embedding = tf.keras.layers.Embedding(
            input_dim=num_users, output_dim=embedding_dim, name="user_embedding"
        )(user_input)

        # Represent the embedding in 1-dimensional array tensor
        user_embedding = tf.keras.layers.Flatten()(user_embedding)
        # Add an input layer for the book IDs with a shape of (1,), which means that it accepts a single integer as input
        book_input = tf.keras.layers.Input(shape=(1,))
        # Create an embedding layer for the book id
        book_embedding = tf.keras.layers.Embedding(
            input_dim=num_books, output_dim=embedding_dim, name="book_embedding"
        )(book_input)
        # Represent the embedding in 1-dimensional array tensor
        book_embedding = tf.keras.layers.Flatten()(book_embedding)
        # Calculates the dot product of the embeddings to produce a single output value representing the predicted rating of the user for the book
        dot = tf.keras.layers.Dot(axes=1)([user_embedding, book_embedding])
        # Create the model structure for the input and output
        self.model = tf.keras.Model(inputs=[user_input, book_input], outputs=dot)
        # Compile the model with Adam optimizer and the mean squared error loss function.
        self.model.compile(optimizer="adam", loss="mean_squared_error")

        return self.model

    def fit_model(self, epoch=10):
        # Create train and test validation set
        train_data, test_data = train_test_split(
            self.ratings, test_size=0.12, random_state=42
        )
        # Fit and validate the model
        self.model.fit(
            [train_data["User-ID"], train_data["ISBN"]],
            train_data["Book-Rating"],
            validation_data=(
                [test_data["User-ID"].values, test_data["ISBN"].values],
                test_data["Book-Rating"].values,
            ),
            epochs=epoch,
        )

    def build_fit_model(self, epoch=10):
        self.build_model()
        self.fit_model(epoch)

    def recommend_books(self, user_id, num_recommendations=10):
        # Get the user embedding weight to calculate the accuracy score of a prediction
        user_embedding = self.model.get_layer(name="user_embedding").get_weights()[0][
            user_id
        ]
        # Calculate the accuracy score of a user for an item
        scores = np.dot(
            user_embedding,
            self.model.get_layer(name="book_embedding").get_weights()[0].T,
        )
        # Sort the score descendingly and return the position of each score
        self.book_indexes = np.argsort(scores)[::-1]

        # Display the recommended books
        for i in range(num_recommendations):
            book_id = self.book_indexes[i]
            # Extract the book details for the book id
            book_title = self.books.loc[
                self.books["ISBN"] == book_id, "Book-Title"
            ].values[0]
            # score = scores[book_id]
            print(
                f"Recommended book {i + 1}: {book_title !r} <====> Score: {scores[book_id]}"
            )

    def recommended_books_table(self, num_recommendation=10):
        return self.books.loc[
            self.books["ISBN"].isin(self.book_indexes[:num_recommendation]), :
        ]


if __name__ == "__main__":
    # Load the book ratings dataset
    ratings = pd.read_csv("Data/BX-Book-Ratings.csv", sep=";", low_memory=False)
    books = pd.read_csv(
        "Data/BX-Books.csv", sep=";", on_bad_lines="skip", low_memory=False
    )
    users = pd.read_csv(
        "Data/BX-Users.csv", sep=";", on_bad_lines="skip", low_memory=False
    )

    ratings = ratings[ratings["ISBN"].isin(books["ISBN"].values)]
    ratings = ratings[ratings["User-ID"].isin(users["User-ID"].values)]

    # Transform both the user and the book IDs
    isbn_transformer = LabelEncoder().fit(books["ISBN"])
    books["ISBN"] = isbn_transformer.transform(books["ISBN"])
    ratings["ISBN"] = isbn_transformer.transform(ratings["ISBN"])
    userid_transformer = LabelEncoder().fit(users["User-ID"])
    users["User-ID"] = userid_transformer.transform(users["User-ID"])
    ratings["User-ID"] = userid_transformer.transform(ratings["User-ID"])

    model = RecommendationSystem(ratings, books, users)

    model.build_fit_model()

    # Recommend books to the user
    model.recommend_books(user_id=0)
