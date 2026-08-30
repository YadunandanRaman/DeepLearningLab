from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


def build_baseline_model():
    """
    The fixed architecture given in the lab manual (Section 3 / Task 3):
    784 -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, Softmax)
    """
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    return model


def build_search_model(hidden_layers=2, hidden_neurons=128, activation="relu",
                        dropout_rate=0.0, optimizer_name="adam", learning_rate=0.001):
    """
    A variable depth MLP used only by the hyperparameter search
    (6_hyperparameter_search.py). Every hyperparameter here corresponds to
    a row in the search space table in Section 7 of the lab manual.

    Uses sparse_categorical_crossentropy (integer labels) rather than
    categorical_crossentropy, since scikit-learn's search tools expect a
    1D target vector, not one-hot vectors. The baseline model above still
    uses one-hot labels and categorical_crossentropy exactly as the
    manual specifies for Task 4.
    """
    model = Sequential()
    model.add(Input(shape=(784,)))

    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation="softmax"))

    if optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
