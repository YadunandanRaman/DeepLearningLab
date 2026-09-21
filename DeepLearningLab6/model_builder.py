import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, SimpleRNN, LSTM, GRU, Dropout, Dense, GlobalAveragePooling2D,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

RECURRENT_LAYERS = {"rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}


def build_sequence_classifier(recurrent_type="rnn", input_shape=(128, 9), num_classes=6,
                               units=32, dropout_rate=0.2, dense_units=16):
    """
    Section 9/10/11's shared classifier shape: one recurrent layer,
    Dropout, a Dense ReLU layer, then a Softmax output. recurrent_type
    is one of "rnn", "lstm", or "gru", the only thing that changes
    between the three models this experiment compares, exactly as
    Section 12 specifies (replace only the recurrent layer, keep
    everything else the same).
    """
    if recurrent_type not in RECURRENT_LAYERS:
        raise ValueError(f"unknown recurrent_type {recurrent_type!r}, choose from {list(RECURRENT_LAYERS)}")

    layer_class = RECURRENT_LAYERS[recurrent_type]

    inputs = Input(shape=input_shape)
    x = layer_class(units)(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"{recurrent_type}_classifier")


def build_cnn_feature_extractor(image_shape=(224, 224, 3)):
    """
    Section 20's CNN feature extractor: MobileNetV2 with its
    classification head removed, followed by Global Average Pooling,
    frozen throughout, used only to turn a video frame into a fixed
    length feature vector. Never trained, only ever used for inference,
    exactly as the manual specifies ("do not train the CNN from
    scratch, freeze the pretrained CNN and extract features first").

    Returns the model plus its output feature dimension D, needed to
    build the recurrent classifier that consumes its output.
    """
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=image_shape)
    base_model.trainable = False

    inputs = Input(shape=image_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    outputs = GlobalAveragePooling2D()(x)

    feature_dim = outputs.shape[-1]
    model = Model(inputs=inputs, outputs=outputs, name="cnn_feature_extractor")
    return model, feature_dim


def build_video_classifier(recurrent_type="lstm", num_frames=10, feature_dim=1280,
                            num_classes=5, units=32):
    """
    Section 21's CNN-LSTM/CNN-GRU model. This takes already extracted
    CNN features as input, shape (num_frames, feature_dim), not raw
    video frames, since Section 20 extracts those features once with a
    frozen CNN before this model ever trains, rather than running the
    CNN as part of this model's own forward pass.
    """
    if recurrent_type not in RECURRENT_LAYERS:
        raise ValueError(f"unknown recurrent_type {recurrent_type!r}, choose from {list(RECURRENT_LAYERS)}")

    layer_class = RECURRENT_LAYERS[recurrent_type]

    inputs = Input(shape=(num_frames, feature_dim))
    x = layer_class(units)(inputs)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"video_{recurrent_type}")


def build_seq2seq_models(vocab_size, max_seq_len, latent_dim=64):
    """
    Section 22's encoder decoder for the synthetic sequence reversal
    task. vocab_size should include one extra slot for a start of
    sequence token used only in the decoder's input, on top of however
    many distinct integers the sequences are drawn from.

    Uses teacher forcing at training time: the decoder receives the
    true previous output token as input at each step, not its own
    previous prediction, the standard way to train a sequence to
    sequence model, discussed in Discussion Question 22.

    Returns three models:
    - training_model: (encoder_input, decoder_input) -> decoder_output,
      used only for training, with teacher forcing.
    - encoder_model: encoder_input -> encoder states, used at inference
      time to get a starting context for the decoder.
    - decoder_model: (decoder_input_step, decoder_states) -> (output_step,
      decoder_states), used at inference time to generate one token at
      a time without teacher forcing, since the true next token is not
      available at inference time.
    """
    # --- training model, with teacher forcing ---
    encoder_inputs = Input(shape=(max_seq_len,), name="encoder_input")
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
    _, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_seq_len,), name="decoder_input")
    decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, latent_dim, mask_zero=True)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation="softmax", name="decoder_output")
    decoder_outputs = decoder_dense(decoder_outputs)

    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_training")

    # --- inference models, generating one token at a time ---
    encoder_model = Model(encoder_inputs, encoder_states, name="seq2seq_encoder")

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_single_input = Input(shape=(1,))
    decoder_single_embedding = decoder_embedding_layer(decoder_single_input)
    decoder_single_outputs, dec_state_h, dec_state_c = decoder_lstm(
        decoder_single_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    decoder_single_outputs = decoder_dense(decoder_single_outputs)
    decoder_model = Model(
        [decoder_single_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_single_outputs, dec_state_h, dec_state_c],
        name="seq2seq_decoder",
    )

    return training_model, encoder_model, decoder_model
