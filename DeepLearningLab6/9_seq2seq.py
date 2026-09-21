import numpy as np
import pandas as pd

from model_builder import build_seq2seq_models

# Synthetic reversal task: [1,4,7,2] -> [2,7,4,1]. Sequences are drawn
# from digits 1 to 9, index 0 is reserved for the decoder's start of
# sequence token (used only as the very first decoder input, at
# inference time as well as during teacher forced training), so it
# never appears as an actual sequence value.
SEQUENCE_LENGTH = 4
DIGIT_RANGE = (1, 10)  # 1 to 9 inclusive
START_TOKEN = 0
VOCAB_SIZE = 10  # 0 = start token, 1 to 9 = digit values

NUM_SEQUENCES = 5000
LATENT_DIM = 64
EPOCHS = 40
BATCH_SIZE = 64

rng = np.random.default_rng(42)
X_all = rng.integers(DIGIT_RANGE[0], DIGIT_RANGE[1], size=(NUM_SEQUENCES, SEQUENCE_LENGTH))
Y_all = X_all[:, ::-1]

# 70/15/15, matching this experiment's other splits
n_total = NUM_SEQUENCES
n_test = int(n_total * 0.15)
n_val = int(n_total * 0.15)
perm = rng.permutation(n_total)
test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

X_train, Y_train = X_all[train_idx], Y_all[train_idx]
X_val, Y_val = X_all[val_idx], Y_all[val_idx]
X_test, Y_test = X_all[test_idx], Y_all[test_idx]
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")


def make_decoder_input(Y):
    # teacher forcing: the decoder's input at each step is the true
    # previous output token, not its own prediction, so its input
    # sequence is the target shifted right by one, with the start
    # token prepended
    return np.concatenate([np.full((len(Y), 1), START_TOKEN), Y[:, :-1]], axis=1)


decoder_input_train = make_decoder_input(Y_train)
decoder_input_val = make_decoder_input(Y_val)

training_model, encoder_model, decoder_model = build_seq2seq_models(
    VOCAB_SIZE, SEQUENCE_LENGTH, latent_dim=LATENT_DIM
)
training_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
total_params = training_model.count_params()
print(f"seq2seq total parameters: {total_params:,}")

history = training_model.fit(
    [X_train, decoder_input_train], Y_train[..., np.newaxis],
    validation_data=([X_val, decoder_input_val], Y_val[..., np.newaxis]),
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
)

final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]
print(f"\nfinal training loss: {final_train_loss:.4f}, final validation loss: {final_val_loss:.4f}")


def predict_sequence(input_seq):
    """
    Generates one output token at a time using the decoder model, no
    teacher forcing, only the model's own previous prediction feeds
    the next step, matching how the model is actually used once there
    is no ground truth target to peek at.
    """
    h, c = encoder_model.predict(input_seq[np.newaxis, :], verbose=0)
    token = np.array([[START_TOKEN]])
    output_tokens = []
    for _ in range(SEQUENCE_LENGTH):
        probs, h, c = decoder_model.predict([token, h, c], verbose=0)
        next_token = int(np.argmax(probs[0, 0]))
        output_tokens.append(next_token)
        token = np.array([[next_token]])
    return output_tokens


print("\nrunning inference on the test set, one token at a time, no teacher forcing")
correct_tokens, total_tokens, correct_sequences = 0, 0, 0
example_rows = []
for i in range(len(X_test)):
    predicted = predict_sequence(X_test[i])
    true = list(Y_test[i])
    correct_tokens += sum(p == t for p, t in zip(predicted, true))
    total_tokens += SEQUENCE_LENGTH
    correct_sequences += int(predicted == true)
    if i < 5:
        example_rows.append({
            "sample": i + 1,
            "input_sequence": list(X_test[i]),
            "predicted_output": predicted,
        })

token_accuracy = correct_tokens / total_tokens
sequence_accuracy = correct_sequences / len(X_test)
print(f"\ntoken accuracy: {token_accuracy:.4f}")
print(f"sequence accuracy: {sequence_accuracy:.4f}")

print("\nfirst 5 test examples:")
for row in example_rows:
    print(f"  sample {row['sample']}: input={row['input_sequence']} -> predicted={row['predicted_output']}")

pd.DataFrame(example_rows).to_csv("outputs/results/seq2seq_examples.csv", index=False)
pd.DataFrame([{
    "token_accuracy": token_accuracy,
    "sequence_accuracy": sequence_accuracy,
    "training_loss": final_train_loss,
    "validation_loss": final_val_loss,
    "parameters": total_params,
}]).to_csv("outputs/results/seq2seq_metrics.csv", index=False)

print("\nsaved seq2seq_examples.csv and seq2seq_metrics.csv to outputs/results")
