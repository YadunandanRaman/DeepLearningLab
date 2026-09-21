import numpy as np
import tensorflow as tf

# Section 8's numerical exercise. Computed two ways: directly from the
# formula, and again using an actual Keras SimpleRNN layer with its
# weights set to match exactly, so the two can be compared rather than
# trusting the formula alone.
x1, x2, x3 = 0.5, 0.7, 0.2
h0 = 0.0
Wx, Wh, b = 0.5, 0.8, 0.1

print("=== computed directly from h_t = tanh(Wx*x_t + Wh*h_(t-1) + b) ===")
h1 = np.tanh(Wx * x1 + Wh * h0 + b)
h2 = np.tanh(Wx * x2 + Wh * h1 + b)
h3 = np.tanh(Wx * x3 + Wh * h2 + b)
print(f"h1 = tanh({Wx}*{x1} + {Wh}*{h0} + {b}) = {h1:.6f}")
print(f"h2 = tanh({Wx}*{x2} + {Wh}*{h1:.6f} + {b}) = {h2:.6f}")
print(f"h3 = tanh({Wx}*{x3} + {Wh}*{h2:.6f} + {b}) = {h3:.6f}")

print("\n=== same computation, using an actual Keras SimpleRNN layer ===")
rnn_layer = tf.keras.layers.SimpleRNN(1, activation="tanh", return_sequences=True, use_bias=True)
inputs = tf.keras.Input(shape=(3, 1))
outputs = rnn_layer(inputs)
model = tf.keras.Model(inputs, outputs)

# kernel is Wx (shape 1x1), recurrent_kernel is Wh (shape 1x1), bias is b (shape 1,)
rnn_layer.set_weights([np.array([[Wx]]), np.array([[Wh]]), np.array([b])])

sequence = np.array([[[x1], [x2], [x3]]])
keras_outputs = model.predict(sequence, verbose=0).flatten()
print(f"h1 = {keras_outputs[0]:.6f}")
print(f"h2 = {keras_outputs[1]:.6f}")
print(f"h3 = {keras_outputs[2]:.6f}")

print("\n=== comparison ===")
manual = np.array([h1, h2, h3])
match = np.allclose(manual, keras_outputs, atol=1e-6)
print(f"manual calculation matches Keras SimpleRNN: {match}")
if not match:
    print("mismatch, difference:", manual - keras_outputs)
