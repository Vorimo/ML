import tensorflow as tf
import os
from rnn.model.my_model import MyModel
import time
from rnn.model.one_step import OneStep


def generate_model(ids_from_chars):
    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = MyModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    return model


def train_model(model, dataset):
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    epochs = 40
    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    return history


def do_prediction(model, chars_from_ids, ids_from_chars, load_saved_model):
    if load_saved_model:
        one_step_model = tf.saved_model.load('./saved_models/one_step')
    else:
        one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    # prediction
    start = time.time()
    states = None
    next_char = tf.constant(['Billy:'])
    result = [next_char]

    # generation of 1000 symbols
    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print('result:')
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)
    print('-----')
    if not load_saved_model:
        tf.saved_model.save(one_step_model, './saved_models/one_step')
