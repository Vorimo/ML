import tensorflow as tf

import os
import time

from text_generation.model.my_model import MyModel
from text_generation.model.one_step import OneStep


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# todo refactoring
# todo readme
if __name__ == '__main__':
    # Read, then decode for py2 compatibility
    text = open('./datasets/dandelion_wine.txt', 'rb').read().decode(encoding='utf-8')

    # length of text is the number of characters in it
    print(f'Length of text: {len(text)} characters')

    # Take a look at the first 250 characters in text
    print(text[:250])

    # The unique characters in the file
    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids).numpy().decode('utf-8'))
    print('-----')

    seq_length = 100
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
    for seq in sequences.take(1):
        print(chars_from_ids(seq))
    print('-----')
    for seq in sequences.take(5):
        print(text_from_ids(seq).numpy())
    print('-----')

    dataset = sequences.map(split_input_target)
    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(input_example).numpy())
        print("Target:", text_from_ids(target_example).numpy())
    print('-----')

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

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

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    print(model.summary())
    print('-----')

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
    print('-----')
    print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
    print('-----')

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)
    print('-----')

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 40
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    # one_step_model = tf.saved_model.load('./saved_models/one_step')

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

    tf.saved_model.save(one_step_model, './saved_models/one_step')
