import tensorflow as tf
from rnn.utils import split_input_target


def read_data(file_name):
    # Read, then decode for py2 compatibility
    text = open(file_name, 'rb').read().decode(encoding='utf-8')

    # length of text is the number of characters in it
    print(f'Length of text: {len(text)} characters')

    # Take a look at the first 250 characters in text
    print('First 250 characters of the text:\n' + text[:250])

    # The unique characters in the file
    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')
    return text, vocab


def generate_dataset(text, ids_from_chars):
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    # Batch size
    batch_size = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    buffer_size = 10000

    dataset = (
        dataset
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset
