import tensorflow as tf
from text_generation.data_utils import read_data, generate_dataset

from text_generation.model_utils import generate_model, train_model, do_prediction

# Switch to False if you want to do a new training (could take > 1 hour)
load_saved_model = True

if __name__ == '__main__':
    text, vocab = read_data('./datasets/dandelion_wine.txt')

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    dataset = generate_dataset(text, ids_from_chars)

    model = generate_model(ids_from_chars)

    if not load_saved_model:
        history = train_model(model, dataset)

    do_prediction(model, chars_from_ids, ids_from_chars, load_saved_model)
