import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim(glove: dict, review: str) -> list:
    # Should do this: "hello test" -> ["hello", "test"] if both are in gloVe
    words = review.split()
    return [word for word in words if word in glove]

def average_embedding(gloVe: dict, data: np.ndarray) -> np.ndarray:

    transformed_data = []
    for label, review in data:
        words = trim(gloVe, review)
        if not words:  # Skip reviews with no words in GloVe
            continue

        sum_vector = np.zeros(VECTOR_LEN)
        for word in words:
            sum_vector += gloVe[word]
        
        avg_vector = sum_vector / len(words)
        transformed_data.append((label, avg_vector))

    return np.array(transformed_data, dtype=object)

def write_to_file(filename: str, data: np.ndarray) -> None:

    with open(filename, 'w', encoding='utf-8') as file:
        for label, avg_vector in data:
            # Convert the label to a string and the avg_vector to a list of strings
            label_str = f"{label:.6f}"
            vector_strs = [f"{v:.6f}" for v in avg_vector]

            # Join the label and the vector components with tabs
            vector_strs.insert(0, label_str)
            line = '\t'.join(vector_strs) 
            file.write(line + '\n')


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    gloVe = load_feature_dictionary(args.feature_dictionary_in)
    train_data = load_tsv_dataset(args.train_input)
    validation_data = load_tsv_dataset(args.validation_input)
    test_data = load_tsv_dataset(args.test_input)

    processed_train_data = average_embedding(gloVe, train_data)
    processed_validation_data = average_embedding(gloVe, validation_data)
    processed_test_data = average_embedding(gloVe, test_data)

    write_to_file(args.train_out, processed_train_data)
    write_to_file(args.validation_out, processed_validation_data)
    write_to_file(args.test_out, processed_test_data)


    
    # file.write(f'error(train): {train_error:.6f}\n')