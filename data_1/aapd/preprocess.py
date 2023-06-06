from bs4 import BeautifulSoup
from topics import topic_num_map
from sklearn.model_selection import train_test_split
import os


def get_binary_label(topics):
    """
    Get a 90-digit binary encoded label corresponding to the given topic list
    :param topics: Set of topics to which an article belongs
    :return: 90-digit binary one-hot encoded label
    """
    category_label = [0 for x in range(len(topic_num_map))]
    for topic in topics:
        if topic.lower() in topic_num_map:
            category_label[topic_num_map[topic.lower()]] = 1
    if sum(category_label) > 0:
        return ''.join(map(str, category_label))
    else:
        print("Label", topics)
        return None


def parse_documents():
    """
    Extract the Reuters-90 dataset from the SGM files in data folder according to the ApteMod splits. This method
    returns the documents that belong to at least one of the categories that have at least one document in both the
    training and the test sets. The dataset has 90 categories with a training set of 7769 documents and a test set of
    3019 documents.
    :return: Two lists containing the train and test splits along with the labels
    """
    train_documents = list()
    with open(os.path.join("data", "text_train")) as text_file, open(os.path.join("data", "label_train")) as topic_file:
        for text, topics in zip(text_file, topic_file):
            if text != "" and topics != "":
                train_documents.append((get_binary_label(topics.strip().split()), text.strip()))

    test_documents = list()
    with open(os.path.join("data", "text_test")) as text_file, open(os.path.join("data", "label_test")) as topic_file:
        for text, topics in zip(text_file, topic_file):
            if text != "" and topics != "":
                test_documents.append((get_binary_label(topics.strip().split()), text.strip()))
    validation_documents = list()
    with open(os.path.join("data", "text_val")) as text_file, open(os.path.join("data", "label_val")) as topic_file:
        for text, topics in zip(text_file, topic_file):
            if text != "" and topics != "":
                validation_documents.append((get_binary_label(topics.strip().split()), text.strip()))
    return train_documents, validation_documents, test_documents


if __name__ == "__main__":
    train_documents, validation_documents, test_documents = parse_documents()
    print("Train, test dataset sizes:", len(train_documents), len(test_documents))
    with open(os.path.join("data", "aapd_train.tsv"), 'w', encoding='utf8') as tsv_file:
        for label, document in train_documents:
            tsv_file.write(label + "\t" + document + "\n")
    with open(os.path.join("data", "aapd_validation.tsv"), 'w', encoding='utf8') as tsv_file:
        for label, document in validation_documents:
            tsv_file.write(label + "\t" + document + "\n")
    with open(os.path.join("data", "aapd_test.tsv"), 'w', encoding='utf8') as tsv_file:
        for label, document in test_documents:
            tsv_file.write(label + "\t" + document + "\n")