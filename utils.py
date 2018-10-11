import numpy as np
import os
from parser.run_parser import run_parser
from sklearn.model_selection import train_test_split
from bidict import bidict


personal_categories_dict = bidict({
    1: 'name',
    2: 'id-number',
    3: 'location',
    4: 'online-id',
    5: 'dob',
    6: 'phone',
    7: 'psychological',
    8: 'professional',
    9: 'genetic',
    10: 'mental',
    11: 'economic',
    12: 'cultural',
    13: 'social',
    14: 'physiological',
})

sensitive_categories_dict = bidict({
    15: 'criminal',
    16: 'origin',
    17: 'health',
    18: 'religion',
    19: 'political',
    20: 'philosophical',
    21: 'unions',
    22: 'sex-life',
    23: 'sex-orientation',
    24: 'biometric',
})

all_categories_dict = bidict({-1: 'other_personal'})
all_categories_dict.putall(personal_categories_dict)
all_categories_dict.putall(sensitive_categories_dict)


def target_to_string(target):
    if target == 2:
        return "sensitive"
    if target == 1:
        return "personal"
    if target == 0:
        return "nonpersonal"


def target_to_string_categories(target):
    return all_categories_dict[target]


def load_dirs_custom(directories, individual=False):
    all_documents = []
    for d in directories:
        all_documents += load_dir_custom(d, individual)
    return all_documents


def document_test_train_split(documents, test_size):
    document_labels = get_documents_labels(documents)
    doc_train, doc_test, _, _ = train_test_split(
        documents, document_labels, test_size=test_size, shuffle=True
    )
    return doc_train, doc_test


def load_dir_custom(directory, individual=False):
    documents = read_dir(directory)
    fill_docs(documents, individual)
    return documents


def fill_docs(documents, individual=False):
    for doc in documents:
        data = np.array([])
        targets = np.array([])
        contexts = np.array([])
        full_text = ''
        for line in doc.lines:
            data = np.append(data, [line.text])
            full_text += line.text
            targets = np.append(
                targets,
                [convert_categories(line.categories, individual)]
            )
            contexts = np.append(
                contexts,
                [convert_categories(line.context, individual)]
            )
        #  import ipdb; ipdb.set_trace()
        doc.data = data
        doc.text = full_text
        doc.contexts = contexts
        doc.targets = targets
        doc.category = classify_doc(targets)


def classify_doc(target_array):
    if len(target_array) == 0:
        return 0
    return max(target_array)


def convert_categories(categories, individual):
    if individual:
        return convert_categories_individual(categories)
    else:
        return convert_categories_buckets(categories)


def convert_categories_individual(categories):
    #  category_list = ['name', 'phone', 'professional']
    category_list = list(all_categories_dict.inv.keys())
    for c in categories:
        if c in category_list:
            return all_categories_dict.inv[c]
        else:
            return -1

    return 0


def convert_categories_buckets(categories):
    for c in sensitive_categories_dict.inv:
        if c in categories:
            return 2
    for c in personal_categories_dict.inv:
        if c in categories:
            return 1
    return 0


def read_dir(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            lines = run_parser(full_path)
            doc = Document(full_path, lines)
            documents.append(doc)

    return documents


def get_documents_labels(documents):
    targets = []
    for doc in documents:
        targets.append(doc.category)
    return targets


def convert_docs_to_lines(documents, context=False):
    targets = np.array([])
    data = np.array([])
    contexts = np.array([])
    for doc in documents:
        targets = np.append(targets, doc.targets)
        data = np.append(data, doc.data)
        contexts = np.append(contexts, doc.contexts)

    if context:
        return data, targets, contexts
    else:
        return data, targets


def n_gram_documents_range(docs, low, high):
    for d in docs:
        n_gram_document_range(d, low, high)
    return docs


def n_gram_document_range(doc, low, high):
    all_data = np.array([])
    all_target = np.array([])
    for i in range(low, high+1):
        data, targets = n_grams(doc.data, doc.targets, i)
        all_data = np.append(all_data, data)
        all_target = np.append(all_target, targets)

    doc.data = all_data
    doc.targets = all_target


def n_gram_documents(docs, n):
    return n_gram_documents_range(docs, n, n)


def n_grams(data_array, target_array, n):
    grams = np.array([])
    targets = np.array([])
    if len(data_array) == 0:
        return grams, targets
        
    if len(data_array) <= n:
        grams = np.append(grams, data_array)
        targets = np.append(targets, [max(target_array)])
        return grams, targets

    for i in range(len(data_array) - n + 1):
        new_str = '\n'.join(data_array[i:i+n])
        grams = np.append(grams, [new_str])
        targets = np.append(targets, [max(target_array[i:i+n])])
    return grams, targets


def label_documents_dir(docpath, clf):
    documents = read_dir(docpath)
    fill_docs(documents)
    for doc in documents:
        label_single_document(doc, clf)


def label_single_document(doc, clf):
        predicted_lines = clf.predict(doc.data)

        new_file_name = "AUTOLABELED/{}_automagic.txt".format(
            os.path.basename(doc.path)
        )
        out_doc = open(new_file_name, "a")
        for i in range(len(predicted_lines)):
            format_string = "{}\n".format(doc.data[i])
            if predicted_lines[i] != 0:
                format_string = "{}\t:::::\t{}\n".format(
                    target_to_string_categories(predicted_lines[i]),
                    doc.data[i],
                )
            out_doc.write(format_string)
        out_doc.close()


class Document:
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines
        self.text = ''
        self.data = np.array([])
        self.targets = np.array([])
        self.contexts = np.array([])
        self.category = -1
