from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import numpy as np
import os

features_folder = 'processed_features'
labels_folder = 'processed_labels'

vectorizer = CountVectorizer()
scaler = StandardScaler(with_mean=False)
svd = TruncatedSVD(n_components=150)
classifier = SGDClassifier()

unique_classes_set = set()

for labels_file in sorted(os.listdir(labels_folder)):
    if labels_file.endswith("_batch_labels.npy"):
        batch_labels = np.load(os.path.join(labels_folder, labels_file), allow_pickle=True)
        unique_classes_set.update(batch_labels)

all_classes = np.array(sorted(unique_classes_set))

def split_batch_features_labels(features, labels, train_ratio=0.8):

    """
    Splits a dataset of features and corresponding labels into training and testing sets.

    :param features: An array or list containing the features.
    :type features: array-like
    :param labels: An array or list containing the corresponding labels.
    :type labels: array-like
    :param train_ratio: The ratio of data to include in the training set (default is 0.8).
    :type train_ratio: float

    :return: A tuple containing four arrays/lists:
        - train_features: The features for the training set.
        - test_features: The features for the testing set.
        - train_labels: The labels for the training set.
        - test_labels: The labels for the testing set.
    :rtype: tuple
    """

    num_train_samples = int(len(features) * train_ratio)
    
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]
    
    train_features = shuffled_features[:num_train_samples]
    test_features = shuffled_features[num_train_samples:]
    train_labels = shuffled_labels[:num_train_samples]
    test_labels = shuffled_labels[num_train_samples:]
    
    return train_features, test_features, train_labels, test_labels

def process_batch(features, labels, vectorizer, scaler, svd, classifier, is_first_batch, all_classes):

     """
    Processes a batch of training data and updates the classifier using incremental learning.

    :param features: The raw feature array for the current batch.
    :type features: array-like
    :param labels: The label array for the current batch.
    :type labels: array-like
    :param vectorizer: An instance of CountVectorizer for text vectorization.
    :type vectorizer: CountVectorizer
    :param scaler: An instance of StandardScaler for feature scaling.
    :type scaler: StandardScaler
    :param svd: An instance of TruncatedSVD for dimensionality reduction.
    :type svd: TruncatedSVD
    :param classifier: An instance of SGDClassifier for incremental learning.
    :type classifier: SGDClassifier
    :param is_first_batch: Flag indicating if this is the first batch of data, triggering fitting of transformers.
    :type is_first_batch: bool
    :param all_classes: An array of all unique class labels in the dataset.
    :type all_classes: array-like

    Each feature in the features array is first converted to a string and then transformed using
    the provided vectorizer, scaler, and SVD transformer. If this is the first batch, the
    transformers will be fit to the data; otherwise, they will be applied using previously learned parameters.
    The classifier is then updated using partial_fit with the transformed features and corresponding labels.

    :returns: None
    """
    
    features_as_strings = [" ".join(map(str, feature)) for feature in features]
    
    if is_first_batch:
        features_transformed = vectorizer.fit_transform(features_as_strings)
        X_scaled = scaler.fit_transform(features_transformed.toarray())
        X_reduced = svd.fit_transform(X_scaled)
    else:
        features_transformed = vectorizer.transform(features_as_strings)
        X_scaled = scaler.transform(features_transformed.toarray())
        X_reduced = svd.transform(X_scaled)
        
    classifier.partial_fit(X_reduced, labels, classes=all_classes)
    
def transform_features(features, vectorizer, scaler, svd):

    """
    Transforms a batch of features using a series of transformers.

    :param features: An array or list of features to be transformed.
    :type features: array-like
    :param vectorizer: An instance of CountVectorizer for text vectorization.
    :type vectorizer: CountVectorizer
    :param scaler: An instance of StandardScaler for feature scaling.
    :type scaler: StandardScaler
    :param svd: An instance of TruncatedSVD for dimensionality reduction.
    :type svd: TruncatedSVD

    :return: The transformed features after applying vectorization, scaling, and dimensionality reduction.
    :rtype: array-like
    """

    features_as_strings = [" ".join(map(str, feature)) for feature in features]
    features_transformed = vectorizer.transform(features_as_strings)
    X_scaled = scaler.transform(features_transformed.toarray())
    X_reduced = svd.transform(X_scaled)
    
    return X_reduced

total_accuracy = 0
batches_processed = 0
is_first_batch = True

for batch_index, features_file in enumerate(sorted(os.listdir(features_folder))):
    
    if features_file.endswith("_batch_features.npy"):
        batch_features = np.load(os.path.join(features_folder, features_file), allow_pickle=True)
        labels_file = features_file.replace('features', 'labels')
        batch_labels = np.load(os.path.join(labels_folder, labels_file), allow_pickle=True)
        
        print(f'Splitting {batch_index} batch into train and test...')
        train_features, test_features, train_labels, test_labels = split_batch_features_labels(batch_features, batch_labels)
        
        print(f'Training {batch_index} batch...')
        process_batch(train_features, train_labels, vectorizer, scaler, svd, classifier, is_first_batch, all_classes)
        is_first_batch = False
        
        print(f'Testing and predicting {batch_index} batch...')
        test_features_transformed = transform_features(test_features, vectorizer, scaler, svd)
        predictions = classifier.predict(test_features_transformed)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Batch {batch_index} Accuracy: {accuracy:.4f}")

        total_accuracy += accuracy
        batches_processed += 1

average_accuracy = total_accuracy / batches_processed
print(f"Average accuracy across all batches: {average_accuracy:.4f}")

dump(scaler, 'scaler.joblib')
dump(svd, 'svd.joblib')
dump(vectorizer, 'vectorizer.joblib')
dump(classifier, 'sign_language_classifier.joblib')
