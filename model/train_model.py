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
