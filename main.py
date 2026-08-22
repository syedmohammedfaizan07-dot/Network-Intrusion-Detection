import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from src.preprocessing import load_data

LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
ATTACK_MAP = {
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'udpstorm': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguest': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
    'normal': 'Normal'
}


def prepare_data_with_smote():
    train_df, test_df = load_data('data/KDDTrain+.txt', 'data/KDDTest+.txt')

    train_df = train_df.drop(columns=['difficulty_level'])
    test_df = test_df.drop(columns=['difficulty_level'])

    y_train = train_df['label'].map(lambda x: ATTACK_MAP.get(x, 'Other'))
    y_test = test_df['label'].map(lambda x: ATTACK_MAP.get(x, 'Other'))

    X_train = train_df.drop(columns=['label'])
    X_test = test_df.drop(columns=['label'])

    X_train = pd.get_dummies(X_train, columns=['protocol_type', 'service', 'flag'])
    X_test = pd.get_dummies(X_test, columns=['protocol_type', 'service', 'flag'])

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Apply SMOTE to rebalance training data fast
    print("Applying SMOTE oversampling to training set...")
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test


def main():
    print("Loading data & balancing classes with SMOTE...")
    X_train, X_test, y_train, y_test = prepare_data_with_smote()

    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = clf.predict(X_test)

    print(f"\nMulti-Class Test Accuracy (with SMOTE): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title('SMOTE-Balanced Intrusion Detection Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("\nConfusion matrix saved as 'confusion_matrix.png'. Execution complete!")


if __name__ == '__main__':
    main()