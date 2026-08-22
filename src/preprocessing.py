import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

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

def load_data(train_path='data/KDDTrain+.txt', test_path='data/KDDTest+.txt'):
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=COLUMNS)
    return train_df, test_df

def preprocess_multiclass(train_df, test_df, apply_smote=True):
    train_df = train_df.drop(columns=['difficulty_level'])
    test_df = test_df.drop(columns=['difficulty_level'])

    y_train = train_df['label'].map(lambda x: ATTACK_MAP.get(x, 'Other'))
    y_test = test_df['label'].map(lambda x: ATTACK_MAP.get(x, 'Other'))

    X_train = train_df.drop(columns=['label'])
    X_test = test_df.drop(columns=['label'])

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Oversample minority classes (R2L, U2R) on training set using SMOTE
    if apply_smote:
        print("Applying SMOTE to rebalance minority attack classes...")
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test