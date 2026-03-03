import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')

# Data
TRAIN_FILE = os.path.join(DATA_DIR, 'application_train', 'application_train.csv')
BUREAU_FILE = os.path.join(DATA_DIR, 'Bureau data', 'bureau.csv')
previous_application_file= os.path.join(DATA_DIR, 'previous_application', 'previous_application.csv')
installments_payments_file=os.path.join(ROOT_DIR, 'notebooks', 'installments_payments', 'installments_payments.csv')
ID_COL = 'SK_ID_CURR'
TARGET_COL = 'TARGET'

# Model
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_TRIALS = 50  # Optuna trials