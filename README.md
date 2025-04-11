## Project Overview
This repository contains a Jupyter Notebook (dlmodelnew.ipynb) that demonstrates the development and evaluation of a deep learning model for multi-class classification using a dataset related to network traffic analysis. The notebook includes data preprocessing, feature engineering, model training, and performance evaluation.

Features
- Data Preprocessing:
  - Handles missing values and outliers.
  - Encodes categorical labels into numerical formats.
  - Balances the dataset using under-sampling and over-sampling techniques (e.g., SMOTE).

- Feature Engineering:
  - Removes unnecessary columns.
  - Scales features using StandardScaler.
    
- Deep Learning Model:
  - Implements a Sequential model using Keras.
  - Includes dense layers with ReLU and softmax activation functions.
  - Trains the model with categorical crossentropy loss and Adam optimizer.
  
- Evaluation Metrics:
  - Accuracy, F1-score, precision, recall.
  - Confusion matrix visualization.
 
## Installation
1. Clone the repository:
  ```bash
  git clone <repository-url>
  cd <repository-folder>
  ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Ensure the dataset (dataset.csv) is placed in the appropriate directory as specified in the notebook.

## Usage
1. Open the notebook:
  ```bash
  jupyter notebook dlmodelnew.ipynb
  ```

2. Execute cells sequentially to preprocess data, train the model, and evaluate its performance.

## Dataset
The notebook uses a dataset containing network traffic data with features such as Flow Duration, Total Fwd Packets, Idle Mean, etc., along with labels like BENIGN, DoS, and PortScan. The dataset has 1,042,557 rows and 79 columns.

### Key Code Sections
- Data Preprocessing:
  ```python
  df = pd.read_csv('/path/to/dataset.csv')
  df = df.dropna(axis=0, how='any')
  df = df.replace([np.inf, -np.inf], np.nan)
  ```

- Model Definition:
  ```python
  classifier = Sequential()
  classifier.add(Dense(42, activation='relu', input_dim=70))
  classifier.add(Dense(42, activation='relu'))
  classifier.add(Dense(4, activation='softmax'))
  classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  ```
  
- Data Preprocessing:
  ```python
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_test, y_pred)
  print('Accuracy:', accuracy)
  ```

### Results
The trained model achieves high accuracy on the test set. Evaluation metrics such as F1-score and precision are computed to measure its performance across different classes.

### Visualization
The notebook includes plots for:
- Model accuracy and loss during training.
- Confusion matrix for multi-class classification.

### Requirements
- Python (>=3.8)
- Libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - Scikit-learn
  - Imbalanced-learn
  - TensorFlow/Keras
