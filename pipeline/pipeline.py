import kfp
import kfp.dsl as dsl
from kfp.components import create_component_from_func
import kfp.components as comp

IMAGE = 'salazar99/python-kubeflow:latest'
DATA_URL = 'https://gs-kubeflow-pipelines.nyc3.digitaloceanspaces.com/clean-spam-data.csv'

# Download data
# def download_data(source_path: str, output_csv: comp.OutputPath('CSV')):
#     import pandas as pd
#     data = pd.read_csv(source_path)
#     print(output_csv)
#     data.to_csv(output_csv, index=False)

# download_op = create_component_from_func(func=download_data,
#                                          base_image=IMAGE)

web_downloader_op = kfp.components.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/web/Download/component.yaml')

# Preprocess and store data
def preprocess_data(source_path: comp.InputPath('CSV'), 
                    x_train_output_path: str,
                    x_test_output_path: str,
                    y_train_output_path: str,
                    y_test_output_path: str):

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    from sklearn.model_selection import train_test_split
    from typing import List
    import pandas as pd
    import numpy as np

    # Load and split data
    data = pd.read_csv(source_path + '.csv')
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)
    
    # Convert to required format
    x_train = list(x_train)
    y_train = y_train.to_numpy()
    x_test = list(x_test)
    y_test = y_test.to_numpy()

    # Function for preprocessing data
    def ngram_vectorize(train_text: List[str], train_labels: np.ndarray, test_text: List[str]):
        # Arguments for vectorizor
        kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
        }

        vectorizer = TfidfVectorizer(**kwargs)
        
        # Vectorize training text
        x_train = vectorizer.fit_transform(train_text)
        # Vectorize test text
        x_test = vectorizer.transform(test_text)

        # Select top k features
        selector = SelectKBest(f_classif, k=TOP_K)
        selector.fit(x_train, train_labels)
        x_train = selector.transform(x_train).astype('float32')
        x_test = selector.transform(x_test).astype('float32')

        return x_train, x_test
    
    # Preprocess data
    x_train, x_test = ngram_vectorize(x_train, y_train, x_test)

    # Save data
    np.save(x_train, x_train_output_path)
    np.save(x_test, x_test_output_path)
    np.save(y_train, y_train_output_path)
    np.save(y_test, y_test_output_path)

preprocess_op = create_component_from_func(func=preprocess_data,
                                           base_image=IMAGE)

# Train model
# Evaluate model
# Save model

# Build pipeline
@dsl.pipeline(
    name="SMS Spam Detection Model Pipeline",
    description="Train an MLP to detect spam messages from csv data"
)
def pipeline(url=DATA_URL):
    download = web_downloader_op(url=url)
    preprocess = preprocess_op(download.outputs['data'], 
                  'x_train.npy', 
                  'x_test.npy', 
                  'y_train.npy', 
                  'y_test.npy').after(download)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='pipeline.yaml'
    )