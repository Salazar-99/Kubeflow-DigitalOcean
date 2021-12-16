from typing import List, Tuple
from absl import logging
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform as tft

LABEL_KEY = 'label'
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
VOCAB_SIZE = 10000
FEATURE_KEYS = [
    'text'
]
FEATURE_SPEC = {
    **{
        feature: tf.io.VarLenFeature(tf.string)
           for feature in FEATURE_KEYS
       },
    LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}

def _input_fn(file_pattern: str,
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 32) -> tf.data.Dataset:

  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(batch_size=batch_size, 
                                     label_key=LABEL_KEY),
                                     schema=schema)

def preprocessing_fn(data):
    """
    Used by Transform component to transform raw features for 
    inference and training
    """
    #TODO: Cant figure out how to compute ngrams and tf-idf from data
    tokens = tf.sparse.from_dense(
        tft.compute_and_apply_vocabulary(
            data['text'], 
            top_k=VOCAB_SIZE, 
            default_value=0, 
            frequency_threshold=2))
    ngrams = tft.ngrams(tokens=tokens, ngram_range=(1,2), separator=" ")
    vocab_index, tf_idf = tft.tfidf(x=ngrams, vocab_size=VOCAB_SIZE, smooth=True)
    return {
        'text_xf': tf_idf
    }

def _apply_preprocessing(raw_features, tft_layer):
    """
`   Used to preprocess data during inference`
    """
    transformed_features = tft_layer(raw_features)
    return transformed_features

def build_model(layers: int, 
                units: int, 
                dropout_rate: float, 
                input_shape: Tuple) -> tf.keras.Model:
    """
    Build MLP model for binary classification
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    for _ in range(layers-1):
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def train_model(train_data,
                validation_data,
                learning_rate=1e-3,
                epochs=100,
                batch_size=32,
                layers=2,
                units=64,
                dropout_rate=0.2):
    """
    Compiles and trains classification model
    """
    # Create model
    model = build_model(layers=layers,
                        units=units, 
                        dropout_rate=dropout_rate, 
                        input_shape=(VOCAB_SIZE,))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = 'binary_crossentropy'
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=['acc', tf.keras.metrics.FalseNegatives(name="fn")])

    # Create early stopping callback
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # Train model
    history = model.fit(train_data, 
                        epochs=epochs,
                        steps_per_epoch=1,
                        callbacks=callbacks,
                        validation_data=validation_data,
                        validation_steps=1,
                        verbose=2,
                        batch_size=batch_size)

    # Print results
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}, false negatives: {fn}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1], fn=history['fn'][-1]))

    return model

def run_fn(fn_args: tfx.components.FnArgs):
    """
    Entrypoint for Trainer component
    """
    schema = schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    train_dataset = generate_dataset(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=TRAIN_BATCH_SIZE)
    eval_dataset = generate_dataset(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=EVAL_BATCH_SIZE)
        
    # Train model
    trained_model = train_model(
        train_dataset,
        eval_dataset)
    
    # Save model with transform graph for preprocessing
    signatures = {'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output)}
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Creates a function to take a serialized tf.example, preprocess it,
    and run a model inference on it
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        # Expected input is a string which is serialized tf.Example format.
        feature_spec = tf_transform_output.raw_feature_spec()
        
        # Filter out any unneeded features
        required_feature_spec = {
            k: v for k, v in feature_spec.items() if k in FEATURE_KEYS
        }
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                                required_feature_spec)

        # Preprocess parsed input
        transformed_features, _ = _apply_preprocessing(parsed_features,
                                                        model.tft_layer)
        # Run inference with ML model.
        return model(transformed_features)

    return serve_tf_examples_fn