def generate_dataset(filepath: str,
                     data_accessor: tfx.components.DataAcessor,
                     schema: schema_pb2.Schema,
                     batch_size: int = 32) -> tf.data.Dataset:
    # Use data_accessor to fetch tfrecord data and create a tf dataset

def build_model() -> tf.keras.Model:
    # Build model

# TFX Trainer component entrypoint
def run_fn(fn_args: tfx.components.FnArgs):
    # Train model here