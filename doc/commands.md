# List of available commands

## Step 1: Dictionary training
Required:
* ```--dict-save-path```: Dictionary save path after training (str)
* ```--dict-train-path```: Dictionary training data folder or file (str)

Optional:
* ```--vocab-size```: Size of vocabulary, (int, default: 30000)
* ```--min-freq```: Minimum token frequency (float, default: 0.0)
* ```--max-freq```: Maximum token frequency (float, default: 1.0)

* ```--file-batch-size```: Dictionary is trained in batches of this size (int, default: 8192)
* ```--prune-every-n```: Dictionary is pruned every n batch (int, default: 200)

## Step 2: Model training
### Path parameters
Required:
* ```--model-dir```: Model checkpoint and log save path (str)
* ```--train-data-path```: Training data folder or file' (str)

Required when creating the model for the first time:
* ```--dict-path```: Path to existing Dictionary (str, default: None)


### Model params
Optional, locked in when creating the model for the first time:
* ```--cell-type```: Cell type, either 'gru' or 'lstm' (str, default: 'lstm
* ```--hidden-dim```: Number of neurons in hidden layers (int, default: 32)
* ```--attn-dim```: Number of neurons in to use in attention. None means attn-dim = hidden-dim (int, default: None)
* ```--embedding-dim```: Embedding dimension (int, default: 16)
* ```--depth```: Number of hidden layers in encoder and decoder (int, default: 2)
* ```--attn-type```: Attention type, either 'bahdanau' or 'luong' (str, default: 'bahdanau
* ```--attn-input-feeding```: Whether attention is fed to decoder inputs (bool, default: True)
* ```--use-residual```: Whether to use residual connection (bool, default: False)
* ```--reverse-source```: Whether to reverse encode input sequences (bool, default: True)
* ```--keep-every-n-hours```: Checkpoint keep interval (int, default: 1)

### Model training params
Optional:
* ```--max-seq-len```: Maximum sequence length (int, default: 100)
* ```--optimizer```: Optimizer to use, should be 'adadelta', 'adam' or 'rmsprop' (str, default: 'adam
* ```--learning-rate```: Learning rate of the optimizer (float, default 0.0001)
* ```--max-gradient-norm```: Maximum norm to clip gradient (float, default: 1.0)
* ```--dropout-rate```: Hidden layer dropout (float, default: 0.2)

### Training params
Optional:´
* ```--batch-size```: Batch size to feed into model (int, default: 32)
* ```--file-batch-size```: Number of rows to read in-memory from each file (int, default: 8192)
* ```--max-file-pool-size```: Maximum number of files to cycle at a time (int, default: 50)
* ```--shuffle-files```: Shuffle files before reading (bool, default: True)
* ```--shuffle-file-batches```: Shuffle file batches before training (bool, default: True)
* ```--save-every-n-batch```: Save model checkpoint every n batch (int, default: 1000)
* ```--validation-data-path```: Validation data folder or file (str, default: None)
* ```--validate-every-n-batch```: Save model checkpoint every n batch (int, default 100)
* ```--validate-n-rows```: Number of rows to read from validation file (int, default: None)

## Step 3: Model decoding
### Path parameters
Required:
* ```--model-dir```: Model checkpoint and log save path (str)
* ```--test-data-path```: Path to source data to decode (str)
* ```--decoded-data-path```: Output file for decoded documents (str)

### Decoder parameters
Optional:
* ```--beam-width```: Number of beams when using beamsearch. When beam-width=1, greedy decoder will be used instead. (int, default: 1)
* ```--max-decode-step```: Maximum sequence length when decoding (int, default: 30)
* ```--batch-size```: Batch size to feed into model (int, default: 32)
* ```--file-batch-size```: Number of rows to read in-memory from each file (int, default: 8192)
* ```--max-file-pool-size```: Maximum number of files to cycle at a time (int, default: 50)

