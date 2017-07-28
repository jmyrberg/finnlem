# List of relevant Python API classes, and their arguments and methods

## dictionary.Dictionary
Args:
* ```vocab_size```: Size of vocabulary, (int, default: 30000)
* ```min_freq```: Minimum token frequency (float, default: 0.0)
* ```max_freq```: Maximum token frequency (float, default: 1.0)

Methods:
* ```fit_batch```: Fit a batch of documents in Dictionary.
* ```fit```: Fit documents in dictionary.
* ```lock```: Lock-in vocabulary to be able to save with batch-fitting.
* ```save```: Save Dictionary object to file.
* ```seq2doc```: Map id sequence to document.
* ```seqs2docs```: Map id sequences to documents.
* ```doc2seq```: Map document to id sequence.
* ```docs2seqs```: Map documents to id sequences.

Attributes:
* ```id2token```: Dictionary of id (int) -> token (str) mappings.
* ```token2id```: Dictionary of token (str) -> id (int) mappings.
* ```counter```: Dictionary of token counts. For most common tokens, use 'Dictionary.counter.most_common()'.
* ```n_docs```: Number of documents the Dictionary has been fitted on.
* ```n_tokens```: Number of tokens in Dictionary.
* ```vocab_size```: Vocabulary size limit, not necessarily the real size.
* ```special_tokens```: List of special tokens.
* ```SOS```: Integer representation for start of sequence token
* ```PAD```: Integer representation for padding token
* ```EOS```: Integer representation for end of sequence token
* ```UNK```: Integer representation for unknown token

## model_wrappers.Seq2Seq
Args:
* ```model_dir```: Model checkpoint and log save path (str)
* ```dict_path```: Path to existing Dictionary (str, default: None)
* ```cell_type```: Cell type, either 'gru' or 'lstm' (str, default: 'lstm
* ```hidden_dim```: Number of neurons in hidden layers (int, default: 32)
* ```attn_dim```: Number of neurons in to use in attention. None means attn_dim = hidden_dim (int, default: None)
* ```embedding_dim```: Embedding dimension (int, default: 16)
* ```depth```: Number of hidden layers in encoder and decoder (int, default: 2)
* ```attn_type```: Attention type, either 'bahdanau' or 'luong' (str, default: 'bahdanau
* ```attn_input_feeding```: Whether attention is fed to decoder inputs (bool, default: True)
* ```use_residual```: Whether to use residual connection (bool, default: False)
* ```reverse_source```: Whether to reverse encode input sequences (bool, default: True)
* ```keep_every_n_hours```: Checkpoint keep interval (int, default: 1)

Methods:
* ```train```: Perform a model training step.
* ```eval```: Perform a model validation step.
* ```decode```: Decode source documents to their target form.

Attributes:
* ```wrapper_config```: Dictionary of wrapper parameters.
* ```model_config```: Dictionary of Seq2Seq model parameters.

## data_utils
Methods for reading from files in batches.
