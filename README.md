# Finnish Neural Lemmatizer (finnlem)

## Training steps
### 1. Fit a dictionary
python -m dict_train ^
		--dict-save-path ./data/dictionaries/lemmatizer.dict ^
		--dict-train-path ./data/dictionaries/lemmatizer.vocab ^
		--vocab-size 50 ^
		--min-freq 0.0 ^
		--max-freq 1.0 ^
		--file-batch-size 8192 ^
		--prune-every-n 200

### 2. Create and train a new model
python -m model_train ^
		--model-dir ./data/models/lemmatizer2 ^
		--dict-path ./data/dictionaries/lemmatizer.dict ^
		--train-data-path ./data/datasets/lemmatizer_train.csv ^
		--optimizer 'adam' ^
		--learning-rate 0.0001 ^
		--dropout-rate 0.2 ^
		--batch-size 128 ^
		--file-batch-size 8192 ^
		--max-file-pool-size 50 ^
		--shuffle-files True ^
		--shuffle-file-batches True ^
		--save-every-n-batch 500 ^
		--validate-every-n-batch 100 ^
		--validation-data-path ./data/datasets/lemmatizer_validation.csv ^
		--validate-n-rows 5000
		
### 3. Make predictions on test set
python -m model_decode ^
		--model-dir ./data/models/lemmatizer ^
		--source-data-path ./data/datasets/lemmatizer_test.csv ^
		--decoded-data-path ./data/decoded/lemmatizer_decoded_1.csv