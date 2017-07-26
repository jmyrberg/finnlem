# Finnish Neural Lemmatizer (finnlem)

## Training steps
### 1. Fit a dictionary
python -m dict_train.py \
	--dict-save-path './data/dicts/lemmatizer.dict' \
	--dict-train-path './data/train