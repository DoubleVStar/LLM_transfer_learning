# Research-Project-Brief-Display
Capstone project of **Locally Deployable Biomedical Pre-trained Language Models for Information Extraction and Classification on Curated Evidence-based Corpus**

A python implementation of data curation and language model fine-tune.


## Requirements

- **Windows** or **Linux**, with **Python = 3.11**

- Details listed in **requirements.txt**

## Related code files

- **NER**:    Data spliting of NER dataset in BioC format from CoDiet dictionary, include the implementation of full fine-tuning and evaluation of BERT-based and LLM-based models on NER task.
- **TEXT-CLS**:    Data spliting of text dataset in CSV format from CoDiet corpus, include the implementation of full fine-tuning and evaluation of BERT-based and LLM-based models on text classification task.
- **Preprocess**: Including data generation and cleaning for NER and classification tasks, including LLM-assisted data curation and IAO section distribution analysis.
- **DATA**: Store the original BioC and json format CoDiet dataset.

## Datasets

This repository only use CoDiet methodology corpus provided by Dr. Joram (j.posma11@imperial.ac.uk).

The NER task requires 'BioC' format txt files and the classification task requires 'json' format files.

```
the	O
principles	O
established	O
by	O
the	O
National	O
Animal	B-methodology
Experimentation	I-methodology
Control	O
Council	O
[	O
17	O
]	O
.	O
```

```
{
  "infons": {
    "section_title_1": "Keywords",
    "iao_name_0": "keywords section",
    "iao_id_0": "IAO:0000630"
  },
  "offset": 114,
  "text": "Lactobacillus gasseriLG-G12; intestinal health; ceftriaxone; gut microbiota; obesity",
  "sentences": [],
  "annotations": [],
  "relations": []
}
...
```

## Training

For customized training: 
**bert.py**

For training example on virtual environment:
```
python bert.py \
  --input-glob '/home/DATA/NER/silver_data_split/cased_bio2_silver_methodology_512/*.txt' \
  --out-dir outputs/test_1 \
  --split 7,1,2 \
  --split-by document \
  --pretrained dmis-lab/biobert-base-cased-v1.1 \
  --max-length 512 \
  --batch-size 16 \
  --lr 3e-5 \
  --num-epochs 30 \
  --autofix-iob
```

The model will be saved by the default path './outputs'

## Evaluation

- Modify the evaluation parameters in main scripts to test the model.

## Performance

- The model evaluation output will be as a table of precision, recall and F1-score.

