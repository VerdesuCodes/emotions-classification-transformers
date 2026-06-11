# Emotion Classification With Transformers

End-to-end NLP project for emotion classification using traditional machine learning and transformer-based models.

## Project Goal

The goal of this project is to build and understand a complete text classification workflow:

- load and explore an NLP dataset
- understand labels and class distribution
- build a traditional machine learning baseline
- evaluate model performance with multiple metrics
- inspect wrong predictions
- extend the project with transformer-based models

This project is being developed as a learning-focused AI/NLP portfolio project.

## Dataset

The project uses the Hugging Face emotion dataset:

```python
load_dataset("dair-ai/emotion")
```

The dataset contains short text examples labeled with one of six emotions:

- sadness
- joy
- love
- anger
- fear
- surprise

Dataset splits:

| Split | Examples |
| --- | ---: |
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

## Current Progress

### Dataset Exploration

The dataset exploration notebook covers:

- dataset structure
- label names
- number of examples
- class distribution
- text length analysis

Notebook:

```text
notebooks/data_exploration.ipynb
```

### Baseline Model

The baseline model uses a traditional NLP pipeline:

```text
Text -> TF-IDF -> Logistic Regression -> Emotion Prediction
```

Notebook:

```text
notebooks/baseline_model.ipynb
```

The baseline model achieved:

```text
Test accuracy: 0.8615
```

The model performed well on larger classes such as `sadness` and `joy`, but had lower performance on smaller or more ambiguous classes such as `love` and `surprise`.

## Key Learning Points

This project currently demonstrates:

- how to use `load_dataset()` from Hugging Face
- how to inspect text classification datasets
- how to convert text into numeric features using TF-IDF
- the difference between `fit_transform()` and `transform()`
- how to train a Logistic Regression classifier
- how to evaluate a model using accuracy, precision, recall, F1-score, and support
- how class imbalance can affect model performance
- why mistake analysis is important in machine learning

## Important Note About Emotion Labels

Emotion labels are subjective. A sentence can include mixed emotions, humor, sarcasm, or missing context.

In this project, the "true emotion" means the label provided by the dataset. Evaluation metrics show how well the model matches those dataset annotations, not perfect human emotional understanding.

## Next Steps

The next phase is to add a transformer-based model using DistilBERT.

Planned transformer learning goals:

- understand tokenization
- use a pre-trained transformer model
- prepare the dataset for transformer fine-tuning
- train or fine-tune DistilBERT
- compare transformer results with the TF-IDF baseline

Notebook started:

```text
notebooks/transformer_model.ipynb
```

## How To Run

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter lab
```

Then open the notebooks inside the `notebooks/` directory.