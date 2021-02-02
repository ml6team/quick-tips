"""
Visualize the model outputs with Streamlit and spaCy.

Source: https://github.com/explosion/projects/blob/v3/tutorials/ner_drugs/scripts/visualize_model.py
"""

import spacy_streamlit
import typer


def main(models: str, default_text: str):
    models = [name.strip() for name in models.split(",")]
    spacy_streamlit.visualize(models, default_text, visualizers=["ner"])


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass