"""
Visualize the data with Streamlit and spaCy.

Based on: https://github.com/explosion/projects/blob/v3/tutorials/ner_drugs/scripts/visualize_data.py
"""
import streamlit as st
import spacy
from spacy import displacy
from spacy.tokens import DocBin
import srsly
import typer


@st.cache(allow_output_mutation=True)
def load_data(filepath):
    nlp = spacy.blank('de')
    doc_bin = DocBin().from_disk(filepath)
    docs = list(doc_bin.get_docs(nlp.vocab))

    n_total_ents = 0
    n_no_ents = 0
    labels = set()

    for doc in docs:
        n_total_ents += len(doc.ents)
        if not doc.ents:
            n_no_ents += 1
        labels.update([ent.label_ for ent in doc.ents])

    return docs, labels, n_total_ents, n_no_ents


def main(file_paths: str):
    files = [p.strip() for p in file_paths.split(",")]
    st.sidebar.title("Data visualizer")
    st.sidebar.markdown(
        "Visualize the annotations using [displaCy](https://spacy.io/usage/visualizers) "
        "and view stats about the datasets. Showing only the first {} docs "
    )
    data_file = st.sidebar.selectbox("Dataset", files)
    docs, labels, n_total_ents, n_no_ents = load_data(data_file)
    n_docs = st.sidebar.slider("# Docs ro visualize", 10, len(docs))

    st.header(f"{data_file} ({len(docs)})")
    wrapper = "<div style='border-bottom: 1px solid #ccc; padding: 20px 0'>{}</div>"
    for doc in docs[:n_docs]:
        html = displacy.render(doc, style='ent').replace("\n\n", "\n")
        st.markdown(wrapper.format(html), unsafe_allow_html=True)

    st.sidebar.markdown(
        f"""
    | `{data_file}` | |
    | --- | ---: |
    | Total examples | {len(docs):,} |
    | Total entities | {n_total_ents:,} |
    | Examples with no entities | {n_no_ents:,} |
    """
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
