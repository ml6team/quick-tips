import typer
from pathlib import Path

from spacy.tokens import DocBin


def main(input_dir: Path, output_dir: Path):
    """Join datasets into single dataset (with separate train/dev/test splits)."""
    for split in ('train', 'dev', 'test'):
        paths = [f for f in input_dir.glob(f'*_{split}.spacy') if f.is_file()]
        doc_bin = DocBin().from_disk(paths[0])
        for path in paths[1:]:
            doc_bin.merge(DocBin().from_disk(path))
        
        doc_bin.to_disk(output_dir / f'{split}.spacy')


if __name__ == "__main__":
    typer.run(main)
