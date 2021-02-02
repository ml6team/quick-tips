import typer
from pathlib import Path
import shutil


def main(input_dir: Path, output_dir: Path):
    """Prepare correctly structured IOB files from input tsv files."""
    for split in ('train', 'dev', 'test'):
        with open(input_dir / f'germaner_{split}.tsv', encoding="utf8") as f:
            tsv_lines = [l for l in f.readlines() if not l.startswith('#')]
            iob_lines = [' '.join(l.split('\t')[1:3]) if len(l.split('\t')) > 1 else ' '.join(l)
                         for l in tsv_lines]

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        with open(output_dir / f'germaner_{split}.iob', 'w', encoding="utf8") as f:
            f.write('\n'.join(iob_lines))


if __name__ == "__main__":
    typer.run(main)
