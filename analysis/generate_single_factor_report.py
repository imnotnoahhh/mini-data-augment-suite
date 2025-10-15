from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from analysis.single_factor_report import generate_report, preview_tables

console = Console()


def main(
    stage_dir: Path = typer.Option(
        Path("outputs/single_factor"),
        "--stage-dir",
        "-s",
        help="Directory containing single-factor runs.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/reports/single_factor"),
        "--output-dir",
        "-o",
        help="Directory to store generated tables and plots.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Torch device override (e.g. cuda, mps, cpu). Defaults to auto-detection.",
    ),
    batch_size: int = typer.Option(256, help="Mini-batch size for validation inference."),
    num_workers: int = typer.Option(0, help="Number of workers for DataLoader."),
    show_preview: bool = typer.Option(
        True,
        "--show-preview/--no-show-preview",
        help="Print abridged summaries to the terminal after generation.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    artifacts = generate_report(
        stage_dir=stage_dir,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    console.print(f"[green]Report generated under[/green] {artifacts.output_root}")

    if show_preview:
        console.print("\n[bold]Preview[/bold]")
        tables = preview_tables(artifacts)
        for title, df in tables.items():
            _print_dataframe(title, df)


def _print_dataframe(title: str, df: pd.DataFrame, max_rows: int = 5) -> None:
    table = Table(title=title.replace("_", " ").title(), show_lines=False)
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.head(max_rows).iterrows():
        table.add_row(*[str(row[col]) for col in df.columns])
    console.print(table)


if __name__ == "__main__":
    typer.run(main)
