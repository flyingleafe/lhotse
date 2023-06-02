from typing import Optional

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_voxpopuli, prepare_voxpopuli
from lhotse.utils import Pathlike


@download.command()
@click.argument("target_dir", type=click.Path())
@click.option("--subset", type=click.STRING, required=True, help="Subset to download")
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def voxpopuli(
    target_dir: Pathlike, subset: str, force_download: Optional[bool] = False
):
    """VoxPopuli download."""
    download_voxpopuli(target_dir, subset=subset, force_download=force_download)


@prepare.command()
@click.argument("target_dir", type=click.Path())
@click.option("--task", type=click.STRING, required=True, help="Task to prepare")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory")
@click.option("--lang", type=click.STRING, default="all", help="Language to prepare")
@click.option("--num-jobs", type=click.INT, default=1, help="Number of jobs")
def voxpopuli(
    target_dir: Pathlike,
    task: str,
    output_dir: Optional[Pathlike] = None,
    lang: str = "all",
    num_jobs: int = 1,
):
    """VoxPopuli preparation."""
    prepare_voxpopuli(
        target_dir, task=task, output_dir=output_dir, lang=lang, num_jobs=num_jobs
    )
