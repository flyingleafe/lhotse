import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.magicdata_conv import download_magicdata_conv, prepare_magicdata_conv
from lhotse.utils import Pathlike

__all__ = ["magicdata_conv"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--dev-pct", type=float, default=0.1, help="Dev set percentage.")
@click.option("--test-pct", type=float, default=0.1, help="Test set percentage.")
def magicdata_conv(corpus_dir: Pathlike, output_dir: Pathlike, dev_pct: float = 0.1, test_pct: float = 0.1):
    """Magicdata ASR data preparation."""
    prepare_magicdata_conv(corpus_dir, output_dir=output_dir, dev_pct=dev_pct, test_pct=test_pct)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def magicdata_conv(target_dir: Pathlike, force_download: bool = True):
    """Magicdata download."""
    download_magicdata_conv(target_dir, force_download=force_download)
