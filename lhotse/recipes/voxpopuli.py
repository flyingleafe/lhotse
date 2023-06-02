import csv
import glob
import gzip
import logging
import os
import tarfile
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "pl",
    "it",
    "ro",
    "hu",
    "cs",
    "nl",
    "fi",
    "hr",
    "sk",
    "sl",
    "et",
    "lt",
    "pt",
    "bg",
    "el",
    "lv",
    "mt",
    "sv",
    "da",
]
LANGUAGES_V2 = [f"{x}_v2" for x in LANGUAGES]

YEARS = list(range(2009, 2020 + 1))

ASR_LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "pl",
    "it",
    "ro",
    "hu",
    "cs",
    "nl",
    "fi",
    "hr",
    "sk",
    "sl",
    "et",
    "lt",
]
ASR_ACCENTED_LANGUAGES = ["en_accented"]

S2S_SRC_LANGUAGES = ASR_LANGUAGES

S2S_TGT_LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "pl",
    "it",
    "ro",
    "hu",
    "cs",
    "nl",
    "fi",
    "hr",
    "sk",
    "sl",
    "et",
    "lt",
    "pt",
    "bg",
    "el",
    "lv",
    "mt",
    "sv",
    "da",
]

S2S_TGT_LANGUAGES_WITH_HUMAN_TRANSCRIPTION = ["en", "fr", "es"]

DOWNLOAD_BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli"

SPLITS = ["train", "dev", "test"]


def download_voxpopuli(
    corpus_dir: Pathlike,
    subset: str,
    force_download: bool = False,
    base_url: str = DOWNLOAD_BASE_URL,
):
    """
    Download the VoxPopuli corpus to a directory.

    :param corpus_dir: Pathlike, the path to download the corpus to.
    :param force_download: bool, if True, always download the corpus.
    :param base_url: str, the base URL to download the corpus from.
    """

    if subset in LANGUAGES_V2:
        languages = [subset.split("_")[0]]
        years = YEARS + [f"{y}_2" for y in YEARS]
    elif subset in LANGUAGES:
        languages = [subset]
        years = YEARS
    else:
        languages = {
            "400k": LANGUAGES,
            "100k": LANGUAGES,
            "10k": LANGUAGES,
            "asr": ["original"],
        }.get(subset, None)
        years = {
            "400k": YEARS + [f"{y}_2" for y in YEARS],
            "100k": YEARS,
            "10k": [2019, 2020],
            "asr": YEARS,
        }.get(subset, None)

    url_list = []
    for l in languages:
        for y in years:
            url_list.append(f"{base_url}/audios/{l}_{y}.tar")

    out_dir = Path(corpus_dir) / "raw_audios"
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Downloading audio, {len(url_list)} files to download...")
    for url in tqdm(url_list):
        tar_path = out_dir / Path(url).name
        completed_detector = out_dir / f".{tar_path.stem}.completed"

        if completed_detector.exists() and not force_download:
            print(f"{tar_path} already downloaded and extracted - skipping.")
            continue

        resumable_download(url, tar_path, force_download=force_download)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, out_dir)

        os.remove(tar_path)
        completed_detector.touch()

    print(f"Downloading metadata...")
    if subset == "asr":
        for lang in ASR_LANGUAGES + ASR_ACCENTED_LANGUAGES:
            out_root = Path(corpus_dir) / "asr" / lang
            out_root.mkdir(exist_ok=True, parents=True)
            url = f"{base_url}/annotations/asr/asr_{lang}.tsv.gz"
            tsv_path = out_root / Path(url).name
            if not tsv_path.exists() or force_download:
                resumable_download(url, tsv_path, force_download=force_download)
    else:
        pass


def prepare_voxpopuli(
    corpus_dir: Pathlike,
    task: str,
    output_dir: Optional[Pathlike] = None,
    lang: str = "all",
    num_jobs: int = 1,
):
    if task == "asr":
        prepare_asr_voxpopuli(corpus_dir, output_dir, lang, num_jobs)
    else:
        pass


def prepare_asr_voxpopuli(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    lang: str = "all",
    num_jobs: int = 1,
):
    if lang == "all":
        languages = ASR_LANGUAGES + ASR_ACCENTED_LANGUAGES
    else:
        languages = [lang]

    in_root = Path(corpus_dir) / "raw_audios" / "original"

    rec_paths = glob.glob(in_root / "**/*_original.ogg", recursive=True)
    recordings = {}
    for p in tqdm(rec_paths, "Collecting recordings..."):
        rec_id = Path(p).stem.rsplit("_", 2)[0]
        recordings[rec_id] = Recording.from_file(path=p, recording_id=rec_id)

    supervisions = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for lng in languages:
        tsv_path = Path(corpus_dir) / "asr" / lng / f"asr_{lng}.tsv.gz"
        with gzip.open(tsv_path, "rt") as f:
            metadata = [x for x in csv.DictReader(f, delimiter="|")]

        if num_jobs > 1:
            with ProcessPoolExecutor(num_jobs) as ex:
                ress = ex.map(
                    partial(_prepare_asr_metadata, in_root, lng),
                    metadata,
                    chunksize=1000,
                )
                for res in tqdm(
                    ress, f"Processing metadata ({lng})", total=len(metadata)
                ):
                    if res is None:
                        continue
                    recording, supervision, split = res
                    recordings[split].setdefault(recording.id, recording)
                    supervisions[split].append(supervision)
        else:
            for r in tqdm(metadata, f"Processing metadata ({lng})"):
                res = _prepare_asr_metadata(in_root, lng, r)
                if res is None:
                    continue

                recording, supervision, split = res
                recordings[split].setdefault(recording.id, recording)
                supervisions[split].append(supervision)

    manifests = {}

    for split in SPLITS:
        cur_recordings = RecordingSet.from_recordings(list(recordings[split].values()))
        cur_supervisions = SupervisionSet.from_segments(supervisions[split])
        validate_recordings_and_supervisions(cur_recordings, cur_supervisions)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            cur_recordings.to_file(
                output_dir / f"voxpopuli_asr_recordings_{split}.jsonl.gz"
            )
            cur_supervisions.to_file(
                output_dir / f"voxpopuli_asr_supervisions_{split}.jsonl.gz"
            )

        manifests[split] = {
            "recordings": cur_recordings,
            "supervisions": cur_supervisions,
        }

    return manifests


def _prepare_asr_metadata(in_root, lng, r):
    split = r["split"]
    if split not in SPLITS:
        # logging.warn(f"Skipping {r['id_']} - unknown split {split}")
        return None

    vad = literal_eval(r["vad"])
    start, end = float(vad[0][0]), float(vad[-1][1])
    if end <= start:
        logging.warn(f"Skipping {lng}_{r['id_']} - invalid VAD: {r['vad']}")
        return None

    vad_alignments = [
        AlignmentItem("", float(x[0]), float(x[1]) - float(x[0])) for x in vad
    ]

    recording_id = r["session_id"]
    year = recording_id[:4]
    recording_path = in_root / year / f"{recording_id}_original.ogg"
    recording = Recording.from_file(path=recording_path, recording_id=recording_id)

    # such complex ids - because in en and en_accented IDs clash
    supervision = SupervisionSegment(
        id=f"asr_{recording_id}_{lng}_{r['id_']}",
        recording_id=recording_id,
        start=start,
        duration=end - start,
        channel=0,
        language=lng,
        speaker=r["speaker_id"],
        text=r["original_text"],
        gender=r["gender"],
    )
    supervision = supervision.with_alignment("vad", vad_alignments)

    return recording, supervision, split
