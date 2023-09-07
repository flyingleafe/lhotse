"""
A collection of open-sourced relatively small speech corpora by MagicData Technology Co., Ltd.
for underresourced languages (not Mandarin Chinese and not English).
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union
from zipfile import ZipFile

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


def download_magicdata_conv(
    target_dir: Pathlike = '.',
    force_download: bool = False,
    url_list: str = 'https://bafybeidxtoq6hed4lkibgrga633nuy6mjjp7xqownube3yx73a3fwtnggq.ipfs.w3s.link/magicdata-conversational-datasets.txt',
):
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    url_list_local = target_dir / 'magicdata-conversational-datasets.txt'

    if not url_list_local.exists() or force_download:
        logging.info('Downloading MagicData conversational datasets list...')
        resumable_download(url_list, url_list_local)
    
    with open(url_list_local, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines() if line.strip()]
    
    for lang, url in pairs:
        part_dir = target_dir / lang
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file() and not force_download:
            logging.info(f"Skipping {lang} because {completed_detector} exists.")
            continue
 
        zip_path = target_dir / f'{lang}.zip' 
        logging.info(f'Downloading MagicData conversational dataset for {lang}...')
        resumable_download(url, zip_path, force_download=force_download)

        part_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Extracting MagicData conversational dataset for {lang}...')
        with ZipFile(zip_path) as zf:
            zf.extractall(path=part_dir)

        completed_detector.touch()

        logging.info(f'Cleaning up...')
        os.remove(zip_path)
    
    return target_dir


def prepare_magicdata_conv(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dev_pct: float = 0.1,
    test_pct: float = 0.1,
):
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    langs = [d.name for d in corpus_dir.iterdir() if d.is_dir()]
    manifests = defaultdict(dict)

    for lang in tqdm(langs, desc="Processing languages"):
        lang_dir = corpus_dir / lang
        
        all_wavs = list(lang_dir.glob('WAV/*.wav'))
        dev_n = max(int(len(all_wavs) * dev_pct), 1)
        test_n = max(int(len(all_wavs) * test_pct), 1)
        split = {
            "train": all_wavs[:-dev_n-test_n],
            "dev": all_wavs[-dev_n-test_n:-test_n],
            "test": all_wavs[-test_n:],
        }

        for part, wavs in split.items():
            recording_set = RecordingSet.from_recordings(Recording.from_file(wav) for wav in wavs)
            txts = [lang_dir / 'TXT' / f'{wav.stem}.txt' for wav in wavs]
            supervisions = []

            for txt_file in txts:
                rec_id = txt_file.stem
                with open(txt_file, 'r') as f:
                    for ix, line in enumerate(f.readlines()):
                        ts, speaker, gender, text = line.split('\t')
                        if speaker == "0":
                            # non-speech segment
                            continue

                        start, end = (float(s) for s in ts.strip()[1:-1].split(','))
                        supervisions.append(SupervisionSegment(
                            id=f'{rec_id}-{ix}',
                            recording_id=rec_id,
                            start=start,
                            duration=end - start,
                            channel=0,
                            language=lang,
                            speaker=speaker.strip(),
                            gender=gender.strip(),
                            text=text.strip(),
                        ))
            
            supervision_set = SupervisionSet.from_segments(supervisions)

            recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                output_dir = Path(output_dir)
                ds_name = f"magicdata-conv-{lang}"
                supervision_set.to_file(
                    output_dir / f"{ds_name}_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(output_dir / f"{ds_name}_recordings_{part}.jsonl.gz")

            manifests[lang][part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests



            
