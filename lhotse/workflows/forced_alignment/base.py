import abc
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial, partialmethod
from itertools import chain
from typing import Dict, Generator, List, Optional, Tuple, Type, Union

import torch
from tqdm.auto import tqdm

from lhotse.cut import Cut, CutSet
from lhotse.supervision import AlignmentItem
from lhotse.utils import fastcopy


class FailedToAlign(RuntimeError):
    pass


class ForcedAligner(abc.ABC):
    """
    ForcedAligner is an abstract base class for forced aligners.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)

    @abc.abstractproperty
    def sample_rate(self) -> int:
        pass

    @abc.abstractmethod
    def normalize_text(
        self, text: str, language: Optional[str] = None
    ) -> Union[str, List[str]]:
        pass

    @abc.abstractmethod
    def align(
        self, audio: torch.Tensor, transcript: Union[str, List[Tuple[str, str]]]
    ) -> List[AlignmentItem]:
        pass

    def __call__(self, cut: Cut, normalize: bool = True) -> Cut:
        cut = fastcopy(cut, supervisions=list(cut.supervisions))

        for idx, subcut in enumerate(cut.trim_to_supervisions(keep_overlapping=False)):
            audio = torch.as_tensor(
                subcut.resample(self.sample_rate).load_audio(), device=self.device
            )
            sup = subcut.supervisions[0]
            transcript = (
                self.normalize_text(sup.text, language=sup.language)
                if normalize
                else sup.text
            )

            try:
                pre_alignment = self.align(audio, transcript)
            except FailedToAlign:
                logging.info(
                    f"Failed to align supervision '{sup.id}' for cut '{cut.id}'. Writing it without alignment."
                )
                continue

            alignment = [
                item._replace(start=item.start + subcut.start) for item in pre_alignment
            ]

            # Important: reference the original supervision before "trim_to_supervisions"
            #            because the new one has start=0 to match the start of the subcut
            sup = cut.supervisions[idx].with_alignment(kind="word", alignment=alignment)
            cut.supervisions[idx] = sup

        return cut


class ForcedAlignmentProcessor:
    """
    TODO: Too much copypaste between VAD and FA processors, need to abstract it out.
    """

    _aligners: Dict[Optional[int], ForcedAligner] = {}

    def __init__(
        self,
        aligner_kls: Type[ForcedAligner],
        bundle_name: str,
        num_jobs: int = 1,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ):
        self._make_aligner = partial(
            aligner_kls, bundle_name=bundle_name, device=torch.device(device)
        )
        self.num_jobs = num_jobs
        self.verbose = verbose

    def _init_aligner(self):
        pid = multiprocessing.current_process().pid
        self._aligners[pid] = self._make_aligner()

    def _process_cut(self, cut: Cut, normalize: bool = True) -> Cut:
        pid = multiprocessing.current_process().pid
        aligner = self._aligners[pid]
        return aligner(cut, normalize=normalize)

    def __call__(
        self, cuts: CutSet, normalize: bool = True
    ) -> Generator[Cut, None, None]:
        if self.num_jobs == 1:
            aligner = self._make_aligner()
            for cut in tqdm(cuts, desc="Aligning", disable=not self.verbose):
                yield aligner(cut, normalize=normalize)

        else:
            pool = ProcessPoolExecutor(
                max_workers=self.num_jobs,
                initializer=self._init_aligner,
                mp_context=multiprocessing.get_context("spawn"),
            )

            with pool as executor:
                try:
                    res = executor.map(
                        partial(self._process_cut, normalize=normalize), cuts
                    )
                    for cut in tqdm(
                        res, desc="Aligning", total=len(cuts), disable=not self.verbose
                    ):
                        yield cut
                except KeyboardInterrupt as exc:  # pragma: no cover
                    pool.shutdown(wait=False)
                    if self.verbose:
                        print("Forced alignment interrupted by the user.")
                    raise exc
                except Exception as exc:  # pragma: no cover
                    pool.shutdown(wait=False)
                    raise RuntimeError(
                        "Forced alignment failed. Please report this issue."
                    ) from exc
                finally:
                    self._aligners.clear()
