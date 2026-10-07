import logging
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
# LOG_PATH = Path("outputs/local")

logger = logging.getLogger(__name__)

class LocalLogger(TensorBoardLogger):
    def __init__(self, save_dir, board_dir=None, **kargs) -> None:
        # NOTE: TB on areis is unstable, save to root and sync later
        if board_dir is None:
            board_dir = save_dir
        super().__init__(save_dir=board_dir, **kargs)
        self.log_path = Path(save_dir)

    @property
    def name(self):
        return "LocalLogger(TensorBoardLogger)"

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # super().log_image(key, images, step, **kargs)
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            try:
                Image.fromarray(image).save(path)
            except Exception:
                logger.exception("Failed saving image: %s", path)
