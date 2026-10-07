import numpy as np
from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        dataset_len: int,
        seed: int = 0,
        total_frames_per_gpu: int = 48,
        possible_v=(2, 3, 4, 6, 8, 12, 16, 24),
        weights=None,
        drop_last: bool = True,
    ):
        self.dataset_len = int(dataset_len)
        self.total_frames = int(total_frames_per_gpu)
        self.possible_v = list(possible_v)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def construct_batch_config(self, input_views):
        max_input = self.total_frames / 2
        if input_views <= max_input:
            batch_size = np.floor(max(1, max_input // input_views)).astype(int)
            target_views = input_views
        else:
            target_views = self.total_frames - input_views
            batch_size = 1
        return input_views, target_views, batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = np.arange(self.dataset_len)
        rng.shuffle(indices)

        i = 0
        while i < len(indices):
            views = int(rng.choice(self.possible_v, p=self._norm_weights()))
            input_views, target_views, batch_size = self.construct_batch_config(views)

            if i + batch_size > len(indices):
                if self.drop_last:
                    break
                batch_size = len(indices) - i

            batch = [
                (int(indices[j]), input_views, target_views)
                for j in range(i, i + batch_size)
            ]
            i += batch_size
            yield batch

    def __len__(self):
        return self.dataset_len

    def _norm_weights(self):
        if self.weights is None:
            return None
        return self.weights / self.weights.sum()
