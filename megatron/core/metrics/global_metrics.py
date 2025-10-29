from typing import Iterator, Optional
import torch

from megatron.core import parallel_state

class GlobalTracker:
    known_metrics = {"num_recurrences"}

    def __init__(self, args, metrics: Optional[list[str]] = None):
        if metrics is None:
            metrics = []
        assert set(metrics) <= self.known_metrics
        self.enabled = False
        self.metrics = sorted(metrics)
        self.args = args

        self.final_metrics_map = {metric: idx for idx, metric in enumerate(self.metrics)}
        self._inv_final_metrics_map = {idx: metric for metric, idx in self.final_metrics_map.items()}

        self.local_gbs = self.args.global_batch_size//self.args.data_parallel_size
        self.reset()

    def reset(self):
        self.current_mbs = 0
        self.partial_results = torch.full((self.local_gbs, len(self.metrics)), torch.inf, dtype=torch.float32, device="cuda")
        self.gathered_final_metrics = None

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


    @torch.no_grad()
    def update(self, x: torch.Tensor, name: str):
        # x.shape = [mbs]
        if len(self.metrics) == 0 or not self.enabled:
            return
        if name not in self.final_metrics_map:
            return
        assert self.gathered_final_metrics is None, "Call tracker.reset() before doing tracker.update() after an aggregation"

        mbs = x.size(0)
        batch_slice = slice(self.current_mbs*mbs, (self.current_mbs + 1)*mbs)
        name_idx = self.final_metrics_map[name]
        assert (self.current_mbs + 1)*mbs <= self.local_gbs
        assert torch.all(torch.isinf(self.partial_results[batch_slice, name_idx]))
        assert torch.all(~torch.isinf(x))
        self.partial_results[batch_slice, name_idx] = x.clone()

        if torch.all(~torch.isinf(self.partial_results[batch_slice, :])):
            self.current_mbs += 1

    @torch.no_grad()
    def aggregate(self):
        if len(self.metrics) == 0:
            return
        assert self.enabled, "Can't aggregate metrics disabled"
        assert self.gathered_final_metrics is None, "Tracker has already aggregated metrics"

        # Now we want to aggregate across DP
        # All metrics can be aggregated simply by taking the avg across DP.
        assert torch.all(~torch.isinf(self.partial_results))
        final_metrics = torch.mean(self.partial_results, dim=0)  # size=(num_metrics,)

        # DP all_reduce.
        dp_group = parallel_state.get_data_parallel_group()
        dp_size = torch.distributed.get_world_size(dp_group)
        if dp_size > 1:
            torch.distributed.all_reduce(final_metrics, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        self.gathered_final_metrics = final_metrics

    def get_final_metrics(self) -> Iterator[tuple[str, float]]:
        if len(self.metrics) == 0:
            return iter([])
        assert self.gathered_final_metrics is not None
        values = self.gathered_final_metrics.tolist()
        for name_idx, value in enumerate(values):
            name = self._inv_final_metrics_map[name_idx]
            yield name, value


_GLOBAL_TRACKER: Optional[GlobalTracker] = None


def init_tracker(args, metrics: list[str]):
    global _GLOBAL_TRACKER
    assert _GLOBAL_TRACKER is None, "tracker already initialized"
    _GLOBAL_TRACKER = GlobalTracker(args, metrics=metrics)


def get_tracker() -> GlobalTracker:
    global _GLOBAL_TRACKER
    assert _GLOBAL_TRACKER is not None, "tracker not initialized"
    return _GLOBAL_TRACKER
