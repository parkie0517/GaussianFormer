from jaxtyping import Float, Int64, Shaped
from torch import Tensor
from einops import reduce
import torch


def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
):
# tuple[
#     Int64[Tensor, "*batch sample"],  # index
#     Float[Tensor, "*batch sample"],  # probability density
# ]
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum")) # kinda like sotfmax but does not use the exponential function 
    cdf = normalized_pdf.cumsum(dim=-1) 
    samples = torch.rand((*batch, num_samples), device=pdf.device) # 왜 랜덤 값을 샘플링하지?
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
):
    # tuple[
    #     Int64[Tensor, "*batch sample"],  # index
    #     Float[Tensor, "*batch sample"],  # probability density
    # ]
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)


class DistributionSampler:
    def sample(
        self,
        pdf: Float[Tensor, "*batch bucket"],
        deterministic: bool,
        num_samples: int,
    ):
    # tuple[
    #     Int64[Tensor, "*batch sample"],  # index
    #     Float[Tensor, "*batch sample"],  # probability density
    # ]
        """Sample from the given probability distribution. USES `inverse transform sampling` technique
        Return sampled indices and their corresponding probability densities.
        """
        if deterministic:
            index, densities = gather_discrete_topk(pdf, num_samples)
        else:
            index, densities = sample_discrete_distribution(pdf, num_samples)
        return index, densities

    def gather(
        self,
        index: Int64[Tensor, "*batch sample"],
        target: Shaped[Tensor, "..."],  # *batch bucket *shape
    ) -> Shaped[Tensor, "..."]:  # *batch *shape
        """Gather from the target according to the specified index. Handle the
        broadcasting needed for the gather to work. See the comments for the actual
        expected input/output shapes since jaxtyping doesn't support multiple variadic
        lengths in annotations.
        """
        bucket_dim = index.ndim - 1
        while len(index.shape) < len(target.shape):
            index = index[..., None]
        broadcasted_index_shape = list(target.shape) # torch.Size([1, 6, 108, 200, 128, 3])
        broadcasted_index_shape[bucket_dim] = index.shape[bucket_dim] # 해당 값은 anchors_per_pixel와 같음 값. 즉, 하나의 ray에서 몇 개를 anchor를 사용할 거냐?
        index = index.broadcast_to(broadcasted_index_shape)

        # Add the ability to broadcast.
        if target.shape[bucket_dim] == 1:
            index = torch.zeros_like(index)

        return target.gather(dim=bucket_dim, index=index) # 우리가 원하는 bin의 vehicle 좌표를 반환
