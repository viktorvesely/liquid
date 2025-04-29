import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import normalized_mutual_info_score

from liquid_ensemble import LiquidEnsembleLayer


def get_regions_classes(X: torch.Tensor) -> torch.Tensor:

    c = torch.zeros(X.shape[0], dtype=torch.int, device=X.device)
    x, y = X[:, 0], X[:, 1]
    mask_ul = (x < 0) & (y > 0)
    mask_ur = (x >= 0) & (y > 0)
    mask_ll = (x < 0) & (y <= 0)
    mask_lr = (x >= 0) & (y <= 0)

    c[mask_ul] = 0
    c[mask_ur] = 1
    c[mask_ll] = 2
    c[mask_lr] = 3

    return c

def inference(council: LiquidEnsembleLayer, dataset: TensorDataset, bs: int = 1_000):

    loader = DataLoader(dataset, bs)

    classifications = []
    Ds = []
    Ps = []

    with torch.no_grad():
        for samples, _ in loader:

            samples = samples.to("cuda")

            c = council(samples)
            classifications.append(c.cpu())
            Ds.append(council.last_D.cpu())
            Ps.append(council.last_power.cpu())

    classifications = torch.cat(classifications, dim=1)
    Ds = torch.cat(Ds, dim=0)
    Ps = torch.cat(Ps, dim=0)

    return classifications, Ds, Ps

def calc_metrics(
        council: LiquidEnsembleLayer,
        dataset: TensorDataset,
        classifications: torch.Tensor,
        Ps: torch.Tensor,
        verbal: bool = False
    ):
    with torch.no_grad():
        hatlabel = council.vote(classifications, Ps)
        accuracy = (dataset.tensors[1]  == hatlabel).to(float).mean().item()

        power_voter = torch.argmax(Ps, 1)
        region_classes = get_regions_classes(dataset.tensors[0])

        region_nmi = normalized_mutual_info_score(
            power_voter.cpu().numpy(), region_classes.cpu().numpy()
        )

        if not isinstance(region_nmi, float):
            region_nmi = region_nmi.item()

        power_entropy = council.power_entropy(Ps).item()
        speaker_entropy = council.speaker_entropy(Ps).item()

    if verbal:
        print(f"accuracy {accuracy:.3f}")
        print(f"power_entropy {power_entropy:.3f}")
        print(f"speaker_entropy {speaker_entropy:.3f}")
        print(f"region_nmi {region_nmi:.3f}")

    return accuracy, power_entropy, speaker_entropy, region_nmi
