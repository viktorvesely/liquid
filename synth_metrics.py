import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import normalized_mutual_info_score

from LE import Liquid
from other.ME import Moe


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

def inference(council: Liquid | Moe, dataset: TensorDataset, bs: int = 1_000):

    loader = DataLoader(dataset, bs)

    classifications = []
    Ps = []

    with torch.no_grad():
        for samples, _ in loader:

            samples = samples.to("cuda")

            c = council.model(samples)
            classifications.append(c.cpu())

            if isinstance(council, Liquid):
                Ps.append(council.liquid_ensemble.last_power.cpu())
            else:
                Ps.append(council.moe_layer.last_gate.cpu())

    classifications = torch.cat(classifications, dim=0)
    Ps = torch.cat(Ps, dim=0)

    return classifications, Ps

def calc_metrics(
        council: Liquid | Moe,
        dataset: TensorDataset,
        classifications: torch.Tensor,
        Ps: torch.Tensor,
        verbal: bool = False
    ):
    with torch.no_grad():

        hatlabel = torch.argmax(classifications, 1)
        accuracy = (dataset.tensors[1]  == hatlabel).to(float).mean().item()

        power_voter = torch.argmax(Ps, 1)
        region_classes = get_regions_classes(dataset.tensors[0])

        region_nmi = normalized_mutual_info_score(
            power_voter.cpu().numpy(), region_classes.cpu().numpy()
        )

        if not isinstance(region_nmi, float):
            region_nmi = region_nmi.item()

        if isinstance(council, Liquid):
            power_entropy = council.liquid_ensemble.power_entropy(Ps).item()
            speaker_entropy = council.liquid_ensemble.speaker_entropy(Ps).item()
        else:
            power_entropy = council.moe_layer.power_entropy(Ps).item()
            speaker_entropy = council.moe_layer.speaker_entropy(Ps).item()

    if verbal:
        print(f"accuracy {accuracy:.3f}")
        print(f"power_entropy {power_entropy:.3f}")
        print(f"speaker_entropy {speaker_entropy:.3f}")
        print(f"region_nmi {region_nmi:.3f}")

    return accuracy, power_entropy, speaker_entropy, region_nmi
