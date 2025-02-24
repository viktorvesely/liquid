import torch
import torch.nn as nn

class Citizen(nn.Module):

    def __init__(self, n_citizens: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=60, out_channels=90, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=90, out_channels=120, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        n_body_f = 120 * 4

        def mid(source, target):
            return int(round(source + (target - source) / 2))


        self.class_head = nn.Sequential(
            nn.Linear(n_body_f, mid(n_body_f, 10)),
            nn.LeakyReLU(),
            nn.Linear(mid(n_body_f, 10), 10),
            nn.Softmax(dim=1)
        )

        self.delegate_head = nn.Sequential(
            nn.Linear(n_body_f, mid(n_body_f, n_citizens)),
            nn.LeakyReLU(),
            nn.Linear(mid(n_body_f, n_citizens), n_citizens),
            nn.Softmax(dim=1)
        )


    def forward(self, x: torch.Tensor):

        b = self.body(x)
        c = self.class_head(b)
        d = self.delegate_head(b)

        return c, d

    @classmethod
    def solve_delegation(cls, Ds: list[torch.Tensor]) -> torch.Tensor:

        # Create a delegation matrix
        Ds_cat_ready = [torch.unsqueeze(d, -1) for d in Ds]
        D = torch.cat(Ds_cat_ready, dim=-1)
        n_batch, n, _ = D.shape

        # Extract the power the model wants to keep for itself
        p = torch.diagonal(D, dim1=-2, dim2=-1)
        mask = torch.ones_like(D)
        batch_indices = torch.arange(n_batch).unsqueeze(1).unsqueeze(2)
        diag_indices = torch.arange(n).unsqueeze(0).expand(n, -1)
        mask[batch_indices, diag_indices, diag_indices] = 0
        D_no_diag = D * mask

        identity = torch.zeros_like(D)
        identity[batch_indices, diag_indices, diag_indices] = 1

        p_column = p.unsqueeze(-1)
        inverse = torch.pinverse(identity - D_no_diag)
        influence = torch.bmm(inverse, p_column)
        influence = torch.squeeze(influence)

        return influence
