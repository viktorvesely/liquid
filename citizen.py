import torch
import torch.nn as nn

class Citizen(nn.Module):

    def __init__(self, n_citizens: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=60, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=60, out_channels=120, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=120, out_channels=180, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=180, out_channels=240, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        n_body_f = 240 * 4

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

