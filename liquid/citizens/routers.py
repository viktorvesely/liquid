class VisionRouter(Citizen):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_citizens: int,
            layers: int,
            max_pool_every: int = 2
        ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.max_pool_every = max_pool_every
        self.n_citizens = n_citizens

        self.layers = monotonically_increasing_cnn(
            in_channels,
            out_channels,
            depth=layers,
            max_pool_every=max_pool_every
        )

        self.out = FinalGlobalHead(out_channels, n_citizens)


    def forward(self, x: torch.Tensor):
        h = self.layers(x)
        return self.out(h)

    def get_constructor(self) -> dict:

        return {
            "in_channels":  self.in_channels,
            "out_channels":  self.out_channels,
            "layers":  self.layers,
            "max_pool_every": self.max_pool_every,
            "n_citizens": self.n_citizens
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance