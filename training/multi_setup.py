from torch.nn.modules.container import ModuleDict
from .setup import Setup


class MultiSetup(Setup):
    def __init__(self, setups):
        super().__init__()
        self.setups = []
        for name, setup in setups.items():
            self.setups.append(setup)
            setattr(self, name, setup)

    def forward(self, x):
        out = self.main(x)
        out = self.monodepth(x, out)
        return out