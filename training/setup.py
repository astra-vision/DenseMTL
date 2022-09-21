import torch
import torch.nn as nn

from utils.logs import cpprint


class Setup(nn.Module):
    def load_setup(self, f):
        cpprint(f'{f} loaded onto setup', c='red')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state = torch.load(f, map_location=device)

        if f == 'models/3ways_checkpoint/best_model_without_opt.pkl':
            ms = state['model_state']
            prefix = 'models.encoder.'
            bstate = {k.split(prefix)[1]: v for k, v in ms.items() if prefix in k}
            self.main.backbone.load_state_dict(bstate)

            prefix = 'models.mtl_decoder.depth_dec.'
            dstate = {k.split(prefix)[1]: v for k, v in ms.items() if prefix in k}
            self.main.heads.disp.load_state_dict(dstate)

            prefix = 'models.mtl_decoder.seg_dec.'
            sstate = {k.split(prefix)[1]: v for k, v in ms.items() if prefix in k}
            diff = self.main.heads.semseg.load_state_dict(sstate, False)
            cpprint(diff)

            assert False

        else:
            diff = self.load_state_dict(state, False)
            cpprint(diff)