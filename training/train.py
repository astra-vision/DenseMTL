from torch.utils.data.dataloader import DataLoader

from dataset import build_dataset
from vendor.advent.func import lr_poly
from training import get_setup
from training.weights import load_weights_
from optim import get_optimizer, get_scheduler
from loss import get_loss
from training.base import TrainerBase
from utils.logs import cpprint


class Trainer(TrainerBase):
    def __init__(self, config):
        super().__init__(config)
        self.model = get_setup(self.cfg, self.device)
        self.loss = get_loss(self.cfg, self.device)

        self.optimizer = get_optimizer(self.cfg, self.model)
        self.scheduler = get_scheduler(self.cfg, self.optimizer)

        load_weights_(self.model, self.cfg, self.device)

    def train_itr(self):
        return self.train_dl

    def saved_modules(self):
        return {'model', 'optimizer', 'scheduler'}

    def train_step(self, inputs, avg_meter):
        outputs = self.inference(inputs)
        loss, viz = self.loss(inputs, outputs, self.viz_train, train=True)
        avg_meter.update(loss)

        self.optimizer.zero_grad()
        loss.total.backward()
        self.model_step_()

        return viz

    def validate_step(self, inputs, outputs, avg_meter):
        loss, viz = self.loss(inputs, outputs, self.viz_valid, train=False)
        avg_meter.update(loss)
        return viz

    def setup_dls(self):
        data = self.cfg.data

        ds_common = dict(cfg=data,
                         load_sequence=data.load_sequences,
                         load_labels=data.load_semantic_gt)

        dl_common = dict(num_workers=self.cfg.training.n_workers,
                         pin_memory=True,
                         drop_last=True)

        self.train_ds = build_dataset(split=data.src_split, **ds_common)
        self.train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=data.train_bs,
            shuffle=self.cfg.training.shuffle_trainset,
            **dl_common)
        cpprint(f' train dataloader = {len(self.train_dl)} batches', c='blue')

        self.valid_ds = build_dataset(split=data.val_split, **ds_common)
        self.valid_dl = DataLoader(
            dataset=self.valid_ds,
            batch_size=data.valid_bs,
            **dl_common)

        xxx=self.train_ds[0]

        cpprint(f' validation dataloader = {len(self.valid_dl)} batches', c='blue')
