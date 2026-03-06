import copy
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coco_dataset_FPN_v2 import COCODatasetFPNv2
from modelFPN_ResNet_v2 import ObjectDetectorFPN, DetectionLossFPN
from metricsFPN_v2 import DetectionMetricsFPNv2
from config_v2 import IMAGE_SIZE

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        d = self.decay
        with torch.no_grad():
            # Paramètres
            for ep, mp in zip(self.ema.parameters(), model.parameters()):
                ep.data.mul_(d).add_(mp.data, alpha=1 - d)
            # Buffers (running_mean / running_var des BN)
            for eb, mb in zip(self.ema.buffers(), model.buffers()):
                if eb.dtype.is_floating_point:
                    eb.data.mul_(d).add_(mb.data, alpha=1 - d)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs,
                 save_dir='./models', resume_from=None, use_wandb=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = DetectionLossFPN().to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_map = 0.0
        self.best_model_path = os.path.join(save_dir, 'best_model.pt')
        self.history = {'train': {}, 'val': {}}
        self.metrics = DetectionMetricsFPNv2()

        self.optimizer = AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=3e-4,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=5 / max(epochs, 6),
            anneal_strategy='cos',
        )
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        self.ema = ModelEMA(self.model, decay=0.9999)
        self.start_epoch = 0
        self.use_wandb = use_wandb and HAS_WANDB

        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.best_map = ckpt.get('best_map', 0.0)
            self.history = ckpt['history']
            self.start_epoch = ckpt['epoch'] + 1
            if 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])

        if self.use_wandb:
            wandb.init(project='yolo-v2', config={
                'epochs': epochs, 'batch_size': train_loader.batch_size,
                'lr': 3e-4, 'optimizer': 'AdamW', 'scheduler': 'OneCycleLR',
                'label_smoothing': 0.1, 'ema_decay': 0.9999,
            })
            wandb.watch(self.model, log='gradients', log_freq=200)

    def _save_checkpoint(self, epoch, val_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'best_map': self.best_map,
            'history': self.history,
            'val_loss': val_loss,
        }, self.best_model_path)

    def _save_plot(self):
        epochs = sorted(self.history['train'].keys(), key=int)
        tl = [self.history['train'][e]['loss'] for e in epochs]
        vl = [self.history['val'][e]['loss'] for e in epochs]
        plt.figure()
        plt.plot(epochs, tl, label='Train')
        plt.plot(epochs, vl, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.close()

    def _to_device(self, batch):
        images, t0, t1, t2 = batch
        images = images.to(self.device, non_blocking=True)
        t0 = tuple(t.to(self.device, non_blocking=True) for t in t0)
        t1 = tuple(t.to(self.device, non_blocking=True) for t in t1)
        t2 = tuple(t.to(self.device, non_blocking=True) for t in t2)
        return images, (t0, t1, t2)

    def _train_epoch(self):
        self.model.train()
        sums = [0.0, 0.0, 0.0, 0.0]
        for batch in tqdm(self.train_loader, leave=False):
            images, targets = self._to_device(batch)
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                preds = self.model(images)
                obj_l, cls_l, box_l, total_l = self.criterion(preds, targets)
            self.scaler.scale(total_l).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self.model)
            sums[0] += obj_l.item()
            sums[1] += cls_l.item()
            sums[2] += box_l.item()
            sums[3] += total_l.item()
        n = len(self.train_loader)
        return [s / n for s in sums]

    def _validate(self, compute_map=False):
        self.ema.ema.eval()
        sums = [0.0, 0.0, 0.0, 0.0]
        all_preds, all_gts = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                images, targets = self._to_device(batch)
                with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    preds = self.ema.ema(images)
                    obj_l, cls_l, box_l, total_l = self.criterion(preds, targets)
                sums[0] += obj_l.item()
                sums[1] += cls_l.item()
                sums[2] += box_l.item()
                sums[3] += total_l.item()
                if compute_map:
                    all_preds.extend(self.metrics.decode_prediction_fpn(preds))
                    all_gts.extend(self.metrics.extract_gt_boxes_fpn(targets))

        n = len(self.val_loader)
        losses = [s / n for s in sums]
        map_score = self.metrics.compute_map_from_decoded(all_preds, all_gts) if compute_map else None
        return losses, map_score

    def train(self):
        print(f'Device : {self.device}')
        try:
            for epoch in range(self.start_epoch, self.epochs):
                do_map = (epoch % 3 == 0) or (epoch >= self.epochs - 5)
                tr = self._train_epoch()
                vl, map_score = self._validate(compute_map=do_map)
                lr = self.optimizer.param_groups[0]['lr']

                self.history['train'][epoch] = {
                    'obj_loss': tr[0], 'class_loss': tr[1],
                    'box_loss': tr[2], 'loss': tr[3],
                }
                self.history['val'][epoch] = {
                    'obj_loss': vl[0], 'class_loss': vl[1],
                    'box_loss': vl[2], 'loss': vl[3],
                }

                log = {
                    'train/obj': tr[0], 'train/cls': tr[1],
                    'train/box': tr[2], 'train/total': tr[3],
                    'val/obj': vl[0], 'val/cls': vl[1],
                    'val/box': vl[2], 'val/total': vl[3],
                    'lr': lr, 'epoch': epoch,
                }
                if map_score is not None:
                    log['val/mAP@0.5'] = map_score
                if self.use_wandb:
                    wandb.log(log)

                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.epochs} | LR {lr:.6f}")
                print(f"Train - total:{tr[3]:.4f}  obj:{tr[0]:.4f}  cls:{tr[1]:.4f}  box:{tr[2]:.4f}")
                print(f"Val   - total:{vl[3]:.4f}  obj:{vl[0]:.4f}  cls:{vl[1]:.4f}  box:{vl[2]:.4f}")
                if map_score is not None:
                    print(f"mAP@0.5 : {map_score:.4f}")

                if map_score is not None and map_score > self.best_map:
                    self.best_map = map_score
                    self._save_checkpoint(epoch, vl[3])
                    print(f"Best model saved (mAP {map_score:.4f})")

        except KeyboardInterrupt:
            print('Training interrupted')
        finally:
            with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
            self._save_plot()
            if self.use_wandb:
                wandb.finish()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    EPOCHS = 100
    BATCH_SIZE = 64

    train_ds = COCODatasetFPNv2(
        '/content/annotations/instances_train2017.json',
        '/content/train2017', train=True,
    )
    val_ds = COCODatasetFPNv2(
        '/content/annotations/instances_val2017.json',
        '/content/val2017', train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model = ObjectDetectorFPN()
    trainer = Trainer(model, train_loader, val_loader, epochs=EPOCHS, use_wandb=True)
    trainer.train()
