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
import matplotlib.pyplot as plt

from coco_dataset import COCODatasetFPN
from model import DetectionLossFPN, ObjectDetectorFPN
from metrics import DetectionMetricsFPN
from config import ANCHORS

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
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
            for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
                if ema_b.dtype.is_floating_point:
                    ema_b.data.mul_(self.decay).add_(model_b.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs,
                 save_dir="./models", resume_from=None, use_wandb=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = DetectionLossFPN(80, ANCHORS, label_smoothing=0.1).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_map = 0.0
        self.best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        self.history = {'train': {}, 'val': {}}
        self.metrics = DetectionMetricsFPN(80, ANCHORS)

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
            checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_map = checkpoint.get('best_map', 0.0)
            self.history = checkpoint['history']
            self.start_epoch = checkpoint['epoch'] + 1
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])

        if self.use_wandb:
            wandb.init(
                project="yolo-detector",
                config={
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "lr": 3e-4,
                    "optimizer": "AdamW",
                    "scheduler": "OneCycleLR",
                    "label_smoothing": 0.1,
                    "ema_decay": 0.9999,
                    "mixed_precision": True,
                    "anchors": ANCHORS,
                },
            )
            wandb.watch(self.model, log="gradients", log_freq=100)

    def _save_model(self, epoch, val_loss, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'best_map': self.best_map,
            'history': self.history,
            'val_loss': val_loss,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

    def _save_plot_history(self, history, save=False):
        epochs = list(history['train'].keys())
        train_losses = [history['train'][e]['loss'] for e in epochs]
        val_losses = [history['val'][e]['loss'] for e in epochs]
        plt.figure()
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.title('Train and Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'metrics_results.png'))
        plt.close()

    def _train_epoch(self):
        self.model.train()
        total_obj_loss, total_class_loss, total_box_loss, total_losses = 0, 0, 0, 0

        for batch in tqdm(self.train_loader, leave=False):
            images, targets_p3, targets_p4, targets_p5 = batch
            images = images.to(self.device, non_blocking=True)
            targets_p3 = tuple(t.to(self.device, non_blocking=True) for t in targets_p3)
            targets_p4 = tuple(t.to(self.device, non_blocking=True) for t in targets_p4)
            targets_p5 = tuple(t.to(self.device, non_blocking=True) for t in targets_p5)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                predictions = self.model(images)
                targets = (targets_p3, targets_p4, targets_p5)
                obj_loss, class_loss, box_loss, total_loss = self.criterion(predictions, targets)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self.model)

            total_obj_loss += obj_loss.detach()
            total_class_loss += class_loss.detach()
            total_box_loss += box_loss.detach()
            total_losses += total_loss.detach()

        n = len(self.train_loader)
        return (total_obj_loss / n).item(), (total_class_loss / n).item(), \
               (total_box_loss / n).item(), (total_losses / n).item()

    def _validate(self, compute_map=False):
        self.ema.ema.eval()
        total_obj_loss, total_class_loss, total_box_loss, total_losses = 0, 0, 0, 0

        if compute_map:
            all_pred_boxes = []
            all_gt_boxes = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                images, targets_p3, targets_p4, targets_p5 = batch
                images = images.to(self.device)
                targets_p3 = tuple(t.to(self.device) for t in targets_p3)
                targets_p4 = tuple(t.to(self.device) for t in targets_p4)
                targets_p5 = tuple(t.to(self.device) for t in targets_p5)

                with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    predictions = self.ema.ema(images)
                    targets = (targets_p3, targets_p4, targets_p5)
                    obj_loss, class_loss, box_loss, total_loss = self.criterion(predictions, targets)

                total_obj_loss += obj_loss.item()
                total_class_loss += class_loss.item()
                total_box_loss += box_loss.item()
                total_losses += total_loss.item()

                if compute_map:
                    batch_preds = self.metrics.decode_prediction_fpn(predictions)
                    batch_gts = self.metrics.extract_gt_boxes_fpn(targets)
                    all_pred_boxes.extend(batch_preds)
                    all_gt_boxes.extend(batch_gts)

        n = len(self.val_loader)
        losses = (total_obj_loss / n, total_class_loss / n,
                  total_box_loss / n, total_losses / n)

        if compute_map:
            map_score = self.metrics.compute_map_from_decoded(all_pred_boxes, all_gt_boxes)
            return losses, map_score

        return losses, None

    def train(self):
        print(f"Device : {self.device}")
        try:
            for epoch in range(self.start_epoch, self.epochs):
                train_obj, train_cls, train_box, train_loss = self._train_epoch()
                (val_obj, val_cls, val_box, val_loss), map_score = self._validate(
                    compute_map=(epoch % 3 == 0 or epoch >= self.epochs - 5)
                )

                current_lr = self.optimizer.param_groups[0]['lr']

                self.history['train'][epoch] = {
                    'obj_loss': train_obj, 'class_loss': train_cls,
                    'box_loss': train_box, 'loss': train_loss,
                }
                self.history['val'][epoch] = {
                    'obj_loss': val_obj, 'class_loss': val_cls,
                    'box_loss': val_box, 'loss': val_loss,
                }

                log_dict = {
                    'train/obj_loss': train_obj, 'train/class_loss': train_cls,
                    'train/box_loss': train_box, 'train/total_loss': train_loss,
                    'val/obj_loss': val_obj, 'val/class_loss': val_cls,
                    'val/box_loss': val_box, 'val/total_loss': val_loss,
                    'lr': current_lr, 'epoch': epoch,
                }
                if map_score is not None:
                    log_dict['val/mAP@0.5'] = map_score

                if self.use_wandb:
                    wandb.log(log_dict)

                print(f"\n{'=' * 60}")
                print(f"Epoque : {epoch + 1}/{self.epochs} | LR : {current_lr:.6f}")
                print(f"Train - Loss : {train_loss:.4f} | Obj : {train_obj:.4f} | "
                      f"Class : {train_cls:.4f} | Box : {train_box:.4f}")
                print(f"Val   - Loss : {val_loss:.4f} | Obj : {val_obj:.4f} | "
                      f"Class : {val_cls:.4f} | Box : {val_box:.4f}")
                if map_score is not None:
                    print(f"mAP@0.5 : {map_score:.4f}")

                if map_score is not None and map_score > self.best_map:
                    self._save_model(epoch, val_loss, self.best_model_path)
                    self.best_map = map_score
                    print(f"Meilleur modele sauvegarde ! (mAP@0.5 : {map_score:.4f})")

        except KeyboardInterrupt:
            print("Entrainement interrompu")
        finally:
            with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=4)
            self._save_plot_history(self.history, save=True)
            if self.use_wandb:
                wandb.finish()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    EPOCHS = 100
    BATCH_SIZE = 64

    train_dataset = COCODatasetFPN('/content/annotations/instances_train2017.json', '/content/train2017', ANCHORS, train=True)
    val_dataset = COCODatasetFPN('/content/annotations/instances_val2017.json', '/content/val2017', ANCHORS, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model = ObjectDetectorFPN(80)
    trainer = Trainer(model, train_loader, val_loader, epochs=EPOCHS, use_wandb=True)
    trainer.train()
