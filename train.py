from coco_dataset import COCODataset
from model import DetectionLoss, ObjectDetector
from metrics import DetectionMetrics
from torch.optim import Adam, lr_scheduler
import os
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

class Trainer: 
    def __init__(self, model, train_loader, val_loader, save_dir="./models"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = DetectionLoss(80).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(self.model.parameters(), lr = 0.001, weight_decay=1e-4)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_loss = float('inf')
        self.best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        self.history = {
            'train': {},
            'val': {}
        }
        self.metrics = DetectionMetrics(80)
    
    def _save_plot_history(self, history, save=False):
        epochs = list(history['train'].keys())
        train_losses = [history['train'][epoch]['loss'] for epoch in epochs]
        val_losses = [history['val'][epoch]['loss'] for epoch in epochs]
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.title('Train and Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if save:
            plt.savefig(os.path.join(self.save_dir, 'metrics_results.png'))
        plt.show()
    
    def _train_epoch(self):
        print("Début de l'epoch...")
        self.model.train()
        total_obj_loss, total_class_loss, total_box_loss, total_losses = 0, 0, 0, 0
        for i, batch in enumerate(tqdm(self.train_loader, leave=False)):
            images, obj_targets, class_targets, box_targets = batch
            images = images.to(self.device, non_blocking=True)
            obj_targets = obj_targets.to(self.device, non_blocking=True)
            class_targets = class_targets.to(self.device, non_blocking=True)
            box_targets = box_targets.to(self.device, non_blocking=True)
            obj_preds, class_preds, box_preds = self.model(images)
            obj_loss, class_loss, box_loss, total_loss = self.criterion((obj_preds, class_preds, box_preds), (obj_targets, class_targets, box_targets))
            total_obj_loss += obj_loss.detach()
            total_class_loss += class_loss.detach()
            total_box_loss += box_loss.detach()
            total_losses += total_loss.detach()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return (total_obj_loss / len(self.train_loader)).item(), \
       (total_class_loss / len(self.train_loader)).item(), \
       (total_box_loss / len(self.train_loader)).item(), \
       (total_losses / len(self.train_loader)).item()

    def _validate(self, compute_map=False):
        self.model.eval()
        total_obj_loss, total_class_loss, total_box_loss, total_losses = 0, 0, 0, 0
        if compute_map:
            all_obj_preds = []
            all_class_preds = []
            all_box_preds = []
            all_obj_targets = []
            all_class_targets = []
            all_box_targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                images, obj_targets, class_targets, box_targets = batch
                images = images.to(self.device)
                obj_targets = obj_targets.to(self.device)
                class_targets = class_targets.to(self.device)
                box_targets = box_targets.to(self.device)
                obj_preds, class_preds, box_preds = self.model(images)
                obj_loss, class_loss, box_loss, total_loss = self.criterion((obj_preds, class_preds, box_preds), (obj_targets, class_targets, box_targets))
                total_obj_loss += obj_loss.item()
                total_class_loss += class_loss.item()
                total_box_loss += box_loss.item()
                total_losses += total_loss.item()
                if compute_map:
                    all_obj_preds.append(obj_preds.cpu())
                    all_class_preds.append(class_preds.cpu())
                    all_box_preds.append(box_preds.cpu())
                    all_obj_targets.append(obj_targets.cpu())
                    all_class_targets.append(class_targets.cpu())
                    all_box_targets.append(box_targets.cpu())
        losses = total_obj_loss / len(self.val_loader), total_class_loss / len(self.val_loader), total_box_loss / len(self.val_loader), total_losses / len(self.val_loader)
        if compute_map:
            obj_preds_all = torch.cat(all_obj_preds)
            class_preds_all = torch.cat(all_class_preds)
            box_preds_all = torch.cat(all_box_preds)
            obj_targets_all = torch.cat(all_obj_targets)
            class_targets_all = torch.cat(all_class_targets)
            box_targets_all = torch.cat(all_box_targets)
            
            map_score = self.metrics.compute_map(
                (obj_preds_all, class_preds_all, box_preds_all),
                (obj_targets_all, class_targets_all, box_targets_all)
            )
            return losses, map_score
        return losses, None
    
    def train(self, epochs):
        print(f"Device : {self.device}")
        print(f"Model on GPU: {next(self.model.parameters()).is_cuda}")
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        try:
            for epoch in range(epochs):
                train_obj_loss, train_class_loss, train_box_loss, train_losses = self._train_epoch()
                val_losses = self._validate(compute_map=(epoch % 5 == 0))
                if isinstance(val_losses, tuple):
                    (val_obj_loss, val_class_loss, val_box_loss, val_loss), map_score = val_losses
                    if map_score is not None:
                        print(f"mAP@0.5: {map_score:.4f}")
                scheduler.step(val_loss)
                print(f"Learning rate : {scheduler.get_last_lr()[0]}")
                self.history['train'][epoch] = {'obj_loss': train_obj_loss, 'class_loss': train_class_loss, 'box_loss': train_box_loss, 'loss': train_losses}
                self.history['val'][epoch] = {'obj_loss': val_obj_loss, 'class_loss': val_class_loss, 'box_loss': val_box_loss, 'loss': val_loss}

                print(f"\n{'=' * 60}")
                print(f"Epoque : {epoch + 1}/{epochs}")
                print(f"\n{'=' * 60}")
                print(f"Train - Loss : {train_losses} | Obj : {train_obj_loss} | Class : {train_class_loss} | Box : {train_box_loss}")
                print(f"Val - Loss : {val_loss} | Obj : {val_obj_loss} | Class : {val_class_loss} | Box : {val_box_loss}")

                if val_loss < self.best_loss:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.best_loss = val_loss
                    print(f"Meilleur modèle sauvegardé ! (Val Loss : {val_loss})")
        except KeyboardInterrupt:
            print("Entraînement interrompu")
        finally:
            print("Sauvegarde de l'historique")
            with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=4)
            self._save_plot_history(self.history, True)
            print("Historique sauvegardé !")
        
            


if __name__ == "__main__":
    train_dataset = COCODataset('coco2017/annotations/instances_train2017.json', 'coco2017/train2017')
    val_dataset = COCODataset('coco2017/annotations/instances_val2017.json', 'coco2017/val2017', train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ObjectDetector(80)
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=100)