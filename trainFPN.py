from coco_dataset_FPN import COCODatasetFPN
from modelFPN import DetectionLossFPN, ObjectDetectorFPN
from metricsFPN import DetectionMetricsFPN
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
        self.criterion = DetectionLossFPN(80).to(self.device)
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
        self.metrics = DetectionMetricsFPN(80)
    
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
        self.model.train()
        total_obj_loss, total_class_loss, total_box_loss, total_losses = 0, 0, 0, 0
        for i, batch in enumerate(tqdm(self.train_loader, leave=False)):
            images, targets_p3, targets_p4, targets_p5 = batch
            images = images.to(self.device, non_blocking=True)
            targets_p3 = tuple(t.to(self.device, non_blocking=True) for t in targets_p3)
            targets_p4 = tuple(t.to(self.device, non_blocking=True) for t in targets_p4)
            targets_p5 = tuple(t.to(self.device, non_blocking=True) for t in targets_p5)
            predictions = self.model(images)
            targets = (targets_p3, targets_p4, targets_p5)
            obj_loss, class_loss, box_loss, total_loss = self.criterion(predictions, targets)
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
            all_preds_p3 = {'obj': [], 'cls': [], 'box': []}
            all_preds_p4 = {'obj': [], 'cls': [], 'box': []}
            all_preds_p5 = {'obj': [], 'cls': [], 'box': []}
            all_targets_p3 = {'obj': [], 'cls': [], 'box': []}
            all_targets_p4 = {'obj': [], 'cls': [], 'box': []}
            all_targets_p5 = {'obj': [], 'cls': [], 'box': []}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                images, targets_p3, targets_p4, targets_p5 = batch
                images = images.to(self.device)
                targets_p3 = tuple(t.to(self.device) for t in targets_p3)
                targets_p4 = tuple(t.to(self.device) for t in targets_p4)
                targets_p5 = tuple(t.to(self.device) for t in targets_p5)
                
                predictions = self.model(images)
                targets = (targets_p3, targets_p4, targets_p5)
                obj_loss, class_loss, box_loss, total_loss = self.criterion(predictions, targets)
                
                total_obj_loss += obj_loss.item()
                total_class_loss += class_loss.item()
                total_box_loss += box_loss.item()
                total_losses += total_loss.item()
                
                if compute_map:
                    all_preds_p3['obj'].append(predictions[0][0].cpu())
                    all_preds_p3['cls'].append(predictions[0][1].cpu())
                    all_preds_p3['box'].append(predictions[0][2].cpu())
                    all_targets_p3['obj'].append(targets_p3[0].cpu())
                    all_targets_p3['cls'].append(targets_p3[1].cpu())
                    all_targets_p3['box'].append(targets_p3[2].cpu())

                    all_preds_p4['obj'].append(predictions[1][0].cpu())
                    all_preds_p4['cls'].append(predictions[1][1].cpu())
                    all_preds_p4['box'].append(predictions[1][2].cpu())
                    all_targets_p4['obj'].append(targets_p4[0].cpu())
                    all_targets_p4['cls'].append(targets_p4[1].cpu())
                    all_targets_p4['box'].append(targets_p4[2].cpu())

                    all_preds_p5['obj'].append(predictions[2][0].cpu())
                    all_preds_p5['cls'].append(predictions[2][1].cpu())
                    all_preds_p5['box'].append(predictions[2][2].cpu())
                    all_targets_p5['obj'].append(targets_p5[0].cpu())
                    all_targets_p5['cls'].append(targets_p5[1].cpu())
                    all_targets_p5['box'].append(targets_p5[2].cpu())
        
        losses = (total_obj_loss / len(self.val_loader), 
                total_class_loss / len(self.val_loader), 
                total_box_loss / len(self.val_loader), 
                total_losses / len(self.val_loader))
        
        if compute_map:
            preds = (
                (torch.cat(all_preds_p3['obj']), torch.cat(all_preds_p3['cls']), torch.cat(all_preds_p3['box'])),
                (torch.cat(all_preds_p4['obj']), torch.cat(all_preds_p4['cls']), torch.cat(all_preds_p4['box'])),
                (torch.cat(all_preds_p5['obj']), torch.cat(all_preds_p5['cls']), torch.cat(all_preds_p5['box']))
            )
            
            targets_all = (
                (torch.cat(all_targets_p3['obj']), torch.cat(all_targets_p3['cls']), torch.cat(all_targets_p3['box'])),
                (torch.cat(all_targets_p4['obj']), torch.cat(all_targets_p4['cls']), torch.cat(all_targets_p4['box'])),
                (torch.cat(all_targets_p5['obj']), torch.cat(all_targets_p5['cls']), torch.cat(all_targets_p5['box']))
            )
            
            map_score = self.metrics.compute_map(preds, targets_all)
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
    train_dataset = COCODatasetFPN('train_labels.json', 'images')
    val_dataset = COCODatasetFPN('val_labels.json', 'images', train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model = ObjectDetectorFPN(80)
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=100)