import torch
import numpy as np


class ChangeDetectionEvaluation:
    def __init__(self, device=torch.device('cpu'), best_f1=0, best_recall=0, best_precision=0, best_accuracy=0):
        self.device = device
        self.tp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.tn = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fn = torch.tensor(0, device=self.device, dtype=torch.float32)

        self.best_f1 = best_f1
        self.best_recall = best_recall
        self.best_precision = best_precision
        self.best_accuracy = best_accuracy

        self.is_best_f1 = False
        self.is_best_recall = False
        self.is_best_precision = False
        self.is_best_accuracy = False

    def update_best(self):
        aux_f1 = self.get_f1().cpu().numpy()
        aux_recall = self.get_recall().cpu().numpy()
        aux_precision = self.get_precision().cpu().numpy()
        aux_accuracy = self.get_accuracy().cpu().numpy()

        if aux_f1 > self.best_f1:
            self.best_f1 = aux_f1

            self.is_best_f1 = True

        if aux_recall > self.best_recall:
            self.best_recall = aux_recall

            self.is_best_recall = True

        if aux_precision > self.best_precision:
            self.best_precision = aux_precision

            self.is_best_precision = True

        if aux_accuracy > self.best_accuracy:
            self.best_accuracy = aux_accuracy

            self.is_best_accuracy = True

    def reset(self):
        self.tp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.tn = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fn = torch.tensor(0, device=self.device, dtype=torch.float32)

        self.is_best_f1 = False
        self.is_best_recall = False
        self.is_best_precision = False
        self.is_best_accuracy = False

    def update(self, ground_truth, prediction):
        prediction = prediction > 0.5

        self.tp += (prediction * ground_truth).sum()
        self.tn += ((prediction == 0) * (ground_truth == 0)).sum()

        self.fn += ((prediction == 0) * (ground_truth == 1)).sum()
        self.fp += ((prediction == 1) * (ground_truth == 0)).sum()

    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp / (self.tp + self.fn)

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()

        return 2 * precision * recall / (precision + recall)

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fn + self.tn + self.fp)

    def get_confusion_matrix(self):
        return np.array([[self.tp.cpu().numpy(), self.fn.cpu().numpy()],
                         [self.fp.cpu().numpy(), self.tn.cpu().numpy()]])

