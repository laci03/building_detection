import torch


class ChangeDetectionEvaluation:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.tp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.tn = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fn = torch.tensor(0, device=self.device, dtype=torch.float32)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fp = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.tn = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.fn = torch.tensor(0, device=self.device, dtype=torch.float32)

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
