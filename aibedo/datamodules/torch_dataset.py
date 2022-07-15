import torch


class AIBEDOTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, targets):
        assert X.shape[0] == targets.shape[0]
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else torch.tensor(X).float()
        self.monsoon_index = torch.tensor(targets)

    def __getitem__(self, i):
        return self.X[i], self.monsoon_index[i]

    def __len__(self):
        return self.X.shape[0]