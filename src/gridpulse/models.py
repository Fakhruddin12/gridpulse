from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _tcn_classes():
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin, cout, kernel, dilation, dropout):
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv1 = nn.Conv1d(cin, cout, kernel, dilation=dilation)
            self.conv2 = nn.Conv1d(cout, cout, kernel, dilation=dilation)
            self.dropout = nn.Dropout(dropout)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
            self.relu = nn.ReLU()

        def forward(self, z):
            residual = self.skip(z)
            z = nn.functional.pad(z, (self.pad, 0))
            z = self.dropout(self.relu(self.conv1(z)))
            z = nn.functional.pad(z, (self.pad, 0))
            z = self.dropout(self.relu(self.conv2(z)))
            return self.relu(z + residual)

    class TCN(nn.Module):
        def __init__(self, input_channels, hidden, kernel, dropout, horizon):
            super().__init__()
            self.blocks = nn.Sequential(
                Block(input_channels, hidden, kernel, 1, dropout),
                Block(hidden, hidden, kernel, 2, dropout),
                Block(hidden, hidden, kernel, 4, dropout),
                Block(hidden, hidden, kernel, 8, dropout),
            )
            self.head = nn.Linear(hidden, horizon)

        def forward(self, z):
            return self.head(self.blocks(z.transpose(1, 2))[:, :, -1])

    return TCN


def fit_ridge(X: np.ndarray, future_calendar: np.ndarray, y: np.ndarray, alpha: float = 10.0):
    features = np.concatenate([X.reshape(len(X), -1), future_calendar.reshape(len(X), -1)], axis=1)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features, y)
    return model


def predict_ridge(model, X: np.ndarray, future_calendar: np.ndarray):
    features = np.concatenate([X.reshape(len(X), -1), future_calendar.reshape(len(X), -1)], axis=1)
    return model.predict(features)


def seasonal_naive(X: np.ndarray, period: int = 168, horizon: int = 24):
    """Forecast using the corresponding hour one week ago (or prior day)."""
    if period < horizon or X.shape[1] < period:
        raise ValueError("Lookback must cover the requested seasonal period")
    start = X.shape[1] - period
    return X[:, start:start + horizon, 0]


def train_tcn(X_train, y_train, X_val, y_val, config: dict, device: str | None = None):
    """Train a small causal dilated TCN with validation early stopping."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(int(config.get("seed", 42)))
    torch.set_num_threads(max(1, int(config.get("num_threads", 4))))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    TCN = _tcn_classes()
    model = TCN(X_train.shape[-1], int(config["hidden_channels"]), int(config["kernel_size"]),
                float(config["dropout"]), y_train.shape[1]).to(device)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    has_validation = X_val is not None and y_val is not None
    vx = torch.tensor(X_val, device=device) if has_validation else None
    vy = torch.tensor(y_val, device=device) if has_validation else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    best, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(int(config["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.huber_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        if has_validation:
            with torch.no_grad():
                val_mae = float(torch.mean(torch.abs(model(vx) - vy)).item())
        else:
            val_mae = float("nan")
        history.append({"epoch": epoch + 1, "val_mae": val_mae})
        criterion = val_mae if has_validation else -float(epoch + 1)
        if criterion < best:
            best, stale = criterion, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if has_validation and stale >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("TCN did not produce a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, device


def predict_tcn(model, X, device):
    import torch
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
