"""Train and evaluate the four-block 1D CNN on pre-split SERS spectra."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from RegressionPlt import RegressionPlt


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        channels = 1
        for kernel, filters in zip((57, 37, 11, 3), (8, 16, 32, 32)):
            blocks.extend((nn.Conv1d(channels, filters, kernel, padding='same'),
                           nn.BatchNorm1d(filters, eps=1e-5), nn.ReLU(),
                           nn.MaxPool1d(2, stride=2, ceil_mode=True)))
            channels = filters
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 1))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.head(self.blocks(x)).squeeze(-1)


def load_data(path):
    """MAT v5/v7 (not v7.3): spectral rows, sample columns, supplied splits."""
    raw = loadmat(path)
    data = {}
    length = None
    for split in ('tra', 'vad', 'tst'):
        for name in ('X_' + split, 'Y_' + split):
            if name not in raw:
                raise ValueError(f'Missing MAT variable: {name}')
        x, y = np.asarray(raw['X_' + split]), np.asarray(raw['Y_' + split])
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError('Spectra and targets must be real.')
        x, y = x.astype(np.float32), y.astype(np.float32).reshape(-1)
        if x.ndim != 2 or x.shape[1] != y.size or y.size == 0:
            raise ValueError(f'{split}: expected X shape [spectral points, samples] and one target per sample.')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'{split}: input contains NaN or infinity.')
        if length is not None and x.shape[0] != length:
            raise ValueError('All splits must have the same spectral length.')
        length = x.shape[0]
        if length < 16:
            raise ValueError('Use at least 16 spectral points for the four pooling blocks.')
        for key, expected in (('N_' + split, y.size), ('N_dim', length)):
            if key in raw and (np.asarray(raw[key]).size != 1 or float(np.asarray(raw[key]).item()) != expected):
                raise ValueError(f'{key} does not agree with array dimensions.')
        data[split] = (torch.from_numpy(np.ascontiguousarray(x.T[:, None, :])), torch.from_numpy(y))
    if 'N_sam' in raw and float(np.asarray(raw['N_sam']).item()) != sum(len(data[s][1]) for s in data):
        raise ValueError('N_sam does not agree with the total number of samples.')
    return data


def demo_data(seed=2026):
    """Synthetic spectra for checking installation, NOT the paper's dataset."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 1, 129)
    peak = np.exp(-((axis - .35) / .04) ** 2) + .6 * np.exp(-((axis - .7) / .07) ** 2)
    result = {}
    for split, n in (('tra', 64), ('vad', 16), ('tst', 16)):
        y = np.tile(np.array([0., .25, .5, 1.]), n // 4).astype('float32')
        rng.shuffle(y)
        x = (y[:, None] * peak + .03 * rng.normal(size=(n, axis.size))).astype('float32')
        result[split] = (torch.from_numpy(x[:, None, :]), torch.from_numpy(y))
    return result


def predict(model, x, device='cpu', batch_size=32):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch.to(device)).cpu() for batch in x.split(batch_size)]).numpy()


def drive_CNN(data_path='data_CNN.mat', *, epochs=500, batch_size=32, seed=2026,
              device='cpu', demo=False, output_dir='results_CNN', plot=True, output_network='best-validation'):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available; use --device cpu.')
    if output_network not in ('best-validation', 'last'):
        raise ValueError('output_network must be best-validation or last.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = demo_data(seed) if demo else load_data(data_path)
    model = CNN().to(device)
    weights = [layer.weight for layer in model.modules() if isinstance(layer, (nn.Conv1d, nn.Linear))]
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm1d):
            weights.extend((layer.weight, layer.bias))
    weight_ids = {id(p) for p in weights}
    other = [p for p in model.parameters() if id(p) not in weight_ids]
    optimizer = torch.optim.Adam([{'params': weights, 'weight_decay': 1e-4}, {'params': other, 'weight_decay': 0.}], lr=1e-3, betas=(.9, .999), eps=1e-8)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data['tra']), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    history, best, best_state = [], float('inf'), None
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Training loss became non-finite. Check input scale and data.')
            loss.backward()
            optimizer.step()
            total += loss.item() * y.numel()
        prediction = predict(model, data['vad'][0], device, batch_size)
        validation = float(np.mean((prediction - data['vad'][1].numpy()) ** 2))
        if not np.isfinite(validation):
            raise RuntimeError('Validation loss became non-finite. Check input scale and data.')
        if validation < best:
            best, best_state = validation, copy.deepcopy(model.state_dict())
        history.append(dict(epoch=epoch+1, training_mse=total/len(data['tra'][1]), validation_mse=validation))
        if (epoch + 1) % 500 == 0:
            for group in optimizer.param_groups:
                group['lr'] *= .5
    if output_network == 'best-validation':
        model.load_state_dict(best_state)
    calibration_x = torch.cat((data['tra'][0], data['vad'][0]))
    calibration_y = torch.cat((data['tra'][1], data['vad'][1])).numpy()
    calibration_prediction = predict(model, calibration_x, device, batch_size)
    test_prediction = predict(model, data['tst'][0], device, batch_size)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values = RegressionPlt(calibration_y, calibration_prediction, data['tst'][1].numpy(), test_prediction, plot=plot, output_dir=output)
    metrics = {name: dict(zip(('calibration', 'test'), [float(v) if np.isfinite(v) else None for v in array])) for name, array in zip(('rho', 'R2', 'BIAS', 'RMSEP', 'LOD'), values)}
    result = dict(dataset='synthetic demonstration' if demo else str(data_path), seed=seed, epochs=epochs,
                  batch_size=batch_size, output_network=output_network, seconds=time.perf_counter()-start, best_validation_mse=best, metrics=metrics, history=history)
    (output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    torch.save({'state_dict': model.cpu().state_dict(), 'spectral_length': data['tra'][0].shape[-1], 'seed': seed}, output/'model.pt')
    np.savez(output/'predictions.npz', calibration_actual=calibration_y, calibration_predicted=calibration_prediction,
             test_actual=data['tst'][1].numpy(), test_predicted=test_prediction)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data_CNN.mat')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--demo', action='store_true', help='Use synthetic spectra, not experimental data.')
    parser.add_argument('--output', default='results_CNN')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output-network', choices=('best-validation', 'last'), default='best-validation')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    result = drive_CNN(args.data, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                      device=args.device, demo=args.demo, output_dir=args.output, plot=not args.no_plots, output_network=args.output_network)
    print(json.dumps({key: value for key, value in result.items() if key != 'history'}, indent=2))


if __name__ == '__main__':
    main()
