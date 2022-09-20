import numpy as np
import pytorch_lightning as pl
from aibedo.models.eof_timeseries.datamodule import AIBEDO_EOF_DataModule
from aibedo.models.eof_timeseries.models import AIBEDO_EOF_MLP, EOF_AFNONet

DATA_DIR = 'C:/Users/salvarc/data'


def run():
    dm = AIBEDO_EOF_DataModule(
        data_dir=DATA_DIR,
        time_lag=4
    )

    mlp = AIBEDO_EOF_MLP(hidden_dims=[128, 128], dropout=0.1, residual=True)
    afno = EOF_AFNONet(num_layers=3)

    # CHOOSE HERE WHICH MODEL TO RUN
    model = mlp

    epochs = 25
    trainer = pl.Trainer(gpus=0, max_epochs=epochs)
    trainer.fit(model, dm)
    yy = trainer.test(model, dataloaders=dm.test_dataloader())
    yy = model.get_preds(dm.test_dataloader())
    # save targets and predictions to npz file
    preds_file = f'{DATA_DIR}/preds-eof-nonorm-{epochs}epochs.npz'
    np.savez(preds_file, **yy)


if __name__ == '__main__':
    # load npz file
    yy = np.load(f'{DATA_DIR}/preds-eof-nonorm.npz')
    print(list(yy.keys()))
    run()
