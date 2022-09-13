import pytorch_lightning as pl
from aibedo.models.eof_timeseries.datamodule import AIBEDO_EOF_DataModule
from aibedo.models.eof_timeseries.models import AIBEDO_EOF_MLP, EOF_AFNONet


def run():
    dm = AIBEDO_EOF_DataModule(time_lag=4)

    mlp = AIBEDO_EOF_MLP(hidden_dims=[128, 128], dropout=0.1, residual=True)
    afno = EOF_AFNONet(num_layers=3)

    model = afno

    trainer = pl.Trainer(gpus=0, max_epochs=25)
    trainer.fit(model, dm)
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    run()