import numpy as np
import pytorch_lightning as pl
import torch

from aibedo.models.eof_timeseries.datamodule import AIBEDO_EOF_DataModule
from aibedo.models.eof_timeseries.models import AIBEDO_EOF_MLP, EOF_AFNONet

DATA_DIR = 'C:/Users/salvarc/data'
DATA_DIR = '../data'

################ DATA REQUIRED FOR TRAINING ################
# aws s3 cp s3://darpa-aibedo/eof.isosph5.nonorm.CESM2.piControl.r1i1p1f1.Input.Exp8.nc .
# aws s3 cp s3://darpa-aibedo/eof.isosph5.nonorm.CESM2.piControl.r1i1p1f1.Output.nc .
# OR
# aws s3 cp s3://darpa-aibedo/eof.isosph5.nonorm.CESM2.historical.r1i1p1f1.Input.Exp8.nc .
# aws s3 cp s3://darpa-aibedo/eof.isosph5.nonorm.CESM2.historical.r1i1p1f1.Output.nc .
############################################################
def run():
    TIME_LAG = 4
    SIMULATION = 'piControl'   # 'historical' OR 'piControl'
    dm = AIBEDO_EOF_DataModule(
        data_dir=DATA_DIR,
        simulation=SIMULATION,
        time_lag=TIME_LAG
    )

    mlp = AIBEDO_EOF_MLP(hidden_dims=[128, 128], dropout=0.1, residual=True)
    afno = EOF_AFNONet(num_layers=3)

    # CHOOSE HERE WHICH MODEL TO RUN
    model = mlp

    epochs = 25
    gpus = 1 if torch.cuda.is_available() else 0
    RUN_NAME = f"{SIMULATION}-{TIME_LAG}lag-{epochs}epochs-{model.name}"
    trainer = pl.Trainer(gpus=gpus, max_epochs=epochs,
                         callbacks=[
                             pl.callbacks.LearningRateMonitor(),
                             pl.callbacks.ModelCheckpoint(dirpath=DATA_DIR + '/EOF_ckpts', monitor='val/loss', mode='min'),
                         ],
                         logger=pl.loggers.WandbLogger(project='EOF-AIBEDO', entity='salv47', name=RUN_NAME),
                         )
    trainer.fit(model, dm)
    # yy = trainer.test(model, dataloaders=dm.test_dataloader())
    yy = model.get_preds(dataloader=dm.test_dataloader())
    # save targets and predictions to npz file
    preds_file = f'{DATA_DIR}/preds-eof-nonorm-{RUN_NAME}.npz'
    np.savez(preds_file, **yy)


if __name__ == '__main__':
    # load npz file
    yy = np.load(f'{DATA_DIR}/preds-eof-nonorm.npz')
    print(list(yy.keys()))
    run()
