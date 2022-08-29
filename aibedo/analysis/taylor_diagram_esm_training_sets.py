import os

import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm

from aibedo.utilities.utils import get_logger
from aibedo.utilities.wandb_api import filter_wandb_runs, has_tags, get_predictions_xarray

log = get_logger(__name__)

tag = "MLP-all-4-constraints"
ref_dataset = 'era5'
save_dir = 'out_dir/taylor'
overrides = [
    '++datamodule.prediction_data=same_as_test',
    f'datamodule.partition=[0.85, 0.15, {ref_dataset}]',
    f'datamodule.num_workers={3}',
    'datamodule.eval_batch_size=10',
    'verbose=False',
    '++model.use_auxiliary_vars=False',
    # f'datamodule.data_dir={DATA_DIR}',
]
run_filters = [
    'has_finished',
    has_tags([tag]),
]
hyperparam_filter = None  # {'seed': 7}
runs = filter_wandb_runs(hyperparam_filter=hyperparam_filter, filter_functions=run_filters)
assert len(runs) > 0, "No runs found"
labels = dict()
corrs, stds, rmses = dict(), dict(), dict()

for i, run in enumerate(runs):
    if i > 3:
        pass  # continue
    run_id = run.id
    preds_xarray = get_predictions_xarray(
        run_id, overrides,
        also_errors=False, also_targets=True,
        split='predict',
        return_normalized_outputs=True,
        variables='all')
    assert preds_xarray.attrs['dataset_name'].lower() == ref_dataset.lower()
    esm_for_training = preds_xarray.attrs['esm_for_training']
    label_name = esm_for_training.upper()
    log.info(f" Evaluating {run_id} (trained on {esm_for_training})")

    out_vars = preds_xarray.attrs['variable_names'].split(';')
    for var in out_vars:
        if var in labels.keys() and label_name in labels[var]:
            continue
        ml_preds = preds_xarray[f'{var}_preds'].values
        ref = preds_xarray[f'{var}_targets'].values

        try:
            taylor_stats = sm.taylor_statistics(predicted=ml_preds.flatten(), reference=ref.flatten(), field='')
        except ValueError as e:
            log.warning(f"{esm_for_training} (ID={run_id}) raised exception on var {var}: {e}")
            break
        # Store statistics in arrays
        if i == 0:
            print(f"{var} stdev of ref:", taylor_stats['sdev'][0])
            stds[var] = [taylor_stats['sdev'][0], taylor_stats['sdev'][1]]
            rmses[var] = [taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]]
            corrs[var] = [taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]]
            labels[var] = [ref_dataset.upper()]
        else:
            assert np.isclose(stds[var][0], taylor_stats['sdev'][0])
            stds[var] += [taylor_stats['sdev'][1]]
            rmses[var] += [taylor_stats['crmsd'][1]]
            corrs[var] += [taylor_stats['ccoef'][1]]
        labels[var] += [label_name]

for var in out_vars:
    stds_var = np.array(stds[var])
    corrs_var = np.array(corrs[var])
    rmses_var = np.array(rmses[var])
    labels_var = labels[var]
    print(len(labels_var), stds_var.shape, corrs_var.shape, rmses_var.shape)
    sm.taylor_diagram(
        stds_var, rmses_var, corrs_var,
        checkstats=True,
        alpha=0.0,
        markersize=6,
        markerlegend='on',
        # markerdisplayed='colorbar',
        markerobs=True,
        markerLabel=labels_var,
        titleobs=ref_dataset.upper(),
        widthcor=0.5, widthrms=0.5, widthstd=0.6,
    )
    save_filename = os.path.join(save_dir, f"taylor_diagram_{ref_dataset.upper()}_{tag}_{var}.png")
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
