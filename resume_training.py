import os
import sys
from aibedo.test import reload_and_test_model

if __name__ == "__main__":
    direc = None
    run_id = None
    # find arg with wandb ID
    args = sys.argv[1:]
    for arg in args:
        print(arg)
        if 'logger.wandb.id' in str(arg):
            run_id = arg.split('=')[1]   # get wandb ID from logger.wandb.id=<wandb_id>
            break
    assert run_id is not None, "No wandb ID found in args. Please provide it as logger.wandb.id=<wandb_id>"
    print("Resuming training og ", run_id)
    if direc is not None and os.path.isdir(os.path.join(direc, run_id)):
        saved_ckpts = [f for f in os.listdir(os.path.join(direc, run_id)) if f.endswith('.ckpt')]
        print(" Checkpoints saved:", saved_ckpts)
        if 'last.ckpt' in saved_ckpts:
            print("Reloading from last.ckpt")
            checkpoint_path = os.path.join(direc, f"{run_id}/last.ckpt")
        else:
            print("Reloading from", saved_ckpts[0])
            checkpoint_path = os.path.join(direc, f"{run_id}/{saved_ckpts[0]}")
    else:
        checkpoint_path = direc

    reload_and_test_model(run_id=run_id, checkpoint_path=checkpoint_path, train_model=True, override_kwargs=args)
