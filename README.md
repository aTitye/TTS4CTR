# TTS-CTR - Paper Release

Essential files for reproducing paper results.

## Files

* `data/tfloader.py` - Data loading
* `modules/models.py` - Model definitions  
* `modules/layers.py` - Custom layers
* `trainer.py` - Training script
* `trainer_group.py` - Group training

## Environment

```BASH
conda env create -f environment.yml
```

usage example:

```BASH
bash avazu_train_group.sh

```
