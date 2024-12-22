# CS 4775 Final Project
The DenseNet training file is available under ‘densenet/model.py‘, and the 
reimplementation training file is under ‘replications/model.py‘. Moreover, 
we provide open-source access to checkpoints for the models used to conduct
inference under the ‘saved_models‘ directory.

1. Create virtual environment and install requirements.txt

2. cd into either the replications or densenet directory.

3. Download data from this link: https://figshare.com/articles/dataset/Archives/8279618/2

4. Modify paths in model.py as needed to run the following commands


# Usage Example:

For gapped data:
```
 python model.py --convert_dataset 1 \
 --gapped 1 \
 --train $GAPPED_PATH/TRAIN.npy \
 --valid $GAPPED_PATH/VALID.npy \
 --test $GAPPED_PATH/TEST.npy \
 -N 4
```


For ungapped data:
```
 python model.py --convert_dataset 1 \
 --gapped 0 \
 --train $UNGAPPED_PATH/TRAIN.npy \
 --valid $UNGAPPED_PATH/VALID.npy \
 --test $UNGAPPED_PATH/TEST.npy \
 -N 4
```