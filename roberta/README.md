# Experiments for RoBERTa
This repo contains the experiment to reproduce the number reported for RoBERTa.

## Repo structure

Please download the data from the attachment and place them under `../../data/` relevant to this directory.


## Install dependencies

```
pip install -r requirements.txt
```

## Training

To train a RoBERTa model, simply do the following:

```
python train_roberta.py --output_dir outputs --model_name MODEL_NAME
```

where `MODEL_NAME` is `roberta-large`. This will train a model and save the checkpoint under the `outputs` directory.


## Trained weights

For reproduction purposes, we include the [link](https://drive.google.com/file/d/1YXp9MfW8hvwd6qbKvthX7ZCcpi-phnc8/view?usp=drive_link) to the trained checkpoints of one run of our experiment.


## Evaluation

To test the performance of the trained model, use the following command:


```
python test_roberta.py --checkpoint_path PATH_TO_TRAINED_CHECKPOINT --model_name MODEL_NAME
```

where `PATH_TO_TRAINED_CHECKPOINT` is the path to the checkpoint that you have trained in the previous step. Note that `MODEL_NAME` should be the same as the name of the model for the checkpoint.