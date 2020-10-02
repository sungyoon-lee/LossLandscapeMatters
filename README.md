# Loss Landscape Matters: Training Certifiably Robust Models with Favorable Loss Landscape

This repository is the official implementation of "Loss Landscape Matters: Training Certifiably Robust Models with Favorable Loss Landscape".

<!----
> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
---->

## Requirements

It requires torch version>=1.3.0.


To install requirements:

```setup
conda env create -f environment.yml
```

<!----
> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
---->

## Training (and Evaluation)

To train and evaluate the model(s) in the paper, run this command:

```train
python train.py --config config/cifar10.json 
python train.py --config config/cifar10.json "training_params:epsilon=0.007843" "training_params:train_epsilon=0.007843" 
python train.py --config config/mnist.json
python train.py --config config/svhn.json


python eval.py --config config/cifar10.json "eval_params:model_paths=cifar_medium_8px"
python eval.py --config config/cifar10.json "eval_params:model_paths=cifar_medium_2px" "eval_params:epsilon=0.007843"
python eval.py --config config/mnist.json "eval_params:model_paths=mnist_large_train04"
python eval.py --config config/svhn.json "eval_params:model_paths=svhn_large_001"


```

<!----
> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
---->
