# Loss Landscape Matters: Training Certifiably Robust Models with Favorable Loss Landscape

This repository is the official implementation of "Loss Landscape Matters: Training Certifiably Robust Models with Favorable Loss Landscape".

<!----
> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
---->

## Requirements

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


## Pre-trained Models

You can download pretrained models here:
<!----
 - [OUR model](https://drive.google.com/file/d/) trained on MNIST.
 - [OUR model](https://drive.google.com/file/d/) trained on CIFAR-10.
---->

<!----
> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
---->

## Evaluation of pretrained models

After downloading the pretrained models to the directory ./pretrained, you are ready to evaluate them.
To evaluate the pretrained model, run:

```eval
python eval.py --config config/mnist.json "eval_params:model_paths=mnist_large_train04" "eval_params:epsilon=0.3"
python eval.py --config config/cifar10.json "eval_params:model_paths=cifar_medium_2px" "eval_params:epsilon=0.007843"
```

<!----
> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
---->

## Results

Our model achieves the following performance against $\ell_\infty$-perturbation :

### MNIST ($\epsilon_{test}$=0.1-0.4)

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     92.54%         |      66.23%       | 48.20%  |
| [CAP](https://arxiv.org/abs/1805.12514)                |     88.39%         |      62.25%       | 43.95%  |
| [LMT](https://arxiv.org/abs/1802.04034)               |     86.48%         |      53.56%       | 40.55%  |

### CIFAR-10 ($\epsilon_{test}$=2,4,6,8,16/255)

Model1

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     65.64         |      59.59%       | 50.27%  |
| [CAP](https://arxiv.org/abs/1805.12514)                |     60.14%         |      55.67%       | 50.29%  |
| [LMT](https://arxiv.org/abs/1802.04034)               |     56.49%         |      49.83%       | 37.20%  |

Model2

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     65.72%         |      60.78%       | 51.30%  |
| [CAP](https://arxiv.org/abs/1805.12514)                |     60.10%         |      56.20%       | 50.87%  |
| [LMT](https://arxiv.org/abs/1802.04034)               |     63.05%         |      58.32%       | 38.11%  |



<!----
> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
---->

<!----
## Contributing
> 📋Pick a licence and describe how to contribute to your code repository. 
---->
