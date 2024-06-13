# scripts

## partition_function.py

Reproduces the paritition function results from the paper. 

```
usage: partition_function.py [-h] -m MODEL [-c CACHE_DIR] [-t | --training | --no-training]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        the model to evaluate. valid choices are ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b',
                        'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b', 'gpt-neox-20b', 'falcon-7b', 'opt-7b',
                        'crystalcoder', 'amber']
  -c CACHE_DIR, --cache_dir CACHE_DIR
                        directory to cache huggingface models defaults to ./cache
  -t, --training, --no-training
                        Use the 21 training checkpoints from the paper (only works with Pythia models)
```

Example:
```
python partition_function.py -m pythia-70m
Calculating Isotropy: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.86s/it]
{
  "": {
    "IW_in": 0.9584758281707764,
    "IW_out": 0.08308906108140945,
    "IW_in_centered": 0.9968440532684326,
    "IW_out_centered": 0.9724733829498291
  }
}
```

Note you must have enough memory to load the model. 

If you want to add another model create a function that will get its embedding. E.g., for Pythia models

```
def gptneox_get_embeddings(
        model_params
        ):
    """_summary_ gets the input and output embeddings for the given pythia model

    Args:
        model_params (_type_): _description_ model paramters 

    Returns:
        _type_: _description_ emb_in, emb_out
    """

    model = AutoModelForCausalLM.from_pretrained(**model_params)

    emb_in = model.gpt_neox.embed_in.weight
    emb_out = model.embed_out.weight

    return emb_in, emb_out
```

and then add your model to the dict of valid model choices. E.g for pythia models

```
MODEL_CONFIGS = {
    "pythia-70m":{
        "model_addr":"EleutherAI/pythia-70M",
        "get_emb_fn":gptneox_get_embeddings
    },
  ...
```

## cosine_improvement.py

A demonstration of how our method improves the time it takes to computer the average cosine similarity between distinct pairs in a set of vectors.

```
usage: cosine_improvement.py [-h] [-ne NUMBER_EMB] [-es EMB_SIZE] [-i | --isotropic | --no-isotropic] [-ss SAMPLE_SIZE]
                              [-b | --batched | --no-batched] [-bs BATCH_SIZE] [-psm PARTIAL_SUM_MAX] [-s SEED]

options:
  -h, --help            show this help message and exit
  -ne NUMBER_EMB, --number_emb NUMBER_EMB
                        The number of random embeddings to generate
  -es EMB_SIZE, --emb_size EMB_SIZE
                        The size of the embeddings
  -i, --isotropic, --no-isotropic
                        Isotropic: emb values in N(0,1), Not Isotropic: emb values in uniform [0,1)
  -ss SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        Number of sample to estime true cosine similarity (will still compute full cosine)
  -b, --batched, --no-batched
                        Use batched methods (may OOM if not using batched methods)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size if using batched methods, default:128
  -psm PARTIAL_SUM_MAX, --partial_sum_max PARTIAL_SUM_MAX
                        Number of values in each partial sum (to prevent floating point errors), if using batched methods. default: 1280
```

### Examples

Isotropic embeddings
