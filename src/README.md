# scripts

## partition_function.py

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
