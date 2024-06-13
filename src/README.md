# scripts

## partition_function.py

```
python partition_function.py -h
usage: partition_function.py [-h] -m MODEL [-c CACHE_DIR] [-t | --training | --no-training]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        the model to evaluate. valid choices are ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b', 'gpt-
                        neox-20b', 'falcon-7b', 'opt-7b', 'crystalcoder', 'amber']
  -c CACHE_DIR, --cache_dir CACHE_DIR
                        directory to cache huggingface models defaults to ./cache
  -t, --training, --no-training
                        Use the 21 training checkpoints from the paper (only works with Pythia models)
```
