# expirements around BPE tokenization

To understand the actual value of adding BPE to my model (the first addition I made to teh baseline transformer), I'm going to run a few experiments to understand how it performs without BPE and with BPE at different numbers of merges.

I will keep all hyperparameters (except merge count) static at the following values:

| Hyperparameter | Value |
|---|---|
| batch_size | 64 |
| block_size | 256 |
| max_iters | 2000 |
| eval_interval | 100 |
| learning_rate | 3e-4 |
| eval_iters | 200 |
| n_embd | 384 |
| n_head | 6 |
| n_layer | 6 |
| dropout | 0.2 |

I would love to run with more iterations but it takes hours per run, so I'm making the tradeoff to run with fewer iterations and see if I can get a signal from the results.


Here are the stats I'll be tracking:
- final loss/validation
- loss curve
- chars-per-token
- wall-clock time per iteration
- generation quality