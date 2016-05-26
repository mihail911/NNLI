import numpy as np
import random

FIXED = {
    "batch_size": 20,
    "max_norm": 40,
    "n_epochs": 20,
    "num_hops": 3,
    "adj_weight_typing": True,
    "shuffle_batch": True
}

VAR = {
    "embedding_size": ("LIN", 10, 100),
    "lr": ("EXP", 0.0001, 0.01),
    "l2_reg": ("EXP", 0.001, 0.1)
}


def generate_run_cmd():
    cmd_str = "python memn2n "

    # Add fixed params
    for name, value in FIXED.iteritems():
        cmd_str += " --" + name + " " + str(value)

    # Add var params
    for name, value in VAR.iteritems():
        if type(value) == tuple:
            sample = value[0]
            start = float(value[1])
            end = float(value[2])

            r = random.uniform(0,1)
            if sample == "LIN":
                diff = end - start
                new_value = int(start + r * diff)
            elif sample == "EXP":
                log_end = np.log(end)
                log_start = np.log(start)
                new_value = np.exp(log_start + (log_end - log_start) * r)


        cmd_str += " --" + name + " " + str(new_value)
    print cmd_str


NUM_RUNS = 5

# Generate a number of strings for runs
for _ in range(NUM_RUNS):
    generate_run_cmd()