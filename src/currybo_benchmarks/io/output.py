import os
import numpy as np
import pandas as pd
import json

from datetime import datetime
from typing import Dict


def save_results(result, cli_args: Dict, what: str):
    timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    reshaped = {i['name']: [[]] for i in cli_args['targets']}

    if cli_args['output_filename'] != "":
        json_filename = os.path.join(cli_args['output_dir'], f"{cli_args['output_filename']}_ARGS.json")
    else:
        json_filename = os.path.join(cli_args['output_dir'], f"currybo-bechmarks_{cli_args['dataset']}_{cli_args['batch_size']}_ARGS_{timestamp}.json")

    if not os.path.isfile(json_filename):
        with open(json_filename, "w") as f:
            json.dump(cli_args, f, indent=2)
        print(f":: Saved parameters to {json_filename}")

    for i in cli_args['targets']:
        field = i['name']
        m = []
        for j in result:
            m.append(j[field])
        reshaped[field] = m

    for field in reshaped:
        if cli_args['output_filename'] != "":
            filename = os.path.join(cli_args['output_dir'], f"{cli_args['output_filename']}_{what}_{field}.csv")
        else:
            filename = os.path.join(cli_args['output_dir'], f"currybo-benchmarks_{cli_args['dataset']}_{cli_args['batch_size']}_{what}_{field}_{timestamp}.csv")

        arr = np.array(reshaped[field]).T

        df = pd.DataFrame(arr, columns=np.arange(arr.shape[1]))

        df.to_csv(filename)
        print(f":: Saved results to {filename}")
