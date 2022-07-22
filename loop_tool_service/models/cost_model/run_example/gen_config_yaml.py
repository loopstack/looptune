import sys
import yaml


if len(sys.argv) != 2: 
    print('Format: python gen_config_yaml.py write_path')
    exit()

sweep_config = {
  "name" : "Cost-sweep",
  "method": "random",
  "metric": {
    "name": "final_performance",
    "goal": "maximize",
  },
  "parameters" : {
    "hidden_size" : {"values": [ 100, 200, 400 ]},
    "layers" : {"values": [ 3, 5, 10]},
    'lr': {
      'distribution': 'log_uniform_values',
      'min': 0.00001,
      'max': 0.01
    },
    "epochs": { "value" : 50 },
    "batch_size": { "value" : 50 },
    "dropout": { "values" : [0, 0.2] },
    "data_size": { "value" : -1 },
  }
}

sweep_params_path = sys.argv[1]
with open(sweep_params_path, 'w') as outfile:
    yaml.dump(sweep_config, outfile, default_flow_style=False)