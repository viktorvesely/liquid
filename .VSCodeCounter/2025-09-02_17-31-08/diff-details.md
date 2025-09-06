# Diff Details

Date : 2025-09-02 17:31:08

Directory c:\\Users\\viktor\\Documents\\liquid

Total : 77 files,  41457 codes, 83 comments, 376 blanks, all 41916 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [liquid/adapter.py](/liquid/adapter.py) | Python | 7 | 1 | 7 | 15 |
| [liquid/cifar10.json](/liquid/cifar10.json) | JSON | 72 | 0 | 0 | 72 |
| [liquid/cifar10\_hyper.json](/liquid/cifar10_hyper.json) | JSON | -52 | 0 | 0 | -52 |
| [liquid/citizens/citizen.py](/liquid/citizens/citizen.py) | Python | 7 | -1 | 3 | 9 |
| [liquid/citizens/delegating\_citizen.py](/liquid/citizens/delegating_citizen.py) | Python | 4 | 0 | 2 | 6 |
| [liquid/citizens/routers.py](/liquid/citizens/routers.py) | Python | -37 | 0 | -9 | -46 |
| [liquid/citizens/vision\_citizen.py](/liquid/citizens/vision_citizen.py) | Python | 11 | 0 | 1 | 12 |
| [liquid/globals.py](/liquid/globals.py) | Python | 10 | 0 | 5 | 15 |
| [liquid/liquid\_ensemble/le\_adapter.py](/liquid/liquid_ensemble/le_adapter.py) | Python | 3 | 0 | 2 | 5 |
| [liquid/liquid\_ensemble/le\_cifar10architecture.py](/liquid/liquid_ensemble/le_cifar10architecture.py) | Python | 6 | 0 | 0 | 6 |
| [liquid/liquid\_ensemble/le\_layer.py](/liquid/liquid_ensemble/le_layer.py) | Python | 7 | 1 | 4 | 12 |
| [liquid/moe/moe\_adapter.py](/liquid/moe/moe_adapter.py) | Python | 2 | 0 | 1 | 3 |
| [liquid/moe/moe\_cifar10architecture.py](/liquid/moe/moe_cifar10architecture.py) | Python | 9 | 0 | 0 | 9 |
| [liquid/moe/moe\_layer.py](/liquid/moe/moe_layer.py) | Python | 3 | 0 | 0 | 3 |
| [liquid/nn\_adapter.py](/liquid/nn_adapter.py) | Python | 73 | 3 | 28 | 104 |
| [liquid/notebooks/le\_best\_rmse\_results\_df.csv](/liquid/notebooks/le_best_rmse_results_df.csv) | CSV | 434 | 0 | 1 | 435 |
| [liquid/notebooks/le\_confidence\_power\_kendall\_results\_df.csv](/liquid/notebooks/le_confidence_power_kendall_results_df.csv) | CSV | 434 | 0 | 1 | 435 |
| [liquid/notebooks/lgbm\_confidence\_quantile\_kendall\_results\_df.csv](/liquid/notebooks/lgbm_confidence_quantile_kendall_results_df.csv) | CSV | 752 | 0 | 1 | 753 |
| [liquid/notebooks/lgbm\_rmse\_results\_df.csv](/liquid/notebooks/lgbm_rmse_results_df.csv) | CSV | 752 | 0 | 1 | 753 |
| [liquid/notebooks/loss\_landscape.ipynb](/liquid/notebooks/loss_landscape.ipynb) | JSON | 324 | 0 | 1 | 325 |
| [liquid/notebooks/moe\_best\_rmse\_results\_df.csv](/liquid/notebooks/moe_best_rmse_results_df.csv) | CSV | 756 | 0 | 1 | 757 |
| [liquid/notebooks/moe\_confidence\_gate\_kendall\_results\_df.csv](/liquid/notebooks/moe_confidence_gate_kendall_results_df.csv) | CSV | 756 | 0 | 1 | 757 |
| [liquid/notebooks/optimize\_nmi\_analysis.ipynb](/liquid/notebooks/optimize_nmi_analysis.ipynb) | JSON | 66 | 0 | 0 | 66 |
| [liquid/notebooks/random\_protein\_eval.ipynb](/liquid/notebooks/random_protein_eval.ipynb) | JSON | 487 | 0 | 0 | 487 |
| [liquid/notebooks/rf\_confidence\_std\_kendall\_results\_df.csv](/liquid/notebooks/rf_confidence_std_kendall_results_df.csv) | CSV | 801 | 0 | 1 | 802 |
| [liquid/notebooks/rf\_rmse\_results\_df.csv](/liquid/notebooks/rf_rmse_results_df.csv) | CSV | 801 | 0 | 1 | 802 |
| [liquid/notebooks/scaling\_laws.ipynb](/liquid/notebooks/scaling_laws.ipynb) | JSON | 333 | 0 | 1 | 334 |
| [liquid/plain/cifar10.py](/liquid/plain/cifar10.py) | Python | 1 | 0 | 0 | 1 |
| [liquid/plain/simple\_adapter.py](/liquid/plain/simple_adapter.py) | Python | 0 | 0 | 2 | 2 |
| [liquid/protein\_best\_conf.json](/liquid/protein_best_conf.json) | JSON | 63 | 0 | 0 | 63 |
| [liquid/protein\_best\_rmse.json](/liquid/protein_best_rmse.json) | JSON | 63 | 0 | 0 | 63 |
| [liquid/scaling\_laws.py](/liquid/scaling_laws.py) | Python | 294 | 14 | 108 | 416 |
| [liquid/scripts/protein\_train\_best.sh](/liquid/scripts/protein_train_best.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [liquid/scripts/sl\_le\_block.sh](/liquid/scripts/sl_le_block.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [liquid/scripts/sl\_le\_long.sh](/liquid/scripts/sl_le_long.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [liquid/scripts/sl\_moe\_block.sh](/liquid/scripts/sl_moe_block.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [liquid/scripts/sl\_moe\_long.sh](/liquid/scripts/sl_moe_long.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [liquid/scripts/sl\_simple.sh](/liquid/scripts/sl_simple.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [liquid/test\_best.py](/liquid/test_best.py) | Python | 77 | 2 | 38 | 117 |
| [liquid/train.py](/liquid/train.py) | Python | 50 | -1 | 16 | 65 |
| [liquid/utils.py](/liquid/utils.py) | Python | 2 | 0 | 2 | 4 |
| [liquid/vae/auto\_encoder.py](/liquid/vae/auto_encoder.py) | Python | -70 | -30 | -19 | -119 |
| [liquid/vae/cifar10\_dataset.py](/liquid/vae/cifar10_dataset.py) | Python | -57 | -26 | -15 | -98 |
| [liquid/vae/configuration.py](/liquid/vae/configuration.py) | Python | -83 | -66 | -25 | -174 |
| [liquid/vae/decoder.py](/liquid/vae/decoder.py) | Python | -50 | -29 | -15 | -94 |
| [liquid/vae/encoder.py](/liquid/vae/encoder.py) | Python | -50 | -29 | -14 | -93 |
| [liquid/vae/evaluator.py](/liquid/vae/evaluator.py) | Python | -40 | -26 | -16 | -82 |
| [liquid/vae/main.py](/liquid/vae/main.py) | Python | -47 | -27 | -12 | -86 |
| [liquid/vae/residual.py](/liquid/vae/residual.py) | Python | -35 | -27 | -9 | -71 |
| [liquid/vae/residual\_stack.py](/liquid/vae/residual_stack.py) | Python | -13 | -26 | -8 | -47 |
| [liquid/vae/trainer.py](/liquid/vae/trainer.py) | Python | -66 | -30 | -17 | -113 |
| [liquid/vae/vector\_quantizer.py](/liquid/vae/vector_quantizer.py) | Python | -32 | -73 | -18 | -123 |
| [liquid/vae/vector\_quantizer\_ema.py](/liquid/vae/vector_quantizer_ema.py) | Python | -47 | -81 | -24 | -152 |
| [liquid/vae/visualize.ipynb](/liquid/vae/visualize.ipynb) | JSON | -164 | 0 | -1 | -165 |
| [liquid/visualizer.py](/liquid/visualizer.py) | Python | 151 | 0 | 30 | 181 |
| [main\_exp.csv](/main_exp.csv) | CSV | 33,601 | 0 | 1 | 33,602 |
| [nmi\_optimize\_4h.csv](/nmi_optimize_4h.csv) | CSV | 76 | 0 | 1 | 77 |
| [obsolete/council.py](/obsolete/council.py) | Python | 16 | 3 | 13 | 32 |
| [obsolete/dictator\_council.py](/obsolete/dictator_council.py) | Python | 27 | 4 | 20 | 51 |
| [obsolete/graph\_methodology.py](/obsolete/graph_methodology.py) | Python | 90 | 4 | 23 | 117 |
| [obsolete/image\_citizen.py](/obsolete/image_citizen.py) | Python | 57 | 0 | 30 | 87 |
| [obsolete/majority\_council.py](/obsolete/majority_council.py) | Python | 34 | 4 | 23 | 61 |
| [obsolete/vae/auto\_encoder.py](/obsolete/vae/auto_encoder.py) | Python | 70 | 30 | 19 | 119 |
| [obsolete/vae/cifar10\_dataset.py](/obsolete/vae/cifar10_dataset.py) | Python | 57 | 26 | 15 | 98 |
| [obsolete/vae/configuration.py](/obsolete/vae/configuration.py) | Python | 83 | 66 | 25 | 174 |
| [obsolete/vae/decoder.py](/obsolete/vae/decoder.py) | Python | 50 | 29 | 15 | 94 |
| [obsolete/vae/encoder.py](/obsolete/vae/encoder.py) | Python | 50 | 29 | 14 | 93 |
| [obsolete/vae/evaluator.py](/obsolete/vae/evaluator.py) | Python | 40 | 26 | 16 | 82 |
| [obsolete/vae/main.py](/obsolete/vae/main.py) | Python | 47 | 27 | 12 | 86 |
| [obsolete/vae/residual.py](/obsolete/vae/residual.py) | Python | 35 | 27 | 9 | 71 |
| [obsolete/vae/residual\_stack.py](/obsolete/vae/residual_stack.py) | Python | 13 | 26 | 8 | 47 |
| [obsolete/vae/trainer.py](/obsolete/vae/trainer.py) | Python | 66 | 30 | 17 | 113 |
| [obsolete/vae/vector\_quantizer.py](/obsolete/vae/vector_quantizer.py) | Python | 32 | 73 | 18 | 123 |
| [obsolete/vae/vector\_quantizer\_ema.py](/obsolete/vae/vector_quantizer_ema.py) | Python | 47 | 81 | 24 | 152 |
| [obsolete/vae/visualize.ipynb](/obsolete/vae/visualize.ipynb) | JSON | 164 | 0 | 1 | 165 |
| [run\_vae\_train.sh](/run_vae_train.sh) | Shell Script | 4 | 7 | 2 | 13 |
| [setup.py](/setup.py) | Python | 6 | 0 | 1 | 7 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details