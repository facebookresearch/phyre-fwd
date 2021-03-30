# Forward Modeling Methods on PHYRE

For installation, see the [README](../README.md)

# Training an agent

All of the agents are trained from a configuration file. Configurations are included for all experiments included in the paper in the `agents/expts` folder. The entry point is `launch.py` which will train an agent based off the configuration file.
E.g., the following command will launch a sweep training Interaction Nets object based forward models:

```(bash)
cd agents
./launch.py -c expts/fwd_models/000_fwd_IN_win.txt
```

Each of the configurations in the sweep (here 10 training runs for the 10 different data folds), will be run.

To launch a single run with a specified parameter combination a test, add the `-l` flag to the command above, ie  ```./launch.py -c expts/fwd_models/000_fwd_IN_win.txt -l```. At runtime, you will be able to chose desired the parameter combination.


## Results

All results from training runs will be stored in `agents/outputs`.

## Downloading Pre-trained Models

The models from the paper are availabe at `https://dl.fbaipublicfiles.com/phyre-fwd-agents`. The file structure is `s3://dl.fbaipublicfiles.com/phyre-fwd-agents/CONFIG_DIRECTORY/CONFIG_FILE_NAME/RUN_ID`, where `RUN_ID` corresponds to the hydra run ID of the paramter configuration. For each of the runs, `RUN_ID % 10` corresponds to the seed and `RUN_ID // 10` corresponds to the `ith` timestep in the sweep.

All results from training runs will be stored in `agents/outputs/expts/CONFIG_DIRECTORY/CONFIG_FILE_NAME/RUN_ID`.

## Running evaluation example

The following is an example of how to run evaluation for the decoder joint 3 frame model, on seed 0 of the ball with template setup, when it was trained to rollout 2 steps during training (`n_forward_times=2`).

1) Download the model
```
cd agents
mkdir -p outputs/expts/joint/001_joint_DEC_1f_win.txt
s3cmd sync --skip-existing s3://dl.fbaipublicfiles.com/phyre-fwd-agents/joint/001_joint_DEC_1f_win.txt/20/
outputs/expts/joint/001_joint_DEC_1f_win.txt/20/
```

2) Run the evaluation
```
./launch.py -l -t -c expts/joint/001_joint_DEC_1f_win.txt
```
If prompted, select run 20. The results will be stored in a `results.json` file in the experiment directory.

Note: Running local evaluation (with `-l` flag) of a pre-trained model may not configure paths correctly for pixel classifiers on object models (for example [`expts/pix_cls/002_pix_cls_TX_win.txt`](expts/pix_cls/002_pix_cls_TX_win.txt)). For testing such models, remove the `-l` flag.



# Baselines

## Forward Models

The configuration files for each of the forward models can be found in the `exps/fwd_models` directory.

| Model | Eval Setup | Config File |
|----------|:-------------:|------:|
| IN |  ball_within_template | [expts/fwd_models/000_fwd_IN_win.txt](expts/fwd_models/000_fwd_IN_win.txt) |
| Tx | ball_within_template | [expts/fwd_models/001_fwd_TX_win.txt](expts/fwd_models/001_fwd_TX_win.txt) |
| STN | ball_within_template | [expts/fwd_models/002_fwd_STN_win.txt](expts/fwd_models/002_fwd_STN_win.txt) |
| DEC | ball_within_template | [expts/fwd_models/003_fwd_DEC_win.txt](expts/fwd_models/003_fwd_DEC_win.txt) |
| FPA Baseline | ball_within_template | [expts/fwd_models/008_fwd_baseline_win.txt](expts/fwd_models/008_fwd_baseline_win.txt) |
| IN |  ball_cross_template | [expts/fwd_models/004_fwd_IN_x.txt](expts/fwd_models/004_fwd_IN_x.txt) |
| Tx | ball_cross_template | [expts/fwd_models/005_fwd_TX_x.txt](expts/fwd_models/005_fwd_TX_x.txt) |
| STN | ball_cross_template | [expts/fwd_models/006_fwd_STN_x.txt](expts/fwd_models/006_fwd_STN_x.txt ) |
| DEC | ball_cross_template | [expts/fwd_models/007_fwd_DEC_x.txt](expts/fwd_models/007_fwd_DEC_x.txt) |
| FPA Baseline | ball_cross_template | [expts/fwd_models/009_fwd_baseline_x.txt](expts/fwd_models/009_fwd_baseline_x.txt) |

## Pixel Based Classifiers

The configuration files for each of the pixel based classfier models can be found in the `exps/pix_cls` directory. These configurations are for the pixel based classifiers on the latent representations for STN and DEC, and the rendered frames for the ground truth, IN, and Tx.

| Fwd Model | Eval Setup | Config File |
|----------|:-------------:|------:|
| None (Ground Truth) |  ball_within_template |[expts/pix_cls/000_pix_cls_GT_win.txt](expts/pix_cls/000_pix_cls_GT_win.txt) |
| IN |  ball_within_template | [expts/pix_cls/001_pix_cls_IN_win.txt](expts/pix_cls/001_pix_cls_IN_win.txt)  |
| Tx | ball_within_template | [expts/pix_cls/002_pix_cls_TX_win.txt](expts/pix_cls/002_pix_cls_TX_win.txt) |
| STN | ball_within_template | [expts/pix_cls/003_pix_cls_STN_win.txt](expts/pix_cls/003_pix_cls_STN_win.txt) |
| DEC |ball_within_template | [expts/pix_cls/004_pix_cls_DEC_win.txt](expts/pix_cls/004_pix_cls_DEC_win.txt)  |
| STN on frame | ball_within_template | [expts/pix_cls/005_pix_cls_STN_frame_win.txt](expts/pix_cls/005_pix_cls_STN_frame_win.txt) |
| DEC on frame | ball_within_template | [expts/pix_cls/006_pix_cls_DEC_frame_win.txt](expts/pix_cls/006_pix_cls_DEC_frame_win.txt)  |
| None (Ground Truth) |  ball_cross_template | [expts/pix_cls/007_pix_cls_GT_x.txt](expts/pix_cls/007_pix_cls_GT_x.txt) |
| IN |  ball_cross_template | [expts/pix_cls/008_pix_cls_IN_x.txt](expts/pix_cls/008_pix_cls_IN_x.txt) |
| Tx | ball_cross_template | [expts/pix_cls/009_pix_cls_TX_x.txt](expts/pix_cls/009_pix_cls_TX_x.txt) |
| STN | ball_cross_template |[expts/pix_cls/010_pix_cls_STN_x.txt](expts/pix_cls/010_pix_cls_STN_x.txt) |
| DEC |ball_cross_template | [expts/pix_cls/011_pix_cls_DEC_x.txt](expts/pix_cls/011_pix_cls_DEC_x.txt)|


## Object Based Classifiers

The configuration files for each of the object based classfier models can be found in the `exps/obj_cls` directory.


| Fwd Model | Eval Setup | Config File |
|----------|:-------------:|------:|
| None (Ground Truth) | ball_within_template |  [expts/obj_cls/000_tx_cls_GT_win.txt]([expts/obj_cls/000_tx_cls_GT_win.txt]) |
| IN | ball_within_template | [expts/obj_cls/001_tx_cls_IN_win.txt](expts/obj_cls/001_tx_cls_IN_win.txt) |
| Tx | ball_within_template | [expts/obj_cls/002_tx_cls_TX_win.txt](expts/obj_cls/002_tx_cls_TX_win.txt) |
| None (Ground Truth) | ball_cross_template |  [expts/obj_cls/003_tx_cls_GT_x.txt](expts/obj_cls/003_tx_cls_GT_x.txt) |

## Joint Models

The configuration files for each of the joint models can be found in the `exps/joint` directory.

| Model | Eval Setup | Config File |
|----------|:-------------:|------:|
| 3 frame Decoder | ball_within_template | [expts/joint/000_joint_DEC_3f_win.txt](expts/joint/000_joint_DEC_3f_win.txt) |
| 3 frame IN | ball_within_template | [expts/joint/003_joint_IN_win.txt](expts/joint/003_joint_IN_win.txt) |
| 3 frame Tx | ball_within_template | [expts/joint/004_joint_TX_win.txt](expts/joint/004_joint_TX_win.txt) |
| 3 frame Decoder w/o rollout loss |ball_within_template |  [expts/joint/002_joint_DEC_3f_win_no_rollout_loss.txt](expts/joint/002_joint_DEC_3f_win_no_rollout_loss.txt) |
| 3 frame Decoder | ball_cross_template | [expts/joint/005_joint_DEC_3f_x.txt](expts/joint/005_joint_DEC_3f_x.txt) |


## 1 Frame models (used for comparison to SOTA)

| Model | Eval Setup | Config File | Fold ID | Num Fwd Times | Download | AUCESS |
|----------|:-------------:|------:|------:|------:|------:|------:|
| 1 frame Decoder | ball_within_template | [expts/joint/001_joint_DEC_1f_win.txt](expts/joint/001_joint_DEC_1f_win.txt) | 0 | 10 | s3://dl.fbaipublicfiles.com/phyre-fwd-agents/expts/joint/001_joint_DEC_1f_win.txt/70 | 79.73 |
| 1 frame Decoder | ball_cross_template | [expts/joint/006_joint_DEC_1f_x.txt](expts/joint/006_joint_DEC_1f_x.txt) | 0 | 10 | s3://dl.fbaipublicfiles.com/phyre-fwd-agents/expts/joint/006_joint_DEC_1f_x.txt/70 | 52.64 |

To run evaluation to compare to the state of the art, we can download the pretrained model and evaluate it locally.

1) Download the model
```
cd agents
mkdir -p outputs/expts/joint/001_joint_DEC_1f_win.txt
s3cmd sync --skip-existing s3://dl.fbaipublicfiles.com/phyre-fwd-agents/joint/001_joint_DEC_1f_win.txt/70/
outputs/expts/joint/001_joint_DEC_1f_win.txt/70/
```

2) Run the evaluation
```
./launch.py -l -t -c expts/joint/001_joint_DEC_1f_win.txt
```
If prompted, select run 70.

# Computing pixel accuracy

We introduce a Forward Prediction Accuracy (FPA) metric for PHYRE to gauge the
pixel accuracy of our forward models. We provide a nifty script to do the same.

## Generating the rollouts

First, generate the rollouts for the model you want to evaluate using the
following command. The `<conf_path>` refers to the config you want to generate
rollouts for.
```bash
./launch.py -c <conf_path> -tv eval.n_fwd_times=20 \
    eval.data_loader.num_workers=10 \
    hydra.launcher.cpus_per_task=10 \
    eval.batch_size=4 num_gpus=1
```

This will launch jobs for each sweep configuration in `<conf_path>`. Usually
the sweep is over the folds and it will generate rollouts for each fold
in the `outputs/<conf_path>/<run_id>/vis/` directory (where `run_id` would
correspond to the fold). You can also use `--run_id` to specify a specific
run you want to visualize, but in that case the outputs will be stored
in `outputs/<conf_path>/vis/` (not in the `run_id` folder). In either case,
it will store both the GT and predicted rollouts in specific folders.

If the conf involves sweep over other parameters than the fold, for instance
in joint models (like [joint/000_joint_DEC_3f_win.txt](expts/joint/000_joint_DEC_3f_win.txt))
we sweep over both the `fold_id` and `n_fwd_times`
(i.e. the rollout length the model is trained for), you can still use the
above command, which will launch the 10 fold x 8 `n_fwd_times` jobs,
and you can kill the jobs corresponding to params you don't need visualizations.
For instance for the above config, if you just want to run visualization for
fold 0 corresponding to the model trained with `n_fwd_times=10`, you can
launch all 80 jobs and kill everything except the job 70 (jobs 0-9 correspond
to all folds for `n_fwd_times=0`, 10-19 correspond to all folds for `n_fwd_times=1`,
and so on).
If you're using a SLURM cluster where the jobs are launched as SLURM arrays,
this can be done by something like the following,
where job_id is what SLURM assigns to your job after it is launched.

```bash
scancel <job_id>_[0-69] <job_id>_[71-79]
```

## Evaluating the accuracy

Once the rollouts have been computed, you can compute the FPA using
`python pixel_acc_utils.py`, after setting the paths in the `main` function
in that python file. See the provided example for the arguments.
For the example config provided in the file (3 frame DEC model,
trained for `n_fwd_times=10`, fold 0; i.e.
[joint/000_joint_DEC_3f_win.txt](expts/joint/000_joint_DEC_3f_win.txt) run 70),
you should get avg FPA over 10s rollout around 96.13.
