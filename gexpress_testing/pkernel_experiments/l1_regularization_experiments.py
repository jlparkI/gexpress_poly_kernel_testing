"""Trains a Bayesian linear model and a poly kernel
model on the data and checks training set performance,
after first filtering the input datasets to ensure
only non-redundant files are included."""
import time
import os
import pickle
import numpy as np
import cupy as cp
from scipy.stats import pearsonr
from exactPolynomial import ExactQuadratic
from exactPolynomial import build_offline_np_dataset
from ..utilities.utilities import get_tt_split


def single_file_evaluation(model, tt_split, data_key, fhandle,
                           model_type, data_type, data_path):
    """Evaluates the pearson r separately for each nonredundant id in
    the validation set."""
    for data_group in ["train", "valid"]:
        all_y, all_preds = [], []

        for idnum, data_batch in tt_split[f"{data_group}_ids"].items():
            batch_y, batch_preds = [], []
            for xfile, yfile in zip(data_batch[data_key], data_batch["y"]):
                xdata, ydata = np.load(xfile), np.load(yfile)
                preds = model.predict(xdata)

                all_y.append(ydata)
                batch_y.append(ydata)
                all_preds.append(preds)
                batch_preds.append(preds)

            batch_pearsonr = pearsonr(np.concatenate(batch_preds),
                                      np.concatenate(batch_y))[0]
            fhandle.write(f"{idnum},{data_group},{batch_pearsonr},"
                        f"{model_type},{data_type},{data_path}\n")

        all_pearsonr = pearsonr(np.concatenate(all_preds),
                                      np.concatenate(all_y))[0]
        fhandle.write(f"ALL_LINES,{data_group},{all_pearsonr},"
                        f"{model_type},{data_type},{data_path}\n")
        fhandle.flush()



def eval_bic_aic(model, xfiles, yfiles):
    """Calculates the bic and aic for the training set."""
    gt_y = np.concatenate([np.load(yfile) for yfile in yfiles])
    preds = np.concatenate([model.predict(np.load(xfile))
                                for xfile in xfiles])
    sse = ( (preds - gt_y)**2 ).sum()
    ndpoints = float(gt_y.shape[0])
    df = ndpoints - float(np.load(xfiles[0]).shape[1])
    sigma2 = sse / df

    if model.device == "cpu":
        n_params = (np.abs(model.weights) > 0).sum()
    else:
        n_params = (np.abs(cp.asnumpy(model.weights))).sum()
    
    loglik_term = ndpoints * np.log(sigma2) + sse / sigma2
    bic = np.log(ndpoints) * n_params + loglik_term
    aic = 2 * n_params + loglik_term
    return bic, aic


def eval_batch(model, xfiles, yfiles):
    """Evaluates the model on a specific batch of files."""
    gt_y = np.concatenate([np.load(yfile) for yfile in yfiles])
    preds = np.concatenate([model.predict(np.load(xfile))
                                for xfile in xfiles])
    return pearsonr(preds, gt_y)[0]


def eval_array(model, xdata, ydata):
    """Evaluates the model on arrays of data loaded into memory."""
    preds = model.predict(xdata, chunk_size=200)
    return pearsonr(preds, ydata)[0]


def hyperparameter_tuning(pfiles, yfiles, nonredundant_ids, output_file):
    """Fits linear and approximated quadratic to 40 cell lines,
    then tests on the rest, using promoters only."""
    tt_split = get_tt_split([], pfiles, yfiles, nonredundant_ids)
    with open(output_file, "a+", encoding="utf-8") as fhandle:
        for hparam in [np.array([1.5]), np.array([2]),
                       np.array([-1])]:
            fit_and_calc_bic(tt_split, fhandle, hparam, "promoters")


def fit_and_calc_bic(tt_split, fhandle, preset_hyperparams,
                          data_type = "promoters"):
    """Fits using L1 regularization and calculates BIC, AIC.

    Args:
        tt_split (dict): A dict containing a list of the training
            and validation files.
        fhandle: A handle to the output file where results on
            the training and validation sets should be written.
        preset_hyperparams: 
        data_type (str): One of 'promoters', 'merged'.

    Returns:
        hparams (np.ndarray): The hyperparameters as a numpy
            array so that caller can recycle them if desired.
    """
    st = time.time()
    
    data_key = "x"
    if data_type == "promoters":
        data_key = "p"
    trainy, trainx = [], []
    validy, validx = [], []

    for nonred_id in tt_split["train_ids"]:
        trainx += tt_split["train_ids"][nonred_id][data_key]
        trainy += tt_split["train_ids"][nonred_id]["y"]

    for nonred_id in tt_split["valid_ids"]:
        validx += tt_split["valid_ids"][nonred_id][data_key]
        validy += tt_split["valid_ids"][nonred_id]["y"]
   
    train_dset = build_offline_np_dataset(trainx, trainy,
                        chunk_size=250, skip_safety_checks=True)

    model = ExactQuadratic(device = "gpu", num_threads = 10)
    model.initialize(train_dset)
    hparams = preset_hyperparams.copy()

    model.fit(train_dset, regularization="l1",
            max_iter=1000, mode = "lbfgs",
            preset_hyperparams = hparams)
    bic, aic = eval_bic_aic(model, trainx, trainy)
    train_r = eval_batch(model, trainx, trainy)
    valid_r = eval_batch(model, validx, validy)

    hparams = "_".join([str(z) for z in model.get_hyperparams().tolist()])
    print(f"Model: ExactQuadratic, Hyperparams: {hparams}", flush=True)
    print(f"train: {train_r}, valid: {valid_r}", flush=True)

    fhandle.write(f"L1 ExactQuadratic,{data_type},"
                f"{train_r},{valid_r},{hparams},{bic},{aic}\n")
    fhandle.flush()
    print(time.time() - st)



def fit_final_model(xfiles, yfiles, output_path, preset_hyperparams):
    """Fits the ExactQuadratic model with L1 regularization to the full
    promoter only dataset.

    Args:
        xfiles (list): A list of promoter feature files.
        yfiles (list): A list of expression value files.
        output_path (str): The path where the final model should
            be saved.
        preset_hyperparams: A numpy array. These "recycled"
            hyperparameters from a previous tuning run are
            used to fit the final model.
    """
    train_dset = build_offline_np_dataset(xfiles, yfiles,
                        chunk_size=500, skip_safety_checks=True)

    model = ExactQuadratic(device = "cpu", num_threads = 5)
    hparams = preset_hyperparams.copy()

    model.fit(train_dset, regularization="l1",
            max_iter=1000, mode = "lbfgs",
            preset_hyperparams = hparams)
    with open(os.path.join(output_path, "l1_exact_quad.pk"), "wb") as fhandle:
        pickle.dump(model, fhandle)
