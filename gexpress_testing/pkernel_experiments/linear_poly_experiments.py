"""Trains a Bayesian linear model and a poly kernel
model on the data and checks training set performance,
after first filtering the input datasets to ensure
only non-redundant files are included."""
import time
import os
import pickle
import numpy as np
from scipy.stats import pearsonr
from xGPR import xGPRegression as xGPReg
from xGPR import build_offline_fixed_vector_dataset, build_online_dataset
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
                preds = model.predict(xdata, get_var=False)

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




def eval_batch(model, xfiles, yfiles, scaler = None):
    """Evaluates the model on a specific batch of files."""
    gt_y = np.concatenate([np.load(yfile) for yfile in yfiles])
    if scaler is None:
        preds = np.concatenate([model.predict(np.load(xfile), get_var=False)
                                for xfile in xfiles])
    else:
        preds = []
        for xfile in xfiles:
            xdata = scaler.transform(np.load(xfile))
            preds.append(model.predict(xdata, get_var=False))
        preds = np.concatenate(preds)
    return pearsonr(preds, gt_y)[0]


def eval_array(model, xdata, ydata, scaler = None):
    """Evaluates the model on arrays of data loaded into memory."""
    if scaler is None:
        preds = model.predict(xdata, get_var=False, chunk_size=200)
    else:
        xtemp = scaler.transform(xdata)
        preds = model.predict(xtemp, get_var=False, chunk_size=200)
    return pearsonr(preds, ydata)[0]


def fit_final_exact_quad(xfiles, yfiles, output_path):
    """Fits the final exact quadratic model to the full dataset.
    This model can be used to extract feature / interaction
    term importance."""
    # The hard-coded hyperparameters here were obtained from
    # tuning using the approximate kernel. TODO: Move these
    # to a constants file.
    fit_final_model(xfiles, yfiles, output_path,
                    np.array([-0.2256119, 0]))


def run_traintest_split(pfiles, yfiles, nonredundant_ids, data_path,
                        output_file):
    """Fits linear and approximated quadratic to 40 cell lines,
    then tests on the rest, using promoters only."""
    tt_split = get_tt_split([], pfiles, yfiles, nonredundant_ids)
    with open(output_file, "a+", encoding="utf-8") as fhandle:
        _ = fit_evaluate_model(tt_split, 2048, fhandle, data_path,
                            "Linear", None, "promoters")
        prom_h = fit_evaluate_model(tt_split, 32768, fhandle, data_path,
                            "Poly", None, "promoters")



def fit_evaluate_model(tt_split, rffs, fhandle, data_path,
                model_type = "Linear", preset_hyperparams = None,
                data_type = "promoters"):
    """Evaluates a linear model on the split supplied by
    caller.

    Args:
        tt_split (dict): A dict containing a list of the training
            and validation files, organized by cell line id.
        rffs (int): The number of rffs. Ignored for linear models.
        fhandle: A handle to the output file where results on
            the training and validation sets should be written.
        model_type (str): One of 'Linear', 'Poly'. If 'poly',
            fit a degree-2 polynomial.
        preset_hyperparams: Either None or a numpy array. If a
            numpy array, these "recycled" hyperparameters from
            a previous tuning run are used; otherwise, hyperparameters
            are tuned.
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

    for nonred_id in tt_split["train_ids"]:
        trainx += tt_split["train_ids"][nonred_id][data_key]
        trainy += tt_split["train_ids"][nonred_id]["y"]

    train_dset = build_offline_fixed_vector_dataset(trainx, trainy,
                        chunk_size=5000)

    # Variance rffs does not matter and is not used. For a linear
    # model the # rffs is ignored. num_threads is ignored if fitting
    # on GPU.
    model = xGPReg(training_rffs = 8192, fitting_rffs = rffs,
                          variance_rffs = 64, kernel_choice = model_type,
                          kernel_specific_params={"intercept":True,
                                           "polydegree":2},
                          verbose = True, device = "gpu",
                          num_threads = 10)
    if model_type == "Linear":
        pre_rank, pre_method = 512, "srht"
        xdim = np.load(trainx[0]).shape[1]
        if xdim < 512:
            pre_rank = 256
        bounds = np.array([[-6.907,2], [-0.001,0.001]])
        _, _, nmll, _ = model.tune_hyperparams_crude_bayes(train_dset, bounds=bounds)
        hparams = model.get_hyperparams()

    else:
        pre_rank, pre_method = 4000, "srht_2"
        if preset_hyperparams is None:
            bounds = np.array([[-6.907,2], [-0.001,0.001]])
            _, _, nmll, _ = model.tune_hyperparams_crude_bayes(train_dset, bounds=bounds)
            hparams = model.get_hyperparams()
        else:
            nmll = "NA"
            hparams = preset_hyperparams.copy()

    print(f"Tuning complete. Hparams: {hparams}", flush=True)
    preconditioner, ratio = model.build_preconditioner(train_dset,
                       max_rank = pre_rank, method = pre_method,
                        preset_hyperparams = hparams)
    print(f"Ratio: {ratio}", flush=True)
    model.fit(train_dset, preconditioner = preconditioner,
                 mode = "cg", tol = 1e-6, suppress_var=True)

    hparams = "_".join([str(z) for z in model.get_hyperparams().tolist()])
    print(f"Model: {model_type}, Hyperparams: {hparams}", flush=True)

    single_file_evaluation(model, tt_split, data_key, fhandle,
                           model_type, data_type, data_path)

    print(time.time() - st)
    return model.get_hyperparams()



def fit_evaluate_eq_model(cv_split, fhandle, preset_hyperparams = None,
                          data_type = "promoters", online_dataset = True):
    """Evaluates the ExactQuadratic kernel on the cv split passed
    by caller.

    Args:
        cv_split (dict): A dict containing a list of the training
            and validation files.
        fhandle: A handle to the output file where results on
            the training and validation sets should be written.
        preset_hyperparams: Either None or a numpy array. If a
            numpy array, these "recycled" hyperparameters from
            a previous tuning run are used; otherwise, hyperparameters
            are tuned.
        data_type (str): One of 'promoters', 'merged'.
        online_dataset (bool): If False, build an offline dataset
            so that data is only loaded one chunk at a time.

    Returns:
        hparams (np.ndarray): The hyperparameters as a numpy
            array so that caller can recycle them if desired.
    """
    st = time.time()
    trainy, validy = cv_split["train_y"], cv_split["valid_y"]
    if data_type == "promoters":
        trainx, validx = cv_split["train_p"], cv_split["valid_p"]
    else:
        trainx, validx = cv_split["train_x"], cv_split["valid_x"]

    #Loading all of the training data into memory is...not great --
    #just doing this as a temporary hack -- will fix this later.
    if online_dataset:
        trainx = np.vstack([np.load(x) for x in trainx]).astype(np.float64)
        trainy = np.concatenate([np.load(y) for y in trainy])
        train_dset = build_online_dataset(trainx, trainy, chunk_size=250)

    else:
        train_dset = build_offline_fixed_vector_dataset(trainx, trainy,
                        chunk_size=250, skip_safety_checks=True)

    # Variance rffs does not matter and is not used. For a linear
    # model the # rffs is ignored. num_threads is ignored if fitting
    # on GPU.
    model = xGPReg(training_rffs = 8192, fitting_rffs = 8192,
                          variance_rffs = 512, kernel_choice = "ExactQuadratic",
                          kernel_specific_params={"intercept":True,
                                           "polydegree":2},
                          verbose = True, device = "gpu",
                          num_threads = 10)
    pre_rank, pre_method = 5000, "srht"
    nmll = "NA"
    hparams = preset_hyperparams.copy()

    preconditioner, ratio = model.build_preconditioner(train_dset,
                       max_rank = pre_rank, method = pre_method,
                        preset_hyperparams = hparams)
    print(f"Ratio: {ratio}", flush=True)
    model.fit(train_dset, preconditioner = preconditioner,
                 max_iter=500, mode = "lbfgs",
                preset_hyperparams = hparams)
    if online_dataset:
        train_r = eval_array(model, trainx, trainy)
    else:
        train_r = eval_batch(model, trainx, trainy)
    valid_r = eval_batch(model, validx, validy)

    hparams = "_".join([str(z) for z in model.get_hyperparams().tolist()])
    print(f"Model: ExactQuadratic, Hyperparams: {hparams}", flush=True)
    print(f"train: {train_r}, valid: {valid_r}", flush=True)

    fhandle.write(f"ExactQuadratic,{model.fitting_rffs},{data_type},"
                f"{train_r},{valid_r},{hparams},{nmll}\n")
    fhandle.flush()
    print(time.time() - st)
    return model.get_hyperparams()



def fit_final_model(xfiles, yfiles, output_path, preset_hyperparams):
    """Fits the ExactQuadratic model to the full promoter only dataset.

    Args:
        xfiles (list): A list of promoter feature files.
        yfiles (list): A list of expression value files.
        output_path (str): The path where the final model should
            be saved.
        preset_hyperparams: A numpy array. These "recycled"
            hyperparameters from a previous tuning run are
            used to fit the final model.
    """
    train_dset = build_offline_fixed_vector_dataset(xfiles, yfiles,
                        chunk_size=500, skip_safety_checks=True)

    model = xGPReg(training_rffs = 8192, fitting_rffs = 8192,
                          variance_rffs = 4096, kernel_choice = "ExactQuadratic",
                          kernel_specific_params={"intercept":True,
                                           "polydegree":2},
                          verbose = True, device = "gpu",
                          num_threads = 10)
    pre_rank, pre_method = 5000, "srht"
    hparams = preset_hyperparams.copy()

    preconditioner, ratio = model.build_preconditioner(train_dset,
                       max_rank = pre_rank, method = pre_method,
                        preset_hyperparams = hparams)
    print(f"Ratio: {ratio}", flush=True)
    model.fit(train_dset, preconditioner = preconditioner,
                 max_iter=1000, mode = "lbfgs",
                preset_hyperparams = hparams)
    with open(os.path.join(output_path, "exact_quad_overlapped_pro.pk"), "wb") as fhandle:
        pickle.dump(model, fhandle)
