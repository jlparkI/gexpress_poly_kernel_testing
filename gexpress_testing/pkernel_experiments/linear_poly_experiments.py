"""Trains a Bayesian linear model and a poly kernel
model on the data and checks training set performance,
after first filtering the input datasets to ensure
only non-redundant files are included."""
import time
import numpy as np
from scipy.stats import pearsonr
from xGPR import xGPRegression as xGPReg
from xGPR import build_offline_fixed_vector_dataset
from ..utilities.utilities import get_cv_splits

def evaluate_model(model, trainx_files, trainy_files, validx_files, validy_files):
    """Evaluates a trained model against the training and validation
    data."""
    return eval_batch(model, trainx_files, trainy_files), \
            eval_batch(model, validx_files, validy_files)


def eval_batch(model, xfiles, yfiles):
    """Evaluates the model on a specific batch of files."""
    preds, gt_y = [], []
    for xfile, yfile in zip(xfiles, yfiles):
        preds.append(model.predict(np.load(xfile), get_var=False))
        gt_y.append(np.load(yfile))
    preds = np.concatenate(preds)
    gt_y = np.concatenate(gt_y)
    return pearsonr(preds, gt_y)[0]


def run_all_cvs(xfiles, pfiles, yfiles, output_file):
    """Runs all of the planned experiments against the nonredundant ids.
    Loop over the CV splits. For each split, train 1) a linear model
    using promoters only, 2) a linear model using promoters and enhancers,
    3) a poly model using promoters only with different RFFs,
    4) a poly model using promoters + enhancers with different RFFs.
    Write all results to the output file."""
    cv_splits = get_cv_splits(xfiles, pfiles, yfiles)
    
    with open(output_file, "a+", encoding="utf-8") as fhandle:
        fhandle.write("Model,RFFs,Input_data,"
                "Train_pearson_r,Valid_pearson_r,hparams,NMLL\n")
        for i, cv_split in enumerate(cv_splits):
            print(f"CV split {i}", flush=True)
            _ = fit_evaluate_model(cv_split, 2048, fhandle, "Linear",
                                   None, "promoters")
            _ = fit_evaluate_model(cv_split, 2048, fhandle, "Linear",
                                   None, "merged")

            prom_h = fit_evaluate_model(cv_split, 16384, fhandle, "Poly",
                                    None, "promoters")
            merge_h = fit_evaluate_model(cv_split, 16384, fhandle, "Poly",
                                    None, "merged")
            for rffs in [32768,65536]:
                _ = fit_evaluate_model(cv_split, rffs, fhandle, "Poly",
                                prom_h, "promoters")
                _ = fit_evaluate_model(cv_split, rffs, fhandle, "Poly",
                                merge_h, "merged")



def fit_evaluate_model(cv_split, rffs, fhandle, model_type = "Linear",
                preset_hyperparams = None, data_type = "promoters"):
    """Evaluates a linear model on the split supplied by
    caller.

    Args:
        cv_split (dict): A dict containing a list of the training
            and validation files.
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
    trainy, validy = cv_split["train_y"], cv_split["valid_y"]
    if data_type == "promoters":
        trainx, validx = cv_split["train_p"], cv_split["valid_p"]
    else:
        trainx, validx = cv_split["train_x"], cv_split["valid_x"]

    train_dset = build_offline_fixed_vector_dataset(trainx, trainy,
                        chunk_size=20000, skip_safety_checks=True)

    # Variance rffs does not matter and is not used. For a linear
    # model the # rffs is ignored. num_threads is ignored if fitting
    # on GPU.
    model = xGPReg(training_rffs = 8192, fitting_rffs = rffs,
                          variance_rffs = 512, kernel_choice = model_type,
                          kernel_specific_params={"intercept":True,
                                           "polydegree":2},
                          verbose = False, device = "gpu",
                          num_threads = 10)
    if model_type == "Linear":
        pre_rank, pre_method = 512, "srht"
        _, _, nmll, _ = model.tune_hyperparams_crude_bayes(train_dset)
        hparams = model.get_hyperparams()

    else:
        pre_rank, pre_method = 4000, "srht_2"
        if preset_hyperparams is None:
            _, _, nmll, _ = model.tune_hyperparams_crude_bayes(train_dset)
            hparams = model.get_hyperparams()
        else:
            nmll = "NA"
            hparams = preset_hyperparams.copy()

    print("Tuning complete.", flush=True)    
    preconditioner, _ = model.build_preconditioner(train_dset,
                       max_rank = pre_rank, method = pre_method,
                        preset_hyperparams = hparams)
    model.fit(train_dset, preconditioner = preconditioner,
                 mode = "cg", tol = 1e-6)
    train_r, valid_r = evaluate_model(model, trainx, trainy,
                                        validx, validy)

    hparams = "_".join([str(z) for z in model.get_hyperparams().tolist()])
    print(f"Model: {model_type}, Hyperparams: {hparams}", flush=True)

    fhandle.write(f"{model_type},{model.fitting_rffs},{data_type},"
                f"{train_r},{valid_r},{hparams},{nmll}\n")
    fhandle.flush()
    print(time.time() - st)
    return model.get_hyperparams()
