"""Trains a Bayesian linear model and a poly kernel
model on the data and checks training / validation set performance,
after first filtering the input datasets to ensure
only non-redundant files are included."""
import time
import numpy as np
from scipy.stats import pearsonr
from xGPR import xGPRegression as xGPReg
from xGPR import build_regression_dataset


def single_file_evaluation(model, tt_split:dict, data_key:str, fhandle,
                    model_type:str, data_type:str, split_description:str):
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
                        f"{model_type},{data_type},{split_description}\n")

        all_pearsonr = pearsonr(np.concatenate(all_preds),
                                      np.concatenate(all_y))[0]
        fhandle.write(f"ALL_LINES_INTERACTIONS,{data_group},{all_pearsonr},"
                        f"{model_type},{data_type},{split_description}\n")
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


def run_traintest_split(tt_split:dict, split_description:str,
                        output_file:str, data_type:str):
    """Fits linear and approximated quadratic to 40 cell lines,
    then tests on the rest."""
    with open(output_file, "a+", encoding="utf-8") as fhandle:
        fit_evaluate_model(tt_split, 2048, fhandle, split_description,
                            "Linear", data_type)
        fit_evaluate_model(tt_split, 32768, fhandle, split_description,
                            "Poly", data_type)


def fit_evaluate_model(tt_split:dict, rffs:int, fhandle, split_description:str,
                model_type:str = "Linear", data_type:str = "promoters"):
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
        data_type (str): One of 'promoters', 'merged'.
    """
    st = time.time()
    data_key = "x"
    if data_type == "promoters":
        data_key = "p"
    trainy, trainx = [], []

    for nonred_id in tt_split["train_ids"]:
        trainx += tt_split["train_ids"][nonred_id][data_key]
        trainy += tt_split["train_ids"][nonred_id]["y"]

    train_dset = build_regression_dataset(trainx, trainy,
                        chunk_size=2000)

    # Variance rffs does not matter and is not used. For a linear
    # model the # rffs is ignored. num_threads is ignored if fitting
    # on GPU.
    model = xGPReg(num_rffs = 8192, variance_rffs = 64, kernel_choice = model_type,
                          kernel_settings={"intercept":True, "polydegree":2},
                          verbose = True, device = "gpu",
                          num_threads = 10)

    hparams, _, nmll = model.tune_hyperparams_crude(train_dset)
    hparams = "_".join([str(z) for z in hparams.tolist()])
    if model_type == "Linear":
        pre_rank, pre_method = 512, "srht"
        xdim = np.load(trainx[0]).shape[1]
        if xdim < 512:
            pre_rank = 256
    else:
        pre_rank, pre_method = 4000, "srht"

    print(f"Tuning complete, nmll {nmll}, hparams: {hparams}", flush=True)
    model.num_rffs = rffs
    preconditioner, ratio = model.build_preconditioner(train_dset,
                       max_rank = pre_rank, method = pre_method)
    print(f"Ratio: {ratio}", flush=True)
    model.fit(train_dset, preconditioner = preconditioner,
                 mode = "cg", tol = 1e-6, suppress_var=True)

    single_file_evaluation(model, tt_split, data_key, fhandle,
                           model_type, data_type, split_description)

    print(time.time() - st, flush=True)
