"""Trains a Bayesian linear model and a poly kernel
model on the data and checks training / validation set performance,
after first filtering the input datasets to ensure
only non-redundant files are included."""
import os
import pickle
import time
import numpy as np
from scipy.stats import pearsonr
from xGPR import xGPRegression as xGPReg
from xGPR import build_regression_dataset
from exactPolynomial import ExactQuadratic
from exactPolynomial import build_offline_np_dataset



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
                preds = model.predict(xdata)

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
        fhandle.write(f"ALL_LINES,{data_group},{all_pearsonr},"
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


def run_gene_group_traintest_split(pfiles:list, output_file:str):
    """This is a wrapper on run_traintest_split which splits up a list
    of files with separate gene groups and runs a 5x CV."""
    # Simple sanity check -- make sure that no nonredundant cell id has more
    # than 5 associated files.
    if max([int(os.path.basename(k).split("_")[1]) for k in pfiles]) > 4:
        raise ValueError("One or more files associated with the gene group split "
                         "has more than 5 associated files which is an error.")

    with open(output_file, "a+", encoding="utf-8") as fhandle:
        for i in range(5):
            train_pfiles = [p for p in pfiles if os.path.basename(p).split("_")[1] != str(i)]
            test_pfiles = [p for p in pfiles if os.path.basename(p).split("_")[1] == str(i)]
            train_yfiles = [p.replace("promoters.npy", "y.npy") for p in train_pfiles]
            test_yfiles = [p.replace("promoters.npy", "y.npy") for p in test_pfiles]

            tt_split = {"train_ids":{"NA":{"p":train_pfiles, "y":train_yfiles }},
                    "valid_ids":{"NA":{"p":test_pfiles, "y":test_yfiles} } }
            #fit_evaluate_model(tt_split, 2048, fhandle, f"Split_{i}",
            #                "Linear", "promoters")
            #fit_evaluate_model(tt_split, 32768, fhandle, f"Split_{i}",
            #                "Poly", "promoters")
            fit_evaluate_model(tt_split, 32768, fhandle, f"Split_{i}",
                            "RBF", "promoters")



def run_traintest_split(tt_split:dict, split_description:str,
                        output_file:str, data_type:str):
    """Fits linear and approximated quadratic to 40 cell lines,
    then tests on the rest."""
    with open(output_file, "a+", encoding="utf-8") as fhandle:
        fit_evaluate_model(tt_split, 2048, fhandle, split_description,
                            "Linear", data_type)
        fit_evaluate_model(tt_split, 32768, fhandle, split_description,
                            "Poly", data_type)





def run_traintest_exact_quad(tt_split:dict, split_description:str,
                            output_file:str, data_type:str,
                            output_path:str):
    """Fits an exact quadratic to 40 cell lines,
    then tests on the rest."""
    with open(output_file, "a+", encoding="utf-8") as fhandle:
        fit_evaluate_eq_model(tt_split, fhandle, split_description,
                    output_path, data_type)



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
        split_description (str): A description of this particular split
            which will be written to file.
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

    # The chunk size shown is a maximum -- most will have significantly
    # less than this.
    train_dset = build_regression_dataset(trainx, trainy,
                        chunk_size=8000)

    # Variance rffs does not matter and is not used. For a linear
    # model the # rffs is ignored. num_threads is ignored if fitting
    # on GPU.
    if model_type == "RBF":
        train_rffs = 4096
    else:
        train_rffs = 8192
    model = xGPReg(num_rffs = train_rffs, variance_rffs = 64, kernel_choice = model_type,
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




def fit_evaluate_eq_model(tt_split:dict, fhandle, split_description:str,
                output_path:str, data_type:str = "promoters"):
    """Evaluates an exact quadratic model on the tt split specified by
    caller.

    Args:
        tt_split (dict): A dict containing a list of the training
            and validation files, organized by cell line id.
        fhandle: A handle to the output file where results on
            the training and validation sets should be written.
        split_description (str): A description of this particular split
            which will be written to file.
        output_path (str): Where to save the weights.
        data_type (str): One of 'promoters', 'merged'.
    """
    data_key = "x"
    if data_type == "promoters":
        data_key = "p"
    trainy, trainx = [], []

    for nonred_id in tt_split["train_ids"]:
        trainx += tt_split["train_ids"][nonred_id][data_key]
        trainy += tt_split["train_ids"][nonred_id]["y"]

    train_dset = build_offline_np_dataset(trainx, trainy,
                        chunk_size=250, skip_safety_checks=True)


    model = ExactQuadratic(device = "gpu", regularization = "l2")
    model.initialize(train_dset, hyperparams = np.array([2.]))

    preconditioner, ratio = model.build_preconditioner(train_dset,
                       max_rank = 4000, method = 'srht')
    print(f"Ratio: {ratio}", flush=True)
    model.fit(train_dset, preconditioner = preconditioner,
                 max_iter=1000, mode = "cg")

    single_file_evaluation(model, tt_split, data_key, fhandle,
                           "EXACT_QUADRATIC", data_type, split_description)

    hparams = "_".join([str(z) for z in model.get_hyperparams().tolist()])
    print(f"Model: ExactQuadratic, Hyperparams: {hparams}", flush=True)

    model.device = "cpu"
    with open(os.path.join(output_path, f"{split_description.split('_fpath_')[0]}_exact_quad.pk"), "wb") as \
            weight_handle:
        output_dict = {"weights":model.weights, "hparam":hparams}
        pickle.dump(output_dict, weight_handle)




def single_group_gcv(xfname:str, yfname:str,
                     output_file:str):
    """Performs a cross-validation on a single pair of x and y
    files."""
    xdata, ydata = np.load(xfname), np.load(yfname)
    rng = np.random.default_rng(123)
    idx = np.array_split(rng.permutation(xdata.shape[0]), 5)


    for model_type in ["Linear", "Poly"]:
        elapsed_time = time.time()

        for i in range(5):
            train_idx = np.concatenate(idx[:i] + idx[(i+1):])

            testx, testy = xdata[idx[i],...], ydata[idx[i]]
            trainx, trainy = xdata[train_idx,...], ydata[train_idx]
            train_dset = build_regression_dataset(trainx, trainy,
                                                  chunk_size=2000)

            # Variance rffs does not matter and is not used. For a linear
            # model the # rffs is ignored. num_threads is ignored if fitting
            # on GPU.
            model = xGPReg(num_rffs = 16384, variance_rffs = 64,
                           kernel_choice = model_type,
                           kernel_settings={"intercept":True, "polydegree":2},
                           verbose = True, device = "gpu",
                           num_threads = 10)

            # Inner nested CV to pick a hyperparameter.
            hps = [1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5, 5.5, 6]
            nmll_vals = []
            for hpv in hps:
                nested_cv_nmll = []
                model.set_hyperparams(np.array([hpv]), dataset=train_dset)
                for j in range(5):
                    subset_idx = np.array_split(rng.permutation(trainx.shape[0]), 5)
                    subset_train_idx = np.concatenate(subset_idx[:j] +
                                                      subset_idx[(j+1):])

                    subset_testx, subset_testy = trainx[subset_idx[j],...], \
                            trainy[subset_idx[j]]
                    subset_trainx, subset_trainy = trainx[subset_train_idx,...], \
                            trainy[subset_train_idx]
                    subset_train_dset = build_regression_dataset(subset_trainx,
                                            subset_trainy, chunk_size=2000)
                    if model_type == "Linear":
                        model.fit(subset_train_dset, mode = "exact", tol=1e-6,
                                  suppress_var = True)
                    else:
                        model.fit(subset_train_dset, mode = "cg", tol = 1e-6,
                                  suppress_var=True)
                    nested_cv_nmll.append(pearsonr(model.predict(subset_testx),
                                                   subset_testy)[0])
                nmll_vals.append(np.mean(nested_cv_nmll))

            print(f"HP: {hps}\nHeld out res: {nmll_vals}", flush=True)

            model.num_rffs = 32768
            model.set_hyperparams(np.array([ hps[np.argmax(nmll_vals)] ]) )

            if model_type == "Linear":
                model.fit(train_dset, mode = "exact", tol=1e-6,
                                  suppress_var = True)
            else:
                model.fit(train_dset, mode = "cg", tol = 1e-6,
                          suppress_var=True)


            trainres = pearsonr(model.predict(trainx, get_var=False), trainy)[0]
            testres = pearsonr(model.predict(testx, get_var=False), testy)[0]

            with open(output_file, "a+", encoding="utf-8") as fhandle:
                fhandle.write(f"{xfname},{i},{model_type},{trainres},{testres}\n")
        elapsed_time = time.time() - elapsed_time
        print(f"For model type {model_type}, elapsed time {elapsed_time}", flush=True)
