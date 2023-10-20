"""Utilities for handling, filtering and sorting file lists."""
import os
import random
import itertools
from math import ceil
import numpy as np
from sklearn.preprocessing import StandardScaler



def get_file_lists(start_dir, prom_dir, ydir, en_dir):
    """Gets the list of all training promoter, enhancer, y files."""
    os.chdir(start_dir)
    os.chdir(prom_dir)
    prom_files = [f for f in os.listdir() if f.endswith("_count_matrix_pro.npy")]
    yfiles = [os.path.join(ydir, f.split("_count_matrix_pro.npy")[0] + ".y.npy")
              for f in prom_files]
    en_files = [os.path.join(en_dir, f.split("_count_matrix_pro.npy")[0] + "_count_matrix_enh.npy")
              for f in prom_files]
    prom_files = [os.path.abspath(f) for f in prom_files]
    os.chdir(start_dir)
    return prom_files, yfiles, en_files



def filter_data(prom_path, ypath, en_path, nonredundant_ids,
                storage_dir):
    """Filters the input data by making copies only of files with
    nonredundant lines. Generates merged files containing both
    promoters and enhancers.

    Args:
        prom_path (str): A filepath to the directory with promoter counts.
        ypath (str): A filepath to the directory with yvalues.
        en_path (str): A filepath to the directory with enhancer counts.
        nonredundant_ids (list): List of the nonredundant cell lines.
        storage_dir (str): A filepath to a temporary storage dir where
            the data is saved. In a cluster environment, this can be
            /scratch.

    Returns:
        out_xfiles (list): A list of numpy files in storage_dir with both
            promoter and enhancer counts; saved as np.ushort.
        out_pfiles (list): A list of numpy files in storage_dir with promoter
            counts only.
        out_yfiles (list): A list of numpy files in storage dir with yvalues.

    Raises:
        ValueError: A ValueError is raised if one or more expected nonredundant
            cell line ids are not found
    """
    out_xfiles, out_pfiles, out_yfiles = [], [], []
    ndpoints = 0

    print("Now retrieving and sorting files...", flush=True)

    for nonred_id in nonredundant_ids:
        pfile = os.path.join(prom_path, f"{nonred_id}_count_matrix_pro.npy")
        yfile = os.path.join(ypath, f"{nonred_id}.y.npy")
        enfile = os.path.join(en_path, f"{nonred_id}_count_matrix_enh.npy")
        try:
            promoters = np.load(pfile)
            ydata = np.load(yfile)
            enhancers = np.load(enfile)
        except:
            raise ValueError(f"Could not find numpy files for nonredundant id {nonred_id}")

        promoters = promoters.astype(np.float32)
        enhancers = enhancers.astype(np.float32)
        ndpoints += promoters.shape[0]

        merged_data = np.concatenate([promoters, enhancers], axis=1)

        out_prom_file = os.path.join(storage_dir, f"{nonred_id}_promoters.npy")
        out_xfile = os.path.join(storage_dir, f"{nonred_id}_merged.npy")
        out_yfile = os.path.join(storage_dir, f"{nonred_id}_y.npy")

        np.save(out_prom_file, promoters)
        np.save(out_xfile, merged_data)
        np.save(out_yfile, ydata)

        out_pfiles.append(out_prom_file)
        out_xfiles.append(out_xfile)
        out_yfiles.append(out_yfile)

    print(f"Total datapoints: {ndpoints}", flush=True)
    return out_xfiles, out_pfiles, out_yfiles



def get_final_data_list(prom_path, ypath, nonredundant_ids,
                storage_dir):
    """Filters the input data by making copies only of files with
    nonredundant lines, and saves data in appropriately-sized chunks
    for the exact quadratic. Uses promoter features only.

    Args:
        prom_path (str): A filepath to the directory with promoter counts.
        ypath (str): A filepath to the directory with yvalues.
        nonredundant_ids (list): List of the nonredundant cell lines.
        storage_dir (str): A filepath to a temporary storage dir where
            the data is saved. In a cluster environment, this can be
            /scratch.

    Returns:
        out_pfiles (list): A list of numpy files in storage_dir with promoter
            counts only.
        out_yfiles (list): A list of numpy files in storage dir with yvalues.

    Raises:
        ValueError: A ValueError is raised if one or more expected nonredundant
            cell line ids are not found
    """
    out_pfiles, out_yfiles = [], []
    ndpoints = 0

    print("Now retrieving and sorting files...", flush=True)

    for nonred_id in nonredundant_ids:
        pfile = os.path.join(prom_path, f"{nonred_id}_count_matrix_pro.npy")
        yfile = os.path.join(ypath, f"{nonred_id}.y.npy")
        try:
            promoters = np.load(pfile)
            ydata = np.load(yfile)
        except:
            raise ValueError(f"Could not find numpy files for nonredundant id {nonred_id}")

        promoters = promoters.astype(np.float32)
        ndpoints += promoters.shape[0]

        chunk_size = 250

        for j in range(0, promoters.shape[0], chunk_size):
            out_prom_file = os.path.join(storage_dir, f"{nonred_id}_{j}_promoters.npy")
            out_yfile = os.path.join(storage_dir, f"{nonred_id}_{j}_y.npy")

            np.save(out_prom_file, promoters[j:j+chunk_size,...])
            np.save(out_yfile, ydata[j:j+chunk_size])

            out_pfiles.append(out_prom_file)
            out_yfiles.append(out_yfile)

    print(f"Total datapoints: {ndpoints}", flush=True)
    return out_pfiles, out_yfiles






def cleanup_storage(xfiles, pfiles, yfiles):
    """Cleans up the temporary files created in the storage dir."""
    for xfile in xfiles:
        os.remove(xfile)
    for yfile in yfiles:
        os.remove(yfile)
    for pfile in pfiles:
        os.remove(pfile)


def get_cv_splits(xfiles, pfiles, yfiles):
    """Breaks the input file lists up into train-test splits
    for cross-validations."""
    idx = list(range(len(xfiles)))
    random.seed(123)
    random.shuffle(idx)
    vset_size = ceil(len(xfiles) / 5)

    chunked_idx = [idx[i:i+vset_size] for i in range(0, len(idx), vset_size)]

    cv_splits = [{} for i in range(5)]
    
    for i, valid_idx  in enumerate(chunked_idx):
        train_idx = itertools.chain.from_iterable(chunked_idx[:i] +
                                                  chunked_idx[i+1:])
        train_idx = list(train_idx)
        cv_splits[i]["train_x"] = [xfiles[j] for j in train_idx]
        cv_splits[i]["train_p"] = [pfiles[j] for j in train_idx]
        cv_splits[i]["train_y"] = [yfiles[j] for j in train_idx]

        cv_splits[i]["valid_x"] = [xfiles[j] for j in valid_idx]
        cv_splits[i]["valid_p"] = [pfiles[j] for j in valid_idx]
        cv_splits[i]["valid_y"] = [yfiles[j] for j in valid_idx]

    return cv_splits
