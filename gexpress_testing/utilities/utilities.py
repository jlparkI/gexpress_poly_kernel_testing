"""Utilities for handling, filtering and sorting file lists."""
import os
import numpy as np



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



def filter_data(prom_path:str, ypath:str, en_path:str, nonredundant_ids:list,
                storage_dir:str, chunk_size:int = 250, promoter_columns = None,
                enhancer_columns = None):
    """Filters the input data by making copies only of files with
    nonredundant lines. Generates merged files containing both
    promoters and enhancers.

    Args:
        prom_path (str): A filepath to the directory with promoter counts.
        ypath (str): A filepath to the directory with yvalues.
        en_path (str): Either None or a valid filepath to the directory
            with enhancer counts.
        nonredundant_ids (list): List of the nonredundant cell lines.
        storage_dir (str): A filepath to a temporary storage dir where
            the data is saved. In a cluster environment, this can be
            /scratch.
        chunk_size (int): The number of datapoints to save to each numpy
            file.
        promoter_columns: Either None or a numpy array containing the
            promoter columns that should be used. If None, all columns are
            used.
        enhancer_columns: Either None or a numpy array containing the
            enhancer columns that should be used. If None, all columns are
            used.

    Returns:
        out_pfiles (list): A list of numpy files in storage_dir with promoter
            counts only.
        out_yfiles (list): A list of numpy files in storage dir with yvalues.
        out_xfiles (list): A list of numpy files in storage_dir with both
            promoter and enhancer counts.

    Raises:
        ValueError: A ValueError is raised if one or more expected nonredundant
            cell line ids are not found
    """
    ignore_enhancers = False
    if en_path is None:
        print("No valid enhancer path supplied. Ignoring enhancers. Promoters only "
              "will be used.", flush=True)
        ignore_enhancers = True

    out_xfiles, out_pfiles, out_yfiles = [], [], []
    ndpoints = 0

    print("Now retrieving and sorting files...", flush=True)

    for nonred_id in nonredundant_ids:
        pfile = os.path.join(prom_path, f"{nonred_id}_count_matrix_pro.npy")
        yfile = os.path.join(ypath, f"{nonred_id}.y.npy")
        try:
            promoters = np.load(pfile)
            ydata = np.load(yfile)
            if not ignore_enhancers:
                enfile = os.path.join(en_path, f"{nonred_id}_count_matrix_enh.npy")
                enhancers = np.load(enfile)
        except:
            raise ValueError(f"Could not find numpy files for nonredundant id {nonred_id}")

        promoters = promoters.astype(np.float32)
        ndpoints += promoters.shape[0]

        if promoter_columns is not None:
            promoters = promoters[:,promoter_columns]
 
        if not ignore_enhancers:
            enhancers = enhancers.astype(np.float32)
            if enhancer_columns is not None:
                enhancers = enhancers[:,enhancer_columns]
            merged_data = np.concatenate([promoters, enhancers], axis=1)

        for j in range(0, promoters.shape[0], chunk_size):
            out_prom_file = os.path.join(storage_dir, f"{nonred_id}_{j}_promoters.npy")
            out_yfile = os.path.join(storage_dir, f"{nonred_id}_{j}_y.npy")

            np.save(out_prom_file, promoters[j:j+chunk_size,...])
            np.save(out_yfile, ydata[j:j+chunk_size])

            out_pfiles.append(out_prom_file)
            out_yfiles.append(out_yfile)
            
            if not ignore_enhancers:
                out_xfile = os.path.join(storage_dir, f"{nonred_id}_{j}_merged.npy")
                out_xfiles.append(out_xfile)
                np.save(out_xfile, merged_data[j:j+chunk_size,...])

    print(f"Total datapoints: {ndpoints}", flush=True)
    return out_pfiles, out_xfiles, out_yfiles




def cleanup_storage(xfiles, pfiles, yfiles):
    """Cleans up the temporary files created in the storage dir."""
    for xfile in xfiles:
        os.remove(xfile)
    for yfile in yfiles:
        os.remove(yfile)
    for pfile in pfiles:
        os.remove(pfile)



def get_tt_split(xfiles:list, pfiles:list, yfiles:list, nonred_ids:list,
                 num_train:int = 40, offset:int = 0):
    """Breaks the input file lists up into a single train-test
    split with num_train cell lines for training. This is typically
    40 (to match experiments conducted with alternative models).

    The first num_train files are used as training unless caller
    specifies an offset (e.g. 40); if an offset is specified,
    the first 'offset' files are used for testing, and the next
    'num_train' are used as training instead. This procedure is
    slightly more cumbersome than just running a CV but necessary
    to match experiments conducted with alternative models.
    """
    if len(pfiles) != len(yfiles):
        raise ValueError("The number of promoter and yfiles "
                         "is not the same when generating a tt split.")
    if offset + num_train > len(nonred_ids):
        raise ValueError("The number of training files requested with "
                         "the requested offset exceeds the number of "
                         "files in the input lists.")
    test_ids = nonred_ids[:offset] + nonred_ids[num_train + offset:]
    train_ids = nonred_ids[offset:offset + num_train]

    print(f"The training nonred ids are: {train_ids}", flush=True)
    print(f"The valid nonred ids are: {test_ids}\n", flush=True)

    tt_split = {"train_ids":[], "valid_ids":[]}

    for split_type, split_ids in [("train", train_ids), ("valid", test_ids)]:
        tt_split[f"{split_type}_ids"] = {}
        for split_id in split_ids:
            tt_split[f"{split_type}_ids"][split_id] = {}
            tt_split[f"{split_type}_ids"][split_id]["p"] = [pfile for pfile in pfiles if
                                os.path.basename(pfile).split("_")[0] == split_id]
            tt_split[f"{split_type}_ids"][split_id]["x"] = [xfile for xfile in xfiles if
                                os.path.basename(xfile).split("_")[0] == split_id]
            tt_split[f"{split_type}_ids"][split_id]["y"] = [yfile for yfile in yfiles if
                                os.path.basename(yfile).split("_")[0] == split_id]

    return tt_split
