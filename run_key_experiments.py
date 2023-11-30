"""Provides a CLI interface to the source code for filtering
input data and running CVs with linear / poly kernel models."""
import os
import sys
import argparse

from gexpress_testing.pkernel_experiments.linear_poly_experiments import run_traintest_split, run_traintest_exact_quad
from gexpress_testing.utilities.utilities import filter_data, get_tt_split, cleanup_storage


def gen_arg_parser():
    """Allows for command line arguments to this file."""
    arg_parser = argparse.ArgumentParser(description="Use this command line app "
            "to run key experiments: ")
    arg_parser.add_argument("--prom_path", type=str, help="A filepath to the "
            "directory containing the promoter motif counts.")
    arg_parser.add_argument("--ypath", type=str, help="A filepath to the "
            "directory containing the gene expression levels.")
    arg_parser.add_argument("--en_path", type=str, help="A filepath to the "
            "directory containing the enhancer motif counts.")
    arg_parser.add_argument("--nonred_fpath", type=str, help="A filepath to the "
            "text file indicating which of the data files are nonredundant "
            "cell lines.")
    arg_parser.add_argument("--storage", type=str, help="A filepath to a "
            "directory where temporary files (the merger of the enhancer "
            "and promoter counts) created while the algorithm is running "
            "can be stored.")
    arg_parser.add_argument("--exp_type", type=str,
                            help="Argument should be one of 40vsrest_split, 40vsrest_exact . "
                            "If 40vsrest_split, it is fitted to 40 cell lines "
                            "then tested on the remainder up to 5x. If 40vsrest_exact, "
                            "the same experiment is performed but using the exact quadratic, "
                            "and weights are saved to disk.")
    return arg_parser


if __name__ == "__main__":
    home_dir = os.path.dirname(os.path.abspath(__file__))
    parser = gen_arg_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        with open(args.nonred_fpath, "r",
                  encoding='utf-8') as fhandle:
            nonredundant_ids = [l.split()[0] for l in fhandle]
    except:
        raise ValueError("You supplied a nonredundant file that does "
                         "not exist.")


    pfiles, xfiles, yfiles = filter_data(args.prom_path, args.ypath,
                args.en_path, nonredundant_ids, args.storage,
                chunk_size = 250)

    # There are 116 cell lines so we set up three splits with different
    # offsets.
    tt_splits = [get_tt_split(xfiles, pfiles, yfiles, nonredundant_ids,
                offset = offset) for offset in [0,20,40,60,76]]
    tset_descriptions = [f"{i}_to_{i+40}_fpath_{args.prom_path}"
                         for i in [0,20,40,60,76]]

    if args.exp_type == "40vsrest":
        output_fpath = os.path.join(home_dir, "results",
                                    "approx_splits_test_results.csv")
        for tt_split, split_description in zip(tt_splits,
                                               tset_descriptions):
            run_traintest_split(tt_split, split_description,
                            output_fpath, "promoters")

    if args.exp_type == "40vsrest_exact":
        output_fpath = os.path.join(home_dir, "results",
                                    "exact_splits_test_results.csv")
        weight_save_dir = os.path.join(home_dir, "results")
        for tt_split, split_description in zip(tt_splits,
                                               tset_descriptions):
            run_traintest_exact_quad(tt_split, split_description,
                        output_fpath, "promoters", weight_save_dir)


    cleanup_storage(xfiles, pfiles, yfiles)
