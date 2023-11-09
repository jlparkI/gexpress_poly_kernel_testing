"""Provides a CLI interface to the source code for filtering
input data and running CVs with linear / poly kernel models."""
import os
import sys
import argparse

#from gexpress_testing.pkernel_experiments.linear_poly_experiments import fit_final_exact_quad
#from gexpress_testing.pkernel_experiments.linear_poly_experiments import run_traintest_split
from gexpress_testing.utilities.utilities import filter_data, cleanup_storage, get_final_data_list
from gexpress_testing.pkernel_experiments.l1_regularization_experiments import hyperparameter_tuning



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
                            help="Argument should be one of 40vsrest, "
                            "final, cv, bic. If 40vsrest, the model is fitted "
                            "to 40 randomly selected cell lines and tested "
                            "on the remainder. If final, the final exact "
                            "quadratic model is trained. If cv, a 5x cv is "
                            "performed on the full dataset.")
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

    if args.exp_type == "bic":
        #For calculating BIC, we fit an exact quadratic using
        #L1 regularization and calculate BIC / AIC using a simple
        #gridsearch across the available hyperparameter.
        xfiles = []
        pfiles, yfiles = get_final_data_list(args.prom_path, args.ypath,
                    nonredundant_ids, args.storage)
        output_fpath = os.path.join(home_dir, "results", "bic_aic_l1_results.csv")
        hyperparameter_tuning(pfiles, yfiles, nonredundant_ids, output_fpath)
        fit_final_exact_quad(pfiles, yfiles, output_fpath)


    elif args.exp_type == "final":
        # For fitting the final model, we create one set of files in
        # the storage folder -- the promoter counts saved as 32 bit
        # floats, divided into small chunks for ease of use.
        print("FINAL FIT", flush=True)
        xfiles = []
        pfiles, yfiles = get_final_data_list(args.prom_path, args.ypath,
                    nonredundant_ids, args.storage)
        output_fpath = os.path.join(home_dir, "results")
        fit_final_exact_quad(pfiles, yfiles, output_fpath)


    elif args.exp_type == "cv":
        # We create two sets of files in the storage folder -- one that uses
        # merged enhancer / promoter counts, and one that uses promoters
        # only -- using the nonredundant cell lines only. This way we
        # can see how model performance changes if providing enhancers
        # in addition to promoters. Note that to save space in
        # the storage folder and speed up fitting, the motif counts are
        # saved as 32-bit floats
        xfiles, pfiles, yfiles = filter_data(args.prom_path, args.ypath,
                    args.en_path, nonredundant_ids, args.storage)
        output_fpath = os.path.join(home_dir, "results", "polyk_tests.csv")
        #run_all_cvs(xfiles, pfiles, yfiles, output_fpath)

    elif args.exp_type == "40vsrest":
        xfiles = []
        pfiles, yfiles = get_final_data_list(args.prom_path, args.ypath,
                    nonredundant_ids, args.storage)
        output_fpath = os.path.join(home_dir, "results",
                                    "40vs_rest_cell_line_results.csv")
        run_traintest_split(pfiles, yfiles, nonredundant_ids, args.prom_path,
                            output_fpath)

    cleanup_storage(xfiles, pfiles, yfiles)
