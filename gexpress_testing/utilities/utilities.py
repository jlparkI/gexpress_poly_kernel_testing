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
                storage_dir, add_interactions = False, key_motifs_only = False):
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
        add_interactions (bool): If True, add a list of interactions.
        key_motifs_only (bool): If True, use only key motifs from a prior
            L1 regularization fit.

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


    interaction_list = ['4_96', '4_203', '5_135', '9_446', '12_277', '12_551', '13_50', '13_130', '13_135', '13_286', '13_294', '13_307', '13_312', '13_501', '13_572', '16_96', '19_163', '19_203', '19_307', '22_31', '22_203', '22_316', '22_425', '23_72', '23_158', '23_316', '23_395', '23_423', '23_437', '23_446', '23_591', '24_44', '27_96', '27_301', '27_446', '30_116', '30_117', '30_163', '30_164', '30_364', '30_407', '30_453', '31_38', '31_44', '31_63', '31_65', '31_123', '31_145', '31_146', '31_153', '31_156', '31_200', '31_213', '31_231', '31_259', '31_273', '31_290', '31_309', '31_312', '31_315', '31_320', '31_326', '31_333', '31_349', '31_356', '31_358', '31_376', '31_382', '31_425', '31_429', '31_437', '31_450', '31_478', '31_486', '31_491', '31_500', '31_505', '31_534', '31_545', '31_548', '31_554', '31_572', '31_573', '31_610', '32_203', '33_96', '33_163', '33_410', '33_500', '34_45', '35_50', '35_106', '35_212', '35_292', '35_307', '35_372', '35_391', '35_421', '35_530', '35_545', '35_554', '38_44', '44_49', '44_65', '44_69', '44_106', '44_119', '44_156', '44_166', '44_168', '44_174', '44_183', '44_200', '44_210', '44_211', '44_223', '44_231', '44_241', '44_244', '44_259', '44_299', '44_318', '44_321', '44_344', '44_354', '44_357', '44_359', '44_376', '44_379', '44_382', '44_397', '44_407', '44_437', '44_447', '44_453', '44_478', '44_484', '44_510', '44_525', '44_533', '44_564', '44_589', '44_614', '45_66', '45_96', '45_116', '45_147', '45_158', '45_231', '45_243', '45_244', '45_256', '45_329', '45_334', '45_338', '45_359', '45_364', '45_423', '45_453', '45_488', '45_500', '45_516', '46_146', '46_163', '46_299', '46_337', '46_344', '46_395', '46_446', '46_453', '49_407', '50_90', '50_200', '50_290', '50_338', '50_344', '50_407', '50_425', '50_449', '50_499', '50_525', '50_540', '50_583', '52_95', '62_415', '62_446', '64_203', '66_96', '66_344', '66_423', '72_168', '72_231', '72_242', '72_244', '72_250', '72_258', '72_294', '72_407', '72_415', '72_530', '72_554', '72_558', '72_583', '72_618', '87_540', '93_530', '94_203', '94_230', '94_258', '94_508', '95_130', '95_142', '95_200', '95_230', '95_250', '95_256', '95_382', '95_410', '95_417', '95_534', '95_540', '96_99', '96_117', '96_139', '96_146', '96_153', '96_156', '96_164', '96_297', '96_354', '96_388', '96_437', '96_442', '96_545', '96_548', '96_551', '96_554', '96_571', '96_572', '96_614', '106_258', '106_326', '106_344', '106_383', '106_453', '116_173', '116_286', '116_319', '116_453', '116_557', '117_139', '117_359', '117_425', '119_500', '121_135', '121_213', '121_344', '124_437', '130_231', '130_540', '134_258', '135_407', '135_425', '135_430', '135_591', '136_203', '136_446', '138_203', '139_203', '139_530', '140_294', '140_334', '140_456', '140_571', '141_212', '141_215', '141_258', '141_316', '141_319', '141_530', '142_382', '142_386', '143_316', '145_203', '145_231', '145_407', '145_456', '146_286', '146_326', '146_344', '146_417', '146_441', '147_203', '147_395', '153_230', '153_243', '153_316', '156_174', '156_396', '156_407', '156_437', '156_545', '156_557', '156_610', '158_193', '158_203', '158_244', '158_256', '158_268', '158_334', '158_337', '158_444', '158_488', '158_546', '158_568', '160_316', '163_213', '163_256', '163_286', '163_294', '163_307', '163_338', '163_383', '163_395', '163_437', '163_484', '163_501', '163_525', '163_530', '163_540', '163_568', '163_583', '163_589', '163_610', '168_290', '173_318', '173_501', '173_551', '174_203', '174_500', '175_540', '182_203', '185_277', '185_294', '188_500', '188_551', '193_230', '193_245', '193_316', '193_382', '193_423', '193_488', '199_530', '200_478', '200_633', '203_231', '203_262', '203_290', '203_294', '203_307', '203_324', '203_326', '203_333', '203_382', '203_391', '203_406', '203_421', '203_423', '203_449', '203_461', '203_500', '203_512', '203_533', '203_545', '203_556', '203_557', '203_583', '203_591', '203_628', '204_437', '204_464', '204_525', '204_530', '210_319', '210_437', '212_213', '212_437', '213_219', '213_231', '213_256', '213_258', '213_269', '213_359', '213_410', '213_453', '213_478', '213_623', '215_318', '215_450', '219_307', '219_334', '219_359', '219_407', '219_447', '220_421', '222_410', '230_231', '230_258', '230_262', '230_299', '230_359', '230_364', '230_377', '230_407', '230_481', '230_488', '230_501', '230_551', '230_583', '230_591', '230_628', '231_337', '231_354', '231_355', '231_430', '231_439', '231_450', '231_453', '231_506', '231_545', '231_546', '231_558', '231_591', '231_630', '232_388', '241_318', '241_425', '242_407', '242_437', '242_478', '243_256', '243_258', '243_286', '243_318', '243_329', '243_395', '243_453', '243_551', '243_558', '244_354', '244_382', '244_383', '244_446', '244_450', '244_453', '244_584', '245_383', '245_453', '256_337', '256_425', '256_446', '256_450', '258_290', '258_320', '258_329', '258_391', '258_410', '258_453', '258_455', '258_486', '258_488', '258_501', '258_557', '258_579', '259_501', '262_425', '262_501', '264_316', '267_268', '268_501', '269_437', '273_407', '273_540', '277_316', '277_318', '278_307', '286_292', '286_338', '286_382', '286_407', '286_425', '286_444', '286_453', '286_478', '286_508', '286_584', '290_437', '290_453', '290_478', '290_591', '292_500', '292_501', '293_316', '293_446', '294_372', '294_633', '301_501', '305_396', '307_339', '307_382', '307_383', '307_545', '307_572', '310_382', '315_396', '316_356', '316_382', '316_385', '316_396', '316_415', '316_459', '316_500', '316_501', '316_524', '316_557', '316_626', '318_364', '318_382', '318_388', '318_391', '318_407', '318_441', '318_453', '319_373', '319_396', '319_499', '319_525', '320_437', '326_437', '326_452', '329_446', '329_450', '329_501', '337_359', '337_446', '337_557', '338_407', '338_530', '339_558', '344_425', '344_500', '353_382', '353_437', '355_446', '355_530', '355_589', '356_446', '357_551', '359_437', '359_450', '359_501', '359_583', '364_551', '372_382', '372_425', '372_540', '373_382', '377_453', '382_391', '382_488', '382_628', '383_410', '383_430', '383_591', '386_551', '388_395', '388_540', '391_453', '391_501', '391_551', '395_446', '395_607', '396_450', '396_453', '396_494', '396_551', '396_557', '396_572', '402_446', '407_453', '407_501', '407_530', '407_540', '407_545', '407_625', '415_446', '421_446', '423_540', '423_611', '425_453', '425_464', '425_478', '425_530', '425_540', '425_613', '425_618', '427_446', '434_437', '437_444', '437_447', '437_453', '437_500', '437_501', '437_516', '437_524', '437_527', '437_530', '437_534', '437_557', '437_610', '441_501', '446_494', '446_501', '446_503', '446_512', '446_514', '446_524', '446_540', '446_557', '446_558', '446_568', '446_607', '446_611', '446_628', '449_530', '450_453', '450_456', '450_614', '453_506', '453_508', '453_520', '453_545', '453_571', '453_591', '453_614', '453_633', '464_558', '466_501', '466_540', '478_614', '481_540', '488_530', '495_551', '499_554', '500_530', '500_540', '500_584', '500_610', '500_623', '501_530', '501_558', '501_583', '501_584', '501_591', '520_530', '524_540', '525_558', '530_534', '530_555', '530_572', '530_579', '530_610', '530_614', '533_540', '540_551', '558_613']



    key_motif_list = [1, 2, 4, 7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 33, 36, 38, 39, 42, 45, 46, 48, 50, 51, 52, 53, 54, 57, 60, 61, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 75, 81, 82, 83, 86, 87, 90, 93, 94, 95, 96, 99, 102, 106, 108, 109, 114, 116, 117, 118, 119, 123, 124, 125, 127, 128, 129, 130, 133, 134, 135, 136, 139, 140, 141, 142, 143, 145, 146, 147, 149, 150, 151, 153, 155, 156, 157, 159, 160, 162, 163, 164, 166, 168, 169, 171, 172, 173, 174, 175, 178, 179, 182, 183, 184, 185, 188, 192, 193, 196, 197, 198, 199, 200, 202, 203, 204, 206, 207, 209, 211, 212, 214, 215, 216, 219, 220, 225, 226, 229, 230, 231, 232, 234, 235, 236, 238, 241, 242, 243, 244, 249, 250, 251, 256, 258, 259, 260, 262, 264, 267, 269, 271, 273, 274, 276, 277, 279, 280, 281, 286, 290, 292, 293, 294, 295, 297, 299, 303, 304, 306, 307, 309, 310, 311, 314, 315, 316, 317, 318, 320, 321, 323, 324, 325, 328, 329, 331, 332, 333, 334, 335, 337, 338, 339, 340, 342, 343, 344, 345, 349, 350, 353, 355, 356, 357, 359, 363, 364, 367, 369, 372, 373, 374, 377, 378, 379, 381, 382, 383, 385, 386, 387, 388, 389, 392, 395, 397, 398, 399, 402, 403, 405, 407, 408, 412, 413, 415, 416, 421, 423, 424, 425, 427, 429, 431, 432, 433, 434, 435, 437, 438, 439, 441, 442, 444, 446, 447, 449, 450, 452, 455, 456, 458, 459, 460, 461, 463, 464, 465, 466, 472, 477, 478, 480, 485, 488, 490, 491, 492, 493, 494, 495, 498, 499, 500, 501, 505, 506, 507, 508, 509, 510, 511, 512, 517, 520, 521, 522, 524, 525, 526, 528, 530, 533, 534, 535, 536, 538, 540, 542, 545, 547, 548, 549, 550, 551, 552, 553, 554, 556, 558, 559, 561, 564, 565, 566, 567, 568, 570, 571, 576, 577, 578, 579, 580, 581, 583, 584, 587, 589, 591, 592, 594, 598, 604, 605, 608, 610, 611, 613, 614, 617, 618, 619, 620, 622, 623, 625, 626, 627, 628, 629, 630, 633]
    key_motif_list = np.array(key_motif_list)


    for nonred_id in nonredundant_ids:
        pfile = os.path.join(prom_path, f"{nonred_id}_count_matrix_pro.npy")
        yfile = os.path.join(ypath, f"{nonred_id}.y.npy")
        try:
            promoters = np.load(pfile)
            ydata = np.load(yfile)
        except:
            raise ValueError(f"Could not find numpy files for nonredundant id {nonred_id}")

        promoters = promoters.astype(np.float32)
        if add_interactions:
            ex_promoters = np.zeros((promoters.shape[0],
                    promoters.shape[1] + len(interaction_list)))
            ex_promoters[:,:promoters.shape[1]] = promoters
            for i, term in enumerate(interaction_list):
                t1, t2 = term.split("_")
                t1, t2 = int(t1), int(t2)
                ex_promoters[:,i+promoters.shape[1]] = \
                        promoters[:,t1] * promoters[:,t2]
            promoters = ex_promoters
        elif key_motifs_only:
            promoters = promoters[:,key_motif_list]

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



def get_tt_split(xfiles, pfiles, yfiles, nonred_ids, num_train=40):
    """Breaks the input file lists up into a single train-test
    split with 40 cell lines for training, using promoter data
    only."""
    train_ids, test_ids = nonred_ids[:num_train], nonred_ids[num_train:]

    print(f"The training nonred ids are: {train_ids}", flush=True)

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
