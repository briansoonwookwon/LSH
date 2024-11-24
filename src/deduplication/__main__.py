import argparse
import time
from utils.utils import read_tsv
from deduplication.LSH import LSH
from deduplication.LSHImproved import LSHImproved
from deduplication.LSHForest import LSHForest
from deduplication.dedup import Baseline
from utils.use_cases import collection_deduplication, nearest_neighbor_search
import logging
import os
import psutil
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("deduplication_log.txt"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to make corpus for assignment 2"
    )
    """
    Parses command-line arguments for creating a corpus.

    This script defines various command-line arguments for processing 
    input files, defining parameters, and choosing methods for the 
    assignment's corpus creation.

    Arguments:
        -d, --indir (str): Required. Directory path of the input file.
        -t, --case (str): Required. Type of use case.
        -s, --save (str): Optional. Whether to output results to a text file ('y' or 'n').
        -e, --example (str): Optional. Document to query.
        -n, --numhash (int): Optional. Number of hash functions to use.
        -b, --numband (int): Optional. Number of bands.
        -r, --row (int): Optional. Number of rows per band.
        -k, --shinlen (int): Optional. Length of shingles.
        -c, --treesize (int): Optional. Size of the tree.
        -m, --method (str): Optional. Default is 'LSH'. Specifies the method to use. 
                            Options: 'baseline', 'LSH', 'LSH_mp', 'LSH_forest'.

    Returns:
        Namespace: An object containing the parsed arguments.
    """
    parser.add_argument("-d", "--indir", required=True, help="Directory of file to input")
    parser.add_argument("-t", "--case", required=True, help="Type of Usecase")
    parser.add_argument("-s", "--save", required=False, help="Output to txt y/n")
    parser.add_argument("-e", "--example", required=False, help="Document to Query")
    parser.add_argument("-n", "--numhash", required=False, help="Number of hash functions")
    parser.add_argument("-b", "--numband", required=False, help="Number of bands")
    parser.add_argument("-r", "--row", required=False, help="Rows per band")
    parser.add_argument("-k", "--shinlen", required=False, help="Length of Shingles")
    parser.add_argument("-c", "--treesize", required=False, help="Tree size")
    parser.add_argument("-m", "--method", required=False, default="LSH", choices=['baseline', 'LSH', 'LSH_mp', 'LSH_forest'], help="Method - choose 'basline', 'LSH', 'LSH_mp' or 'LSH_forest'")

    args = parser.parse_args()
    method = args.method

    def log_memory_usage(message="Memory usage"):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"{message}: {mem_info.rss / (1024 * 1024):.2f} MB")
    
    if method == "LSH_forest":
        default_num_hashes = 200
        default_num_bands = 10
        default_rows_per_band = 4
        default_k = 8
        default_num_trees = 5
    else:
        default_num_hashes = 100
        default_num_bands = 20
        default_rows_per_band = 5
        default_k = 8
        default_num_trees = 1

    # Assign either the provided values or the default ones
    num_hashes = int(args.numhash) if args.numhash is not None else default_num_hashes
    num_bands = int(args.numband) if args.numband is not None else default_num_bands
    rows_per_band = int(args.row) if args.row is not None else default_rows_per_band
    k = int(args.shinlen) if args.shinlen is not None else default_k
    num_trees = int(args.treesize) if args.treesize is not None else default_num_trees
    
    # Check whether the initial condition is met
    if method == "LSH":
        if num_hashes != num_bands * rows_per_band:
            logging.error("Hash functions must equal bands * rows_per_band")
            sys.exit(1)

    elif method == "LSH_forest":
        if num_hashes/num_trees != num_bands * rows_per_band:
            logging.error("Invalid tree size")
            sys.exit(1)
        
    def model(docs, num_hashes=num_hashes, num_bands=num_bands, rows_per_band=rows_per_band, k=k, method=method, num_trees = num_trees):
        """Use LSH for collection deduplication."""

        start_time = time.time() 
        if method == "LSH_mp":
            logging.info("Initializing LSH with %d hashes, %d bands, and %d rows per band", num_hashes, num_bands, rows_per_band)
            logging.info("Using LSHImproved with multi-probe lookup.")
            lsh = LSHImproved(num_hashes=num_hashes, num_bands=num_bands, rows_per_band=rows_per_band, k=k)
        
        elif method == "LSH_forest":
            logging.info("Initializing LSH with %d hashes, %d bands, %d rows per band, and %d trees", num_hashes, num_bands, rows_per_band, num_trees)
            logging.info("Using LSH Forest.")
            lsh = LSHForest(num_hashes=num_hashes, num_bands=num_bands, rows_per_band=rows_per_band, k=k, num_trees=num_trees)
        
        else:
            logging.info("Initializing LSH with %d hashes, %d bands, and %d rows per band", num_hashes, num_bands, rows_per_band)
            logging.info("Using basic LSH.")
            lsh = LSH(num_hashes=num_hashes, num_bands=num_bands, rows_per_band=rows_per_band, k=k)
       
        initialization_time = time.time() - start_time  #step 1 
        logging.info("LSH initialization time: %.2f seconds", initialization_time)

        logging.info("Computing MinHash signatures for the documents.")
        
        start_time_minhash = time.time()  # Start timing MinHash signature computation
        signatures = lsh.compute_minhash_signatures(docs)
        end_time_minhash = time.time()  # End timing MinHash signature computation
        logging.info("MinHash signatures computed in %.2f seconds.", end_time_minhash - start_time_minhash)

        logging.info("Applying LSH banding technique.")
        start_time_banding = time.time()  # Start timing LSH banding
        lsh.banding(signatures)
        end_time_banding = time.time()  # End timing LSH banding
        logging.info("LSH banding completed in %.2f seconds.", end_time_banding - start_time_banding)

        return lsh

    logging.info("Reading input file from %s", args.indir)
    start_time_reading = time.time()  # Start timing file reading
    tsv_dict = read_tsv(args.indir)
    end_time_reading = time.time()  # End timing file reading
    logging.info("Input file read in %.2f seconds.", end_time_reading - start_time_reading)

    logging.info("Starting the deduplication process.")
    start_time_total = time.time()  # Start timing the entire deduplication process
    if method == 'baseline':
        lsh = Baseline()
    else:
        lsh = model(tsv_dict)
    
    if (args.case).lower() == 'deduplication':
        start_time_deduplication = time.time()  # Start timing deduplication
        if method =='baseline':
            clusters = lsh.collection_deduplication(tsv_dict)
        else:
            clusters = collection_deduplication(lsh)
        end_time_deduplication = time.time()  # End timing deduplication

        if args.save == 'y':
            name = os.path.splitext(os.path.basename(args.indir))[0]
            method = args.method
            with open(f'./output/{name}-{method}.txt', 'w') as f:
                for cluster_id, doc_ids in clusters.items():
                    # Join the doc_ids with spaces and write to the file
                    doc_ids_str = ' '.join(map(str, doc_ids))
                    f.write(f"{doc_ids_str}\n")

        logging.info("Deduplication process completed in %.2f seconds.", end_time_deduplication - start_time_deduplication)
        logging.info("Input Documents: %d", len(tsv_dict))
        if method =='baseline':
            pass
        else:
            logging.info("Unique Documents: %d", len(lsh.unique_docs))       
        logging.info("Clusters Formed: %d", len(clusters))
    elif (args.case).lower() == 'ann':
        start_time_ann = time.time()  # Start timing nearest neighbor search
        candidates = nearest_neighbor_search(args.example, lsh)
        end_time_ann = time.time()  # End timing nearest neighbor search
        logging.info("Nearest neighbors search completed in %.2f seconds.", end_time_ann - start_time_ann)
        logging.info(f"Nearest neighbors for the query : {candidates}")
    else:
        logging.error("Enter either deduplication or ann for -t")

    end_time_total = time.time()  # End timing the entire process
    logging.info("Total deduplication process took %.2f seconds.", end_time_total - start_time_total)

    log_memory_usage("Final memory usage")


# python -m deduplication -d './data/onek.tsv' -t 'deduplication' -s 'y'
# python -m deduplication -d './data/threehundred.tsv' -t 'deduplication'
# python -m deduplication -d './data/threehundred.tsv' -t 'deduplication' -m 'baseline'
# python -m deduplication -d './data/hundred.tsv' -t 'ann' -e 'this is a blank statement'
# python -m deduplication -d './data/onek.tsv' -t 'deduplication' -m "LSH_forest"
# python -m deduplication -d './data/onek.tsv' -t 'deduplication' -m "LSH_forest" -n 200 -b 10 -r 5 -c 4