from collections import defaultdict
from itertools import combinations
from deduplication.LSH import LSH
from utils.utils import split_dict, majority_vote

class LSHForest(LSH):
    """
    Locality Sensitive Hashing Forest using MinHash and Banding techniques.

    The LSHForest class extends the LSH class to support the creation of multiple 
    trees in the forest, allowing for more robust detection of candidate pairs. 
    Each tree corresponds to an independent LSH index, with results combined via majority voting.
    """
    def __init__(self, num_hashes=200, num_bands=10, rows_per_band=4, num_trees=5, k=5):
        """
        Initializes an LSHForest instance.

        Args:
            num_hashes (int): Number of hash functions used in MinHashing.
            num_bands (int): Number of bands for the banding technique.
            rows_per_band (int): Number of rows in each band.
            num_trees (int): Number of LSH trees in the forest.
            k (int): Parameter for the LSH superclass, indicating the number of nearest neighbors to consider.
        """
        super().__init__(num_hashes, num_bands, rows_per_band, k)
        self.num_trees = num_trees
        """num_trees (int): Number of LSH trees in the forest."""
        
    def banding(self, signatures):
        """
        Performs the banding technique across multiple LSH trees and identifies candidate pairs.

        The signatures are split across multiple trees (LSH indices), and candidate pairs 
        are identified within each tree using the banding technique. The candidate pairs 
        from all trees are combined using a majority voting mechanism to form the final set 
        of candidate pairs.

        Args:
            signatures (dict): A dictionary where keys are document IDs and values are MinHash signatures (lists of hash values).

        Returns:
            set: A set of candidate document pairs (as tuples of document IDs) that share similar bands.
        """
        
        # Split the signature dictionary across the specified number of trees.
        signature_lists = split_dict(signatures, self.num_trees)
        candidate_sets = []
        
        # Iterate over each subset of signatures, treating each as a separate LSH tree.
        for sig in signature_lists:
            index = defaultdict(list)
            candidate_pairs = set()
            
            # Process each document's signature within the current tree.
            for doc_id, sig in sig.items():
                for band_idx in range(self.num_bands):
                    start = band_idx * self.rows_per_band
                    band = tuple(sig[start:start + self.rows_per_band])
                    index[(band_idx, band)].append(doc_id)

            # Identify candidate pairs from documents that share the same band entry.
            for doc_ids in index.values():
                if len(doc_ids) > 1:
                    candidate_pairs.update(combinations(doc_ids, 2))
            
            # Add the candidate pairs from the current tree to the list of candidate sets.
            candidate_sets.append(candidate_pairs)
        
        # Use majority voting across all candidate sets from the different trees.
        self.candidate_pairs = majority_vote(candidate_sets)
        
        return self.candidate_pairs