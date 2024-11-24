import hashlib
import re
from collections import defaultdict
from itertools import combinations
from utils.utils import clean_document, shingle, minhash
import numpy as np
from joblib import Parallel, delayed

class LSHImproved:
    """
    Multi-Probe Locality Sensitive Hashing with MinHash and Banding.

    LSH Improved uses a multi-probe approach to traditional locality-sensitive
    hashing to reduce the number of required hash tables, ultimately saving space 
    and improving efficiency. This class supports various banding methods for probing
    nearby bands, allowing customizable deduplication and similarity search.
    """
    def __init__(self, num_hashes=100, num_bands=20, rows_per_band=5, k=5, num_probes = 4, banding_method='nearby_banding'):
        """
        Initializes the LSHImproved instance with the specified number of hash functions, bands, and shingle size, number of probes, and banding strategy.

        Raises 
            AssertionError: If the number of hash functions does not equal `num_bands * rows_per_band`.
        """

        self.num_hashes = num_hashes
        """num_hashes (int): Total number of hash functions used for MinHash signature computation."""
        self.num_bands = num_bands
        """num_bands (int): Number of bands used for LSH banding."""
        self.rows_per_band = rows_per_band
        """rows_per_band (int): Number of rows in each band (bands * rows_per_band must equal num_hashes)."""
        self.index = defaultdict(list)
        """index (defaultdict): A dictionary used to store bands and document IDs for candidate identification."""
        self.unique_docs = {}
        """unique_docs (dict): A dictionary of unique documents (after removing exact duplicates)."""
        self.cleaned_docs = {}
        """cleaned_docs (dict): A dictionary of cleaned documents after pre-processing."""
        self.candidate_pairs = set()
        """candidate_pairs (set): A set of candidate document pairs found using LSH."""
        self.exact_duplicates = {}
        """exact_duplicates (dict): A dictionary of exact duplicates, mapping original doc IDs to duplicate doc IDs."""
        self.banding_method = banding_method
        """banding_method: The banding method to use for hash perturbation.
            Options are 'nearby_banding', 'bit_flip', and 'gaussian'."""

        self.k = k
        """k (int): The shingle size (number of words or characters in each shingle)."""
        assert self.num_hashes == self.num_bands * self.rows_per_band, "Hash functions must equal bands * rows_per_band"
    
    # Keeping remove duplicates the same
    def remove_duplicates(self, docs):
        """Removes exact duplicates from the provided documents and stores the unique ones.
        
        This method scans through the provided documents, identifying and removing exact duplicates.
        It stores the first occurrence of each document in the `unique_docs` attribute and tracks duplicates
        in the `exact_duplicates` attribute.
        
        Args:
            docs (dict): A dictionary where keys are document IDs and values are document contents.
        
        Side Effects:
            - Populates the `unique_docs` dictionary with unique documents.
            - Populates the `exact_duplicates` dictionary with mappings from original document IDs to their duplicates.
        """
        seen_docs = {}
        for doc_id, doc in docs.items():
            if doc not in seen_docs:
                self.unique_docs[doc_id] = doc
                seen_docs[doc] = doc_id  # Track first occurrence of the document
            else:
                # Track exact duplicates
                original_id = seen_docs[doc]
                if original_id not in self.exact_duplicates:
                    self.exact_duplicates[original_id] = []
                self.exact_duplicates[original_id].append(doc_id)

    def compute_minhash_signatures(self, docs):
        """Computes MinHash signatures for each unique document using parallel processing.
        
        This method performs the following steps:
        1. Removes exact duplicates from the documents using `remove_duplicates`.
        2. Cleans each document (e.g., tokenization, removing punctuation).
        3. Generates shingles for each document based on the specified shingle size (`k`).
        4. Computes MinHash signatures for each document using the `minhash` function.
        
        Args:
            docs (dict): A dictionary where keys are document IDs and values are document contents.
        
        Returns:
            dict: A dictionary where keys are document IDs and values are the MinHash signatures (lists of hash values).
        
        Side Effects:
            - Populates the `cleaned_docs` dictionary with cleaned versions of the documents.
            - Populates the `shingle_sets` dictionary with shingles for each document.
            - Populates the `signatures` dictionary with MinHash signatures for each document.
        """
        self.remove_duplicates(docs)
        self.cleaned_docs = {doc_id: clean_document(doc) for doc_id, doc in self.unique_docs.items()}
        self.shingle_sets = {doc_id: shingle(doc, self.k) for doc_id, doc in self.cleaned_docs.items()}
        
        # Parallel computation of MinHash signatures
        signatures = Parallel(n_jobs=-1)( delayed(minhash)(shingles, self.num_hashes) for doc_id, shingles in self.shingle_sets.items())
        self.signatures = dict(zip(self.shingle_sets.keys(), signatures))

        return self.signatures
    
    
    # Creating nearby_banding
    def nearby_banding(self, band, num_probes):
        """
        This method creates multiple variations of the input band by applying 
        perturbations to its elements via modulus calculations. 

        Args:
            band (tuple): A tuple representing the original hash band, where each 
                        element corresponds to a hash value.
            num_probes (int): The number of perturbations (nearby bands) to generate. 

        Returns:
            list: A list of tuples, each representing a perturbed version of the 
                original band. The number of tuples returned is equal to 
                `num_probes`.
        """
    
        # We identify "nearby" neighbors by making perturbations in the hash key
        perturbed_bands = []

        for _ in range(num_probes):
            changed_band = list(band)

            #picking random band to change
            i = np.random.randint(0, len(changed_band))
            changed_band[i] = (changed_band[i] + np.random.randint(1, 10)) % 100 
            # ^^ used modules perturbation in this case, but there other options we can change

            #appending to list 
            perturbed_bands.append(tuple(changed_band))
        
        return perturbed_bands

    def bit_flip(self, band, num_probes):
        """
        This method creates multiple variations of the input band by applying 
        perturbations to its elements via randomized bit flips within the hash token. 

        Args:
            band (tuple): A tuple representing the original hash band, where each 
                        element corresponds to a hash value.
            num_probes (int): The number of perturbations (nearby bands) to generate. 

        Returns:
            list: A list of tuples, each representing a perturbed version of the 
                original band. The number of tuples returned is equal to 
                `num_probes`.
        """
        perturbed_bands = []

        #flipping the bits in the hash 
        for _ in range(num_probes):
            p_band = list(band)
            i = np.random.randint(0, len(p_band)) 
            p_band[i] ^= 1  # Flip a bit at the random index
            perturbed_bands.append(tuple(p_band))

        return perturbed_bands
    
    def gaussian(self, band, num_probes, std_dev=1.0):
        """
        This method creates multiple variations of the input band by applying 
        perturbations to its elements via gaussian noise. 

        Args:
            band (tuple): A tuple representing the original hash band, where each 
                        element corresponds to a hash value.
            num_probes (int): The number of perturbations (nearby bands) to generate. 

        Returns:
            list: A list of tuples, each representing a perturbed version of the 
                original band. The number of tuples returned is equal to 
                `num_probes`.
        """
        perturbed_bands = []

        #adding gaussian "noise" so that hash is slightly altered
        for _ in range(num_probes):
            p_band = list(band)
            i = np.random.randint(0, len(p_band))
            noise = np.random.normal(0, std_dev)  # adding Gaussian noise
            p_band[i] += noise
            perturbed_bands.append(tuple(p_band))

        return perturbed_bands

    #Changing the banding method so it checks neighboring bands
    def banding(self, signatures):
        """
        This method organizes the MinHash signatures into bands and applies a perturbation 
        strategy to generate nearby bands for each document. It populates an index of candidate 
        pairs of documents that are likely to be similar based on their perturbed bands.

        The banding method used is determined by the `self.banding_method` attribute, which 
        can be one of the following: 'nearby_banding', 'bit_flip', or 'gaussian'.

        Args:
            signatures (dict): A dictionary where keys are document IDs and values are MinHash signatures (lists or tuples of hash values) for each document.

        Returns:
            set: A set of tuples, where each tuple contains two document IDs that are identified 
            as candidate pairs based on their bands and perturbations.
        """
        if self.banding_method == 'nearby_banding':
            perturb_fn = self.nearby_banding
        elif self.banding_method == 'bit_flip':
            perturb_fn = self.bit_flip
        elif self.banding_method == 'gaussian':
            perturb_fn = lambda band, num_probes: self.gaussian(band, num_probes)
        else:
            raise ValueError(f"Unknown banding method: {self.banding_method}")

        # Process signatures and apply perturbations
        for doc_id, sig in signatures.items():
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                band = tuple(sig[start:start + self.rows_per_band])
                self.index[(band_idx, band)].append(doc_id)

                neighboring_bands = perturb_fn(band, self.num_probes)
                for neighbor in neighboring_bands:
                    self.index[(band_idx, neighbor)].append(doc_id)

        for doc_ids in self.index.values():
            if len(doc_ids) > 1:
                self.candidate_pairs.update(combinations(doc_ids, 2))

        return self.candidate_pairs