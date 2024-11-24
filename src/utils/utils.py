import hashlib
import re
import xxhash
from collections import Counter

# Helper functions
def clean_document(text):
    """Clean and normalize the document by lowercasing and removing special characters.
    
    This function processes the input text by converting it to lowercase and removing any non-alphabetical 
    characters (such as numbers, punctuation, or special symbols), leaving only lowercase alphabetic characters 
    and spaces. It helps normalize the text for further processing like shingling or hashing.
    
    Args:
        text (str): The input document text to be cleaned.
    
    Returns:
        str: A cleaned version of the input text, containing only lowercase letters and spaces.
    
    Example:
        >>> clean_document("Hello, World!123")
        'hello world'
    """
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return text

def shingle(text, k=5):
    """Generate k-shingles (or k-grams) from a given text.
    
    A shingle is a contiguous sequence of k words. This function splits the cleaned text into words and
    generates all possible k-shingles from those words. Shingling helps capture local word context for use 
    in similarity detection and MinHashing.
    
    Args:
        text (str): The input document text that has been pre-processed.
        k (int): The number of words in each shingle. Default is 5.
    
    Returns:
        set: A set of k-shingles, where each shingle is a string of k words joined by spaces.
    
    Example:
        >>> shingle("this is an example of a document", k=3)
        {'this is an', 'is an example', 'an example of', 'example of a', 'of a document'}
    """
    split = text.split()
    return {' '.join(split[i:i+k]) for i in range(len(split) - k + 1)}

def minhash(shingles, num_hashes=100):
    """Generate a MinHash signature for the given set of shingles.
    
    MinHash is used to estimate the Jaccard similarity between two sets efficiently. This function computes 
    a MinHash signature for a given set of shingles by generating `num_hashes` hash functions and returning 
    the minimum hash value for each function.
    
    Args:
        shingles (set): A set of k-shingles (strings) representing the document.
        num_hashes (int): The number of hash functions to use in generating the MinHash signature. Default is 100.
    
    Returns:
        list: A MinHash signature represented as a list of `num_hashes` integers, where each entry is the minimum 
              hash value for that hash function over the shingles.
    
    Example:
        >>> shingles = {"this is an example", "example of a document", "a document example"}
        >>> minhash(shingles, num_hashes=3)
        [210933193411394158751014314048262142248, 7042737232409725043981610333941128952, 113933543564394106622393219348875832082]
    """
    signature = []
    for i in range(num_hashes):
        # hash_vals = [int(hashlib.md5((str(s) + str(i)).encode()).hexdigest(), 16) for s in shingles]
        hash_vals = [int(xxhash.xxh64_intdigest(str(s), i)) for s in shingles]
        signature.append(min(hash_vals))
    return signature

class UnionFind:
    """Union-Find (Disjoint Set) data structure with path compression for efficient merging and finding.
    
    Union-Find (also known as Disjoint Set) is a data structure that efficiently handles merging of sets 
    and finding the root or representative element of a set. It uses path compression to flatten the tree structure, 
    improving the efficiency of future find operations.
    
    Attributes:
        parent (dict): A dictionary mapping each element to its parent element. If an element is its own parent, 
                       it is the root of its set.
    
    Methods:
        find(x): Finds the root of the set containing element `x`, with path compression to improve efficiency.
        union(x, y): Merges the sets containing elements `x` and `y`.
    """
    
    def __init__(self):
        self.parent = {}
        """Initializes an empty Union-Find structure."""

    def find(self, x):
        """Find the root of the set containing element `x` with path compression.
        
        Path compression is used to make future queries faster by setting each element's parent directly to the root.
        If the element `x` is not yet in the Union-Find structure, it is initialized as its own root.
        
        Args:
            x: The element whose set root is being queried.
        
        Returns:
            The root element of the set containing `x`.
        
        Example:
            >>> uf = UnionFind()
            >>> uf.union(1, 2)
            >>> uf.find(1)
            2
        """
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing elements `x` and `y`.
        
        This method merges the sets that contain elements `x` and `y`. If the elements are already in the same set, 
        no change is made. Otherwise, one set's root is updated to point to the other set's root.
        
        Args:
            x: The first element to union.
            y: The second element to union.
        
        Example:
            >>> uf = UnionFind()
            >>> uf.union(1, 2)
            >>> uf.union(2, 3)
            >>> uf.find(3)
            2
        """
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

def read_tsv(tsv):
    """Read a TSV (tab-separated values) file and return a dictionary of its contents.
    
    This function reads a TSV file where each line consists of an index and a text value separated by a tab.
    The index is used as the key, and the text as the value in the returned dictionary.
    
    Args:
        tsv (str): The file path to the TSV file.
    
    Returns:
        dict: A dictionary where keys are the integer indices from the TSV file, and values are the text content.
    
    Example:
        >>> read_tsv('documents.tsv')
        {1: 'This is the first document.', 2: 'Another document with different content.'}
    """
    with open(tsv, 'r', encoding='utf-8') as file:
        tsv_dict = {}
        for line in file:
            if line.strip():  # To skip empty lines
                index, text = line.split('\t', 1)
                tsv_dict[int(index)] = text
    return tsv_dict

def split_dict(input_dict, num_splits):
    """
    Splits a dictionary into a specified number of smaller dictionaries.

    Args:
        input_dict (dict): The input dictionary to be split.
        num_splits (int): The number of splits.

    Returns:
        list: A list of dictionaries with split entries.
    """
    # Calculate the split size for each smaller dictionary
    split_size = len(next(iter(input_dict.values()))) // num_splits
    result = []
    
    for i in range(num_splits):
        # Calculate the start and end indices for the current split
        start = i * split_size
        end = (i + 1) * split_size
        
        # Create a new dictionary for each split
        split_dict = {key: value[start:end] for key, value in input_dict.items()}
        result.append(split_dict)

    return result

def majority_vote(candidate_sets):
    """
    This function performs a majority vote on a list of sets of candidate pairs.

    Args:
        candidate_sets (list of sets): List of sets containing candidate pairs.

    Returns:
        list: The pairs that appear in the majority of sets.
    """
    # Flatten the list of sets into a list of pairs
    all_pairs = [pair for candidate_set in candidate_sets for pair in candidate_set]

    # Count occurrences of each pair
    pair_counts = Counter(all_pairs)

    # Determine the majority threshold (more than half of the sets)
    majority_threshold = len(candidate_sets) / 2

    # Select pairs that meet the majority threshold
    majority_pairs = [pair for pair, count in pair_counts.items() if count > majority_threshold]

    return majority_pairs
