# from collections import defaultdict
# from utils.utils import UnionFind, clean_document, shingle, minhash

# #Use Case 1
# def collection_deduplication(lsh):
#     # Step 5: Use Union-Find to cluster documents
#     uf = UnionFind()
#     for doc1, doc2 in lsh.candidate_pairs:
#         uf.union(doc1, doc2)
    
#     # Group documents by their root in Union-Find
#     clusters = defaultdict(list)
#     for doc_id in lsh.unique_docs:
#         root = uf.find(doc_id)
#         clusters[root].append(doc_id)
    
#     # Now include the exact duplicates
#     for original_id, duplicate_ids in lsh.exact_duplicates.items():
#         root = uf.find(original_id)
#         clusters[root].extend(duplicate_ids)  # Add the duplicates to the cluster of their original doc

#     return clusters

# # Use Case 2
# def nearest_neighbor_search(query_doc, lsh):
#     """Find approximate nearest neighbors for a given query document."""
#     query_doc_cleaned = clean_document(query_doc)
#     query_shingles = shingle(query_doc_cleaned, lsh.k)
#     query_signature = minhash(query_shingles, lsh.num_hashes)
    
#     candidate_pairs = set()
#     # Find candidate pairs from the index
#     for band_idx in range(lsh.num_bands):
#         start = band_idx * lsh.rows_per_band
#         band = tuple(query_signature[start:start + lsh.rows_per_band])
#         if (band_idx, band) in lsh.index:
#             candidate_pairs.update(lsh.index[(band_idx, band)])
    
#     return candidate_pairs

from collections import defaultdict
from utils.utils import UnionFind, clean_document, shingle, minhash

# Use Case 1
def collection_deduplication(lsh):
    """Clusters documents using LSH results and Union-Find, removing duplicates.
    
    This function clusters documents into groups of near-duplicates based on the candidate pairs 
    found using Locality Sensitive Hashing (LSH) and the Union-Find data structure. It further
    incorporates exact duplicates into the clusters.
    
    The process works as follows:
    1. It initializes a Union-Find structure to manage clustering.
    2. For each pair of candidate documents (found via LSH banding), it unites them in the Union-Find.
    3. The documents are grouped by their root in Union-Find to form clusters of similar documents.
    4. Exact duplicates (if any) are added to the clusters containing their original document.

    Args:
        lsh (LSH): An instance of the LSH class containing precomputed MinHash signatures, 
                   candidate pairs, and exact duplicate information.

    Returns:
        dict: A dictionary where the keys are root document IDs and values are lists of document IDs 
              representing clusters of similar documents (including exact duplicates).

    Example:
        >>> clusters = collection_deduplication(lsh)
        >>> for root, docs in clusters.items():
            >>> print(f"Cluster with root document {root}: {docs}")
    """
    # Step 5: Use Union-Find to cluster documents
    uf = UnionFind()
    for doc1, doc2 in lsh.candidate_pairs:
        uf.union(doc1, doc2)
    
    # Group documents by their root in Union-Find
    clusters = defaultdict(list)
    for doc_id in lsh.unique_docs:
        root = uf.find(doc_id)
        clusters[root].append(doc_id)
    
    # Now include the exact duplicates
    for original_id, duplicate_ids in lsh.exact_duplicates.items():
        root = uf.find(original_id)
        clusters[root].extend(duplicate_ids)  # Add the duplicates to the cluster of their original doc

    return clusters


# Use Case 2
def nearest_neighbor_search(query_doc, lsh):
    """Finds approximate nearest neighbors for a given query document using LSH.
    
    This function performs an approximate nearest neighbor search for the input query document.
    It uses the following steps:
    
    1. Cleans the query document by removing stop words, punctuation, and normalizing the text.
    2. Computes shingles from the cleaned document using the shingle size `lsh.k`.
    3. Calculates the MinHash signature of the query document based on the specified number of hash functions.
    4. Finds candidate nearest neighbors by comparing the query document's MinHash signature against 
       the existing signatures in the LSH index. The candidate pairs are documents that share at least 
       one band with the query document.

    Args:
        query_doc (str): The text of the query document for which to find approximate nearest neighbors.
        lsh (LSH): An instance of the LSH class containing the precomputed MinHash signatures 
                   and the banding index for efficient nearest neighbor search.

    Returns:
        set: A set of document IDs representing the candidate nearest neighbors for the query document.
    
    Example:
        >>> query_document = "This is a sample document text for searching."
        >>> candidate_neighbors = nearest_neighbor_search(query_document, lsh)
        >>> print("Candidate Nearest Neighbors:", candidate_neighbors)
    """
    query_doc_cleaned = clean_document(query_doc)
    query_shingles = shingle(query_doc_cleaned, lsh.k)
    query_signature = minhash(query_shingles, lsh.num_hashes)
    
    candidate_pairs = set()
    # Find candidate pairs from the index
    for band_idx in range(lsh.num_bands):
        start = band_idx * lsh.rows_per_band
        band = tuple(query_signature[start:start + lsh.rows_per_band])
        if (band_idx, band) in lsh.index:
            candidate_pairs.update(lsh.index[(band_idx, band)])
    
    return candidate_pairs
