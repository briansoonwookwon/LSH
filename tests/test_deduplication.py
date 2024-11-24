from deduplication.dedup import Baseline
from deduplication.LSH import LSH
from utils.use_cases import collection_deduplication, nearest_neighbor_search
from utils.utils import UnionFind

def test_exact_duplicates():
    documents = [
        "We like McDonalds.",
        "KFC is better.",
        "We like McDonalds."  # Duplicate
    ]

    baseline = Baseline()
    _, duplicates = baseline.detect_duplicates(documents)
    
    assert len(duplicates) == 1  # One duplicate found
    assert duplicates[0] == "We like McDonalds."  # The duplicate should be correctly identified

def test_no_duplicates():
    documents = [
        "We like McDonalds.",
        "KFC is better.",
        "Sonic garlic burger is the best."
    ]

    baseline = Baseline()
    _, duplicates = baseline.detect_duplicates(documents)
    
    assert len(duplicates) == 0  # No duplicates found

def test_length_based_duplicates():
    documents = [
        "We like McDonalds.",
        "KFC is better.",
        "We love McDonalds."  # Duplicate based on length
    ]

    baseline = Baseline()
    _, duplicates = baseline.length_based_baseline(documents)
    
    assert len(duplicates) == 1  # One duplicate found
    assert duplicates[0] in ["We like McDonalds.", "We love McDonalds."]  # The correct duplicate pair should be identified
    

def test_word_count_based_duplicates():
    documents = [
        "we like McDonalds",
        "McDonalds like we",  # Duplicate (word overlap)
        "Sonic garlic burger is the best"
    ]

    baseline = Baseline()
    duplicates = baseline.word_count_baseline(documents, threshold=0.8)
    
    assert len(duplicates) == 1  # One duplicate found
    # assert duplicates[0] == ("We like McDonalds.", "McDonalds like we.")  # The correct duplicate pair should be identified


from deduplication.bloom_filter import BloomFilter

def test_bloom_filter_add_query():
    bf = BloomFilter(n=100, f=0.01)

    bf.add("Test document")

    assert bf.query("Test document") == True

    # Query a different document (should return False)
    assert bf.query("Unknown document") == False

def test_bloom_filter_false_positive_rate():
    bf = BloomFilter(n=100, f=0.01)
    
    # Add 50 unique documents
    for i in range(50):
        bf.add(f"Document {i}")
    
    # Check for false positives (items not added)
    false_positives = 0
    for i in range(50, 100):
        if bf.query(f"Document {i}"):
            false_positives += 1
    
    # Assert that the false positive rate is within expected range
    assert false_positives / 50 <= 0.01

def test_bloom_filter_no_false_negatives():
    bf = BloomFilter(n=100, f=0.01)
    
    # Add a document
    bf.add("Another document")
    
    # Query for the same document
    assert bf.query("Another document") == True

    # Query a different document
    
    assert bf.query("Non-existent document") == False


def test_base_lsh_dedup_bytest():
    docs = {
    1: "the quick brown fox jumps over the lazy dog",
    2: "the quick brown fox jumps over the lazy lazy dog",
    3: "a fast dark brown fox leaps over the lazy hound",
    4: "the lazy dog jumps over the quick brown fox",
    }

    lsh = LSH(num_hashes=100, num_bands=20, rows_per_band=5, k=3)

    signatures = lsh.compute_minhash_signatures(docs)

    lsh.banding(signatures)

    clusters = collection_deduplication(lsh)

    assert len(clusters) == 3

def test_base_lsh_nn():
    docs = {
        1: "the quick brown fox jumps over the lazy dog",
        2: "the quick brown fox jumped over the lazy dog ",
        3: "a fast dark brown fox leaps over the lazy hound",
        4: "the lazy dog jumps over the quick brown fox",
    }
    lsh = LSH(num_hashes=100, num_bands=20, rows_per_band=5, k=3)

    signatures = lsh.compute_minhash_signatures(docs)

    lsh.banding(signatures)

    # clusters = collection_deduplication(lsh)

    # clusters, lsh = collection_deduplication(docs)
    # lsh = LSH(num_hashes=num_hashes, num_bands=num_bands, rows_per_band=rows_per_band)
    query_doc = "the quick brown fox jumps over the lazy"
    candidates = nearest_neighbor_search(query_doc, lsh)

    assert candidates == {1}

def test_union_find():
    uf = UnionFind()

    # Test that a new element is its own parent
    assert uf.find(1) == 1
    assert uf.find(2) == 2

    # Union two elements and test if they have the same root
    uf.union(1, 2)
    assert uf.find(1) == uf.find(2)

    # Test adding a new element and union it with an existing one
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3)

    # Check if a new element is still its own parent when not yet unioned
    assert uf.find(4) == 4

    # Perform another union and check the resulting structure
    uf.union(4, 1)
    assert uf.find(4) == uf.find(1)
    assert uf.find(4) == uf.find(3)
