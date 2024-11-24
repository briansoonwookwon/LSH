import hashlib
from collections import Counter
from collections import defaultdict

class Baseline:
    def __init__(self):
        """
        Initialize the baseline deduplication class.
        """
        self.hash_set = set()
        """
        A set to store unique MD5 hashes of documents for deduplication.
        Used to track unique document hashes and identify duplicates in the dataset.
        """

    def collection_deduplication(self,documents_dict):
        """
        Groups documents into clusters based on identical content.
        
        Args:
            documents_dict: Dictionary where keys are document IDs and values are document strings.
        
        Returns:
            clusters: Dictionary where each key is a cluster ID and the value is a list of document IDs
                    that are considered duplicates or similar.
        """
        # Create a dictionary to store clusters (hash to list of document IDs)
        hash_to_docs = defaultdict(list)
        
        # Group documents by their MD5 hash (you can replace this with LSH or any other similarity function)
        for doc_id, document in documents_dict.items():
            hash_value = hashlib.md5(document.encode()).hexdigest()
            hash_to_docs[hash_value].append(doc_id)
        
        # Convert the hash groupings into cluster IDs
        clusters = {}
        cluster_id = 1  # Simple counter for cluster IDs
        
        for doc_ids in hash_to_docs.values():
            clusters[cluster_id] = doc_ids
            cluster_id += 1
        
        return clusters

    # Exact Duplicates Baseline
    def detect_duplicates(self, documents):
        """
        This function computes MD5 hashes of each document and identifies duplicates
        based on matching hash values.

        Args:
            documents: list of document strings to check for duplicates.

        Returns:
            hash_set: a set containing unique MD5 hashes of the documents.
            duplicates: a list of duplicate documents.
        """
        duplicates = []
        hash_set = set()
        
        for document in documents:
            hash_value = hashlib.md5(document.encode()).hexdigest()
            if hash_value not in hash_set:
                hash_set.add(hash_value)
            else:
                print(f"Duplicate found: {document}")
                duplicates.append(document)

        return hash_set, duplicates

    # Length-Based Baseline
    def length_based_baseline(self, documents):
        """
        This function considers documents as duplicates if they have the same length.
        It compares the lengths of documents and identifies duplicates based on that.

        Args:
            documents: list of document strings to check for duplicates.

        Returns:
            length_map: a dictionary that maps document lengths to documents.
            duplicates: a list of duplicate documents based on length.
        """
        length_map = {}
        duplicates = []

        for document in documents:
            length = len(document)
            if length in length_map:
                print(f"Duplicate found based on length: '{document}' matches '{length_map[length]}'")
                duplicates.append(document)
            else:
                length_map[length] = document

        return length_map, duplicates

    # Token-Based Baseline
    def tokenize(self, document):
        """
        Tokenizes a document by splitting on spaces and returns a Counter of word frequencies.
        
        Args:
            document: A string representing a document.
        
        Returns:
            A Counter object with word frequencies.
        """
        tokens = document.split()
        return Counter(tokens)

    def word_count_baseline(self, documents, threshold=0.8):
        """
        This function uses word count similarity to find duplicates. If two documents have
        a word overlap greater than the threshold, they are considered duplicates.
        
        Args:
            documents: A list of document strings.
            threshold: The fraction of word overlap required to consider two documents as duplicates.
        
        Returns:
            duplicates: A list of tuples where each tuple contains two documents that are considered duplicates.
        """
        duplicates = []

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1_tokens = self.tokenize(documents[i])
                doc2_tokens = self.tokenize(documents[j])

                total_words = sum((doc1_tokens | doc2_tokens).values())
                common_words = sum((doc1_tokens & doc2_tokens).values())

                overlap_ratio = common_words / total_words

                if overlap_ratio >= threshold:
                    print(f"Duplicate found between: '{documents[i]}' and '{documents[j]}' with overlap ratio: {overlap_ratio}")
                    duplicates.append((documents[i], documents[j]))

        return duplicates
