import mmh3
import math
import bitarray
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

class BloomFilter:
    def __init__(self, n: int, f: float, k: int = None):
        """
        Creates a Bloom Filter object that takes k.
        """
        self.n = n
        """int: Maximum number of elements the filter is designed to hold."""
        self.f = f
        """float: Desired false positive rate for the filter."""
        # Size of bit array
        self.m = int(-math.log(self.f) * self.n / (math.log(2)**2))
        """int: Size of the bit array, calculated based on `f` and `n` to meet the desired false positive rate."""
        # Number of hash functions
        self.k = k
        """int: Number of hash functions used in the filter. If not provided, it is calculated."""
        # Create the bit array
        self.bit_array = bitarray.bitarray(self.m)
        """bitarray: Bit array used to store hashed indices, with a size of `m`."""
        self.bit_array.setall(0)  # Set all bits to 0

    def add(self, item: str):
        """
        Adds an item to the bloom filter.
        
        Args:
            item: the string to add
        """
        for i in range(self.k):
            index = mmh3.hash(item, i) % self.m
            self.bit_array[index] = 1

    def query(self, item: str) -> bool:
        """
        Checks if an item might be in the bloom filter (may have false positives).
        
        Args:
            item: the string to query
            
        Returns:
            bool: True if item might be in the bloom filter, False if it definitely isn't
        """
        for i in range(self.k):
            index = mmh3.hash(item, i) % self.m
            if not self.bit_array[index]:
                return False
        return True

def calculate_false_positive_rate(n, f, k):
    """
    Calculates the false positive rate for a Bloom Filter given n, f, and k.
    
    Args:
        n: number of elements
        f: desired false positive rate
        k: number of hash functions
    
    Returns:
        The observed false positive rate.
    """
    bf = BloomFilter(n, f, k)
    
    # Add n random elements to the Bloom filter
    inserted_elements = [str(random.randint(0, 10**12)) for _ in range(n)]
    for elem in inserted_elements:
        bf.add(elem)

    # Perform random lookups to calculate false positive rate
    false_positives = 0
    num_lookups = 10000
    for _ in range(num_lookups):
        elem = str(random.randint(0, 10**12))
        if bf.query(elem):
            false_positives += 1

    return false_positives / num_lookups

def plot_false_positive_rate_vs_hash_functions(n, f, max_k):
    """
    Plots the false positive rate vs. the number of hash functions.
    
    Args:
        n: number of elements
        f: desired false positive rate
        max_k: maximum number of hash functions to test
    """
    ks = range(1, max_k + 1)
    false_positive_rates = [calculate_false_positive_rate(n, f, k) for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, false_positive_rates, marker='o')
    plt.xlabel('Number of Hash Functions (k)')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate vs. Number of Hash Functions')
    plt.grid(True)

    plt.savefig('../images/hash_functions_FP_rate.png', format='png')
    
    plt.show()

# if __name__ == "__main__":
#     n = 10**7     # Number of elements
#     f = 0.02      # Desired false positive rate
#     max_k =12     # Maximum number of hash functions to test

#     plot_false_positive_rate_vs_hash_functions(n, f, max_k)