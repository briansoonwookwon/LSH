import math
import mmh3
import bitarray
from nltk import ngrams
import random

class BloomFilter:
    def __init__(self, n: int, f: float):
        """
        Creates a Bloom Filter object.
        """
        self.n = n
        """n (int): Max # of elements"""
        self.f = f
        """f (int): Desired false positive rate"""
        self.m = int(-math.log(self.f) * self.n / (math.log(2)**2))
        """m (int): Size of bit array"""
        self.k = int(self.m * math.log(2) / self.n)
        """k (int): Number of hash functions"""
        self.n_bytes = (self.m + 7) // 8
        """n_bytes (int): Number of bytes required to store bit array"""
        self.bit_array = bitarray.bitarray(self.n_bytes * 8)
        """bit_array: Create the bit array"""
        self.bit_array.setall(0)  # Set all bits to 0

    def add(self, item: str):
        """
        Adds an item to the bloom filter.
        
        Args:
            item: the string to add
        """
        tokens = item.lower().split()
        for i in range(self.k):
            for n in range(1, 4):  # You could make this range flexible
                n_grams = ngrams(tokens, n)
                for piece in n_grams:
                    index = mmh3.hash(" ".join(piece), i) % self.m
                    self.bit_array[index] = 1

    def query(self, item: str) -> bool:
        """
        Checks if an item might be in the bloom filter (may have false positives).
        
        Args:
            item: the string to query
        
        Returns:
            bool: True if item might be in the bloom filter, False if it definitely isn't
        """
        tokens = item.lower().split()
        for i in range(self.k):
            for n in range(1, 4):
                n_grams = ngrams(tokens, n)
                for piece in n_grams:
                    index = mmh3.hash(" ".join(piece), i) % self.m
                    if not self.bit_array[index]:
                        return False
        return True



class BloomFilter_KM_Opt:
    # This class includes the Bloom Filter with the Kirsch-Mitzenmacher-Optimization

    def __init__(self, n: int, f: float):
        """
        Creates a Bloom Filter object with Kirsch-Mitzenmacher-Optimization.
        """
        self.n = n
        """n (int): Max # of elements"""
        self.f = f
        """f (int): Desired false positive rate"""
        self.m = int(-math.log(self.f) * self.n / (math.log(2)**2))
        """m (int): Size of bit array"""
        self.k = int(self.m * math.log(2) / self.n)
        """k (int): Number of hash functions"""
        self.n_bytes = (self.m + 7) // 8
        """n_bytes (int): Number of bytes required to store bit array"""
        self.bit_array = bitarray.bitarray(self.n_bytes * 8)
        """bit_array: Create the bit array"""
        self.bit_array.setall(0)
        """Set all bits to 0"""

    def add(self, item: str):
        """
        Adds an item to the Bloom filter using the Kirsch-Mitzenmacher optimization.
        
        Args:
            item: the string to add
        """
        # Compute the two primary hash values
        h1 = mmh3.hash(item, 0)
        h2 = mmh3.hash(item, 1)

        # Use linear combinations of h1 and h2 to set k indices
        for i in range(self.k):
            index = (h1 + i * h2) % self.m
            self.bit_array[index] = 1

    def query(self, item: str) -> bool:
        """
        Checks if an item might be in the bloom filter using the Kirsch-Mitzenmacher optimization.
        
        Args:
            item: the string to query
        
        Returns:
            bool: True if item might be in the bloom filter, False if it definitely isn't
        """
        # Compute the two primary hash values
        h1 = mmh3.hash(item, 0)
        h2 = mmh3.hash(item, 1)

        # Check the linear combinations of h1 and h2 for the k indices
        for i in range(self.k):
            index = (h1 + i * h2) % self.m
            if not self.bit_array[index]:
                return False
        return True
    

class BloomFilter_Uni_Hash:
    # This class includes the Bloom Filter with the Universal Hashing

    def __init__(self, n: int, f: float):
        """
        Creates a Bloom Filter object.
        """
        self.n = n
        """n (int): Max # of elements"""
        self.f = f
        """f (int): Desired false positive rate"""
        self.m = int(-math.log(self.f) * self.n / (math.log(2)**2))
        """m (int): Size of bit array"""
        self.k = int(self.m * math.log(2) / self.n)
        """k (int): Number of hash functions"""
        self.n_bytes = (self.m + 7) // 8
        """n_bytes (int): Number of bytes required to store bit array"""
        self.bit_array = bitarray.bitarray(self.n_bytes * 8)
        """bit_array: Create the bit array"""
        self.bit_array.setall(0)  # Set all bits to 0
        self.seeds = [random.randint(0, 10**6) for _ in range(self.k)]

    def add(self, item: str):
        """
        Adds an item to the bloom filter Universal Hashing.
        
        Args:
            item: the string to add
        """
        for i in range(self.k): 
            index = mmh3.hash(item, self.seeds[i]) % self.m 
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
            index = mmh3.hash(item, self.seeds[i]) % self.m 
            if self.bit_array[index] == 0: 
                return False 
        return True
    


class BloomFilter_QF:
    # This class includes the Bloom Filter with the Quotient Filter

    def __init__(self, n: int, f: float):
        """
        Creates a Bloom Filter object.
        """
        self.n = n
        """n (int): Max # of elements"""
        self.f = f
        """f (int): Desired false positive rate"""
        self.m = int(-math.log(self.f) * self.n / (math.log(2)**2))
        """m (int): Size of bit array"""
        self.quotient_size = int(math.log2(self.m)) 
        self.remainder_size = 32 - self.quotient_size # Assuming 32-bit hash values 
        self.filter = [None] * (2**self.quotient_size)     

    def _hash(self, item): 
        h = mmh3.hash(item) 
        quotient = h >> self.remainder_size 
        remainder = h & ((1 << self.remainder_size) - 1) 
        
        return quotient, remainder

    def add(self, item: str):
        """
        Adds an item to the bloom filter.
        
        Args:
            item: the string to add
        """
        quotient, remainder = self._hash(item) 
        self.filter[quotient] = remainder 

    def query(self, item: str) -> bool:
        """
        Checks if an item might be in the bloom filter (may have false positives).
        
        Args:
            item: the string to query
        
        Returns:
            bool: True if item might be in the bloom filter, False if it definitely isn't
        """
        quotient, remainder = self._hash(item) 
        return self.filter[quotient] == remainder