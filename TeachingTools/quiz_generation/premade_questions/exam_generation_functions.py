import random
import math
from typing import Dict, List


class QuickFunctions:
  
  @classmethod
  def add_spaces_to_str(cls, input_str, every=4):
    if len(input_str) < 2*every:
      return input_str
    return ' '.join([input_str[i:i+every] for i in range(0, len(input_str), every)])
  
  @classmethod
  def random_binary_number(cls, num_bits):
    # return random.randrange(0, int(math.pow(2, num_bits)))
    return '0b ' + cls.add_spaces_to_str(cls.random_binary_bits(num_bits))
  
  @classmethod
  def random_binary_bits(cls, num_bits):
    return ''.join(random.choices("01", k=num_bits))
  @classmethod
  def random_hex_number(cls, num_digits):
    return '0x' + cls.random_hex_digits(num_digits)
  
  @classmethod
  def random_hex_digits(cls, num_digits, prevent_zero=False):
    result = ''.join(random.choices("0123456789abcdef".upper(), k=num_digits))
    if prevent_zero:
      while result == '0':
        result = ''.join(random.choices("0123456789abcdef".upper(), k=num_digits))
    return result
  
  @classmethod
  def pick_replacement_algo(cls):
    return cls.pick_a_choice(["LRU", "FIFO", "Belady"])
  
  @classmethod
  def pick_a_choice(cls, list_of_choices):
    return str(random.choice(list_of_choices))
    
  @classmethod
  def shuffle_list(cls, list_to_shuffle):
    return '\n'.join(random.sample(list_to_shuffle, len(list_to_shuffle)))
  
  @classmethod
  def number_in_range(cls, lower_bound, upper_bound):
    return random.randrange(lower_bound, upper_bound)
  
  @classmethod
  def print_as_hex(cls, in_number, pad_to_length=0, show_prefix=True, add_spaces=False):
    out_str = f"{in_number : x}".zfill(pad_to_length)
    if add_spaces:
      out_str = ' ' + cls.add_spaces_to_str(out_str)
    if show_prefix:
      out_str = '0x' + out_str
    return out_str
  
  @classmethod
  def print_as_binary(cls, in_number, pad_to_length=0, show_prefix=True, add_spaces=False, every=4):
    out_str = f"{in_number : b}".zfill(pad_to_length)
    if add_spaces:
      out_str = ' ' + cls.add_spaces_to_str(out_str)
    if show_prefix:
      out_str = '0b' + out_str
    return out_str

  @classmethod
  def generate_BNF_reversepolish(cls, num_to_generate=10, max_length = 20):
    class BNF:
      class GeneratedString:
        def __init__(self, starting_string : str):
          self.versions = [starting_string]
        
        def __str__(self):
          if self.versions[-1] == "":
            return "\"\""
          return self.versions[-1]
        
        def replace(self, target, replacement, *args, **kwargs):
          self.versions.append(self.versions[-1].replace(target, replacement, *args, **kwargs))
          return self.versions[-1]
        
        def __contains__(self, item):
          return item in self.versions[-1]
        
        def count(self, item):
          return self.versions[-1].count(item)
        
        # These two for set compatibility
        def __hash__(self):
          return self.versions[-1].__hash__()
        def __eq__(self, other):
          return self.versions[-1].__eq__(other.versions[-1])
      
        def __len__(self):
          return len(self.versions[-1])
      
      def __init__(self, productions : Dict[str,List[str]], starting_nonterminal: str):
        self.productions = productions
        self.starting_nonterminal = starting_nonterminal
      
      def is_complete(self, str_to_test : GeneratedString):
        return not any([nonterminal in str_to_test for nonterminal in self.productions.keys()])
      
      def get_string(self, max_depth = 20):
        generated_str = BNF.GeneratedString(self.starting_nonterminal)
        while (not self.is_complete(generated_str)):
          for rule in self.productions.keys():
            for _ in range(generated_str.count(rule)):
              generated_str.replace(rule, random.choice(self.productions[rule]), 1)
              if len(generated_str.versions) > max_depth:
                return ""
        return generated_str
      
      def get_n_unique_strings(self, n: int):
        unique_strings = set()
        while (len(unique_strings) < n):
          unique_strings.add(self.get_string())
        return unique_strings
    
    good_bnf = BNF(
      productions={
        "<A>" : ["<B> <B> <D>"],
        "<B>" : ["<C>", "<A>"],
        "<C>" : ["<E>", "<E><C>"],
        "<D>" : ["+", "-", "*", "/"],
        "<E>" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
      },
      starting_nonterminal="<A>"
    )
    
    bad_bnf = BNF(
      productions={
        "<A>" : ["<B> <D> <B>"],
        "<B>" : ["<C>", "<A>"],
        "<C>" : ["<E>", "<E><C>"],
        "<D>" : ["+", "-", "*", "/"],
        "<E>" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
      },
      starting_nonterminal="<A>"
    )
    
    strings = set()
    while (len(strings) < num_to_generate):
      for generator in [good_bnf, bad_bnf]:
        str_to_add = generator.get_string()
        while len(str_to_add) >= max_length:
          str_to_add = generator.get_string()
        strings.add(str(str_to_add))
      
      for generator in [good_bnf]:
        str_to_add = generator.get_string()
        while len(str_to_add) >= max_length:
          str_to_add = generator.get_string()
        if isinstance(str_to_add, str): continue
        strings.add(str(str_to_add.versions[-2]))
      if "" in strings:
        strings.remove("")
    
    return sorted(
      strings,
      key=(lambda _: random.random())
    )[:num_to_generate]
    