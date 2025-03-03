#!env python
from __future__ import annotations

import enum
from typing import List, Optional

from TeachingTools.quiz_generation.question import Question, Answer, TableGenerator, QuestionRegistry

import random
import math
import collections

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class MemoryQuestion(Question):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MEMORY)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class CachingQuestion(MemoryQuestion):
  
  class Kind(enum.Enum):
    FIFO = enum.auto()
    LRU = enum.auto()
    Belady = enum.auto()
    def __str__(self):
      return self.name
  
  class Cache:
    def __init__(self, kind : CachingQuestion.Kind, cache_size: int, all_requests : List[int]=None):
      self.kind = kind
      self.cache_size = cache_size
      self.all_requests = all_requests
      
      self.cache_state = [] # queue.Queue(maxsize=cache_size)
      self.last_used = collections.defaultdict(lambda: -math.inf)
      self.frequency = collections.defaultdict(lambda: 0)
    
    def query_cache(self, request, request_number):
      was_hit = request in self.cache_state
      
      evicted = None
      if was_hit:
        # hit!
        pass
      else:
        # miss!
        if len(self.cache_state) == self.cache_size:
          # Then we are full and need to evict
          evicted = self.cache_state[0]
          self.cache_state = self.cache_state[1:]
        
        # Add to cache
        self.cache_state.append(request)
      
      # update state variable
      self.last_used[request] = request_number
      self.frequency[request] += 1
      
      # update cache state
      if self.kind == CachingQuestion.Kind.FIFO:
        pass
      elif self.kind == CachingQuestion.Kind.LRU:
        self.cache_state = sorted(
          self.cache_state,
          key=(lambda e: self.last_used[e]),
          reverse=False
        )
      # elif self.kind == CachingQuestion.Kind.LFU:
      #   self.cache_state = sorted(
      #     self.cache_state,
      #     key=(lambda e: (self.frequency[e], e)),
      #     reverse=False
      #   )
      elif self.kind == CachingQuestion.Kind.Belady:
        upcoming_requests = self.all_requests[request_number+1:]
        self.cache_state = sorted(
          self.cache_state,
          # key=(lambda e: (upcoming_requests.index(e), e) if e in upcoming_requests else (-math.inf, e)),
          key=(lambda e: (upcoming_requests.index(e), -e) if e in upcoming_requests else (math.inf, -e)),
          reverse=True
        )
      
      return (was_hit, evicted, self.cache_state)
  
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_elements = kwargs.get("num_elements", 5)
    self.cache_size = kwargs.get("cache_size", 3)
    self.num_requests = kwargs.get("num_requests", 10)
    
    self.instantiate()
  
  def instantiate(self, rng_seed=None, previous : Optional[CachingQuestion]=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    self.answers = []
    if previous is None:
      log.debug("picking new caching policy")
      self.cache_policy = random.choice(list(self.Kind))
    else:
      log.debug("Reusing previous caching policy")
      self.cache_policy = previous.cache_policy
    
    log.debug(f"self.caching_policy: {self.cache_policy}")
    
    self.requests = (
        list(range(self.cache_size)) # Prime the cache with the compulsory misses
        + random.choices(population=list(range(self.cache_size-1)), k=1) # Add in one request to an earlier  that will differentiate clearly between FIFO and LRU
        + random.choices(population=list(range(self.cache_size, self.num_elements)), k=1) ## Add in the rest of the requests
        + random.choices(population=list(range(self.num_elements)), k=(self.num_requests-2)) ## Add in the rest of the requests
    )
    
    self.cache = CachingQuestion.Cache(self.cache_policy, self.cache_size, self.requests)
    
    self.request_results = {}
    number_of_hits = 0
    for (request_number, request) in enumerate(self.requests):
      was_hit, evicted, cache_state = self.cache.query_cache(request, request_number)
      if was_hit:
        number_of_hits += 1
      self.request_results[request_number] = {
        "request" : (f"[answer__request]", request),
        "hit" : (f"[answer__hit-{request_number}]", ('hit' if was_hit else 'miss')),
        "evicted" : (f"[answer__evicted-{request_number}]", ('-' if evicted is None else f"{evicted}")),
        "cache_state" : (f"[answer__cache_state-{request_number}]", ','.join(map(str, cache_state)))
      }
      self.answers.extend([
        Answer(f"answer__hit-{request_number}",         ('hit' if was_hit else 'miss'),          Answer.AnswerKind.BLANK),
        Answer(f"answer__evicted-{request_number}",     ('-' if evicted is None else f"{evicted}"),      Answer.AnswerKind.BLANK),
        Answer(f"answer__cache_state-{request_number}", ','.join(map(str, cache_state)),  Answer.AnswerKind.BLANK),
      ])
      
      # log.debug(f"{request:>2} | {'hit' if was_hit else 'miss':<4} | {evicted if evicted is not None else '':<3} | {str(cache_state):<10}")
      
    self.hit_rate = 100 * number_of_hits / (self.num_requests)
    # self.hit_rate_var = VariableFloat("Hit Rate (%)", true_value=self.hit_rate)
    # self.blank_vars["hit_rate"] = self.hit_rate_var
    self.answers.extend([
      Answer("answer__hit_rate", f"{self.hit_rate:0.2f}", Answer.AnswerKind.BLANK)
    ])
  
  def get_body_lines(self, *args, **kwargs) -> List[str]:
    # return ["question"]
    lines = [
      f"Assume we are using a <b>{self.cache_policy}</b> caching policy and a cache size of <b>{self.cache_size}</b>."
      "",
      "Given the below series of requests please fill in the table.",
      "For the hit/miss column, please write either \"hit\" or \"miss\".",
      "For the eviction column, please write either the number of the evicted page or simply a dash (e.g. \"-\").",
      # "For the cache state, please enter the cache contents in the order suggested in class, separated by commas with no spaces (e.g. \"1,2,3\").",
      # "As a reminder, in class we kept them ordered in the order in which we would evict them from the cache (including Belady, although I was too lazy to do that in class)",
      "",
    ]
    
    table_headers = ["Page Requested", "Hit/Miss", "Evicted", "Cache State"]
    lines.extend(
      self.get_table_generator(
        { request_number :
          [
            self.requests[request_number],
            f"[answer__hit-{request_number}]",
            f"[answer__evicted-{request_number}]",
            f"[answer__cache_state-{request_number}]"
          ]
          for request_number in sorted(self.request_results.keys())
        },
        table_headers,
        sorted_keys=sorted(self.request_results.keys()),
        hide_keys=True
      )
    )
    
    lines.extend([
      "Hit rate, excluding compulsory misses: [answer__hit_rate]%"
    ])
    
    # log.debug('\n'.join(lines))
    
    return lines
  
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    lines = [
      # "Apologies for the below table not including the eviction data, but technical limitations prevented me from including it.  "
      # "Instead, it can be inferred from the change in the cache state.",
      "The full table can be seen below.",
      ""
    ]
    
    table_headers = ["Page", "Hit/Miss", "Evicted", "Cache State"]
    lines.extend(
      self.get_table_generator(
        { request_number :
          [
            self.request_results[request]["request"][1],
            self.request_results[request]["hit"][1],
            f'{self.request_results[request]["evicted"][1]}',
            f'{self.request_results[request]["cache_state"][1]}',
          ]
          for (request_number, request) in enumerate(sorted(self.request_results.keys()))
        },
        table_headers,
        sorted_keys=sorted(self.request_results.keys()),
        hide_keys=True
      )
    )
    
    lines.extend([
      "",
      "To calculate the hit rate we calculate the percentage of requests that were cache hits out of the total number of requests. "
      "In this case we are counting all requests, excluding compulsory misses."
    ])
    
    return lines
  
  def is_interesting(self) -> bool:
    # todo: interesting is more likely based on whether I can differentiate between it and another algo, so maybe rerun with a different approach but same requests?
    return (self.hit_rate / 100.0) < 0.5


class MemoryAccessQuestion(MemoryQuestion):
  PROBABILITY_OF_VALID = .875


@QuestionRegistry.register()
class Paging(MemoryAccessQuestion):
  
  MIN_OFFSET_BITS = 3
  MIN_VPN_BITS = 3
  MIN_PFN_BITS = 3
  
  MAX_OFFSET_BITS = 8
  MAX_VPN_BITS = 8
  MAX_PFN_BITS = 16
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    self.instantiate()
  
  def instantiate(self, rng_seed=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    self.num_offset_bits = random.randint(self.MIN_OFFSET_BITS, self.MAX_OFFSET_BITS)
    self.num_vpn_bits = random.randint(self.MIN_VPN_BITS, self.MAX_VPN_BITS)
    self.num_pfn_bits = random.randint(max([self.MIN_PFN_BITS, self.num_vpn_bits]), self.MAX_PFN_BITS)
    
    self.virtual_address = random.randint(0, 2**(self.num_vpn_bits + self.num_offset_bits))
    
    # Calculate these two
    self.offset = self.virtual_address % (2**(self.num_offset_bits))
    self.vpn = self.virtual_address // (2**(self.num_offset_bits))
    
    # Generate this randomly
    self.pfn = random.randint(0, 2**(self.num_pfn_bits))
    
    # Calculate this
    self.physical_address = self.pfn * (2**self.num_offset_bits) + self.offset
    
    if random.choices([True, False], weights=[(self.PROBABILITY_OF_VALID), (1-self.PROBABILITY_OF_VALID)], k=1)[0]:
      self.is_valid = True
      # Set our actual entry to be in the table and valid
      self.pte = self.pfn + (2**(self.num_pfn_bits))
      # self.physical_address_var = VariableHex("Physical Address", self.physical_address, num_bits=(self.num_pfn_bits+self.num_offset_bits), default_presentation=VariableHex.PRESENTATION.BINARY)
      # self.pfn_var = VariableHex("PFN", self.pfn, num_bits=self.num_pfn_bits, default_presentation=VariableHex.PRESENTATION.BINARY)
    else:
      self.is_valid = False
      # Leave it as invalid
      self.pte = self.pfn
      # self.physical_address_var = Variable("Physical Address", "INVALID")
      # self.pfn_var = Variable("PFN",  "INVALID")
    
    # self.pte_var = VariableHex("PTE", self.pte, num_bits=(self.num_pfn_bits+1), default_presentation=VariableHex.PRESENTATION.BINARY)
    
    self.answers.extend([
      Answer("answer__vpn",     self.vpn,     variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=self.num_vpn_bits),
      Answer("answer__offset",  self.offset,  variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=self.num_offset_bits),
      Answer("answer__pte",     self.pte,     variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=(self.num_pfn_bits + 1)),
    ])
    
    if self.is_valid:
      self.answers.extend([
        Answer("answer__is_valid",          "VALID"),
        Answer("answer__pfn",               self.pfn,               variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=self.num_pfn_bits),
        Answer("answer__physical_address",  self.physical_address,  variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=(self.num_pfn_bits + self.num_offset_bits)),
      ])
    else:
      self.answers.extend([
        Answer("answer__is_valid",          "INVALID"),
        Answer("answer__pfn",               "INVALID"),
        Answer("answer__physical_address",  "INVALID"),
      ])
  
  def get_body_lines(self, *args, **kwargs) -> List[str|TableGenerator]:
    lines = [
      "Given the below information please calculate the equivalent physical address of the given virtual address, filling out all steps along the way.",
      "Remember, we typically have the MSB representing valid or invalid."
    ]
    
    lines.extend([
      TableGenerator(
        value_matrix=[
          ["Virtual Address", f"0b{self.virtual_address:0{self.num_vpn_bits + self.num_offset_bits}b}"],
          ["# VPN bits", f"{self.num_vpn_bits}"],
          ["# PFN bits", f"{self.num_pfn_bits}"],
        ]
      )
    ])
    
    # Make values for Page Table
    table_size = random.randint(5,10)
    table_bottom = self.vpn - random.randint(0, table_size)
    if table_bottom < 0:
      table_bottom = 0
    table_top = min([table_bottom + table_size, 2**self.num_vpn_bits])
    
    page_table = {}
    page_table[self.vpn] = self.pte
    
    # Fill in the rest of the table
    # for vpn in range(2**self.num_vpn_bits):
    for vpn in range(table_bottom, table_top):
      if vpn == self.vpn: continue
      pte = page_table[self.vpn]
      while pte in page_table.values():
        pte = random.randint(0, 2**self.num_pfn_bits-1)
        if random.choices([True, False], weights=[(1-self.PROBABILITY_OF_VALID), self.PROBABILITY_OF_VALID], k=1)[0]:
          # Randomly set it to be valid
          pte += (2**(self.num_pfn_bits))
      # Once we have a unique random entry, put it into the Page Table
      page_table[vpn] = pte
    
    
    lines.extend([
      TableGenerator(
        headers=["VPN", "PTE"],
        value_matrix=[
          [f"0b{vpn:0{self.num_vpn_bits}b}", f"0b{pte:0{(self.num_pfn_bits+1)}b}"]
          for vpn, pte in sorted(page_table.items())
        ]
      )
    ])
    
    lines.extend([
      "- VPN: [answer__vpn]",
      "- Offset: [answer__offset]",
      "- PTE: [answer__pte]",
      "- PFN: [answer__pfn]",
      "- Physical Address: [answer__physical_address]",
    ])
    
    return lines
  
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    
    lines = [
      "The core idea of Paging is we want to break the virtual address into the VPN and the offset.  "
      "From here, we get the Page Table Entry corresponding to the VPN, and check the validity of the entry.  "
      "If it is valid, we clear the metadata and attach the PFN to the offset and have our physical address.",
      "",
      "Don't forget to pad with the appropriate number of 0s (the appropriate number is the number of bits)!",
      "",
      f"Virtual Address = VPN | offset",
      f"<tt>0b{self.virtual_address:0{self.num_vpn_bits+self.num_offset_bits}b}</tt> = <tt>0b{self.vpn:0{self.num_vpn_bits}b}</tt> | <tt>0b{self.offset:0{self.num_offset_bits}b}</tt>",
      ""
    ]
    
    lines.extend([
      "We next use our VPN to index into our page table and find the corresponding entry."
      f"Our Page Table Entry is:",
      "",
      f"<tt>0b{self.pte:0{(self.num_pfn_bits+1)}b}</tt>"
      f"which we found by looking for our VPN in the page table.",
      "",
    ])
    
    is_valid = (self.pte // (2**self.num_pfn_bits) == 1)
    if is_valid:
      lines.extend([
        f"In our PTE we see that the first bit is <b>{self.pte // (2**self.num_pfn_bits)}</b> meaning that the translation is <b>VALID</b>"
      ])
    else:
      lines.extend([
        f"In our PTE we see that the first bit is <b>{self.pte // (2**self.num_pfn_bits)}</b> meaning that the translation is <b>INVALID</b>.",
        "Therefore, we just write \"INVALID\" as our answer.",
        "If it were valid we would complete the below steps.",
        "",
        "<hr>"
        "\n",
      ])
    
    lines.extend([
      "Next, we convert our PTE to our PFN by removing our metadata.  In this case we're just removing the leading bit.  We can do this by applying a binary mask.",
      f"PFN = PTE & mask",
      f"which is,",
      "",
      f"<tt>{self.pfn:0{self.num_pfn_bits}b}</tt> = <tt>0b{self.pte:0{self.num_pfn_bits+1}b}</tt> & <tt>0b{(2**self.num_pfn_bits)-1:0{self.num_pfn_bits+1}b}</tt>"
    ])
    
    lines.extend([
      "We then add combine our PFN and offset",
      "",
      "Physical Address = PFN | offset",
      f"{'<tt><b>' if is_valid else ''}0b{self.physical_address:0{self.num_pfn_bits+self.num_offset_bits}b}{'</b></tt>' if is_valid else ''} = <tt>0b{self.pfn:0{self.num_pfn_bits}b}</tt> | <tt>0b{self.offset:0{self.num_vpn_bits}b}</tt>",
      "",
      "",
      "Note: Strictly speaking, this calculation is:",
      f"{'<tt><b>' if is_valid else ''}0b{self.physical_address:0{self.num_pfn_bits+self.num_offset_bits}b}{'</b></tt>' if is_valid else ''} = <tt>0b{self.pfn:0{self.num_pfn_bits}b}{0:0{self.num_offset_bits}}</tt> + <tt>0b{self.offset:0{self.num_offset_bits}b}</tt>",
      "But that's a lot of extra 0s, so I'm splitting them up for succinctness",
      ""
    ])
    return lines

#########################
# todo: below this line #
#########################

class VirtualAddress_parts(Question):
  MAX_BITS = 64
  
  def __init__(
      self,
      num_va_bits=None,
      num_offset_bits=None,
      num_vpn_bits=None
  ):
    
    if (num_va_bits is None) and (num_offset_bits is None) and (num_vpn_bits is None):
      num_va_bits = random.randint(1, self.MAX_BITS)
    
    # If we have a VA (or have generated one)
    if (num_offset_bits is None) and (num_vpn_bits is None):
      num_offset_bits = random.randint(1, num_va_bits)
    
    if (num_vpn_bits is None):
      num_vpn_bits = num_va_bits - num_offset_bits
    
    self.num_va_bits = Variable("# VA bits", num_va_bits)
    self.num_offset_bits = Variable("# offset bits", num_offset_bits)
    self.num_vpn_bits = Variable("# VPN bits", num_vpn_bits)
    
    super().__init__(
      given_vars=[
        self.num_va_bits,
        self.num_offset_bits,
        self.num_vpn_bits
      ]
    )
  
  def get_explanation(self) -> List[str]:
    
    line_to_add = ""
    if self.num_va_bits in self.target_vars:
      line_to_add += f"***{self.num_va_bits.true_value}***"
    else:
      line_to_add += f"{self.num_va_bits.true_value}"
    
    line_to_add += " = "
    
    if self.num_vpn_bits in self.target_vars:
      line_to_add += f"***{self.num_vpn_bits.true_value}***"
    else:
      line_to_add += f"{self.num_vpn_bits.true_value}="
    
    line_to_add += " + "
    
    if self.num_offset_bits in self.target_vars:
      line_to_add += f"***{self.num_offset_bits.true_value}***"
    else:
      line_to_add += f"{self.num_offset_bits.true_value}"
    
    return [
      "VA = VPN + offset",
      line_to_add
    ]


class BaseAndBounds(MemoryAccessQuestion):
  MAX_BITS = 32
  MAX_BOUNDS_BITS = 16
  
  def __init__(
      self
  ):
    
    use_binary = random.randint(0,1) % 2 ==0
    
    bounds_bits = random.randint(1, self.MAX_BOUNDS_BITS)
    base_bits = self.MAX_BITS - bounds_bits
    
    self.bounds = int(math.pow(2, bounds_bits))
    self.base = random.randint(1, int(math.pow(2, base_bits))) * self.bounds
    self.virtual_address = random.randint(1, int(self.bounds / self.PROBABILITY_OF_VALID))
    
    self.bounds_var = VariableHex("Bounds", self.bounds, default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX))
    self.base_var = VariableHex("Base", self.base, default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX))
    self.virtual_address_var = VariableHex("Virtual Address", self.virtual_address, default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX))
    
    logging.debug(f"bounds: {self.bounds}")
    logging.debug(f"base: {self.base}")
    logging.debug(f"va: {self.virtual_address}")
    
    if self.virtual_address < self.bounds:
      self.physical_address_var = VariableHex(
        "Physical Address",
        num_bits=math.ceil(math.log2(self.base + self.virtual_address)),
        true_value=(self.virtual_address + self.base),
        default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)
      )
    else:
      self.physical_address_var = Variable("Physical Address", "INVALID")
    
    logging.debug(f"pa: {self.virtual_address + self.base}")
    logging.debug(f"pa: {self.physical_address_var}")
    
    
    super().__init__(
      given_vars=[
        self.base_var,
        self.bounds_var,
        self.virtual_address_var
      ],
      target_vars=[
        self.physical_address_var
      ]
    )
  
  def get_explanation(self) -> List[str]:
    explanation_lines = [
      "There's two steps to figuring out base and bounds.",
      "1. Are we within the bounds?",
      "2. If so, add to our base.",
      "",
    ]
    if self.virtual_address < self.bounds:
      explanation_lines.extend([
        f"Step 1: {self.virtual_address_var.true_value} < {self.bounds_var.true_value} --> {'***VALID***' if (self.virtual_address < self.bounds) else 'INVALID'}",
        "",
        f"Step 2: Since the previous check passed, we calculate {self.base_var.true_value} + {self.virtual_address_var.true_value} = ***0x{self.base + self.virtual_address:X}***.  If it had been invalid we would have simply written INVALID"
      ]
      )
    else:
      explanation_lines.extend([
        f"Step 1: {self.virtual_address_var.true_value} < {self.bounds_var.true_value} --> {'VALID' if (self.virtual_address < self.bounds) else '***INVALID***'}",
        "",
        f"Step 2: Since the previous check failed, we simply write ***INVALID***.  If it had been valid, we would have calculated {self.base_var.true_value} + {self.virtual_address_var.true_value} = 0x{self.base + self.virtual_address:X}"
      ]
      )
    return explanation_lines
  
  
  
  def get_explanation(self, *args, **kwargs) -> List[str]:
    explanation_lines = [
      "There's two steps to figuring out base and bounds.",
      "1. Are we within the bounds?",
      "2. If so, add to our base.",
      "",
    ]
    if self.virtual_address < self.bounds:
      explanation_lines.extend([
        f"Step 1: {self.virtual_address_var.true_value} < {self.bounds_var.true_value} --> {'<b>>VALID</b>' if (self.virtual_address < self.bounds) else 'INVALID'}",
        "",
        f"Step 2: Since the previous check passed, we calculate {self.base_var.true_value} + {self.virtual_address_var.true_value} = <b>{self.physical_address_var.true_value}</b>.",
        "If it had been invalid we would have simply written INVALID"
      ]
      )
    else:
      explanation_lines.extend([
        f"Step 1: {self.virtual_address_var.true_value} < {self.bounds_var.true_value} --> {'VALID' if (self.virtual_address < self.bounds) else '<b>INVALID</b>'}",
        "",
        f"Step 2: Since the previous check failed, we simply write <b>INVALID</b>.",
        f"If it had been valid, we would have calculated {self.base_var.true_value} + {self.virtual_address_var.true_value} = {self.physical_address_var.true_value}"
      ]
      )
    return explanation_lines


class Segmentation_canvas(MemoryAccessQuestion, CanvasQuestion__fill_in_the_blanks):
  MAX_BITS = 32
  MIN_VIRTUAL_BITS = 5
  MAX_VIRTUAL_BITS = 10
  
  def __within_bounds(self, segment, offset, bounds):
    if segment == "unallocated":
      return False
    elif bounds < offset:
      return False
    else:
      return True
  
  def __init__(self, *args, **kwargs):
    super().__init__(given_vars=[], target_vars=[], *args, **kwargs)
    
    use_binary = random.randint(0,1) % 2 ==0
    
    self.base = {
      "code" : 0,
      "heap" : 0,
      "stack" : 0,
    }
    self.bounds = {
      "code" : 0,
      "heap" : 0,
      "stack" : 0,
    }
    
    self.virtual_bits = random.randint(self.MIN_VIRTUAL_BITS, self.MAX_VIRTUAL_BITS)
    self.physical_bits = random.randint(self.virtual_bits+1, self.MAX_BITS)
    
    min_bounds = 2
    max_bounds = int(2**(self.virtual_bits - 2))
    
    def segment_collision(base, bounds):
      # lol, I think this is probably silly, but should work
      return 0 != len(set.intersection(*[
        set(range(base[segment], base[segment]+bounds[segment]+1))
        for segment in base.keys()
      ]))
    
    self.base["unallocated"] = 0
    self.bounds["unallocated"] = 0
    
    
    while (segment_collision(self.base, self.bounds)):
      for segment in self.base.keys():
        self.bounds[segment] = random.randint(min_bounds, max_bounds)
        self.base[segment] = random.randint(0, (2**self.physical_bits - self.bounds[segment]))
    
    self.segment = random.choice(list(self.base.keys()) + ["unallocated"])
    self.segment_bits = {
      "code" : 0,
      "heap" : 1,
      "unallocated" : 2,
      "stack" : 3
    }[self.segment]
    
    try:
      self.virtual_address_offset = random.randint(0,
        min([
          ((self.virtual_bits-2)**2)-1,
          int(self.bounds[self.segment] / self.PROBABILITY_OF_VALID)
        ])
      )
    except KeyError:
      self.virtual_address_offset = random.randint(0, ((self.virtual_bits-2)**2)-1)
    
    self.virtual_address = (
        (self.segment_bits << (self.virtual_bits - 2))
        + self.virtual_address_offset
    )
    
    self.physical_address = self.base[self.segment] + self.virtual_address_offset
    if not self.__within_bounds(self.segment, self.virtual_address_offset, self.bounds[self.segment]):
      self.physical_address_var = Variable("Physical Address", "INVALID")
    else:
      self.physical_address_var = VariableHex("Physical Address", true_value=self.physical_address, num_bits=self.physical_bits, default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX))
    
    
    self.virtual_address_var = VariableHex("Virtual Address", true_value=self.virtual_address, default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX))
    self.segment_var = Variable("Segment", true_value=self.segment, default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX))
    self.base_vars = {
      "code" : VariableHex("Code Base", true_value=self.base["code"], default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)),
      "heap" : VariableHex("Heap Base", true_value=self.base["heap"], default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)),
      "stack" : VariableHex("Stack Base", true_value=self.base["stack"], default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)),
    }
    
    self.bounds_vars = {
      "code" : VariableHex("Code Bound", true_value=self.bounds["code"], default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)),
      "heap" : VariableHex("Heap Bound", true_value=self.bounds["heap"], default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)),
      "stack" : VariableHex("Stack Bound", true_value=self.bounds["stack"], default_presentation=(VariableHex.PRESENTATION.BINARY if use_binary else VariableHex.PRESENTATION.HEX)),
    }
    
    self.blank_vars.update({
      "segment_val" : self.segment_var,
      "physical_address_val" : self.physical_address_var
    })
  
  def get_question_body(self) -> List[str]:
    question_lines = [
      f"Given a virtual address space of {self.virtual_bits}bits, and a physical address space of {self.physical_bits}bits, what is the physical address associated with the virtual address {self.virtual_address_var}?",
      "If it is invalid simply type INVALID.",
      "Note: assume that the stack grows in the same way as the code and the heap."
    ]
    
    question_lines.extend(
      self.get_table_lines(
        table_data={
          "code": [self.base_vars["code"], self.bounds_vars["code"]],
          "heap": [self.base_vars["heap"], self.bounds_vars["heap"]],
          "stack": [self.base_vars["stack"], self.bounds_vars["stack"]],
        },
        sorted_keys=[
          "code", "heap", "stack"
        ],
        headers=["base", "bounds"],
        add_header_space=True
      )
    )
    
    question_lines.extend([
      "Segment name: [segment_val]",
      "Physical Address: [physical_address_val]"
    ])
    
    return question_lines
  
  def get_explanation(self, *args, **kwargs) -> List[str]:
    explanation_lines = [
      "The core idea to keep in mind with segmentation is that you should always check the first two bits of the virtual address to see what segment it is in and then go from there."
      "Keep in mind, we also may need to include padding if our virtual address has a number of leading zeros left off!",
      ""
    ]
    
    explanation_lines.extend([
      f"In this problem our virtual address, converted to binary and including padding, is 0b{self.virtual_address:0{self.virtual_bits}b}.",
      f"From this we know that our segment bits are 0b{self.segment_bits:02b}, meaning that we are in the <b>{self.segment_var}</b> segment.",
      ""
    ])
    
    if self.segment == "unallocated":
      explanation_lines.extend([
        "Since this is the unallocated segment there are no possible valid translations, so we enter <b>INVALID</b>."
      ])
    else:
      explanation_lines.extend([
        f"Since we are in the {self.segment_var} segment, we see from our table that our bounds are {self.bounds_vars[self.segment]}. "
        f"Remember that our check for our {self.segment_var} segment is: ",
        f"<code> if (offset > bounds({self.segment})) : INVALID</code>",
        "which becomes"
        f"<code> if ({self.virtual_address_offset:0b} > {self.bounds[self.segment]:0b}) : INVALID</code>"
      ])
      
      if not self.__within_bounds(self.segment, self.virtual_address_offset, self.bounds[self.segment]):
        # then we are outside of bounds
        explanation_lines.extend([
          "We can therefore see that we are outside of bounds so we should put <b>INVALID</b>.",
          "If we <i>were</i> requesting a valid memory location we could use the below steps to do so."
          "<hr>"
        ])
      else:
        explanation_lines.extend([
          "We are therefore in bounds so we can calculate our physical address, as we do below."
        ])
      
      explanation_lines.append("")
      
      explanation_lines.extend([
        "To find the physical address we use the formula:",
        "<code>physical_address = base(segment) + offset</code>",
        "which becomes",
        f"<code>physical_address = {self.base[self.segment]:0b} + {self.virtual_address_offset:0b}</code>.",
        ""
      ])
      
      explanation_lines.extend([
        "Lining this up for ease we can do this calculation as:",
        "<pre><code>",
        f"  0b{self.base[self.segment]:0{self.physical_bits}b}",
        f"<u>+ 0b{self.virtual_address_offset:0{self.physical_bits}b}</u>",
        f"  0b{self.physical_address:0{self.physical_bits}b}"
        "</code></pre>"
      ])
    
    
    
    return explanation_lines
    