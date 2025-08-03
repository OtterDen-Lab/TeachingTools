#!env python
from __future__ import annotations

import abc
import collections
import copy
import enum
import logging
import math
from typing import List, Optional

from TeachingTools.quiz_generation.misc import ContentAST
from TeachingTools.quiz_generation.question import Question, Answer, QuestionRegistry

log = logging.getLogger(__name__)


class MemoryQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MEMORY)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register("VirtualAddressParts")
class VirtualAddressParts(MemoryQuestion):
  MAX_BITS = 64
  
  class Target(enum.Enum):
    VA_BITS = "# VA Bits"
    VPN_BITS = "# VPN Bits"
    OFFSET_BITS = "# Offset Bits"
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    # Generate baselines, if not given
    self.num_va_bits = kwargs.get("num_va_bits", self.rng.randint(2, self.MAX_BITS))
    self.num_offset_bits = self.rng.randint(1, self.num_va_bits-1)
    self.num_vpn_bits = self.num_va_bits - self.num_offset_bits
    
    self.possible_answers = {
      self.Target.VA_BITS : Answer("answer__num_va_bits", self.num_va_bits, variable_kind=Answer.VariableKind.INT),
      self.Target.OFFSET_BITS : Answer("answer__num_offset_bits", self.num_offset_bits, variable_kind=Answer.VariableKind.INT),
      self.Target.VPN_BITS : Answer("answer__num_vpn_bits", self.num_vpn_bits, variable_kind=Answer.VariableKind.INT)
    }
    
    # Select what kind of question we are going to be
    self.blank_kind = self.rng.choice(list(self.Target))
    
    self.answers['answer'] = self.possible_answers[self.blank_kind]
    
    return
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        "Given the information in the below table, please complete the table as appropriate."
      ])
    )
    
    body.add_element(
      ContentAST.Table(
        headers=[t.value for t in list(self.Target)],
        data=[[
          f"{self.possible_answers[t].display} bits"
          if t != self.blank_kind
          else ContentAST.Element([
            ContentAST.Answer(self.possible_answers[t], " bits")
          ])
          for t in list(self.Target)
        ]]
      )
    )
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "Remember, when we are calculating the size of virtual address spaces, "
        "the number of bits in the overall address space is equal to the number of bits in the VPN "
        "plus the number of bits for the offset.",
        "We don't waste any bits!"
      ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        ContentAST.Text(f"{self.num_va_bits}", emphasis=(self.blank_kind == self.Target.VA_BITS)),
        ContentAST.Text(" = "),
        ContentAST.Text(f"{self.num_vpn_bits}", emphasis=(self.blank_kind == self.Target.VPN_BITS)),
        ContentAST.Text(" + "),
        ContentAST.Text(f"{self.num_offset_bits}", emphasis=(self.blank_kind == self.Target.OFFSET_BITS))
      ])
    )
    
    return explanation
    

@QuestionRegistry.register()
class CachingQuestion(MemoryQuestion):
  
  class Kind(enum.Enum):
    FIFO = enum.auto()
    LRU = enum.auto()
    BELADY = enum.auto()
    def __str__(self):
      return self.name
  
  class Cache:
    def __init__(self, kind : CachingQuestion.Kind, cache_size: int, all_requests : List[int] = None):
      self.kind = kind
      self.cache_size = cache_size
      self.all_requests = all_requests
      
      self.cache_state = []
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
      elif self.kind == CachingQuestion.Kind.BELADY:
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
    
    # First set a random algo, then try to see if we should use a different one
    self.cache_policy_generator = (lambda: self.rng.choice(list(self.Kind)))
    
    policy_str = (kwargs.get("policy") or kwargs.get("algo"))
    if policy_str:
      try:
        self.cache_policy_generator = (lambda: self.Kind[policy_str.upper()])
      except KeyError:
        log.warning(
          f"Invalid cache policy '{policy_str}'. Valid options are: {[k.name for k in self.Kind]}. Defaulting to random"
        )
    
    self.cache_policy = self.cache_policy_generator()
  
  def refresh(self, previous : Optional[CachingQuestion]=None, *args, hard_refresh : bool = False, **kwargs):
    # Check to see if we are using the existing caching policy or a brand new one
    if not hard_refresh:
      self.rng_seed_offset += 1
    else:
      self.cache_policy = self.cache_policy_generator()
    super().refresh(*args, **kwargs)
    
    self.requests = (
        list(range(self.cache_size)) # Prime the cache with the compulsory misses
        + self.rng.choices(population=list(range(self.cache_size-1)), k=1) # Add in one request to an earlier  that will differentiate clearly between FIFO and LRU
        + self.rng.choices(population=list(range(self.cache_size, self.num_elements)), k=1) ## Add in the rest of the requests
        + self.rng.choices(population=list(range(self.num_elements)), k=(self.num_requests-2)) ## Add in the rest of the requests
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
      
      self.answers.update({
        f"answer__hit-{request_number}":          Answer(f"answer__hit-{request_number}",         ('hit' if was_hit else 'miss'),          Answer.AnswerKind.BLANK),
        f"answer__evicted-{request_number}":      Answer(f"answer__evicted-{request_number}",     ('-' if evicted is None else f"{evicted}"),      Answer.AnswerKind.BLANK),
        f"answer__cache_state-{request_number}":  Answer(f"answer__cache_state-{request_number}", copy.copy(cache_state), variable_kind=Answer.VariableKind.LIST),
      })
      
    self.hit_rate = 100 * number_of_hits / (self.num_requests)
    self.answers.update({
      "answer__hit_rate": Answer("answer__hit_rate", self.hit_rate, variable_kind=Answer.VariableKind.AUTOFLOAT)
    })
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        f"Assume we are using a <b>{self.cache_policy}</b> caching policy and a cache size of <b>{self.cache_size}</b>."
        "",
        "Given the below series of requests please fill in the table.",
        "For the hit/miss column, please write either \"hit\" or \"miss\".",
        "For the eviction column, please write either the number of the evicted page or simply a dash (e.g. \"-\")."
      ])
    )
    
    body.add_element(
      ContentAST.TextHTML(
        "For the cache state, please enter the cache contents in the order suggested in class, "
        "which means separated by commas with no spaces (e.g. \"1,2,3\")"
        "and with the left-most being the next to be evicted. "
        "In the case where there is a tie, order by increasing number."
      )
    )
    
    body.add_element(
      ContentAST.Table(
        headers=["Page Requested", "Hit/Miss", "Evicted", "Cache State"],
        data=[
          [
            f"{self.requests[request_number]}",
            ContentAST.Answer(self.answers[f"answer__hit-{request_number}"], blank_length=2),
            ContentAST.Answer(self.answers[f"answer__evicted-{request_number}"], blank_length=2),
            ContentAST.Answer(self.answers[f"answer__cache_state-{request_number}"], blank_length=2)
          ]
          for request_number in sorted(self.request_results.keys())
        ]
      )
    )
    
    body.add_element(
      ContentAST.AnswerBlock(
        ContentAST.Answer(
          answer=self.answers["answer__hit_rate"],
          label=f"Hit rate, excluding compulsory misses.  If appropriate, round to {Answer.DEFAULT_ROUNDING_DIGITS} decimal digits.",
          unit="%"
        )
      )
    )
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(ContentAST.Paragraph(["The full caching table can be seen below."]))
    
    explanation.add_element(
      ContentAST.Table(
        headers=["Page", "Hit/Miss", "Evicted", "Cache State"],
        data=[
          [
            self.request_results[request]["request"][1],
            self.request_results[request]["hit"][1],
            f'{self.request_results[request]["evicted"][1]}',
            f'{self.request_results[request]["cache_state"][1]}',
          ]
          for (request_number, request) in enumerate(sorted(self.request_results.keys()))
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        "To calculate the hit rate we calculate the percentage of requests "
        "that were cache hits out of the total number of requests. "
        f"In this case we are counting only all but {self.cache_size} requests, "
        f"since we are excluding compulsory misses."
      ])
    )
    
    return explanation
  
  def is_interesting(self) -> bool:
    # todo: interesting is more likely based on whether I can differentiate between it and another algo,
    #  so maybe rerun with a different approach but same requests?
    return (self.hit_rate / 100.0) < 0.7


class MemoryAccessQuestion(MemoryQuestion, abc.ABC):
  PROBABILITY_OF_VALID = .875


@QuestionRegistry.register()
class BaseAndBounds(MemoryAccessQuestion):
  MAX_BITS = 32
  MIN_BOUNDS_BIT = 5
  MAX_BOUNDS_BITS = 16
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    bounds_bits = self.rng.randint(self.MIN_BOUNDS_BIT, self.MAX_BOUNDS_BITS)
    base_bits = self.MAX_BITS - bounds_bits
    
    self.bounds = int(math.pow(2, bounds_bits))
    self.base = self.rng.randint(1, int(math.pow(2, base_bits))) * self.bounds
    self.virtual_address = self.rng.randint(1, int(self.bounds / self.PROBABILITY_OF_VALID))
    
    if self.virtual_address < self.bounds:
      self.answers["answer"] = Answer(
        key="answer__physical_address",
        value=(self.base + self.virtual_address),
        variable_kind=Answer.VariableKind.BINARY_OR_HEX,
        length=math.ceil(math.log2(self.base + self.virtual_address))
      )
    else :
      self.answers["answer"] = Answer(
        key="answer__physical_address",
        value="INVALID",
        variable_kind=Answer.VariableKind.STR
      )
  
  def get_body(self) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        "Given the information in the below table, "
        "please calcuate the physical address associated with the given virtual address.",
        "If the virtual address is invalid please simply write ***INVALID***."
      ])
    )
    
    body.add_element(
      ContentAST.Table(
        headers=["Base", "Bounds", "Virtual Address", "Physical Address"],
        data=[[
          f"0x{self.base:X}",
          f"0x{self.bounds:X}",
          f"0x{self.virtual_address:X}",
          ContentAST.Answer(self.answers["answer"])
        ]],
        transpose=True
      )
    )
    
    return body
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "There's two steps to figuring out base and bounds.",
        "1. Are we within the bounds?\n",
        "2. If so, add to our base.\n",
        "",
      ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        f"Step 1: 0x{self.virtual_address:X} < 0x{self.bounds:X} "
        f"--> {'***VALID***' if (self.virtual_address < self.bounds) else 'INVALID'}"
      ])
    )
    
    if self.virtual_address < self.bounds:
      explanation.add_element(
        ContentAST.Paragraph([
          f"Step 2: Since the previous check passed, we calculate "
          f"0x{self.base:X} + 0x{self.virtual_address:X} "
          f"= ***0x{self.base + self.virtual_address:X}***.",
          "If it had been invalid we would have simply written INVALID"
        ])
      )
    else:
      explanation.add_element(
        ContentAST.Paragraph([
          f"Step 2: Since the previous check failed, we simply write ***INVALID***.",
          "***If*** it had been valid, we would have calculated "
          f"0x{self.base:X} + 0x{self.virtual_address:X} "
          f"= 0x{self.base + self.virtual_address:X}.",
        ])
      )
    
    return explanation


@QuestionRegistry.register()
class Segmentation(MemoryAccessQuestion):
  MAX_BITS = 20
  MIN_VIRTUAL_BITS = 5
  MAX_VIRTUAL_BITS = 10
  
  def __within_bounds(self, segment, offset, bounds):
    if segment == "unallocated":
      return False
    elif bounds < offset:
      return False
    else:
      return True
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Pick how big each of our address spaces will be
    self.virtual_bits = self.rng.randint(self.MIN_VIRTUAL_BITS, self.MAX_VIRTUAL_BITS)
    self.physical_bits = self.rng.randint(self.virtual_bits+1, self.MAX_BITS)
    
    # Start with blank base and bounds
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
    
    min_bounds = 4
    max_bounds = int(2**(self.virtual_bits - 2))
    
    def segment_collision(base, bounds):
      # lol, I think this is probably silly, but should work
      return 0 != len(set.intersection(*[
        set(range(base[segment], base[segment]+bounds[segment]+1))
        for segment in base.keys()
      ]))
    
    self.base["unallocated"] = 0
    self.bounds["unallocated"] = 0
    
    # Make random placements and check to make sure they are not overlapping
    while (segment_collision(self.base, self.bounds)):
      for segment in self.base.keys():
        self.bounds[segment] = self.rng.randint(min_bounds, max_bounds-1)
        self.base[segment] = self.rng.randint(0, (2**self.physical_bits - self.bounds[segment]))
    
    # Pick a random segment for us to use
    self.segment = self.rng.choice(list(self.base.keys()))
    self.segment_bits = {
      "code" : 0,
      "heap" : 1,
      "unallocated" : 2,
      "stack" : 3
    }[self.segment]
    
    # Try to pick a random address within that range
    try:
      self.offset = self.rng.randint(1,
        min([
          max_bounds-1,
          int(self.bounds[self.segment] / self.PROBABILITY_OF_VALID)
        ])
      )
    except KeyError:
      # If we are in an unallocated section, we'll get a key error (I think)
      self.offset = self.rng.randint(0, max_bounds-1)
    
    # Calculate a virtual address based on the segment and the offset
    self.virtual_address = (
        (self.segment_bits << (self.virtual_bits - 2))
        + self.offset
    )
    
    # Calculate physical address based on offset
    self.physical_address = self.base[self.segment] + self.offset
    
    # Set answers based on whether it's in bounds or not
    if self.__within_bounds(self.segment, self.offset, self.bounds[self.segment]):
      self.answers["answer__physical_address"] = Answer(
        key="answer__physical_address",
        value=self.physical_address,
        variable_kind=Answer.VariableKind.BINARY_OR_HEX,
        length=self.physical_bits
      )
    else :
      self.answers["answer__physical_address"] = Answer(
        key="answer__physical_address",
        value="INVALID",
        variable_kind=Answer.VariableKind.STR
      )
    
    self.answers["answer__segment"] = Answer(
      key="answer__segment", value=self.segment, variable_kind=Answer.VariableKind.STR
    )
  
  def get_body(self) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        f"Given a virtual address space of {self.virtual_bits}bits, "
        f"and a physical address space of {self.physical_bits}bits, "
        "what is the physical address associated with the virtual address "
        f"0b{self.virtual_address:0{self.virtual_bits}b}?",
        "If it is invalid simply type INVALID.",
        "Note: assume that the stack grows in the same way as the code and the heap."
      ])
    )
    
    body.add_element(
      ContentAST.Table(
        headers=["", "base", "bounds"],
        data=[
          ["code", f"0b{self.base['code']:0{self.physical_bits}b}", f"0b{self.bounds['code']:0b}"],
          ["heap", f"0b{self.base['heap']:0{self.physical_bits}b}", f"0b{self.bounds['heap']:0b}"],
          ["stack", f"0b{self.base['stack']:0{self.physical_bits}b}", f"0b{self.bounds['stack']:0b}"]
        ]
      )
    )
    
    body.add_element(
      ContentAST.AnswerBlock([
        ContentAST.Answer(
          self.answers["answer__segment"],
          label="Segment name"
        ),
        ContentAST.Answer(
          self.answers["answer__physical_address"],
          label="Physical Address"
        )
      ])
    )
    return body
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "The core idea to keep in mind with segmentation is that you should always check ",
        "the first two bits of the virtual address to see what segment it is in and then go from there."
        "Keep in mind, "
        "we also may need to include padding if our virtual address has a number of leading zeros left off!"
      ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        f"In this problem our virtual address, "
        f"converted to binary and including padding, is 0b{self.virtual_address:0{self.virtual_bits}b}.",
        f"From this we know that our segment bits are 0b{self.segment_bits:02b}, "
        f"meaning that we are in the ***{self.segment}*** segment.",
        ""
      ])
    )
    
    if self.segment == "unallocated":
      explanation.add_element(
        ContentAST.Paragraph([
          "Since this is the unallocated segment there are no possible valid translations, so we enter ***INVALID***."
        ])
      )
    else:
      explanation.add_element(
        ContentAST.Paragraph([
          f"Since we are in the {self.segment} segment, "
          f"we see from our table that our bounds are {self.bounds[self.segment]}. "
          f"Remember that our check for our {self.segment} segment is: ",
          f"`if (offset > bounds({self.segment})) : INVALID`",
          "which becomes"
          f"`if ({self.offset:0b} > {self.bounds[self.segment]:0b}) : INVALID`"
        ])
      )
      
      if not self.__within_bounds(self.segment, self.offset, self.bounds[self.segment]):
        # then we are outside of bounds
        explanation.add_element(
          ContentAST.Paragraph([
            "We can therefore see that we are outside of bounds so we should put ***INVALID***.",
            "If we <i>were</i> requesting a valid memory location we could use the below steps to do so."
            "<hr>"
          ])
        )
      else:
        explanation.add_element(
          ContentAST.Paragraph([
            "We are therefore in bounds so we can calculate our physical address, as we do below."
          ])
        )
      
      explanation.add_element(
        ContentAST.Paragraph([
          "To find the physical address we use the formula:",
          "<code>physical_address = base(segment) + offset</code>",
          "which becomes",
          f"<code>physical_address = {self.base[self.segment]:0b} + {self.offset:0b}</code>.",
          ""
        ])
      )
      
      explanation.add_element(
        ContentAST.Paragraph([
          "Lining this up for ease we can do this calculation as:"
        ])
      )
      explanation.add_element(
        ContentAST.Code(
          f"  0b{self.base[self.segment]:0{self.physical_bits}b}\n"
          f"<u>+ 0b{self.offset:0{self.physical_bits}b}</u>\n"
          f"  0b{self.physical_address:0{self.physical_bits}b}\n"
        )
      )
    
    return explanation


@QuestionRegistry.register()
class Paging(MemoryAccessQuestion):
  
  MIN_OFFSET_BITS = 3
  MIN_VPN_BITS = 3
  MIN_PFN_BITS = 3
  
  MAX_OFFSET_BITS = 8
  MAX_VPN_BITS = 8
  MAX_PFN_BITS = 16
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    self.num_offset_bits = self.rng.randint(self.MIN_OFFSET_BITS, self.MAX_OFFSET_BITS)
    self.num_vpn_bits = self.rng.randint(self.MIN_VPN_BITS, self.MAX_VPN_BITS)
    self.num_pfn_bits = self.rng.randint(max([self.MIN_PFN_BITS, self.num_vpn_bits]), self.MAX_PFN_BITS)
    
    self.virtual_address = self.rng.randint(1, 2**(self.num_vpn_bits + self.num_offset_bits))
    
    # Calculate these two
    self.offset = self.virtual_address % (2**(self.num_offset_bits))
    self.vpn = self.virtual_address // (2**(self.num_offset_bits))
    
    # Generate this randomly
    self.pfn = self.rng.randint(0, 2**(self.num_pfn_bits))
    
    # Calculate this
    self.physical_address = self.pfn * (2**self.num_offset_bits) + self.offset
    
    if self.rng.choices([True, False], weights=[(self.PROBABILITY_OF_VALID), (1-self.PROBABILITY_OF_VALID)], k=1)[0]:
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
    
    self.answers.update({
      "answer__vpn": Answer("answer__vpn",     self.vpn,     variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=self.num_vpn_bits),
      "answer__offset": Answer("answer__offset",  self.offset,  variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=self.num_offset_bits),
      "answer__pte": Answer("answer__pte",     self.pte,     variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=(self.num_pfn_bits + 1)),
    })
    
    if self.is_valid:
      self.answers.update({
        "answer__is_valid":         Answer("answer__is_valid",          "VALID"),
        "answer__pfn":              Answer("answer__pfn",               self.pfn,               variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=self.num_pfn_bits),
        "answer__physical_address": Answer("answer__physical_address",  self.physical_address,  variable_kind=Answer.VariableKind.BINARY_OR_HEX, length=(self.num_pfn_bits + self.num_offset_bits)),
      })
    else:
      self.answers.update({
        "answer__is_valid":         Answer("answer__is_valid",          "INVALID"),
        "answer__pfn":              Answer("answer__pfn",               "INVALID"),
        "answer__physical_address": Answer("answer__physical_address",  "INVALID"),
      })
  
  def get_body(self, *args, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        "Given the below information please calculate the equivalent physical address of the given virtual address, filling out all steps along the way.",
        "Remember, we typically have the MSB representing valid or invalid."
      ])
    )
    
    body.add_element(
      ContentAST.Table(
        data=[
          ["Virtual Address", f"0b{self.virtual_address:0{self.num_vpn_bits + self.num_offset_bits}b}"],
          ["# VPN bits", f"{self.num_vpn_bits}"],
          ["# PFN bits", f"{self.num_pfn_bits}"],
        ]
      )
    )
    
    
    # Make values for Page Table
    table_size = self.rng.randint(5,8)
    
    lowest_possible_bottom = max([0, self.vpn - table_size])
    highest_possible_bottom = min([2**self.num_vpn_bits - table_size, self.vpn])
    
    table_bottom = self.rng.randint(lowest_possible_bottom, highest_possible_bottom)
    table_top = table_bottom + table_size
    
    page_table = {}
    page_table[self.vpn] = self.pte
    
    # Fill in the rest of the table
    # for vpn in range(2**self.num_vpn_bits):
    for vpn in range(table_bottom, table_top):
      if vpn == self.vpn: continue
      pte = page_table[self.vpn]
      while pte in page_table.values():
        pte = self.rng.randint(0, 2**self.num_pfn_bits-1)
        if self.rng.choices([True, False], weights=[(1-self.PROBABILITY_OF_VALID), self.PROBABILITY_OF_VALID], k=1)[0]:
          # Randomly set it to be valid
          pte += (2**(self.num_pfn_bits))
      # Once we have a unique random entry, put it into the Page Table
      page_table[vpn] = pte
    
    # Add in ellipses before and after page table entries, if appropriate
    value_matrix = []
    
    if min(page_table.keys()) != 0:
      value_matrix.append(["...", "..."])
    
    value_matrix.extend([
      [f"0b{vpn:0{self.num_vpn_bits}b}", f"0b{pte:0{(self.num_pfn_bits+1)}b}"]
      for vpn, pte in sorted(page_table.items())
    ])
    
    if (max(page_table.keys()) + 1) != 2**self.num_vpn_bits:
      value_matrix.append(["...", "..."])
    
    body.add_element(
      ContentAST.Table(
        headers=["VPN", "PTE"],
        data=value_matrix
      )
    )
    
    body.add_element(
      ContentAST.AnswerBlock([
        
        ContentAST.Answer(self.answers["answer__vpn"],              label="VPN"),
        ContentAST.Answer(self.answers["answer__offset"],           label="Offset"),
        ContentAST.Answer(self.answers["answer__pte"],              label="PTE"),
        ContentAST.Answer(self.answers["answer__is_valid"],         label="VALID or INVALID?"),
        ContentAST.Answer(self.answers["answer__pfn"],              label="PFN"),
        ContentAST.Answer(self.answers["answer__physical_address"], label="Physical Address"),
      ])
    )
    
    return body
  
  def get_explanation_lines(self, *args, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "The core idea of Paging is we want to break the virtual address into the VPN and the offset.  "
        "From here, we get the Page Table Entry corresponding to the VPN, and check the validity of the entry.  "
        "If it is valid, we clear the metadata and attach the PFN to the offset and have our physical address.",
      ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        "Don't forget to pad with the appropriate number of 0s (the appropriate number is the number of bits)!",
      ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        f"Virtual Address = VPN | offset",
        f"<tt>0b{self.virtual_address:0{self.num_vpn_bits+self.num_offset_bits}b}</tt> "
        f"= <tt>0b{self.vpn:0{self.num_vpn_bits}b}</tt> | <tt>0b{self.offset:0{self.num_offset_bits}b}</tt>",
      ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        "We next use our VPN to index into our page table and find the corresponding entry."
        f"Our Page Table Entry is ",
        f"<tt>0b{self.pte:0{(self.num_pfn_bits+1)}b}</tt>"
        f"which we found by looking for our VPN in the page table.",
      ])
    )
    
    if self.is_valid:
      explanation.add_element(
        ContentAST.Paragraph([
          f"In our PTE we see that the first bit is <b>{self.pte // (2**self.num_pfn_bits)}</b> meaning that the translation is <b>VALID</b>"
        ])
      )
    else:
      explanation.add_element(
        ContentAST.Paragraph([
          f"In our PTE we see that the first bit is <b>{self.pte // (2**self.num_pfn_bits)}</b> meaning that the translation is <b>INVALID</b>.",
          "Therefore, we just write \"INVALID\" as our answer.",
          "If it were valid we would complete the below steps.",
          "<hr>"
        ])
      )
    
    explanation.add_element(
      ContentAST.Paragraph([
        "Next, we convert our PTE to our PFN by removing our metadata.  "
        "In this case we're just removing the leading bit.  We can do this by applying a binary mask.",
        f"PFN = PTE & mask",
        f"which is,"
      ])
    )
    explanation.add_element(ContentAST.Equation(
        f"<tt>{self.pfn:0{self.num_pfn_bits}b}</tt> "
        f"= <tt>0b{self.pte:0{self.num_pfn_bits+1}b}</tt> "
        f"& <tt>0b{(2**self.num_pfn_bits)-1:0{self.num_pfn_bits+1}b}</tt>"
      )
    )
    
    explanation.add_elements([
        ContentAST.Paragraph([
          "We then add combine our PFN and offset, "
          "Physical Address = PFN | offset",
        ]),
      ContentAST.Equation(
        f"{'<tt><b>' if self.is_valid else ''}0b{self.physical_address:0{self.num_pfn_bits+self.num_offset_bits}b}{'</b></tt>' if self.is_valid else ''} = <tt>0b{self.pfn:0{self.num_pfn_bits}b}</tt> | <tt>0b{self.offset:0{self.num_vpn_bits}b}</tt>",
      )
      ]
    )
    
    explanation.add_element(
    )
    
    explanation.add_elements([
      ContentAST.Paragraph(["Note: Strictly speaking, this calculation is:",]),
      ContentAST.Equation(
        f"{'<tt><b>' if self.is_valid else ''}0b{self.physical_address:0{self.num_pfn_bits+self.num_offset_bits}b}{'</b></tt>' if self.is_valid else ''} = <tt>0b{self.pfn:0{self.num_pfn_bits}b}{0:0{self.num_offset_bits}}</tt> + <tt>0b{self.offset:0{self.num_offset_bits}b}</tt>",
      ),
      ContentAST.Paragraph(["But that's a lot of extra 0s, so I'm splitting them up for succinctness"])
    ])
    
    return explanation
