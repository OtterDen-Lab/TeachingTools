#!env python
import logging
import math
import random
from typing import List

from TeachingTools.quiz_generation.question import Question, QuestionRegistry, Answer
from TeachingTools.quiz_generation.misc import ContentAST

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class MathQuestion(Question):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class BitsAndBytes(MathQuestion):
  
  MIN_BITS = 3
  MAX_BITS = 49
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.instantiate()
  
  def instantiate(self, rng_seed=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    self.from_binary = 0 == random.randint(0,1)
    self.num_bits = random.randint(self.MIN_BITS, self.MAX_BITS)
    self.num_bytes = int(math.pow(2, self.num_bits))
    
    if self.from_binary:
      self.answers = [Answer("num_bytes", self.num_bytes, Answer.AnswerKind.BLANK)]
    else:
      self.answers = [Answer("num_bits", self.num_bits, Answer.AnswerKind.BLANK)]
  
  def get_body_lines(self, *args, **kwargs) -> List[str]:
    lines = []
    
    lines = [
      f"Given that we have {self.num_bits if self.from_binary else self.num_bytes} {'bits' if self.from_binary else 'bytes'}, "
      f"how many {'bits' if not self.from_binary else 'bytes'} "
      f"{'do we need to address our memory' if not self.from_binary else 'of memory can be addressed'}?"
    ]
    
    lines.extend([
      "",
      f"{'Address space size' if self.from_binary else 'Number of bits in address'}: [{self.answers[0].key}] {'bits' if not self.from_binary else 'bytes'}"
    ])
    
    return lines
  
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    explanation_lines = [
      "Remember that for these problems we use one of these two equations (which are equivalent)",
      "",
      r"- $log_{2}(\text{#bytes}) = \text{#bits}$",
      r"- $2^{(\text{#bits})} = \text{#bytes}$",
      "",
      "Therefore, we calculate:",
    ]
    
    if self.from_binary:
      explanation_lines.extend([
        f"$2 ^ {{{self.num_bits}bits}} = \\textbf{{{self.num_bytes}}}\\text{{bytes}}$"
      ])
    else:
      explanation_lines.extend([
        f"$log_{{2}}({self.num_bytes} \\text{{bytes}}) = \\textbf{{{self.num_bits}}}\\text{{bits}}$"
      ])
    
    return explanation_lines


@QuestionRegistry.register()
class HexAndBinary(MathQuestion):
  
  MIN_HEXITS = 1
  MAX_HEXITS = 8
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.instantiate()
  
  def instantiate(self, rng_seed=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    self.from_binary = random.choice([True, False])
    self.number_of_hexits = random.randint(1, 8)
    self.value = random.randint(1, 16**self.number_of_hexits)
    
    self.hex_val = f"0x{self.value:0{self.number_of_hexits}X}"
    self.binary_val = f"0b{self.value:0{4*self.number_of_hexits}b}"
    
    if self.from_binary:
      self.answers = [Answer("hex_val", self.hex_val, Answer.AnswerKind.BLANK)]
    else:
      self.answers = [Answer("binary_val", self.binary_val, Answer.AnswerKind.BLANK)]
  
  def get_body_lines(self, *args, **kwargs) -> List[str]:
    lines = [
      f"Given the number {self.hex_val if not self.from_binary else self.binary_val} please convert it to {'hex' if self.from_binary else 'binary'}.",
      "Please include base indicator all padding zeros as appropriate (e.g. 0x01 should be 0b00000001",
      "",
      f"Value in {'hex' if self.from_binary else 'binary'}: [{self.answers[0].key}]"
    ]
    return lines
  
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    explanation_lines = [
      "The core idea for converting between binary and hex is to divide and conquer.  "
      "Specifically, each hexit (hexadecimal digit) is equivalent to 4 bits.  "
    ]
    
    if self.from_binary:
      explanation_lines.extend([
        "Therefore, we need to consider each group of 4 bits together and convert them to the appropriate hexit."
      ])
    else:
      explanation_lines.extend([
        "Therefore, we need to consider each hexit and convert it to the appropriate 4 bits."
      ])
    
    binary_str = f"{self.value:0{4*self.number_of_hexits}b}"
    hex_str = f"{self.value:0{self.number_of_hexits}X}"
    explanation_lines.extend(
      self.get_table_generator(
        table_data={
          "0b" : [binary_str[i:i+4] for i in range(0, len(binary_str), 4)],
          "0x" : hex_str
        },
        sorted_keys=["0b", "0x"][::(1 if self.from_binary else -1)],
        add_header_space=False
      )
    )
    if self.from_binary:
      explanation_lines.extend([
        f"Which gives us our hex value of: 0x{hex_str}"
      ])
    else:
      explanation_lines.extend([
        f"Which gives us our binary value of: 0b{binary_str}"
      ])
    
    return explanation_lines


@QuestionRegistry.register()
class AverageMemoryAccessTime(MathQuestion):
  
  CHANCE_OF_99TH_PERCENTILE = 0.75
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.instantiate()
  
  def instantiate(self, rng_seed=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    # Figure out how many orders of magnitude different we are
    orders_of_magnitude_different = self.rng.randint(1,4)
    self.hit_latency = random.randint(1,9)
    self.miss_latency = int(random.randint(1, 9) * math.pow(10, orders_of_magnitude_different))
    
    
    if self.rng.random() < self.CHANCE_OF_99TH_PERCENTILE:
      # Then let's make it very close to 99%
      self.hit_rate = (99 + self.rng.random()) / 100
    else:
      self.hit_rate = random.random()
      
    # Calculate the hit rate
    self.hit_rate = round(self.hit_rate, 4)
    
    # Calculate the AverageMemoryAccessTime (which is the answer itself)
    self.amat = self.hit_rate * self.hit_latency + (1 - self.hit_rate) * self.miss_latency
    
    self.answers = [
      Answer("answer__amat", self.amat, Answer.AnswerKind.BLANK, variable_kind=Answer.VariableKind.FLOAT)
    ]
    
    # Finally, do the randomizing of the question, to avoid these being non-deterministic
    self.show_miss_rate = self.rng.random() > 0.5
    
    # At this point, everything in the question should be set.
    pass
  
  def get_question(self, **kwargs) -> ContentAST.Question:
    # todo: this is a BAD way to do this because we might be missing out on equations and the like.
    return ContentAST.Question(
      body=self.get_body(),
      explanation=self.get_explanation()
    )
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    # Add in background information
    body.add_text_element([
      "Please calculate the Average Memory Access Time given the below information. "
      f"Please round your answer to {Answer.DEFAULT_ROUNDING_DIGITS} decimal points. ",
      f"- Hit Latency: {self.hit_latency} cycles",
      f"- Miss Latency: {self.miss_latency} cycles"
    ])
    
    # Add in either miss rate or hit rate -- we only need one of them
    if self.show_miss_rate:
      body.add_text_element(f"- Miss Rate: {100 * (1 - self.hit_rate): 0.2f}%")
    else:
      body.add_text_element(f"- Hit Rate: {100 * self.hit_rate: 0.2f}%")
    
    # Add in answer line
    body.add_text_element("Average Memory Access Time: [answer__amat]cycles", new_paragraph=True)
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    # Add in General explanation
    explanation.add_text_element([
      "Remember that to calculate the Average Memory Access Time "
      "we weight both the hit and miss times by their relative likelihood.",
      "That is, we calculate:"
    ])
    
    # Add in equations
    ContentAST.Equation.make_block_equation__multiline_equals(
      lhs="AMAT",
      rhs=[
        r"(hit\_rate)*(hit\_cost) + (1 - hit\_rate)*(miss\_cost)",
        f"$({self.hit_rate: 0.4f})*({self.hit_latency}) + ({1 - self.hit_rate: 0.4f})*({self.miss_latency}) = {self.amat: {Answer.DEFAULT_ROUNDING_DIGITS}f}\\text{{cycles}}$"
      ]
    )
    
    return explanation
  