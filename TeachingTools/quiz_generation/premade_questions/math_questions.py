#!env python
import logging
import math
import random
from typing import List

from TeachingTools.quiz_generation.question import Question, QuestionRegistry, Answer

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
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.instantiate()
  
  def instantiate(self, rng_seed=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    orders_of_magnitude_different = random.randint(1,4)
    self.hit_latency = random.randint(1,9)
    self.miss_latency = int(random.randint(1, 9) * math.pow(10, orders_of_magnitude_different))
    
    if random.random() > 0.5:
      # Then let's make it very close to 99%
      self.hit_rate = (99 + random.random()) / 100
    else:
      self.hit_rate = random.random()
    self.hit_rate = round(self.hit_rate, 4)
    self.amat = self.hit_rate * self.hit_latency + (1 - self.hit_rate) * self.miss_latency
    
    self.answers = [
      Answer("amat", self.amat, Answer.AnswerKind.BLANK)
    ]
  
  def get_body_lines(self, *args, **kwargs) -> List[str]:
    lines = [
      "Please calculate the Average Memory Access Time given the below information.  Please round your answer to 2 decimal points.",
      "",
    ]
    
    info_lines = [
      f"- Hit Latency: {self.hit_latency} cycles",
      f"- Miss Latency: {self.miss_latency} cycles",
    ]
    if random.random() > 0.5:
      info_lines.append(f"- Hit Rate: {100 * self.hit_rate: 0.2f}%")
    else:
      info_lines.append(f"- Miss Rate: {100 * (1 - self.hit_rate): 0.2f}%")
    
    lines.extend(random.sample(info_lines, len(info_lines)))
    lines.append("")
    
    lines.extend([
      "",
      "Average Memory Access Time: [amat]cycles"
    ])
    
    return lines
  
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    lines = [
      "Remember that to calculate the Average Memory Access Time we weight both the hit and miss times by their relative likelihood.",
      "That is, we calculate `(hit_rate)*(hit_cost) + (1 - hit_rate)*(miss_cost)`."
      "",
      "In this case, that calculation becomes:",
      f"$({self.hit_rate: 0.4f})*({self.hit_latency}) + ({1 - self.hit_rate: 0.4f})*({self.miss_latency}) = {self.amat:0.2f}\\text{{cycles}}$"
    ]
    return lines
