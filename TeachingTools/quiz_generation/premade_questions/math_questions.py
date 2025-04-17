#!env python
import logging
import math
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
    self.refresh()
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Generate the important parts of the problem
    self.from_binary = (0 == self.rng.randint(0,1))
    self.num_bits = self.rng.randint(self.MIN_BITS, self.MAX_BITS)
    self.num_bytes = int(math.pow(2, self.num_bits))
    
    if self.from_binary:
      self.answers = {"answer" : Answer("num_bytes", self.num_bytes, Answer.AnswerKind.BLANK)}
    else:
      self.answers = {"answer" : Answer("num_bits", self.num_bits, Answer.AnswerKind.BLANK)}
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    body.add_text_element([
      f"Given that we have {self.num_bits if self.from_binary else self.num_bytes} {'bits' if self.from_binary else 'bytes'}, "
      f"how many {'bits' if not self.from_binary else 'bytes'} "
      f"{'do we need to address our memory' if not self.from_binary else 'of memory can be addressed'}?"
    ])
    
    if self.from_binary:
      body.add_elements([
        ContentAST.Text("Address space size: "),
        ContentAST.Answer(self.answers['answer']),
        ContentAST.Text("Bytes")
      ])
    else:
      body.add_elements([
        ContentAST.Text("Number of bits in address: "),
        ContentAST.Answer(self.answers['answer']),
        ContentAST.Text("bits")
      ])
      
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_text_element(
      "Remember that for these problems we use one of these two equations (which are equivalent)"
    )
    explanation.add_elements([
      ContentAST.Equation(r"log_{2}(\text{#bytes}) = \text{#bits}"),
      ContentAST.Equation(r"2^{(\text{#bits})} = \text{#bytes}")
    ])
    
    explanation.add_text_element("Therefore, we calculate:")
    
    if self.from_binary:
      explanation.add_element(
        ContentAST.Equation(f"2 ^ {{{self.num_bits}bits}} = \\textbf{{{self.num_bytes}}}\\text{{bytes}}")
      )
    else:
      explanation.add_element(
        ContentAST.Equation(f"log_{{2}}({self.num_bytes} \\text{{bytes}}) = \\textbf{{{self.num_bits}}}\\text{{bits}}")
      )
    
    return explanation

@QuestionRegistry.register()
class HexAndBinary(MathQuestion):
  
  MIN_HEXITS = 1
  MAX_HEXITS = 8
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.refresh()
  
  def refresh(self, **kwargs):
    super().refresh(**kwargs)
    
    self.from_binary = self.rng.choice([True, False])
    self.number_of_hexits = self.rng.randint(1, 8)
    self.value = self.rng.randint(1, 16**self.number_of_hexits)
    
    self.hex_val = f"0x{self.value:0{self.number_of_hexits}X}"
    self.binary_val = f"0b{self.value:0{4*self.number_of_hexits}b}"
    
    if self.from_binary:
      self.answers['answer'] = Answer("hex_val", self.hex_val, Answer.AnswerKind.BLANK)
    else:
      self.answers['answer'] = Answer("binary_val", self.binary_val, Answer.AnswerKind.BLANK)
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_text_element([
      f"Given the number {self.hex_val if not self.from_binary else self.binary_val} please convert it to {'hex' if self.from_binary else 'binary'}.",
      "Please include base indicator all padding zeros as appropriate (e.g. 0x01 should be 0b00000001",
      "",
      f"Value in {'hex' if self.from_binary else 'binary'}: "
    ])
    
    body.add_element(ContentAST.Answer(self.answers['answer']))
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_text_element([
      "The core idea for converting between binary and hex is to divide and conquer.  "
      "Specifically, each hexit (hexadecimal digit) is equivalent to 4 bits.  "
    ])
    
    if self.from_binary:
      explanation.add_text_element(
        "Therefore, we need to consider each group of 4 bits together and convert them to the appropriate hexit."
      )
    else:
      explanation.add_text_element(
        "Therefore, we need to consider each hexit and convert it to the appropriate 4 bits."
      )
    
    binary_str = f"{self.value:0{4*self.number_of_hexits}b}"
    hex_str = f"{self.value:0{self.number_of_hexits}X}"
    
    explanation.add_element(
      ContentAST.Table(
        data=[
          ["0b"] + [binary_str[i:i+4] for i in range(0, len(binary_str), 4)],
          ["0x"] + list(hex_str)
        ],
      )
    )
    
    if self.from_binary:
      explanation.add_text_element(
        f"Which gives us our hex value of: 0x{hex_str}"
      )
    else:
      explanation.add_text_element(
        f"Which gives us our binary value of: 0b{binary_str}"
      )
      
    return explanation
  


@QuestionRegistry.register()
class AverageMemoryAccessTime(MathQuestion):
  
  CHANCE_OF_99TH_PERCENTILE = 0.75
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.refresh()
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    # Figure out how many orders of magnitude different we are
    orders_of_magnitude_different = self.rng.randint(1,4)
    self.hit_latency = self.rng.randint(1,9)
    self.miss_latency = int(self.rng.randint(1, 9) * math.pow(10, orders_of_magnitude_different))
    
    # Add in a complication of making it sometimes very, very close
    if self.rng.self.rng() < self.CHANCE_OF_99TH_PERCENTILE:
      # Then let's make it very close to 99%
      self.hit_rate = (99 + self.rng.self.rng()) / 100
    else:
      self.hit_rate = self.rng.self.rng()
      
    # Calculate the hit rate
    self.hit_rate = round(self.hit_rate, 4)
    
    # Calculate the AverageMemoryAccessTime (which is the answer itself)
    self.amat = self.hit_rate * self.hit_latency + (1 - self.hit_rate) * self.miss_latency
    
    self.answers = {
      "amat": Answer("answer__amat", self.amat, Answer.AnswerKind.BLANK, variable_kind=Answer.VariableKind.FLOAT)
    }
    
    # Finally, do the self.rngizing of the question, to avoid these being non-deterministic
    self.show_miss_rate = self.rng.self.rng() > 0.5
    
    # At this point, everything in the question should be set.
    pass
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    # Add in background information
    body.add_elements([
      ContentAST.Text("Please calculate the Average Memory Access Time given the below information. "),
      ContentAST.Text(
        f"Please round your answer to {Answer.DEFAULT_ROUNDING_DIGITS} decimal points. ",
        hide_from_latex=True
      ),
      ContentAST.Text(f"- Hit Latency: {self.hit_latency} cycles"),
        ContentAST.Text(f"- Miss Latency: {self.miss_latency} cycles")
    ])
    
    # Add in either miss rate or hit rate -- we only need one of them
    if self.show_miss_rate:
      body.add_text_element(f"- Miss Rate: {100 * (1 - self.hit_rate): 0.2f}%")
    else:
      body.add_text_element(f"- Hit Rate: {100 * self.hit_rate: 0.2f}%")
    
    # Add in answer line
    body.add_elements([
        ContentAST.Text("Average Memory Access Time:"),
        ContentAST.Answer(self.answers["amat"]),
        ContentAST.Text("cycles")
      ],
      new_paragraph=True
    )
    
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
  