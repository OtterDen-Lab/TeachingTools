#!env python
from __future__ import annotations

import re
import fractions
import itertools
import math
from typing import List, Dict

import enum

import pypandoc
import pylatex

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class OutputFormat(enum.Enum):
  LATEX = enum.auto(),
  CANVAS = enum.auto()


class Answer():
  DEFAULT_ROUNDING_DIGITS = 4
  
  class AnswerKind(enum.Enum):
    BLANK = "fill_in_multiple_blanks_question"
    MULTIPLE_ANSWER = "multiple_answers_question"  # todo: have baffles?
    ESSAY = "essay_question"
    
  class VariableKind(enum.Enum): # todo: use these for generate variations?
    STR = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()
    BINARY = enum.auto()
    HEX = enum.auto()
    BINARY_OR_HEX = enum.auto()
    AUTOFLOAT = enum.auto()
    LIST = enum.auto()
    
    
  def __init__(
      self, key:str,
      value,
      kind : Answer.AnswerKind = AnswerKind.BLANK,
      variable_kind : Answer.VariableKind = VariableKind.STR,
      display=None,
      length=None,
      correct=True
  ):
    self.key = key
    self.value = value
    self.kind = kind
    self.variable_kind = variable_kind
    self.display = display if display is not None else value
    self.length = length # Used for bits and hex to be printed appropriately
    self.correct = correct
  
  def get_for_canvas(self) -> List[Dict]:
    if self.variable_kind == Answer.VariableKind.FLOAT:
      return [{
        "blank_id": self.key,
        "answer_text": f"{self.value:0.{self.DEFAULT_ROUNDING_DIGITS}f}",
        "answer_weight": 100,
      }]
    elif self.variable_kind == Answer.VariableKind.BINARY:
      return [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.HEX:
      return [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{(self.length // 8) + 1 if self.length is not None else 0}X}",
          "answer_weight": 100,
        },{
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{(self.length // 8) + 1 if self.length is not None else 0}X}",
          "answer_weight": 100,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.BINARY_OR_HEX:
      return [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{math.ceil(self.length / 8) if self.length is not None else 0}X}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{math.ceil(self.length / 8) if self.length is not None else 0}X}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value}",
          "answer_weight": 100,
        },
      
      ]
    elif self.variable_kind == Answer.VariableKind.AUTOFLOAT:
      value_fraction = fractions.Fraction(self.value).limit_denominator(3*4*5) # For process questions, these are the numbers of jobs we'd have
      
      possible_answers = [{
        "blank_id": self.key,
        "answer_text": value_fraction,
        "answer_weight": 100,
      }]
      if not value_fraction.is_integer():
        possible_answers.extend([
          {
            "blank_id": self.key,
            "answer_text": f"{value_fraction.numerator / value_fraction.denominator:0.{self.DEFAULT_ROUNDING_DIGITS}f}",
            "answer_weight": 100,
          },
          {
            "blank_id": self.key,
            "answer_text":
              f"{value_fraction.numerator // value_fraction.denominator} {value_fraction.numerator % value_fraction.denominator}/{value_fraction.denominator}",
            "answer_weight": 100,
          },
        ])
      
      return possible_answers
    elif self.variable_kind == Answer.VariableKind.LIST:
      possible_answers = [
        {
          "blank_id": self.key,
          "answer_text": ','.join(map(str, possible_state)),
          "answer_weight": 100,
        }
        for possible_state in [self.value] #itertools.permutations(self.value)
      ]
      
      return possible_answers
    
    canvas_answer = {
      "blank_id": self.key,
      "answer_text": self.value,
      "answer_weight": 100 if self.correct else 0,
    }
    return [canvas_answer]


class ContentAST:
  
  class Element:
    def __init__(self, elements=None):
      self.elements : List[ContentAST.Element] = elements or []
    
    def __str__(self):
      return self.render_markdown()
    
    def add_element(self, element):
      self.elements.append(element)
    
    def add_elements(self, elements, new_paragraph=False):
      if new_paragraph:
        self.elements.append(ContentAST.Text(""))
      self.elements.extend(elements)
    
    def add_text_element(self, content: str|List[str], new_paragraph=False):
      """Helper function to add text to reduce the amount of boilerplate"""
      # todo this function is probably now unpredictable.  Whoops.
      #  it should be fixed by changing it to always have the same impact on the AST (i.e. always a TextBlock)
      if new_paragraph:
        self.add_element(ContentAST.Text(""))
      if isinstance(content, str):
        self.add_element(ContentAST.Text(content))
      else:
        self.add_element(
          ContentAST.TextBlock(map(lambda c: ContentAST.Text(c), content))
        )
          
    def convert_markdown(self, str_to_convert, output_format):
      try:
        output = pypandoc.convert_text(
          str_to_convert,
          output_format,
          format='md',
          extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"]
        )
        if output_format == "html":
          output = re.sub(r'^<p>(.*)</p>$', r'\1', output, flags=re.DOTALL)
        return output
      except RuntimeError as e:
        log.warning(f"Specified conversion format '{output_format}' not recognized by pypandoc. Defaulting to markdown")
      return None
    
    def render(self, output_format):
      method_name = f"render_{output_format}"
      if hasattr(self, method_name):
        return getattr(self, method_name)()
      
      return self.render_markdown()  # Fallback to markdown
    
    def render_markdown(self):
      return "".join(element.render("markdown") for element in self.elements)
    
    def render_html(self):
      html = "".join(element.render("html") for element in self.elements)
      return html
    
    def render_latex(self):
      return "".join(element.render("latex") for element in self.elements)
  
    def is_mergeable(self, other: ContentAST.Element):
      return False
  
  class Section(Element):
    """A child class representing a specific section of a question"""
    pass
  
  class Text(Element):
    def __init__(self, content : str, hide_from_latex=False, emphasis=False):
      super().__init__()
      self.content = content
      self.hide_from_latex = hide_from_latex
      self.emphasis = emphasis
    
    def render_markdown(self):
      return f"{'***' if self.emphasis else ''}{self.content}{'***' if self.emphasis else ''}"

    def render_html(self):
      # If the super function returns None then we just return content as is
      return super().convert_markdown(self.content, "html") or self.content
    
    def render_latex(self):
      if self.hide_from_latex:
        return ""
      content = super().convert_markdown(self.content, "latex") or self.content
      return content
    
    def is_mergeable(self, other: ContentAST.Element):
      if not isinstance(other, ContentAST.Text):
        return False
      if self.hide_from_latex != other.hide_from_latex:
        return False
      return True
    
    def merge(self, other: ContentAST.Text):
      self.content = self.render_markdown() + " " + other.render_markdown()
      self.emphasis = False
  
  class TextBlock(Element):
    """A block of text that will combine all child elements together.  i.e. a paragraph"""
    def render(self, output_format):
      # Merge all Text nodes before we render
      merged_elements = []
      previous_node = ContentAST.Text("")
      for elem in self.elements:
        if not previous_node.is_mergeable(elem):
          # Then we can't merge, so we should move on
          merged_elements.append(previous_node)
          previous_node = elem
          continue
        
        # Otherwise, we can merge
        previous_node.merge(elem)
      merged_elements.append(previous_node)
      
      self.elements = merged_elements
      
      return "\n\n" + super().render(output_format)
      
  class Equation(Element):
    def __init__(self, latex):
      super().__init__()
      self.latex = latex
    
    def render_markdown(self):
      return r"$$ \displaystyle " + f"{self.latex}" + r" \frac{}{}$$"
    
    def render_html(self):
      return f"<div class='math'>$$ \\displaystyle{self.latex} \\frac{{}}{{}}$$</div>"
    
    def render_latex(self):
      return f"\\begin{{equation}}\n{self.latex}\n\\end{{equation}}"
  
    @classmethod
    def make_block_equation__multiline_equals(cls, lhs : str, rhs : List[str]):
      equation_lines = []
      equation_lines.extend([
        r"\begin{array}{l}",
        f"{lhs} = {rhs[0]} \\\\",
      ])
      equation_lines.extend([
        f"\\phantom{{{lhs}}} = {eq} \\\\"
        for eq in rhs[1:]
      ])
      equation_lines.extend([
        r"\end{array}",
      ])
      
      return cls('\n'.join(equation_lines))
    
  class Table(Element):
    def __init__(self, data, headers=None, alignments=None, padding=False, transpose=False):
      """
      Generates a ContentAST.Table element
      :param data: data in the table.  List[List[Element]]
      :param headers: headers for the tables
      :param alignments: how should columns be aligned
      :param padding: Add in padding around cells in html table
      :param transpose: apple transpose to table
      """
      # todo: fix alignments
      # todo: implement transpose
      super().__init__()
      self.data = data
      self.headers = headers
      self.alignments = alignments
      self.padding = padding
    
    def render_markdown(self):
      # Basic markdown table implementation
      result = []
      
      if self.headers:
        result.append("| " + " | ".join(str(h) for h in self.headers) + " |")
        
        if self.alignments:
          align_row = []
          for align in self.alignments:
            if align == "left":
              align_row.append(":---")
            elif align == "right":
              align_row.append("---:")
            else:  # center
              align_row.append(":---:")
          result.append("| " + " | ".join(align_row) + " |")
        else:
          result.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
      
      for row in self.data:
        result.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
      return "\n".join(result)
    
    def render_html(self):
      # HTML table implementation
      result = ["<table border=\"1\" style=\"border-collapse: collapse; width: 100%;\">"]
      
      if self.headers:
        result.append("  <thead>")
        result.append("    <tr>")
        for i, header in enumerate(self.headers):
          align_attr = ""
          if self.alignments and i < len(self.alignments):
            align_attr = f' align="{self.alignments[i]}"'
          result.append(f"      <th{align_attr}>{header}</th>")
        result.append("    </tr>")
        result.append("  </thead>")
      
      result.append("  <tbody>")
      for row in self.data:
        result.append("    <tr>")
        for i, cell in enumerate(row):
          if isinstance(cell, ContentAST.Element):
            cell = cell.render_html()
          align_attr = ""
          if self.alignments and i < len(self.alignments):
            align_attr = f' align="{self.alignments[i]}"'
          result.append(f"      <td  style=\"padding: {'5px' if self.padding else '0x'} ; {align_attr};\">{cell}</td>")
        result.append("    </tr>")
      result.append("  </tbody>")
      result.append("</table>")
      
      return "\n".join(result)
    
    def render_latex(self):
      # LaTeX table implementation
      if self.alignments:
        col_spec = "".join("l" if a == "left" else "r" if a == "right" else "c"
                           for a in self.alignments)
      else:
        col_spec = '|'.join(["c"] * (len(self.headers) if self.headers else len(self.data[0])))
      
      result = [f"\\begin{{tabular}}{{{col_spec}}}"]
      result.append("\\hline")
      
      if self.headers:
        result.append(" & ".join(pylatex.escape_latex(str(h)) for h in self.headers) + " \\\\")
        result.append("\\hline")
      
      for row in self.data:
        rendered_row = [cell.render_latex() if isinstance(cell, ContentAST.Element) else cell for cell in row]
        result.append(" & ".join(rendered_row) + " \\\\")
      result.append("\\hline")
      result.append("\\end{tabular}")
      
      return "\n\n" + "\n".join(result)
  
  class Picture(Element):
    def __init__(self, path, caption=None, width=None):
      super().__init__()
      self.path = path
      self.caption = caption
      self.width = width
    
    def render_markdown(self):
      if self.caption:
        return f"![{self.caption}]({self.path})"
      return f"![]({self.path})"
    
    def render_html(self):
      attrs = []
      if self.width:
        attrs.append(f'width="{self.width}"')
      
      img = f'<img src="{self.path}" {" ".join(attrs)} alt="{self.caption or ""}">'
      
      if self.caption:
        return f'<figure>\n  {img}\n  <figcaption>{self.caption}</figcaption>\n</figure>'
      return img
    
    def render_latex(self):
      options = []
      if self.width:
        options.append(f"width={self.width}")
      
      result = ["\\begin{figure}[h]"]
      result.append(f"\\centering")
      result.append(f"\\includegraphics[{','.join(options)}]{{{self.path}}}")
      
      if self.caption:
        result.append(f"\\caption{{{self.caption}}}")
      
      result.append("\\end{figure}")
      return "\n".join(result)
  
  class Question(Element):
    def __init__(
        self,
        body: ContentAST.Section,
        explanation: ContentAST.Section,
        name=None,
        value=1,
        interest=1.0
    ):
      super().__init__()
      self.name = name
      self.explanation = explanation
      self.body = body
      self.value = value
      self.interest = interest
      
    def render(self, output_format):
      # Generate content from all elements
      content = self.body.render(output_format)
      
      # If output format is latex, add in minipage and question environments
      if output_format == "latex":
        latex_lines = [
          r"\noindent\begin{minipage}{\textwidth}",
          r"\question{" + str(int(self.value)) + r"}",
          r"\noindent\begin{minipage}{0.9\textwidth}",
          content,
          r"\end{minipage}",
          r"\end{minipage}"
        ]
        content = '\n'.join(latex_lines)
        
      return content
    
  class Answer(Element):
    def __init__(self, answer : Answer, trailing_text: str = ""):
      super().__init__()
      self.answer = answer
      self.trailing_text = trailing_text
    
    def render_markdown(self):
      return f"[{self.answer.key}] {self.trailing_text}".rstrip()
    
    def render_html(self):
      return self.render_markdown()
    
    def render_latex(self):
      return r"\answerblank{2}"
  
  class Document(Element):
    """Root document class that adds document-level rendering"""
    def __init__(self, title=None):
      super().__init__()
      self.title = title
    
    def render(self, output_format):
      # Generate content from all elements
      content = super().render(output_format)
      
      # Add title if present
      if self.title and output_format == "markdown":
        content = f"# {self.title}\n\n{content}"
      elif self.title and output_format == "html":
        content = f"<h1>{self.title}</h1>\n{content}"
      elif self.title and output_format == "latex":
        content = f"\\section{{{self.title}}}\n{content}"
      
      return content
  