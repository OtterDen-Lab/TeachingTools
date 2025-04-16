#!env python
from __future__ import annotations

import abc
import dataclasses
import datetime
import enum
import importlib
import itertools
import pathlib
import pkgutil
import random
import re
import pypandoc
import yaml
from typing import List, Dict, Any, Tuple, Optional
import canvasapi.course, canvasapi.quiz
import pytablewriter

from TeachingTools.quiz_generation.misc import OutputFormat, Answer

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclasses.dataclass
class TableGenerator:
  headers : List[str] = None
  value_matrix : List[List[str]] = None
  transpose : bool = False # todo: make actually do something
  
  @staticmethod
  def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
      '&': r'\&',
      '%': r'\%',
      '$': r'\$',
      '#': r'\#',
      '_': r'\_',
      '{': r'\{',
      '}': r'\}',
      '~': r'\textasciitilde{}',
      '^': r'\^{}',
      '\\': r'\textbackslash{}',
      '<': r'\textless{}',
      '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)
  
  def generate(self, output_format: OutputFormat) -> str:
    if output_format == OutputFormat.CANVAS:
      html_lines = [
        "<table border=\"1\" style=\"border-collapse: collapse; width: 100%;\">",
      ]
      
      html_lines.append("<tr>")
      if self.headers is not None:
        html_lines.extend([
          f"<th>{header_text}</th>"
          for header_text in self.headers
        ])
      html_lines.append("</tr>")
      
      for row in self.value_matrix:
        html_lines.append("<tr>")
        for val in row:
          html_lines.append(f"<td style=\"padding: 5px;\">{val}</td>")
        html_lines.append("</tr>")
      
      html_lines.append("</table>")
      return '\n'.join(html_lines)
      
    elif output_format == OutputFormat.LATEX:
      table_lines = [
        # r"\begin{table}[h!]",
        # r"\centering",
        r"\begin{tabular}{" + '|l' * len(self.value_matrix[0]) + '|}',
        r"\toprule",
      ]
      if self.headers is not None:
        table_lines.extend([
          ' & '.join([self.tex_escape(element) for element in self.headers]) + r" \\",
          r"\midrule"
        ])
      table_lines.extend([
        ' & '.join([self.tex_escape(element) for element in line]) + r" \\"
        for line in self.value_matrix
      ])
      table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
      ])
      return '\n'.join(table_lines)

  def __str__(self):
    return self.generate(OutputFormat.CANVAS)

class QuestionRegistry:
  _registry = {}
  _scanned = False
  
  @classmethod
  def register(cls, question_type=None):
    def decorator(subclass):
      # Use the provided name or fall back to the class name
      name = question_type.lower() if question_type else subclass.__name__.lower()
      cls._registry[name] = subclass
      return subclass
    return decorator
    
  @classmethod
  def create(cls, question_type, **kwargs):
    """Instantiate a registered subclass."""
    # If we haven't already loaded our premades, do so now
    if not cls._scanned:
      cls.load_premade_questions()
    # Check to see if it's in the registry
    if question_type.lower() not in cls._registry:
      raise ValueError(f"Unknown question type: {question_type}")
    
    return cls._registry[question_type.lower()](**kwargs)
    
    
  @classmethod
  def load_premade_questions(cls):
    package_name = "TeachingTools.quiz_generation.premade_questions"  # Fully qualified package name
    package_path = pathlib.Path(__file__).parent / "premade_questions"
    # log.debug(f"package_path: {package_path}")
    
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
      # Import the module
      module = importlib.import_module(f"{package_name}.{module_name}")
      # log.debug(f"Loaded module: {module.__name__}")


class Question(abc.ABC):
  """
  A question base class that will be able to output questions to a variety of formats.
  """
  
  class Topic(enum.Enum):
    PROCESS = enum.auto()
    MEMORY = enum.auto()
    CONCURRENCY = enum.auto()
    IO = enum.auto()
    PROGRAMMING = enum.auto()
    MATH = enum.auto()
    LANGUAGES = enum.auto()
    SECURITY = enum.auto()
    MISC = enum.auto()
    
    @classmethod
    def from_string(cls, string) -> Question.Topic:
      mappings = {
        member.name.lower() : member for member in cls
      }
      mappings.update({
        "processes": cls.PROCESS,
        "threads": cls.CONCURRENCY,
        "persistance": cls.IO,
        "persistence": cls.IO,
        "programming" : cls.PROGRAMMING,
        "misc": cls.MISC,
      })
      
      if string.lower() in mappings:
        return mappings.get(string.lower())
      return cls.MISC
  
  def __init__(self, name: str, points_value: float, topic: Question.Topic = Topic.MISC, *args, **kwargs):
    if name is None:
      name = self.__class__.__name__
    self.name = name
    self.points_value = points_value
    self.kind = topic
    self.spacing = kwargs.get("spacing", 3)
    
    self.extra_attrs = kwargs # clear page, etc.
    
    self.answers = []
    self.possible_variations = float('inf')
    
    self.rng_seed_offset = kwargs.get("rng_seed_offset", 0)
  
  def get__latex(self, *args, **kwargs):
    concrete_question = self.generate(OutputFormat.LATEX)
    return re.sub(r'\[answer.+]', r"\\answerblank{3}", concrete_question.question_text)

  def get__canvas(self, course: canvasapi.course.Course, quiz : canvasapi.quiz.Quiz, interest_threshold=1.0, *args, **kwargs):
    
    concrete_question = None
    while True:
      concrete_question : ConcreteQuestion = self.generate(
        OutputFormat.CANVAS,
        course=course,
        quiz=quiz,
        previous=(
          None if concrete_question is None
          else concrete_question.question
        )
      )
      if concrete_question.interest >= interest_threshold:
        break
    
    question_type, answers = self.get_answers(*args, **kwargs)
    return {
      "question_name": f"{self.name} ({datetime.datetime.now().strftime('%m/%d/%y %H:%M:%S.%f')})",
      "question_text": concrete_question.question_text,
      "question_type": question_type.value, #e.g. "fill_in_multiple_blanks"
      "points_possible": self.points_value,
      "answers": answers,
      "neutral_comments_html": concrete_question.explanation_text
    }
  
  def get_header(self, output_format : OutputFormat, *args, **kwargs) -> str:
    lines = []
    if output_format == OutputFormat.LATEX:
      lines.extend([
        r"\noindent\begin{minipage}{\textwidth}",
        r"\question{" + str(int(self.points_value)) + r"}",
        r"\noindent\begin{minipage}{0.9\textwidth}",
      ])
    elif output_format == OutputFormat.CANVAS:
      pass
    return '\n'.join(lines)

  def get_footer(self, output_format : OutputFormat, *args, **kwargs) -> str:
    lines = []
    if output_format == OutputFormat.LATEX:
      if self.spacing is not None:
        lines.append(f"\\vspace{{{self.spacing}cm}}")
      lines.extend([
        r"\end{minipage}",
        r"\end{minipage}"
      ])
    elif output_format == OutputFormat.CANVAS:
      pass
    return '\n'.join(lines)

  @staticmethod
  def get_table_generator(
      table_data: Dict[str,List[str]],
      headers: List[str] = None,
      sorted_keys: List[str] = None,
      add_header_space: bool = False,
      hide_keys: bool = False,
      html_out = False
  ) -> List[str|TableGenerator]:
    
    if sorted_keys is None:
      sorted_keys = sorted(table_data.keys())
    if add_header_space and headers is not None:
      headers = [""] + headers
    
    return [
      TableGenerator(
        headers = headers,
        value_matrix=[
          ([key] if not hide_keys else []) + [str(d) for d in table_data[key]]
          for key in sorted_keys
        ])
    ]
  
  @classmethod
  def from_yaml(cls, path_to_yaml):
    with open(path_to_yaml) as fid:
      question_dicts = yaml.safe_load_all(fid)
  
  @abc.abstractmethod
  def get_body_lines(self, *args, **kwargs) -> List[str|TableGenerator]:
    pass
  
  @staticmethod
  def convert_from_lines_to_text(lines, output_format: OutputFormat):
    
    parts = []
    curr_part = ""
    for line in lines:
      if isinstance(line, TableGenerator):
        
        parts.append(
          pypandoc.convert_text(
            curr_part,
            ('html' if output_format == OutputFormat.CANVAS else 'latex'),
            format='md', extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"]
          )
        )
        curr_part = ""
        parts.append('\n' + line.generate(output_format) + '\n')
      else:
        if output_format == OutputFormat.LATEX:
          line = re.sub(r'\[answer\S+]', r"\\answerblank{3}", line)
        curr_part += line + '\n'
    
    parts.append(
      pypandoc.convert_text(
        curr_part,
        ('html' if output_format == OutputFormat.CANVAS else 'latex'),
        format='md', extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"]
      )
    )
    body = '\n'.join(parts)
    if output_format == OutputFormat.LATEX:
      body = re.sub(r'\[answer\S+]', r"\\answerblank{3}", body)
    return body
  
  def get_body(self, output_format:OutputFormat):
    # lines should be in markdown
    lines = self.get_body_lines(output_format=output_format)
    return self.convert_from_lines_to_text(lines, output_format)
    
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    log.warning("get_explanation using default implementation!  Consider implementing!")
    return []
  
  def get_explanation(self, output_format:OutputFormat, *args, **kwargs):
    # lines should be in markdown
    lines = self.get_explanation_lines(*args, **kwargs)
    return self.convert_from_lines_to_text(lines, output_format)
  
  def get_answers(self, *args, **kwargs) -> Tuple[Answer.AnswerKind, List[Dict[str,Any]]]:
    # log.warning("get_answers using default implementation!  Consider implementing!")
    return Answer.AnswerKind.BLANK, list(itertools.chain(*[a.get_for_canvas() for a in self.answers]))

  def instantiate(self, rng_seed=None, *args, **kwargs):
    """If it is necessary to regenerate aspects between usages, this is the time to do it
    :param rng_seed: random number generator seed to use when regenerating question
    :param *args:
    :param **kwargs:
    """
    self.answers = []
    if rng_seed is None:
      random.seed(rng_seed)
    else:
      random.seed(rng_seed + self.rng_seed_offset)
    
  def generate(self, output_format: OutputFormat, rng_seed=None, *args, **kwargs) -> ConcreteQuestion:
    # Renew the problem as appropriate
    self.instantiate(rng_seed, *args, **kwargs)
    
    # while (not self.is_interesting()):
    #   log.debug("Still not interesting...")
    #   self.instantiate()
    
    question_body = self.get_header(output_format)
    question_explanation = ""
    
    # Generation body and explanation based on the output format
    if output_format == OutputFormat.CANVAS:
      question_body += self.get_body(output_format)
      question_explanation = pypandoc.convert_text(self.get_explanation(output_format, *args, **kwargs), 'html', format='md', extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"])
    elif output_format == OutputFormat.LATEX:
      question_body += self.get_body(output_format)
    question_body += self.get_footer(output_format)
    
    # Return question body, explanation, and answers
    return ConcreteQuestion(
      question_text=question_body,
      answer_text=self.get_answers(),
      explanation_text=question_explanation,
      value=self.points_value,
      interest=(1.0 if self.is_interesting() else 0.0),
      question=self
    )
  
  def is_interesting(self) -> bool:
    return True

  @classmethod
  def make_block_equation(cls, str):
    return "\n" r"$$ \displaystyle " + str + r"\frac{}{}$$" "\n"

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
    
    return cls.make_block_equation('\n'.join(equation_lines))
    

@dataclasses.dataclass
class ConcreteQuestion():
  question_text : str
  answer_text : str
  explanation_text : str
  value: float
  interest : float
  question: Question


class QuestionGroup():
  
  def __init__(self, questions_in_group: List[Question], pick_once : bool):
    self.questions = questions_in_group
    self.pick_once = pick_once
  
    self._current_question : Optional[Question] = None
    
  def instantiate(self, *args, **kwargs):
    
    # todo: Make work with rng_seed (or at least verify)
    random.seed(kwargs.get("rng_seed", None))
    
    if not self.pick_once or self._current_question is None:
      self._current_question = random.choice(self.questions)
    
    self._current_question.instantiate(*args, **kwargs)
    
    
  def __getattr__(self, name):
    if self._current_question is None or name == "generate":
      self.instantiate()
    try:
      attr = getattr(self._current_question, name)
    except AttributeError:
      raise AttributeError(
        f"Neither QuestionGroup nor Question has attribute '{name}'"
      )
    
    if callable(attr):
      def wrapped_method(*args, **kwargs):
        return attr(*args, **kwargs)
      return wrapped_method
    
    return attr