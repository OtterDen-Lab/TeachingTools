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

from TeachingTools.quiz_generation.misc import OutputFormat, Answer, TextAST

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
  def get_question(self, **kwargs) -> TextAST.Question:
    pass
  
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
    
  def is_interesting(self) -> bool:
    return True

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