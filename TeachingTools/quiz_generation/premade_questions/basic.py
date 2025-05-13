#!env python
from __future__ import annotations

import datetime
import inspect
import pprint
import random
import re

import canvasapi.course
import canvasapi.quiz
import pypandoc

import yaml
from typing import List, Dict, Any, Tuple
import jinja2

import logging

from TeachingTools.quiz_generation.misc import OutputFormat, ContentAST
from TeachingTools.quiz_generation.question import Question, QuestionRegistry, Answer, TableGenerator
from TeachingTools.quiz_generation.premade_questions.exam_generation_functions import QuickFunctions

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@QuestionRegistry.register()
class FromText(Question):
  
  def __init__(self, *args, text, **kwargs):
    super().__init__(*args, **kwargs)
    self.text = text
    self.answers = []
    self.possible_variations = 1
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    return ContentAST.Section([ContentAST.Text(self.text)])
  
  def get_answers(self, *args, **kwargs) -> Tuple[Answer.AnswerKind, List[Dict[str,Any]]]:
    return Answer.AnswerKind.ESSAY, []


@QuestionRegistry.register()
class FromGenerator(FromText):
  
  def __init__(self, *args, generator=None, text=None, **kwargs):
    if generator is None and text is None:
      raise TypeError(f"Must supply either generator or text kwarg for {self.__class__.__name__}")
    
    if generator is None:
      generator = text
    
    super().__init__(*args, text="", **kwargs)
    self.possible_variations = kwargs.get("possible_variations", float('inf'))
    
    def attach_function_to_object(obj, function_code, function_name='get_body_lines'):
      # log.debug(f"\ndef {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()))
      
      # Define the function dynamically using exec
      exec(f"def {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()), globals(), locals())
      
      # Get the function and bind it to the object
      function = locals()[function_name]
      setattr(obj, function_name, function.__get__(obj))
    
    self.generator_text = generator
    # Attach the function dynamically
    attach_function_to_object(self, generator, "generator")
    
    self.answers = {}
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    return super().get_body()
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    try:
      generated_text = self.generator()
      self.text = generated_text
    except TypeError as e:
      log.error(f"Error generating from text: {e}")
      log.debug(self.generator_text)
      exit(8)


class TrueFalse(FromText):
  pass