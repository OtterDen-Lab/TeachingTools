#!env python
from __future__ import annotations

import datetime
import inspect
import pprint
import random

import canvasapi.course
import canvasapi.quiz

import yaml
from typing import List, Dict, Any, Tuple
import jinja2

import logging

from TeachingTools.quiz_generation.misc import OutputFormat
from TeachingTools.quiz_generation.question import Question, QuestionRegistry, Answer, TableGenerator
from TeachingTools.quiz_generation.premade_questions.exam_generation_functions import QuickFunctions

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

@QuestionRegistry.register()
class Question_legacy(Question):
  _jinja_env = None
  
  def __init__(self, name: str, value: float, topic: Question.Topic, text: str, *args, **kwargs):
    super().__init__(name, value, topic, *args, **kwargs)
    self.text = text
    
    if self._jinja_env is None:
      self._jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("templates"),
        block_start_string='<BLOCK>',
        block_end_string='</BLOCK>',
        variable_start_string='<VAR>',
        variable_end_string='</VAR>',
        comment_start_string='<COMMENT>',
        comment_end_string='</COMMENT>',
      )
      functions = [method for name, method in inspect.getmembers(QuickFunctions, lambda m: inspect.ismethod(m))]
      for func in functions:
        self._jinja_env.globals[func.__name__] = getattr(QuickFunctions, func.__name__)
  
  def get_body_lines(self, output_format: OutputFormat, *args, **kwargs) -> List[str]:
    lines = []
    if output_format == OutputFormat.LATEX:
      lines.extend([
        self._jinja_env.from_string(self.text.replace("[answer]", "\\answerblank{3}")).render() #.replace('[', '{[').replace(']', ']}')
      ])
      if self.extra_attrs.get("clear_page", False):
        lines.append(r"\vspace{10cm}")
    elif output_format == OutputFormat.CANVAS:
      lines.extend([
        self._jinja_env.from_string(self.text).render().replace(r"\answerblank{3}", "[answer]"),
      ])
    
    return lines
  
  def get__canvas(self, course: canvasapi.course.Course, quiz : canvasapi.quiz.Quiz, *args, **kwargs):
    
    question_text, explanation_text, answers = self.generate(OutputFormat.CANVAS, *args, **kwargs)
    def replace_answers(input_str):
      counter = 1
      replacements = []
      while "[answer]" in input_str:
        placeholder = f"answer{counter}"
        input_str = input_str.replace("[answer]", f"[{placeholder}]", 1)
        replacements.append(placeholder)
        counter += 1
      return input_str, replacements
    
    question_text, occurances = replace_answers(question_text)
    answers = [
      {
        "blank_id" : a,
        "answer_text" : str(random.random()) # make it so there is always an answer
      }
      for a in occurances
    ]
    
    # question_type, answers = self.get_answers(*args, **kwargs)
    return {
      "question_name": f"{self.name} ({datetime.datetime.now().strftime('%m/%d/%y %H:%M:%S.%f')})",
      "question_text": question_text,
      "question_type": Answer.AnswerKind.BLANK.value, #e.g. "fill_in_multiple_blanks"
      "points_possible": 1,
      "answers": answers,
      "neutral_comments_html": explanation_text
    }
  
  def get_answers(self, *args, **kwargs) -> Tuple[str, List[Dict[str,Any]]]:
    answers = []
    answers.append({
      "blank_id": "answer",
      "answer_text": "variation",
      "answer_weight": 100,
    })
    return "fill_in_multiple_blanks_question", answers
  
  
  @classmethod
  def from_yaml(cls, path_to_yaml):
    questions = []
    with open(path_to_yaml) as fid:
      question_dicts = yaml.safe_load_all(fid)
      for question_dict in question_dicts:
        log.debug(question_dict)
        if not question_dict.get("enabled", True):
          continue
        extra_attrs = {
          key : question_dict[key]
          for key in question_dict.keys()
          if key not in ["value", "topic", "kind", "text", "name"]
        }
        repeat = question_dict.get("repeat", 1)
        for _ in range(repeat):
          questions.append(
            cls(
              name=question_dict.get("name", "(question)"),
              value=question_dict["value"],
              topic=Question.Topic.from_string(question_dict["subject"]),
              text=question_dict["text"],
              **extra_attrs
            )
          )
    return questions
  
  def get_explanation_lines(self, *args, **kwargs) -> List[str]:
    return []


@QuestionRegistry.register()
class Question_autoyaml(Question):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.input_vars = {}
    self.intermediate_vars = {}
  
  def get_body_lines(self, *args, **kwargs) -> List[str]:
    # will be overwritten by the loading (hopefully)
    pass
  
  
  @classmethod
  def from_yaml(cls, path_to_yaml):
    with open(path_to_yaml) as fid:
      question_dicts = list(yaml.safe_load_all(fid))
    
    questions = []
    for question_dict in question_dicts:
      log.debug(pprint.pformat(question_dict))
      q = Question_autoyaml(
        name=question_dict.get("name", "AutoYaml"),
        value=question_dict.get("value", 1),
        type=question_dict.get("category", 'misc')
      )
      
      # Use exec to attach the function to the object
      def attach_function_to_object(obj, function_code, function_name='get_body_lines'):
        log.debug(f"\ndef {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()))
        
        # Define the function dynamically using exec
        exec(f"def {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()), globals(), locals())
        
        # Get the function and bind it to the object
        function = locals()[function_name]
        setattr(obj, function_name, function.__get__(obj))
      
      # Attach the function dynamically
      attach_function_to_object(q, question_dict["functions"]["instantiate"], "instantiate")
      attach_function_to_object(q, question_dict["functions"]["get_body_lines"], "get_body_lines")
      attach_function_to_object(q, question_dict["functions"]["get_explanation_lines"], "get_explanation_lines")
      attach_function_to_object(q, question_dict["functions"]["get_answers"], "get_answers")
      
      questions.append(q)
    return questions


@QuestionRegistry.register()
class FromText(Question):
  
  def __init__(self, *args, text, **kwargs):
    super().__init__(*args, **kwargs)
    self.text = text
    self.answers = []
    self.possible_variations = 1
  
  def get_body_lines(self, *args, **kwargs) -> List[str|TableGenerator]:
    return [self.text]
  
  def get_answers(self, *args, **kwargs) -> Tuple[Answer.AnswerKind, List[Dict[str,Any]]]:
    return Answer.AnswerKind.ESSAY, []


@QuestionRegistry.register()
class FromGenerator(FromText):
  
  def __init__(self, *args, generator, **kwargs):
    super().__init__(*args, text="", **kwargs)
    self.possible_variations = kwargs.get("possible_variations", float('inf'))
    
    def attach_function_to_object(obj, function_code, function_name='get_body_lines'):
      log.debug(f"\ndef {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()))
      
      # Define the function dynamically using exec
      exec(f"def {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()), globals(), locals())
      
      # Get the function and bind it to the object
      function = locals()[function_name]
      setattr(obj, function_name, function.__get__(obj))
    
    self.generator_text = generator
    # Attach the function dynamically
    attach_function_to_object(self, generator, "generator")
    
    self.answers = []
  
  def instantiate(self, *args, **kwargs):
    super().instantiate()
    try:
      self.text = self.generator()
    except TypeError:
      log.debug(self.generator_text)
      exit(8)
  