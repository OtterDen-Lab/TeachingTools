#!env python
from __future__ import annotations

import argparse
import collections
import itertools
import logging
import os.path
import random
import shutil
import subprocess
import tempfile
from typing import List, Dict

import yaml

from TeachingTools.quiz_generation.misc import OutputFormat
from TeachingTools.quiz_generation.question import Question, QuestionRegistry

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class Quiz:
  """
  A quiz object that will build up questions and output them in a range of formats (hopefully)
  It should be that a single quiz object can contain multiples -- essentially it builds up from the questions and then can generate a variety of questions.
  """
  
  def __init__(self, name, possible_questions: List[dict|Question], practice, *args, **kwargs):
    self.name = name
    self.possible_questions = possible_questions
    self.questions : List[Question] = []
    self.instructions = kwargs.get("instructions", "")
    self.question_sort_order = None
    self.practice = practice
    
    # Plan: right now we just take in questions and then assume they have a score and a "generate" button
  
  def __iter__(self):
    def sort_func(q):
      if self.question_sort_order is not None:
        try:
          return (-q.points_value, self.question_sort_order.index(q.kind))
        except ValueError:
          return (-q.points_value, float('inf'))
      return -q.points_value
    return iter(sorted(self.questions, key=sort_func))
    
  def describe(self):
    counter = collections.Counter([q.points_value for q in self.questions])
    log.info(f"{self.name} : {sum(map(lambda q: q.points_value, self.questions))}points : {len(self.questions)} / {len(self.possible_questions)} questions picked.  {list(counter.items())}")
    
    sort_order = self.question_sort_order
    if sort_order is None:
      sort_order = Question.Topic
    for topic in sort_order:
      log.info(f"{topic} : {sum(map(lambda q: q.points_value, filter(lambda q: q.kind == topic, self.questions)))} points")
    
  def select_questions(self, total_points=None, exam_outline: List[Dict]=None):
    # The exam_outline object should contain a description of the kinds of questions that we want.
    # It will be a list of dictionaries that has "num questions" and then the appropriate filters.
    # We will walk through it and pick an appropriate set of questions, ensuring that we only select each once (unless we can pick more than once)
    # After we've gone through all the rules, we can backfill with whatever is left

    if total_points is None:
      self.questions = self.possible_questions
      return
    
    questions_picked = set()
    
    possible_questions = set(self.possible_questions)
    
    if exam_outline is not None:
      for requirements in exam_outline:
        # Filter out to only get appropriate questions
        log.debug(requirements["filters"])
        appropriate_questions = list(filter(
          lambda q: all([getattr(q, attr_name) == attr_val for (attr_name, attr_val) in requirements["filters"].items()]),
          possible_questions
        ))
        
        log.debug(f"{len(appropriate_questions)} appropriate questions")
        
        # Pick the appropriate number of questions
        questions_picked.update(
          random.sample(appropriate_questions, min(requirements["num_to_pick"], len(appropriate_questions)))
        )
        
        # Remove any questions that were just picked so we don't pick them again
        possible_questions = set(possible_questions).difference(set(questions_picked))
        
    log.debug(f"Selected due to filters: {len(questions_picked)} ({sum(map(lambda q: q.points_value, questions_picked))}points)")
    
    if total_points is not None:
      # Figure out how many points we have left to select
      num_points_left = total_points - sum(map(lambda q: q.points_value, questions_picked))
      
      
      # To pick the remaining points, we want to take our remaining questions and select a subset that adds up to the required number of points
  
      # Find all combinations of objects that match the target value
      log.debug("Finding all matching sets...")
      matching_sets = []
      for r in range(1, len(possible_questions) + 1):
        for combo in itertools.combinations(possible_questions, r):
          if sum(q.points_value for q in combo) == num_points_left:
            matching_sets.append(combo)
            if len(matching_sets) > 1000:
              break
        if len(matching_sets) > 1000:
          break
      
      # Pick a random matching set
      if matching_sets:
        random_set = random.choice(matching_sets)
      else:
        log.error("Cannot find any matching sets")
        random_set = []
    
      questions_picked.update(random_set)
    else:
      # todo: I know this snippet is repeated.  Oh well.
      questions_picked = self.possible_questions
    self.questions = questions_picked
  
  def get_latex(self) -> str:
    text = self.get_header(OutputFormat.LATEX) + "\n\n"
    for question in self:
      text += question.get__latex() + "\n\n"
    text += self.get_footer(OutputFormat.LATEX)
    return text
  
  def get_header(self, output_format: OutputFormat, *args, **kwargs) -> str:
    lines = []
    if output_format == OutputFormat.LATEX:
      lines.extend([
      
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper, margin=1in]{geometry}",
        r"\usepackage{times}",
        r"\usepackage{tcolorbox}",
        r"\usepackage{graphicx} % Required for inserting images",
        r"\usepackage{booktabs}",
        r"\usepackage[final]{listings}",
        r"\usepackage[nounderscore]{syntax}",
        r"\usepackage{caption}",
        r"\usepackage{booktabs}",
        r"\usepackage{multicol}",
        r"\usepackage{subcaption}",
        r"\usepackage{enumitem} % Ensure this is loaded only once",
        r"\usepackage{setspace}",
        r"\usepackage{longtable}",
        r"\usepackage{arydshln}",
        r"\usepackage{ragged2e}\let\Centering\flushleft",
        
        r"% Custom commands",
        r"\newcounter{NumQuestions}",
        r"\newcommand{\question}[1]{ %",
        r"\vspace{0.5cm}",
        r"\stepcounter{NumQuestions} %",
        r"\noindent\textbf{Question \theNumQuestions:} \hfill \rule{0.5cm}{0.15mm} / #1",
        r"\par",
        r"\vspace{0.1cm}",
        r"}",
        r"\newcommand{\answerblank}[1]{\rule[-1.5mm]{#1cm}{0.15mm}}",
        r"\setlist[itemize]{itemsep=10pt, parsep=5pt} % Adjust these values as needed",
        
        
        r"\providecommand{\tightlist}{%",
        r"\setlength{\itemsep}{10pt}\setlength{\parskip}{10pt}",
        r"}",
        
        r"\title{" + self.name + r"}",
        
        r"\begin{document}",
        r"\noindent\Large " + self.name + r"\hfill \normalsize Name: \answerblank{5}",
        r"\vspace{0.5cm}"
      ])
      if len(self.instructions):
        lines.extend([
          "",
          r"\noindent\textbf{Instructions:}",
          r"\noindent " + self.instructions
        ])
      lines.extend([
        r"\onehalfspacing"
      ])
    else:
      lines.extend([
        f"{self.name}",
        "Name:"
      ])
    return '\n'.join(lines)
  
  def get_footer(self, output_format: OutputFormat, *args, **kwargs) -> str:
    lines = []
    if output_format == OutputFormat.LATEX:
      lines.extend([
        r"\end{document}"
      ])
    return '\n'.join(lines)
  
  def set_sort_order(self, sort_order):
    self.question_sort_order = sort_order

  @classmethod
  def from_yaml(cls, path_to_yaml) -> List[Quiz]:
    
    quizes_loaded : List[Quiz] = []
    
    with open(path_to_yaml) as fid:
      list_of_exam_dicts = list(yaml.safe_load_all(fid))
    log.debug(list_of_exam_dicts)
    
    for exam_dict in list_of_exam_dicts:
      # Get general quiz information from the dictionary
      name = exam_dict.get("name", "Unnamed Exam")
      practice = exam_dict.get("practice", False)
      sort_order = list(map(lambda t: Question.Topic.from_string(t), exam_dict.get("sort order", [])))
      sort_order = sort_order + list(filter(lambda t: t not in sort_order, Question.Topic))
      
      # Load questions from the quiz dictionary
      questions_for_exam = []
      for question_value, question_definitions in exam_dict["questions"].items():
        # todo: I can also add in "extra credit" and "mix-ins" as other keys to indicate extra credit or questions that can go anywhere
        log.info(f"Parsing {question_value} point questions")
        
        def make_question(q_name, q_data):
          kwargs= {
            "name" : q_name,
            "points_value" : question_value,
            **q_data.get("kwargs", {})
          }
          if "topic" in q_data:
            kwargs["topic"] = Question.Topic.from_string(q_data["topic"])
          elif "kind" in q_data:
            kwargs["topic"] = Question.Topic.from_string(q_data["kind"])
          new_question = QuestionRegistry.create(
            q_data["class"],
            **kwargs
          )
          return new_question
        
        for q_name, q_data in question_definitions.items():
          log.debug(f"{q_name} : {q_data}")
          if "pick" in q_data:
            num_to_pick = q_data["pick"]
            del q_data["pick"]
            questions_for_exam.extend(
              make_question(name, data) for name, data in
              random.sample(list(q_data.items()), num_to_pick)
            )
          else:
            questions_for_exam.extend([
              make_question(q_name, q_data)
              for _ in range(q_data.get("repeat", 1))
            ]
            )
          
      quiz_from_yaml = Quiz(name, questions_for_exam, practice)
      quiz_from_yaml.set_sort_order(sort_order)
      quizes_loaded.append(quiz_from_yaml)
    return quizes_loaded

  def generate_latex(self, remove_previous=False):
    
    if remove_previous:
      if os.path.exists('out'): shutil.rmtree('out')
    
    tmp_tex = tempfile.NamedTemporaryFile('w')
    
    tmp_tex.write(self.get_latex())
    tmp_tex.flush()
    tmp_tex.flush()
    shutil.copy(f"{tmp_tex.name}", "debug.tex")
    p = subprocess.Popen(
      f"latexmk -pdf -output-directory={os.path.join(os.getcwd(), 'out')} {tmp_tex.name}",
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
    try:
      p.wait(30)
    except subprocess.TimeoutExpired:
      logging.error("Latex Compile timed out")
      p.kill()
      tmp_tex.close()
      return
    proc = subprocess.Popen(
      f"latexmk -c {tmp_tex.name} -output-directory={os.path.join(os.getcwd(), 'out')}",
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE
    )
    proc.wait(timeout=30)
    tmp_tex.close()

def main():
  pass
  

if __name__ == "__main__":
  main()
  