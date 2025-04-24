#!env python
from __future__ import annotations

import collections
import itertools
import logging
import os.path
import random
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

import yaml

from TeachingTools.quiz_generation.misc import OutputFormat, ContentAST
from TeachingTools.quiz_generation.question import Question, QuestionRegistry, QuestionGroup

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ConcreteQuestionSet:
  def __init__(self, questions: List[Question], rng_seed, previous_question_set : Optional[ConcreteQuestionSet]= None):
    self.questions : List[ContentAST.Question] = []
    for i, question in enumerate(questions):
      self.questions.append(
        question.get_question(
          # rng_seed=rng_seed,
          previous=(
            None if previous_question_set is None
            else previous_question_set.questions[i].question
          ) # Keeps scheduling questions across all types
        )
      )
    
  def interesting_score(self, *, weighted=False):
    overall_score = 0.0
    for q in self.questions:
      overall_score += q.interest if not weighted else q.value
    return overall_score / (len(self.questions) if not weighted else sum([q.value for q in self.questions]))
  
  
  def get_latex(self, sort_order) -> str:
    text = ""
    for question in sorted(self.questions, key=lambda q: (-q.value, sort_order.index(q.topic))):
      text += question.render("latex") + "\n\n"
    return text
  
  def get_rubric(self):
    return {
      i : {
        "Value": question.value,
        "Answer" : question.answer_text,
        "Explanation": question.explanation_text
      }
      for i, question in enumerate(self.questions)
    }

class Quiz:
  """
  A quiz object that will build up questions and output them in a range of formats (hopefully)
  It should be that a single quiz object can contain multiples -- essentially it builds up from the questions and then can generate a variety of questions.
  """
  
  INTEREST_THRESHOLD = 1.0
  
  def __init__(self, name, possible_questions: List[dict|Question], practice, *args, **kwargs):
    self.name = name
    self.possible_questions = possible_questions
    self.questions : List[Question] = []
    self.instructions = kwargs.get("instructions", "")
    self.question_sort_order = None
    self.practice = practice
    
    self.used_seeds = []
    
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
    
    # Print out title
    print(f"Title: {self.name}")
    total_points = sum(map(lambda q: q.points_value, self.questions))
    total_questions = len(self.questions)
    
    # Print out overall information
    print(f"{total_points} points total, {total_questions} / {len(self.possible_questions)} questions picked")
    
    # Print out the per-value information
    points_counter = collections.Counter([q.points_value for q in self.questions])
    for points in sorted(points_counter.keys(), reverse=True):
      print(f"{points_counter.get(points)} x {points}points")
    
    # Either get the sort order or default to the order in the enum class
    sort_order = self.question_sort_order
    if sort_order is None:
      sort_order = Question.Topic
      
    # Build per-topic information
    
    topic_information = {}
    topic_strings = {}
    for topic in sort_order:
      topic_strings = {"name": topic.name}
      
      question_count = len(list(map(lambda q: q.points_value, filter(lambda q: q.kind == topic, self.questions))))
      topic_points = sum(map(lambda q: q.points_value, filter(lambda q: q.kind == topic, self.questions)))
      
      # If we have questions add in some states, otherwise mark them as empty
      if question_count != 0:
        topic_strings["count_str"] = f"{question_count} questions ({ 100 * question_count / total_questions:0.1f}%)"
        topic_strings["points_str"] = f"{topic_points:2} points ({ 100 * topic_points / total_points:0.1f}%)"
      else:
        topic_strings["count_str"] = "--"
        topic_strings["points_str"] = "--"
      
      topic_information[topic] = topic_strings
    
    
    # Get padding string lengths
    paddings = collections.defaultdict(lambda: 0)
    for field in topic_strings.keys():
      paddings[field] = max(len(information[field]) for information in topic_information.values())
    
    # Print out topics information using the padding
    for topic in sort_order:
      topic_strings = topic_information[topic]
      print(f"{topic_strings['name']:{paddings['name']}} : {topic_strings['count_str']:{paddings['count_str']}} : {topic_strings['points_str']:{paddings['points_str']}}")
    
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
  
  def set_sort_order(self, sort_order):
    self.question_sort_order = sort_order

  @classmethod
  def from_yaml(cls, path_to_yaml) -> List[Quiz]:
    
    quizes_loaded : List[Quiz] = []
    
    with open(path_to_yaml) as fid:
      list_of_exam_dicts = list(yaml.safe_load_all(fid))
    
    for exam_dict in list_of_exam_dicts:
      # Get general quiz information from the dictionary
      name = exam_dict.get("name", "Unnamed Exam") # + f" ({datetime.now().strftime("%a %b %d %I:%M %p")})"
      practice = exam_dict.get("practice", False)
      sort_order = list(map(lambda t: Question.Topic.from_string(t), exam_dict.get("sort order", [])))
      sort_order = sort_order + list(filter(lambda t: t not in sort_order, Question.Topic))
      
      # Load questions from the quiz dictionary
      questions_for_exam = []
      for question_value, question_definitions in exam_dict["questions"].items():
        # todo: I can also add in "extra credit" and "mix-ins" as other keys to indicate extra credit or questions that can go anywhere
        log.info(f"Parsing {question_value} point questions")
        
        def make_question(q_name, q_data, **kwargs):
          # Build up the kwargs that we're going to pass in
          # todo: this is currently a mess due to legacy things, so before I tell others to use this make it cleaner
          kwargs= {
            "name": q_name,
            "points_value": question_value,
            **q_data.get("kwargs", {}),
            **q_data,
            **kwargs,
          }
          
          # If we are passed in a topic then use it, otherwise don't set it which will have it set to a default
          if "topic" in q_data:
            kwargs["topic"] = Question.Topic.from_string(q_data.get("topic", "Misc"))
          
          # Add in a default, where if it isn't specified we're going to simply assume it is a text
          question_class = q_data.get("class", "FromText")
          
          new_question = QuestionRegistry.create(
            question_class,
            **kwargs
          )
          return new_question
        
        for q_name, q_data in question_definitions.items():
          # Set defaults for config
          question_config = {
            "group" : False,
            "num_to_pick" : 1,
            "random_per_student" : False,
            "repeat": 1,
            "topic": "MISC"
          }
          
          # Update config if it exists
          question_config.update(
            q_data.get("_config", {})
          )
          q_data.pop("_config", None)
          q_data.pop("pick", None) # todo: don't use this anymore
          q_data.pop("repeat", None) # todo: don't use this anymore
          
          # Check if it is a question group
          if question_config["group"]:
            
            # todo: Find a way to allow for "num_to_pick" to ensure lack of duplicates when using duplicates.
            #    It's probably going to be somewhere in the instantiate and get_attr fields, with "_current_questions"
            #    But will require changing how we add concrete questions (but that'll just be everything returns a list
            questions_for_exam.append(
              QuestionGroup(
                questions_in_group=[
                  make_question(name, data | {"topic" : question_config["topic"]}) for name, data in q_data.items()
                ],
                pick_once=(not question_config["random_per_student"])
              )
            )
            
          else: # Then this is just a single question
            questions_for_exam.extend([
              make_question(
                q_name,
                q_data,
                rng_seed_offset=repeat_number
              )
              for repeat_number in range(question_config["repeat"])
            ])
      log.debug(f"len(questions_for_exam): {len(questions_for_exam)}")
      quiz_from_yaml = Quiz(name, questions_for_exam, practice)
      quiz_from_yaml.set_sort_order(sort_order)
      quizes_loaded.append(quiz_from_yaml)
    return quizes_loaded
  
  
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
        r"\lstset{",
        r"  basicstyle=\ttfamily,",
        r"columns=fullflexible,",
        r"frame=single,",
        r"breaklines=true,",
        r"postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space},",
        r"}",
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
        r"\usepackage{array}",
        
        r"\usepackage{verbatim}",
        r"\let\oldverbatim\verbatim",
        r"\let\endoldverbatim\endverbatim",
        r"\renewenvironment{verbatim}",
        r"{\small\linespread{0.8}\selectfont\oldverbatim}",
        r"{\endoldverbatim}",
        
        r"% Custom commands",
        r"\newcounter{NumQuestions}",
        r"\newcommand{\question}[1]{ %",
        r"\vspace{0.5cm}",
        r"\stepcounter{NumQuestions} %",
        r"\noindent\textbf{Question \theNumQuestions:} \hfill \rule{0.5cm}{0.15mm} / #1",
        r"\par",
        r"\vspace{0.1cm}",
        r"}",
        
        r"\newcommand{\answerblank}[1]{\rule{0pt}{10mm}\rule[-1.5mm]{#1cm}{0.15mm}}",
        
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
  
  def generate_latex(self, remove_previous=False):
    
    if remove_previous:
      if os.path.exists('out'): shutil.rmtree('out')
    
    question_set = None
    
    while True:
      # Pick a new random seed
      rng_seed = random.randint(0, 1_000_000_000_000)
      
      # Check if it's been used already
      if rng_seed in self.used_seeds:
        continue
      else:
        self.used_seeds.append(rng_seed)
      
      # Make a quiz
      question_set = ConcreteQuestionSet(self.questions, rng_seed=rng_seed, previous_question_set=question_set)
      
      # Check if it's interesting enough
      if question_set.interesting_score() >= self.INTEREST_THRESHOLD:
        log.debug(f"Found a good seed: {rng_seed}")
        break
      else:
        log.debug(f"Seed {rng_seed} not interesting enough ({question_set.interesting_score()})")
    
    tmp_tex = tempfile.NamedTemporaryFile('w')
    
    tmp_tex.write(self.get_header(OutputFormat.LATEX) + "\n\n")
    tmp_tex.write(question_set.get_latex(self.question_sort_order))
    tmp_tex.write(self.get_footer(OutputFormat.LATEX))
    
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
  