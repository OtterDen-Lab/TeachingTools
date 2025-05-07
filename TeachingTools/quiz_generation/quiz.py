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


class Quiz:
  """
  A quiz object that will build up questions and output them in a range of formats (hopefully)
  It should be that a single quiz object can contain multiples -- essentially it builds up from the questions and then can generate a variety of questions.
  """
  
  INTEREST_THRESHOLD = 1.0
  
  def __init__(self, name, questions: List[dict|Question], practice, *args, **kwargs):
    self.name = name
    self.questions = questions
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
  
  @classmethod
  def from_yaml(cls, path_to_yaml) -> List[Quiz]:
    
    quizes_loaded : List[Quiz] = []
    
    with open(path_to_yaml) as fid:
      list_of_exam_dicts = list(yaml.safe_load_all(fid))
    
    for exam_dict in list_of_exam_dicts:
      # Get general quiz information from the dictionary
      name = exam_dict.get("name", f"Unnamed Exam ({datetime.now().strftime('%a %b %d %I:%M %p')})")
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
      quiz_from_yaml = cls(name, questions_for_exam, practice)
      quiz_from_yaml.set_sort_order(sort_order)
      quizes_loaded.append(quiz_from_yaml)
    return quizes_loaded
  
  def get_quiz(self, **kwargs) -> ContentAST.Document:
    quiz = ContentAST.Document(title=self.name)
    
    quiz.add_elements(
      question.get_question(**kwargs)
      for question in self.questions
    )
    
    return quiz
  
  def describe(self):
    
    # Print out title
    print(f"Title: {self.name}")
    total_points = sum(map(lambda q: q.points_value, self.questions))
    total_questions = len(self.questions)
    
    # Print out overall information
    print(f"{total_points} points total, {total_questions} questions")
    
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
    
  def set_sort_order(self, sort_order):
    self.question_sort_order = sort_order

def main():
  pass
  

if __name__ == "__main__":
  main()
  