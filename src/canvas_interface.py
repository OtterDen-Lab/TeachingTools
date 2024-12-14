#!env python

import argparse
import collections.abc
import pprint
import time
import typing
from datetime import datetime, timezone
from typing import Dict, Set, List

import canvasapi
import canvasapi.course
import canvasapi.quiz
import canvasapi.assignment
import canvasapi.submission
import dotenv, os
import sys

from quiz import Quiz, Question

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


logger = logging.getLogger("canvasapi")
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setLevel(logging.WARNING)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)


QUESTION_VARIATIONS_TO_TRY = 1000


class CanvasInterface:
  def __init__(self, *, course_id : int, prod=False):
    super().__init__()
    dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
    log.debug(os.environ.get("CANVAS_API_URL"))
    if prod:
      log.warning("Using canvas PROD!")
      self.canvas = canvasapi.Canvas(os.environ.get("CANVAS_API_URL_prod"), os.environ.get("CANVAS_API_KEY_prod"))
    else:
      log.info("Using canvas DEV")
      self.canvas = canvasapi.Canvas(os.environ.get("CANVAS_API_URL"), os.environ.get("CANVAS_API_KEY_prod"))
    self.course = self.canvas.get_course(course=course_id)
  
  def create_assignment_group(self, name="dev") -> canvasapi.course.AssignmentGroup:
    for assignment_group in self.course.get_assignment_groups():
      if assignment_group.name == name:
        log.info("Found group existing, returning")
        return assignment_group
    assignment_group = self.course.create_assignment_group(
      name="dev",
      group_weight=0.0,
      position=1,
    )
    return assignment_group
  
  def add_quiz(
      self,
      assignment_group: canvasapi.course.AssignmentGroup,
      title = None,
      *,
      is_practice=False
  ):
    if title is None:
      title = f"New Quiz {datetime.now().strftime('%m/%d/%y %H:%M:%S.%f')}"
    
    q = self.course.create_quiz(quiz={
      "title": title,
      "hide_results" : None,
      "show_correct_answers": True,
      "scoring_policy": "keep_highest",
      "allowed_attempts": -1,
      "shuffle_answers": True,
      "assignment_group_id": assignment_group.id,
      "quiz_type" : "assignment" if not is_practice else "practice_quiz",
      "description": """
        This quiz is aimed to help you practice skills.
        Please take it as many times as necessary to get full marks!
        Please note that although the answers section may be a bit lengthy,
        below them is often an in-depth explanation on solving the problem!
      """
    })
    return q

  def push_quiz_to_canvas(
      self,
      quiz: Quiz,
      num_variations: int,
      title: typing.Optional[str] = None,
      is_practice = False
  ):
    assignment_group = self.create_assignment_group()
    canvas_quiz = self.add_quiz(assignment_group, title, is_practice=is_practice)
    
    all_variations = set()
    for question_i, question in enumerate(quiz):
      log.debug(f"Generating #{question_i} ({question.name})")
  
      group : canvasapi.quiz.QuizGroup = canvas_quiz.create_question_group([
        {
          "name": f"{question.name}",
          "pick_count": 1,
          "question_points": question.points_value
        }
      ])
      
      # Track all variations across every question, in case we have duplicate questions
      variation_count = 0
      for attempt_number in range(QUESTION_VARIATIONS_TO_TRY):
        
        # Get the question in a format that is ready for canvas (e.g. json)
        question_for_canvas = question.get__canvas(self.course, canvas_quiz)
        question_fingerprint = question_for_canvas["question_text"]
        try:
          question_fingerprint += ''.join([str(a["answer_text"]) for a in question_for_canvas["answers"]])
        except TypeError as e:
          log.error(e)
          log.warning("Continuing anyway")
          
        
        # if it is in the variations that we have already seen then skip ahead, else track
        if question_fingerprint in all_variations:
          continue
        all_variations.add(question_fingerprint)
        
        # Set group ID to add it to the question group
        question_for_canvas["quiz_group_id"] = group.id
        
        # Push question to canvas
        log.debug(f"Pushing #{question_i} ({question.name}) {variation_count+1} / {num_variations} to canvas...")
        try:
          canvas_quiz.create_question(question=question_for_canvas)
        except canvasapi.exceptions.CanvasException as e:
          log.warning("Encountered Canvas error.")
          log.warning(e)
          log.warning("Sleeping for 1s...")
          time.sleep(1)
          continue
        
        # Update and check variations already seen
        variation_count += 1
        if variation_count >= num_variations:
          break
        if variation_count >= question.possible_variations:
          break
        
  def get_assignments(self, **kwargs):
    assignments = self.course.get_assignments(**kwargs)
    return assignments
  
  
  def get_submissions(self, assignments: List[canvasapi.assignment.Assignment]):
    submissions : List[canvasapi.submission.Submission] = []
    for assignment in assignments:
      submissions.extend(assignment.get_submissions())
    return submissions
  
  def get_username(self, user_id: int):
    return self.course.get_user(user_id).name
  
class CanvasHelpers:
  @staticmethod
  def get_closed_assignments(interface: CanvasInterface) -> List[canvasapi.assignment.Assignment]:
    closed_assignments : List[canvasapi.assignment.Assignment] = []
    for assignment in interface.get_assignments(
      include=["all_dates"], 
      order_by="name"
    ):
      if not assignment.published:
        continue
      if assignment.lock_at is not None:
        # Then it's the easy case because there's no overrides
        if datetime.fromisoformat(assignment.lock_at) < datetime.now(timezone.utc):
          # Then the assignment is past due
          closed_assignments.append(assignment)
          continue
      elif assignment.all_dates is not None:
        
        # First we need to figure out what the latest time this assignment could be available is
        # todo: This could be done on a per-student basis
        last_lock_datetime = None
        for dates_dict in assignment.all_dates:
          if dates_dict["lock_at"] is not None:
            lock_datetime = datetime.fromisoformat(dates_dict["lock_at"])
            if (last_lock_datetime is None) or (lock_datetime >= last_lock_datetime):
              last_lock_datetime = lock_datetime
        
        # If we have found a valid lock time, and it's in the past then we lock
        if last_lock_datetime is not None and last_lock_datetime <= datetime.now(timezone.utc):
          closed_assignments.append(assignment)
          continue
          
      else:
        log.warning(f"Cannot find any lock dates for assignment {assignment.name}!")
    
    return closed_assignments
  @staticmethod
  def get_unsubmitted_submissions(interface: CanvasInterface, assignment: canvasapi.assignment.Assignment) -> List[canvasapi.submission.Submission]:
    submissions : List[canvasapi.submission.Submission] = list(filter(
      lambda s: s.submitted_at is None and s.score is None and not s.excused,
      assignment.get_submissions()
    ))
    return submissions
  
  @classmethod
  def clear_out_missing(cls, interface: CanvasInterface):
    assignments = cls.get_closed_assignments(interface)
    for assignment in assignments:
      # if "PA6" not in assignment.name: continue
      missing_submissions = cls.get_unsubmitted_submissions(interface, assignment)
      if len(missing_submissions) == 0:
        continue
      log.info(f"Assignment: ({assignment.quiz_id if hasattr(assignment, 'quiz_id') else assignment.id}) {assignment.name} {assignment.published}")
      for submission in missing_submissions:
        log.info(f"{submission.user_id} ({interface.get_username(submission.user_id)}) : {submission.workflow_state} : {submission.missing} : {submission.score} : {submission.grader_id} : {submission.graded_at}")
        submission.edit(submission={"late_policy_status" : "missing"})
      log.info("")
  
 
def main():
  interface = CanvasInterface(course_id=25523, prod=False)
  

if __name__ == "__main__":
  main()
