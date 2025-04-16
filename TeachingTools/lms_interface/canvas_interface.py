#!env python
from __future__ import annotations

import time
import typing
from datetime import datetime, timezone
from typing import List, Optional

import canvasapi
import canvasapi.course
import canvasapi.quiz
import canvasapi.assignment
import canvasapi.submission
import canvasapi.exceptions
import dotenv, os
import sys
import requests
import io

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.classes import LMSWrapper, Student, Submission, Submission__Canvas

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


class CanvasInterface(LMSWrapper):
  def __init__(self, *, prod=False):
    dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
    log.debug(os.environ.get("CANVAS_API_URL"))
    if prod:
      log.warning("Using canvas PROD!")
      self.canvas_url = os.environ.get("CANVAS_API_URL_prod")
      self.canvas_key = os.environ.get("CANVAS_API_KEY_prod")
    else:
      log.info("Using canvas DEV")
      self.canvas_url = os.environ.get("CANVAS_API_URL")
      self.canvas_key = os.environ.get("CANVAS_API_KEY_prod")
    self.canvas = canvasapi.Canvas(self.canvas_url, self.canvas_key)
    
    super().__init__(_inner=self.canvas)
    
  def get_course(self, course_id: int) -> CanvasCourse:
    return CanvasCourse(
      canvas_interface = self,
      canvasapi_course = self.canvas.get_course(course_id)
    )


class CanvasCourse(LMSWrapper):
  def __init__(self, *args, canvas_interface : CanvasInterface, canvasapi_course : canvasapi.course.Course, **kwargs):
    self.canvas_interface = canvas_interface
    self.course = canvasapi_course
    super().__init__(_inner=self.course)
  
  def create_assignment_group(self, name="dev") -> canvasapi.course.AssignmentGroup:
    for assignment_group in self.course.get_assignment_groups():
      if assignment_group.name == name:
        if name == "dev":
          assignment_group.delete()
          break
        log.info("Found group existing, returning")
        return assignment_group
    assignment_group = self.course.create_assignment_group(
      name="dev",
      group_weight=0.0,
      position=0,
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
  
  def get_assignment(self, assignment_id : int) -> Optional[CanvasAssignment]:
    try:
      return CanvasAssignment(
        canvasapi_interface=self.canvas_interface,
        canvasapi_course=self,
        canvasapi_assignment=self.course.get_assignment(assignment_id)
      )
    except canvasapi.exceptions.ResourceDoesNotExist:
      log.error(f"Assignment {assignment_id} not found in course \"{self.name}\"")
      return None
    
  def get_assignments(self, **kwargs) -> List[CanvasAssignment]:
    assignments : List[CanvasAssignment] = []
    for canvasapi_assignment in self.course.get_assignments(**kwargs):
      assignments.append(
        CanvasAssignment(
          canvasapi_interface=self.canvas_interface,
          canvasapi_course=self,
          canvasapi_assignment=canvasapi_assignment
        )
      )
    
    assignments = self.course.get_assignments(**kwargs)
    return assignments
  
  def get_username(self, user_id: int):
    return self.course.get_user(user_id).name
  
  def get_students(self) -> List[Student]:
    return [Student(s.name, s.id, s) for s in self.course.get_users(enrollment_type=["student"])]


class CanvasAssignment(LMSWrapper):
  def __init__(self, *args, canvasapi_interface: CanvasInterface, canvasapi_course : CanvasCourse, canvasapi_assignment: canvasapi.assignment.Assignment, **kwargs):
    self.canvas_interface = canvasapi_interface
    self.canvas_course = canvasapi_course
    self.assignment = canvasapi_assignment
    super().__init__(_inner=canvasapi_assignment)
  
  def push_feedback(self, user_id, score: float, comments: str, attachments=None, keep_previous_best=True, clobber_feedback=False):
    log.debug(f"Adding feedback for {user_id}")
    if attachments is None:
      attachments = []
    
    # Get the previous score to check to see if we should reuse it
    try:
      submission = self.assignment.get_submission(user_id)
      if keep_previous_best and score is not None and submission.score is not None and submission.score > score:
        log.warning(f"Current score ({submission.score}) higher than new score ({score}).  Going to use previous score.")
        score = submission.score
    except requests.exceptions.ConnectionError as e:
      log.warning(f"No previous submission found for {user_id}")
    
    # Update the assignment
    # Note: the bulk_update will create a submission if none exists
    try:
      self.assignment.submissions_bulk_update(
        grade_data={
          'submission[posted_grade]' : score
        },
        student_ids=[user_id]
      )
      
      submission = self.assignment.get_submission(user_id)
    except requests.exceptions.ConnectionError as e:
      log.error(e)
      log.debug(f"Failed on user_id = {user_id})")
      log.debug(f"username: {self.canvas_course.get_user(user_id)}")
      return
    
    # Push feedback to canvas
    submission.edit(
      submission={
        'posted_grade':score,
      },
    )
    
    # If we should overwrite previous comments then remove all the previous submissions
    if clobber_feedback:
      log.debug("Clobbering...")
      # todo: clobbering should probably be moved up or made into a different function for cleanliness.
      for comment in submission.submission_comments:
        # log.debug(f"Existing comment: {comment}")
        comment_id = comment['id']
        # Construct the URL to delete the comment
        delete_url = f"{self.canvas_interface.canvas_url}/api/v1/courses/{self.canvas_course.course.id}/assignments/{self.assignment.id}/submissions/{user_id}/comments/{comment_id}"
        
        # Make the DELETE request to delete the comment
        response = requests.delete(delete_url, headers={"Authorization": f"Bearer {self.canvas_interface.canvas_key}"})
        if response.status_code == 200:
          log.info(f"Deleted comment {comment_id}")
        else:
          log.warning(f"Failed to delete comment {comment_id}: {response.json()}")
    
    
    def upload_buffer_as_file(buffer, name):
      with io.FileIO(name, 'w+') as ffid:
        ffid.write(buffer)
        ffid.flush()
        ffid.seek(0)
        submission.upload_comment(ffid)
      os.remove(name)
    
    if len(comments) > 0:
      upload_buffer_as_file(comments.encode('utf-8'), "feedback.txt")
    
    for i, attachment_buffer in enumerate(attachments):
      upload_buffer_as_file(attachment_buffer.read(), attachment_buffer.name)
  
  def get_submissions(self, only_include_most_recent: bool = True, **kwargs) -> List[Submission__Canvas]:
    """
    Gets submission objects (in this case Submission__Canvas objects) that have students and potentially attachments
    :param only_include_most_recent: Include only the most recent submission
    :param kwargs:
    :return:
    """
    
    if "limit" in kwargs and kwargs["limit"] is not None:
      limit = kwargs["limit"]
    else:
      limit = 1_000_000 # magically large number
    
    submissions: List[Submission__Canvas] = []
    
    # Get all submissions and their history (which is necessary for attachments when students can resubmit)
    for student_index, canvaspai_submission in enumerate(self.assignment.get_submissions(include='submission_history', **kwargs)):
      
      # Get the student object for the submission
      student = Student(
        self.canvas_course.get_username(canvaspai_submission.user_id),
        user_id=canvaspai_submission.user_id,
        _inner=self.canvas_course.get_user(canvaspai_submission.user_id)
      )
      
      log.info(f"Checking submissions for {student.name} ({len(canvaspai_submission.submission_history)} submissions)")
      
      # Walk through submissions in the reverse order, so we'll default to grabbing the most recent submission first
      # This is important when we are going to be only including most recent
      for student_submission_index, student_submission in (
          reversed(list(enumerate(canvaspai_submission.submission_history)))):
        log.debug(f"Submission: {student_submission['workflow_state']} " +
                  (f"{student_submission['score']:0.2f}" if student_submission['score'] is not None else "None"))
        
        try:
          attachments = student_submission["attachments"]
        except KeyError:
          log.warning(f"No submissions found for {student.name}")
          continue
        
        # Add submission to list
        submissions.append(
          Submission__Canvas(
            student=student,
            status=Submission.Status.from_string(student_submission["workflow_state"], student_submission['score']),
            attachments=attachments,
            submission_index=student_submission_index
          )
        )
        
        # Check if we should only include the most recent
        if only_include_most_recent: break
      
      # Check if we are limiting how many students we are checking
      if student_index >= (limit - 1): break
      
    # Reverse the submissions again so we are preserving temporal order.  This isn't necessary but makes me feel happy.
    submissions = list(reversed(submissions))
    return submissions
  
  def get_students(self):
    return self.canvas_course.get_students()


class CanvasHelpers:
  @staticmethod
  def get_closed_assignments(interface: CanvasCourse) -> List[canvasapi.assignment.Assignment]:
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
  def get_unsubmitted_submissions(interface: CanvasCourse, assignment: canvasapi.assignment.Assignment) -> List[canvasapi.submission.Submission]:
    submissions : List[canvasapi.submission.Submission] = list(filter(
      lambda s: s.submitted_at is None and s.score is None and not s.excused,
      assignment.get_submissions()
    ))
    return submissions
  
  @classmethod
  def clear_out_missing(cls, interface: CanvasCourse):
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
  
  
  @staticmethod
  def deprecate_assignment(canvas_course: CanvasCourse, assignment_id) -> List[canvasapi.assignment.Assignment]:
    
    log.debug(canvas_course.__dict__)
    
    # for assignment in canvas_course.get_assignments():
    #   print(assignment)
    
    canvas_assignment : CanvasAssignment = canvas_course.get_assignment(assignment_id=assignment_id)
    
    canvas_assignment.assignment.edit(
      assignment={
        "name": f"{canvas_assignment.assignment.name} (deprecated)",
        "due_at": f"{datetime.now(timezone.utc).isoformat()}",
        "lock_at": f"{datetime.now(timezone.utc).isoformat()}"
      }
    )
  
  @staticmethod
  def mark_future_assignments_as_ungraded(canvas_course: CanvasCourse):
    
    for assignment in canvas_course.get_assignments(
        include=["all_dates"],
        order_by="name"
    ):
      if assignment.unlock_at is not None:
        if datetime.fromisoformat(assignment.unlock_at) > datetime.now(timezone.utc):
          log.debug(assignment)
          for submission in assignment.get_submissions():
          #   log.debug(submission.__dict__)
            submission.mark_unread()
          
    
