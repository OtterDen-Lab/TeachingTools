#!env python
from __future__ import annotations

import abc
import time

from typing import List

from TeachingTools.grading_assistant.assignment import Assignment
from TeachingTools.lms_interface.classes import Feedback

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Grader(abc.ABC):
  pass

  @abc.abstractmethod
  def grade(self, assignment : Assignment, *args, **kwargs) -> None:
    """
    Takes an assignment and walks through its submissions and grades each.
    :param assignment: Takes in an assignment.Assignment object to grade
    :return:
    """
    pass


class Grader__Dummy(Grader):
  def grade(self, assignment :Assignment, *args, **kwargs) -> None:
    for submission in assignment.submissions:
      log.info(f"Grading {submission}...")
      submission.feedback = Feedback(42.0, "Excellent job")
  