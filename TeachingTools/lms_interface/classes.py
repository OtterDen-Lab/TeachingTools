#!env python
from __future__ import annotations

import enum
import logging
import dataclasses
import functools
import io
from typing import Optional, List

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



@dataclasses.dataclass
class Student:
  name : str
  user_id : int


class Submission:
  
  class Status(enum.Enum):
    UNGRADED = enum.auto()
    GRADED = enum.auto()
    
  def __init__(
      self,
      *,
      student : Student = None,
      status : Submission.Status = Status.UNGRADED,
      **kwargs
  ):
    self.__student: Optional[Student] = student
    self.status = status
    self.input_files = None
    self.__files = None
    self.feedback : Optional[Feedback] = None
  
  @property
  def student(self):
    return self.__student
  
  @student.setter
  def student(self, student):
    self.__student = student
  
  def __str__(self):
    try:
      return f"S({self.student.name} : {self.feedback})"
    except AttributeError:
      return f"S({self.student} : {self.feedback})"

  @property
  def files(self):
    return self.__files
  

@functools.total_ordering
@dataclasses.dataclass
class Feedback:
  score: Optional[float] = None
  comments: str = ""
  attachments: List[io.BytesIO] = dataclasses.field(default_factory=list)
  
  def __str__(self):
    return f"Feedback({self.score}, ...)"
  
  def __eq__(self, other):
    if not isinstance(other, Feedback):
      return NotImplemented
    return self.score == other.score
  
  def __lt__(self, other):
    if not isinstance(other, Feedback):
      return NotImplemented
    if self.score is None:
      return False  # None is treated as greater than any other value
    if other.score is None:
      return True
    return self.score < other.score
