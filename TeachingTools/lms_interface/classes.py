#!env python
from __future__ import annotations

import enum
import logging
import dataclasses
import functools
import io
import os
import urllib.request
from typing import Optional, List, Dict

import canvasapi.canvas

log = logging.getLogger(__name__)



class LMSWrapper():
  def __init__(self, _inner):
    self._inner = _inner
  
  def __getattr__(self, name):
    try:
      # Try to get the attribute from the inner instance
      return getattr(self._inner, name)
    except AttributeError:
      # Handle the case where the inner instance also doesn't have the attribute
      print(f"Warning: '{name}' not found in either wrapper or inner class")
      # You can raise the error again, return None, or handle it however you want
      return lambda *args, **kwargs: None  # Returns a no-op function for method calls



@dataclasses.dataclass
class Student(LMSWrapper):
  name : str
  user_id : int
  _inner : canvasapi.canvas.User
  
  

class Submission:
  
  class Status(enum.Enum):
    MISSING = "unsubmitted"
    UNGRADED = ("submitted", "pending_review")
    GRADED = "graded"
    
    @classmethod
    def from_string(cls, status_string, current_score):
      for status in cls:
        if status is not cls.MISSING and current_score is None:
          return cls.UNGRADED
        if isinstance(status.value, tuple):
          if status_string in status.value:
            return status
        elif status_string == status.value:
          return status
      return cls.MISSING  # Default
    
    
  def __init__(
      self,
      *,
      student : Student = None,
      status : Submission.Status = Status.UNGRADED,
      **kwargs
  ):
    self._student: Optional[Student] = student
    self.status = status
    self.input_files = None
    self._files = None
    self.feedback : Optional[Feedback] = None
    self.extra_info = {}
  
  @property
  def student(self):
    return self._student
  
  @student.setter
  def student(self, student):
    self._student = student
  
  def __str__(self):
    try:
      return f"Submission({self.student.name} : {self.feedback})"
    except AttributeError:
      return f"Submission({self.student} : {self.feedback})"

  @property
  def files(self):
    return self._files
  
  def set_extra(self, extras_dict: Dict):
    self.extra_info.update(extras_dict)


class Submission__Canvas(Submission):
  def __init__(self, *args, attachments : Optional[List], **kwargs):
    super().__init__(*args, **kwargs)
    self._attachments = attachments
    self.submission_index = kwargs.get("submission_index", None)
  
  @property
  def files(self):
    # Check if we have already downloaded the files locally and return if we have
    if self._files is not None:
      return self._files
    
    # If we haven't downloaded the files yet, check if we have attachments and can download them
    if self._attachments is not None:
      self._files = []
      for attachment in self._attachments:
        
        # Generate a local file name with a number of options
        # local_file_name = f"{self.student.name.replace(' ', '-')}_{self.student.user_id}_{attachment['filename']}"
        local_file_name = f"{attachment['filename']}"
        with urllib.request.urlopen(attachment['url']) as response:
          buffer = io.BytesIO(response.read())
          buffer.name = local_file_name
          self._files.append(buffer)
    
    return self._files


@functools.total_ordering
@dataclasses.dataclass
class Feedback:
  score: Optional[float] = None
  comments: str = ""
  attachments: List[io.BytesIO] = dataclasses.field(default_factory=list)
  
  def __str__(self):
    short_comment = self.comments[:10].replace('\n', '\\n')
    ellipsis = '...' if len(self.comments) > 10 else ''
    return f"Feedback({self.score}, {short_comment}{ellipsis})"

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
