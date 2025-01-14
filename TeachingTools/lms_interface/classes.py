#!env python
from __future__ import annotations

import enum
import logging
import dataclasses
import functools
import io
import os
import urllib
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
    MISSING = enum.auto()
    UNGRADED = enum.auto()
    GRADED = enum.auto()
    
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
  

class Submission__Canvas(Submission):
  def __init__(self, *args, attachments : Optional[List], **kwargs):
    super().__init__(*args, **kwargs)
    self._attachments = attachments
  
  @property
  def files(self):
    # Check if we have already downloaded the files locally and return if we have
    if self._files is not None:
      return self._files
    
    # If we haven't downloaded the files yet, check if we have attachments and can download them
    if self._attachments is not None:
      self._files = []
      download_dir = "files"
      if not os.path.exists(download_dir):
        os.mkdir(download_dir)
      for attachment in self._attachments:
        
        # Generate a local file name with a number of options
        local_file_name = f"{self.student.name.replace(' ', '-')}_{self.student.user_id}_{attachment['filename']}"
        local_path = os.path.join(download_dir, local_file_name)
        urllib.request.urlretrieve(attachment['url'], local_path)
        
        self._files.append(local_path)
        
        
    return self._files



@functools.total_ordering
@dataclasses.dataclass
class Feedback:
  score: Optional[float] = None
  comments: str = ""
  attachments: List[io.BytesIO] = dataclasses.field(default_factory=list)
  
  def __str__(self):
    return f"Feedback({self.score}, {self.comments[:10].replace('\n','\\n')}{'...' if len(self.comments) > 10 else ''})"
  
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
