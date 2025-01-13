#!env python

import logging
import dataclasses

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



@dataclasses.dataclass
class Student:
  name : str
  user_id : int
