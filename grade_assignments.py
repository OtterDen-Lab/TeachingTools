#!env python
import argparse
import os
import pprint

import yaml

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.canvas_interface import CanvasInterface, CanvasCourse, CanvasAssignment
from TeachingTools.grading_assistant.assignment import Assignment__Exam, Assignment__ProgrammingAssignment
from TeachingTools.grading_assistant.grader import Grader__Dummy, Grader__CST334

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/programming_assignments.yaml"))
  
  parser.add_argument("--limit", default=None, type=int)
  
  return parser.parse_args()
  
  parser.add_argument("--root_dir", required=True)
  parser.add_argument("--pdf_in_dir", default="00-input")
  parser.add_argument("--limit", default=None, type=int)
  
  parser.add_argument("--prepare", action="store_true")
  parser.add_argument("--finalize", action="store_true")
  
  parser.add_argument("--course_id", required=True, type=int)
  parser.add_argument("--assignment_id", required=True, type=int)
  parser.add_argument("--push", action="store_true")
  parser.add_argument("--prod", action="store_true")
  
  parser.add_argument("--skip_name_check", action="store_false", dest="use_ai")
  
  return parser.parse_args()


def main():
  
  args = parse_args()
  
  # Load overall YAML
  with open(args.yaml) as fid:
    grader_info = yaml.safe_load(fid)
  
  log.debug(grader_info)
  
  # Pull flags from YAML file that will be applied to all submissions
  use_prod = grader_info.get('prod', False)
  push_grades = grader_info.get('push', False)
  root_dir = grader_info.get('root_dir') # todo: if not specified then use a temp dir
  
  # Create the LMS interface
  lms_interface = CanvasInterface(prod=use_prod)
  
  # Walk through all defined courses, error if we don't have required information
  for yaml_course in grader_info.get('courses', []):
    try:
      course_id = yaml_course['id']
    except KeyError as e:
      log.error("No course ID specified.  Please update.")
      log.error(f"{pprint.pformat(yaml_course)}")
      log.error(e)
      return
    
    # Create course object if found
    course = lms_interface.get_course(course_id)
    
    # Walk through assignments in course to grade, error if we don't have required information
    for yaml_assignment in yaml_course.get('assignments', []):
      if yaml_assignment.get('disabled', False):
        continue
      try:
        assignment_id = yaml_assignment['id']
      except KeyError as e:
        log.error("No assignment ID specified.  Please update.")
        log.error(f"{pprint.pformat(yaml_course)}")
        log.error(e)
        return
      
      # Create assignment object if we have enough information
      assignment = course.get_assignment(assignment_id)
      assignment_grading_kwargs = yaml_assignment.get('kwargs', {})
      
      # Focus on the given assignment
      with Assignment__ProgrammingAssignment(lms_assignment=assignment, grading_root_dir=root_dir) as assignment:
        assignment.prepare(
          limit=args.limit,
          regrade=True
        )
        grader = Grader__CST334(assignment_path=yaml_assignment.get('name'))
        grader.grade(assignment, **assignment_grading_kwargs)
        for submission in assignment.submissions:
          log.debug(submission)
        assignment.finalize(push=push_grades)
  
  
  return
  root_dir = args.root_dir
  pdf_in_dir = args.pdf_in_dir
  prepare = args.prepare
  finalize = args.finalize
  course_id = args.course_id
  assignment_id = args.assignment_id
  
  with Assignment__Exam(root_dir) as assignment:
    canvas_interface = CanvasCourse(prod=args.prod, course_id=course_id)
    canvas_assignment = canvas_interface.get_assignment(assignment_id=assignment_id)
    if prepare:
      assignment.prepare(input_directory=pdf_in_dir, limit=args.limit, canvas_interface=canvas_interface, use_ai=args.use_ai)
    if finalize:
      assignment.finalize("grades.intermediate.csv", canvas_assignment, push=args.push)




if __name__ == "__main__":
  main()