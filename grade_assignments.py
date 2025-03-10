#!env python
import argparse
import contextlib
import fcntl
import os
import pprint
import tempfile

import yaml

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.canvas_interface import CanvasInterface, CanvasCourse, CanvasAssignment
from TeachingTools.grading_assistant.assignment import AssignmentRegistry
from TeachingTools.grading_assistant.grader import GraderRegistry

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/grading__programming_assignments.yaml"))
  parser.add_argument("--limit", default=None, type=int)
  parser.add_argument("--regrade", "--do_regrade", dest="do_regrade", action="store_true")
  
  return parser.parse_args()


@contextlib.contextmanager
def working_directory(directory=None):
  """
  Context manager that either:
  1. Creates a temporary directory if no directory is provided
  2. Uses the provided directory if one is given
  
  In both cases, it yields the directory path and handles cleanup only for temp dirs
  
  Help from Claude: https://claude.ai/share/f5dc7e5a-23ab-4b7d-bef7-e6234587956a
  """
  temp_dir = None
  
  try:
    if directory is None:
      # Create a temporary directory if none is provided
      temp_dir = tempfile.TemporaryDirectory()
      directory = temp_dir.name
    else:
      directory = os.path.expanduser(directory)
      if not os.path.exists(directory):
        os.mkdir(directory)
    
    # Store the original working directory to return to
    original_dir = os.getcwd()
    
    # Change to the working directory
    os.chdir(directory)
    
    # Yield the path of the working directory
    yield directory
  
  finally:
    # Always return to the original directory
    os.chdir(original_dir)
    
    # Clean up the temporary directory if we created one
    if temp_dir is not None:
      temp_dir.cleanup()


def main():
  
  # First, check to make sure we are the only version running, since this can cause problems with docker and canvas otherwise
  lockfile = "/tmp/TeachingTools.grade_assignments.lock"
  lock_fd = open(lockfile, "w")
  try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError:
    log.warning("Early exiting because another instance is already running")
    return

  # Otherwise, continue with normal flow
  args = parse_args()
  
  # Load overall YAML
  with open(args.yaml) as fid:
    grader_info = yaml.safe_load(fid)
  
  log.debug(grader_info)
  
  # Pull flags from YAML file that will be applied to all submissions
  use_prod = grader_info.get('prod', False)
  push_grades = grader_info.get('push', False)
  root_dir = grader_info.get('root_dir', None)
  
  # Create the LMS interface
  lms_interface = CanvasInterface(prod=use_prod)
  
  # Walk through all defined courses, error if we don't have required information
  for yaml_course in grader_info.get('courses', []):
    try:
      course_id = int(yaml_course['id'])
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
      do_regrade = args.do_regrade
      
      # Get the grader from the registry
      grader = GraderRegistry.create(
        yaml_assignment.get("grader", "Dummy"),
        assignment_path=yaml_assignment.get('name')
      )
      
      with working_directory(root_dir) as working_dir:
        # Focus on the given assignment
        with AssignmentRegistry.create(
            yaml_assignment['kind'],
            lms_assignment=assignment,
            grading_root_dir=working_dir,
            **yaml_assignment.get('assignment_kwargs', {})
        ) as assignment:
          
          # If the grader doesn't need preparation (e.g. we have already graded and are just finalizing), then skip the prep step
          if grader.assignment_needs_preparation():
            # If manual, double check that we should clobber the files
            if yaml_assignment.get("grader", "Dummy").lower() in ["manual"]:
              prepare_assignment = input("Would you like to prepare assignment? (y/N) ").strip().lower()
              if prepare_assignment not in ['y', 'yes']:
                log.info("Aborting execution based on response")
                return
            assignment.prepare(
              limit=args.limit,
              do_regrade=do_regrade,
              **yaml_assignment.get("kwargs", {})
            )
            
          grader.grade_assignment(assignment, **assignment_grading_kwargs)
          
          for submission in assignment.submissions:
            log.debug(submission)
          
          if grader.ready_to_finalize:
            if yaml_assignment.get("grader", "Dummy").lower() in ["manual"]:
              finalize_assignment = input("Would you like to finalize assignment? (y/N) ").strip().lower()
              if finalize_assignment not in ['y', 'yes']:
                log.info("Aborting execution based on response")
                return
            assignment.finalize(push=push_grades)
      
      grader.cleanup()
  



if __name__ == "__main__":
  main()