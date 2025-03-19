#!env python
import argparse
import os

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.canvas_interface import CanvasInterface, CanvasCourse, CanvasHelpers

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
  # Create the top-level parser
  parser = argparse.ArgumentParser(description="Canvas Helper runners")
  
  # Add global arguments
  parser.add_argument("--prod", action="store_true", help="Run in production mode")
  
  # Create a parent parser with common arguments
  parent_parser = argparse.ArgumentParser(add_help=False)
  parent_parser.add_argument("--course_id", required=True, type=int, help="Course ID")
  
  # Create subparsers
  subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
  
  # Create the parser for the "MARK_LATE" command, inheriting from parent
  mark_late_parser = subparsers.add_parser("MARK_LATE",
    help="Mark assignments as late",
    parents=[parent_parser])  # Include parent arguments
  mark_late_parser.add_argument("--assignment_id", required=True, type=int, help="Assignment ID")
  
  # Create the parser for the "GET_STUDENTS" command, inheriting from parent
  get_students_parser = subparsers.add_parser("GET_STUDENTS",
    help="Get list of students",
    parents=[parent_parser])  # Include parent arguments
  get_students_parser.add_argument("--fields_to_print", nargs="+", default=["name"])
  
  # Parse arguments
  args = parser.parse_args()
  return args

def main():
  
  args = parse_args()
  
  canvas_interface = CanvasInterface(prod=args.prod)
  canvas_course = canvas_interface.get_course(args.course_id)
  
  if args.command == "MARK_LATE":
    CanvasHelpers.mark_future_assignments_as_ungraded(canvas_course)
    
  elif args.command == "GET_STUDENTS":
    
    # Get students
    students = canvas_course.get_students()
    
    # Print out course information
    print(f"{canvas_course.name} ({len(students)} students)")
    
    # Calculate how long the maximum length of each field is for the students we got
    field_lengths = {
      field: max(len(str(getattr(s, field))) for s in students)
      for field in args.fields_to_print
    }
    
    # Print out request information for each student
    for student in students:
      info_str = ' : '.join(
        f"{getattr(student, field):{field_lengths[field]}}"
        for field in args.fields_to_print
      )
      print(info_str)
  
if __name__ == "__main__":
  main()