#!env python
import argparse
import os

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.canvas_interface import CanvasAssignment
from TeachingTools.grading_assistant.assignment import Assignment__exams_pdf

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def parse_args():
  parser = argparse.ArgumentParser()
  
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
  
  root_dir = args.root_dir
  pdf_in_dir = args.pdf_in_dir
  prepare = args.prepare
  finalize = args.finalize
  course_id = args.course_id
  assignment_id = args.assignment_id
  
  with Assignment__exams_pdf(root_dir) as assignment:
    canvas_interface = CanvasAssignment(prod=args.prod, course_id=course_id, assignment_id=assignment_id)
    if prepare:
      assignment.prepare(input_directory=pdf_in_dir, limit=args.limit, canvas_interface=canvas_interface, use_ai=args.use_ai)
    if finalize:
      assignment.finalize("grades.intermediate.csv", canvas_interface, push=args.push)




if __name__ == "__main__":
  main()