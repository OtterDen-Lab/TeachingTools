#!env python
import argparse
import os

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.canvas_interface import CanvasInterface

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--prod", action="store_true")
  parser.add_argument("--course_id", default=25523, type=int)
  
  parser.add_argument("--quiz_yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/exam.yaml"))
  parser.add_argument("--num_canvas", default=0, type=int)
  parser.add_argument("--num_pdfs", default=0, type=int)
  
  args = parser.parse_args()
  return args

def main():
  
  args = parse_args()
  
  quizzes = Quiz.from_yaml(args.quiz_yaml)
  for quiz in quizzes:
    quiz.select_questions()
    
    for q in quiz:
      log.debug(q.kind)
    
    for i in range(args.num_pdfs):
      quiz.generate_latex(remove_previous=(i==0))
    
    if args.num_canvas > 0:
      interface = CanvasInterface(prod=args.prod, course_id=args.course_id)
      interface.push_quiz_to_canvas(quiz, args.num_canvas, title=quiz.name, is_practice=quiz.practice)
    
    quiz.describe()


if __name__ == "__main__":
  main()