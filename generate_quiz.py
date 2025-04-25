#!env python
import argparse
import os
import shutil
import subprocess
import tempfile

from TeachingTools.quiz_generation.quiz import Quiz
from TeachingTools.lms_interface.canvas_interface import CanvasInterface, CanvasCourse

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--prod", action="store_true")
  parser.add_argument("--course_id", type=int)
  
  parser.add_argument("--quiz_yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/exam_generation.yaml"))
  parser.add_argument("--num_canvas", default=0, type=int)
  parser.add_argument("--num_pdfs", default=0, type=int)
  
  subparsers = parser.add_subparsers(dest='command')
  test_parser = subparsers.add_parser("TEST")
  
  
  args = parser.parse_args()
  
  if args.num_canvas > 0 and args.course_id is None:
    log.error("Must provide course_id when pushing to canvas")
    exit(8)
  
  return args


def test():
  log.info("Running test...")
  pass
  
  
def generate_latex(latex_text, remove_previous=False):
  
  if remove_previous:
    if os.path.exists('out'): shutil.rmtree('out')
  
  tmp_tex = tempfile.NamedTemporaryFile('w')
  
  tmp_tex.write(latex_text)
  
  tmp_tex.flush()
  shutil.copy(f"{tmp_tex.name}", "debug.tex")
  p = subprocess.Popen(
    f"latexmk -pdf -output-directory={os.path.join(os.getcwd(), 'out')} {tmp_tex.name}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
  try:
    p.wait(30)
  except subprocess.TimeoutExpired:
    logging.error("Latex Compile timed out")
    p.kill()
    tmp_tex.close()
    return
  proc = subprocess.Popen(
    f"latexmk -c {tmp_tex.name} -output-directory={os.path.join(os.getcwd(), 'out')}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
  )
  proc.wait(timeout=30)
  tmp_tex.close()


def main():
  
  args = parse_args()
  
  if args.command == "TEST":
    test()
    return
  
  quizzes = Quiz.from_yaml(args.quiz_yaml)
  for quiz in quizzes:
    
    for i in range(args.num_pdfs):
      latex_text = quiz.get_quiz().render_latex()
      generate_latex(latex_text, remove_previous=(i==0))
    
    if args.num_canvas > 0:
      canvas_interface = CanvasInterface(prod=args.prod)
      canvas_course = canvas_interface.get_course(course_id=args.course_id)
      canvas_course.push_quiz_to_canvas(quiz, args.num_canvas, title=quiz.name, is_practice=quiz.practice)
    
    quiz.describe()


if __name__ == "__main__":
  main()