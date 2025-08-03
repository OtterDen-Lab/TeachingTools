#!env python
import argparse
import os
import shutil
import subprocess
import tempfile
from lms_interface.canvas_interface import CanvasInterface, CanvasCourse

from TeachingTools.quiz_generation.quiz import Quiz


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

def generate_quiz(
    path_to_quiz_yaml,
    num_pdfs=0,
    num_canvas=0,
    use_prod=False,
    course_id=None
):
  
  quizzes = Quiz.from_yaml(path_to_quiz_yaml)
  for quiz in quizzes:
    
    for i in range(num_pdfs):
      log.debug(f"Generating PDF {i+1}/{num_pdfs}")
      latex_text = quiz.get_quiz().render_latex()
      generate_latex(latex_text, remove_previous=(i==0))
    
    if num_canvas > 0:
      canvas_interface = CanvasInterface(prod=use_prod)
      canvas_course = canvas_interface.get_course(course_id=course_id)
      canvas_course.push_quiz_to_canvas(quiz, num_canvas, title=quiz.name, is_practice=quiz.practice)
    
    quiz.describe()

def main():
  
  args = parse_args()
  
  if args.command == "TEST":
    test()
    return
  
  generate_quiz(
    args.quiz_yaml,
    num_pdfs=args.num_pdfs,
    num_canvas=args.num_canvas,
    use_prod=args.prod,
    course_id=args.course_id
  )


if __name__ == "__main__":
  main()