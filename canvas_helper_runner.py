#!env python
import argparse
import glob
import logging
import os
import shutil
import subprocess

from TeachingTools.lms_interface.canvas_interface import CanvasInterface, CanvasHelpers

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
  # Create the top-level parser
  parser = argparse.ArgumentParser(description="Canvas Helper runners")
  
  # Add global arguments
  parser.add_argument("--prod", action="store_true", help="Run in production mode")
  parser.add_argument("--limit", type=int, default=None)
  
  # Create a parent parser with common arguments
  parent_parser = argparse.ArgumentParser(add_help=False)
  parent_parser.add_argument("--course_id", required=True, type=int, help="Course ID")
  
  # Create subparsers
  subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
  
  # Create the parser for the "MARK_LATE" command, inheriting from parent
  mark_late_parser = subparsers.add_parser(
    "MARK_LATE",
    help="Mark assignments as late",
    parents=[parent_parser]
  )  # Include parent arguments
  mark_late_parser.add_argument("--assignment_id", required=True, type=int, help="Assignment ID")
  
  # Create the parser for the "GET_STUDENTS" command, inheriting from parent
  get_students_parser = subparsers.add_parser(
    "GET_STUDENTS",
    help="Get list of students",
    parents=[parent_parser]
  )  # Include parent arguments
  get_students_parser.add_argument("--fields_to_print", nargs="+", default=["name"])
  
  # Create the parser for the "GET_STUDENTS" command, inheriting from parent
  get_all_submissions_parser = subparsers.add_parser(
    "GET_SUBMISSIONS",
    help="Download submissions",
    parents=[parent_parser]
  )  # Include parent arguments
  get_all_submissions_parser.add_argument("--assignment_id", required=True, type=int, help="Assignment ID")
  get_all_submissions_parser.add_argument("--base_files", nargs="+")
  
  # Create the parser for the "GET_STUDENTS" command, inheriting from parent
  get_all_submissions_parser = subparsers.add_parser(
    "RUN_MOSS",
    help="Run MOSS script on submissions",
    parents=[parent_parser]
  )  # Include parent arguments
  get_all_submissions_parser.add_argument("--assignment_id", required=True, type=int, help="Assignment ID")
  get_all_submissions_parser.add_argument("--base_files", nargs="+")
  
  # Parse arguments
  args = parser.parse_args()
  return args


def submit_to_moss(language, directory, base_files=None, max_matches=3):
  """
  Submit files to MOSS plagiarism detection service.
  
  Args:
      language (str): Programming language ('c', 'cc', 'java', 'ml', 'pascal', 'ada', 'lisp', 'scheme', 'haskell',
                      'fortran', 'ascii', 'vhdl', 'perl', 'matlab', 'python', 'mips', 'prolog', 'spice', 'vb',
                      'csharp', 'modula2', 'a8086', 'javascript', 'plsql')
      directory (str): Directory containing files to check
      base_files (list): List of base files that students were given
      max_matches (int): Maximum number of matches to show
  
  Returns:
      str: URL to the MOSS results
  """
  # Path to moss.pl script (you need to download this)
  moss_script = os.path.expanduser("~/scripts/moss.pl")
  
  # Make sure the script is executable
  os.chmod(moss_script, 0o755)
  
  # Start building the command
  cmd = [moss_script, "-l", language, "-m", str(max_matches)]
  
  # Add base files if provided
  if base_files:
    for base in base_files:
      cmd.extend(["-b", base])
  
  # Add all .c files from the directory
  c_files = glob.glob(os.path.join(directory, "*.c"))
  cmd.extend(c_files)
  
  # Execute the command and capture output
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  stdout, stderr = process.communicate()
  
  if process.returncode != 0:
    raise Exception(f"MOSS submission failed: {stderr}")
  
  # The last line of stdout should contain the URL
  lines = stdout.strip().split('\n')
  result_url = lines[-1]
  
  return result_url


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
  
  elif args.command in ["GET_SUBMISSIONS", "RUN_MOSS"]:
    download_dir = "files"
    assignment = canvas_course.get_assignment(assignment_id=args.assignment_id)
    
    # Erase old files
    shutil.rmtree(download_dir, ignore_errors=True)
    os.mkdir(download_dir)
    
    # Download each submission, indexing them as they were updated
    for submission in assignment.get_submissions(limit=args.limit):
      log.debug(f"{submission.student.name} ({submission.submission_index + 1})")
      for f in submission.files:
        filename = os.path.join(
          download_dir,
          f"{submission.student.name.replace(' ', '')}_{submission.submission_index}_{f.name}"
        )
        with open(filename, 'wb') as fid:
          fid.write(f.read())
    
    # Running MOSS if we are running that subparser
    if args.command == "RUN_MOSS":
      moss_url = submit_to_moss("c", download_dir, base_files=args.base_files)
      print(moss_url)


if __name__ == "__main__":
  main()
