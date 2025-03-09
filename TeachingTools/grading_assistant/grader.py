#!env python
from __future__ import annotations

import abc
import ast
import importlib
import pathlib
import pkgutil
import pprint
import time

import docker
import docker.errors
import docker.models.images
import docker.models.containers
import io
import tarfile
import os
import textwrap
import json
import shutil
import yaml
from collections import defaultdict
import pandas as pd

from typing import List, Tuple, Optional

from TeachingTools.grading_assistant.assignment import Assignment
from TeachingTools.lms_interface.classes import Feedback, Submission

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class GraderRegistry:
  _registry = {}
  _scanned = False
  
  @classmethod
  def register(cls, grader_type=None):
    log.debug("Registering...")
    
    def decorator(subclass):
      # Use the provided name or fall back to the class name
      name = grader_type.lower() if grader_type else subclass.__name__.lower()
      cls._registry[name] = subclass
      return subclass
    
    return decorator
  
  @classmethod
  def create(cls, grader_type, **kwargs):
    """Instantiate a registered subclass."""
    
    # If we haven't already loaded our premades, do so now
    if not cls._scanned:
      cls.load_premade_graders()
    # Check to see if it's in the registry
    if grader_type.lower() not in cls._registry:
      raise ValueError(f"Unknown grader type: {grader_type}")
    
    return cls._registry[grader_type.lower()](**kwargs)
  
  
  @classmethod
  def load_premade_graders(cls):
    package_name = "TeachingTools.grading_assistant"  # Fully qualified package name
    package_path = pathlib.Path(__file__).parent / "grader"
    log.debug(f"package_path: {package_path}")
    
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
      # Import the module
      module = importlib.import_module(f"{package_name}.{module_name}")
      log.debug(f"Loaded module: {module}")


@GraderRegistry.register("Dummy")
class Grader(abc.ABC):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.ready_to_finalize = True

  def grade_assignment(self, assignment: Assignment, *args, **kwargs) -> None:
    """
    Takes an assignment and walks through its submissions and grades each.
    :param assignment: Takes in an assignment.Assignment object to grade
    :return:
    """
    for submission in assignment.submissions:
      if submission.files is None or len(submission.files) == 0:
        submission.feedback = Feedback(0.0, "Assignment submission files missing")
        continue
      submission.feedback = self.grade_submission(submission, **kwargs)

  def grade_submission(self, submission: Submission, **kwargs) -> Feedback:
    """
    Takes in a submission, grades it, and returns back a Feedback
    :param submission: A Submission object that may have files associated with it
    :param kwargs:
    :return: returns a Feedback object for the submission
    """
    return Feedback(score=0.0, comments="(grade_submission not implemented)")

  def assignment_needs_preparation(self):
    return True

  def prepare(self, *args, **kwargs):
    """
    Anything that is needed to take the assignment and prepare it for grading.
    For example, making a CSV file from the submissions for manual grading
    :param args:
    :param kwargs:
    :return:
    """
  
  def finalize(self, *args, **kwargs):
    """
    anything that is needed to connect the grades/feedback to the submissions after grading.
    For example, loading up the CSV and connecting grades to the submissions
    :param args:
    :param kwargs:
    :return:
    """


@GraderRegistry.register("Manual")
class Grader__Manual(Grader):
  
  CSV_NAME = "grades.intermediate.csv"
  
  def is_grading_complete(self):
    """
    Checks to see if grading is complete.  Currently just looks for whether there is a `total` score for each entry.
    :return:
    """
    if not os.path.exists(self.CSV_NAME): return False
    
    grades_df = pd.read_csv(self.CSV_NAME)
    
    # Clean out the extra columns not associated with any submission
    grades_df = grades_df[grades_df["document_id"].notna()]
    
    # If there are entries missing a `total` column then we should get a different count and are incomplete
    return grades_df[grades_df["total"].notna()].shape == grades_df.shape
  
  def prepare(self, assignment: Assignment, *args, **kwargs):
    self.ready_to_finalize = False
    log.debug("Preparing manual grading")
    # Make a dataframe
    df = pd.DataFrame([
      {
        **submission.extra_info,
        #"page_mappings" : page_mappings_by_user[submission.document_id],
        "name" : submission.student.name if submission.student is not None else "",
        "user_id" : submission.student.user_id if submission.student is not None else "",
        "total" : None
      }
      for submission in assignment.submissions
    ])
    print(df.head())
    df = df.sort_values(by="document_id")
    
    df.to_csv(self.CSV_NAME, index=False)
  
  def finalize(self, assignment, *args, **kwargs):
    log.debug("Finalizing manual grading")
    if not self.is_grading_complete():
      log.error("It seems like some entries do not have scores.  Please correct and rerun.")
      exit(4)
    
    # Steps:
    # 1. Recreate submissions
    # 2. Pass back to assignment to remerge
    # 3. Generate grades and feedback
    
    # Load CSV
    grades_df = pd.read_csv(self.CSV_NAME)
    # Remove any extra information (because I like tracking my progress)
    grades_df = grades_df[grades_df["document_id"].notna()]
    
    # Get list of students from canvas
    # todo: this should probably be done in the `assignment`
    canvas_students_by_id = { s.user_id : s for s in assignment.lms_assignment.get_students() }
    
    graded_submissions : List[Submission] = []
    
    # Make submission objects for students who have already been matched
    for _, row in grades_df[grades_df["user_id"].notna()].iterrows():
      log.debug(canvas_students_by_id[int(row["user_id"])])
      submission = Submission(
        student=canvas_students_by_id[int(row["user_id"])],
        status=Submission.Status.GRADED,
      )
      # todo: get PDFs and comments.
      submission.feedback = Feedback(
        score=row["total"],
        comments="(comments_to_be_added later)",
        attachments=[]
      )
      submission.set_extra({
        "page_mappings": ast.literal_eval(row["page_mappings"]),
        "document_id": row["document_id"]
      })
      graded_submissions.append(submission)
      del canvas_students_by_id[int(row["user_id"])]
    
    log.info(f"There were {len(graded_submissions)} matched canvas users.")
    log.info(f"There are {len(canvas_students_by_id)} unmatched canvas users")
    log.debug("\n" + pprint.pformat(canvas_students_by_id))
    
    # If we have unmatched students, exit because they should be manually matched.
    if grades_df[grades_df["user_id"].isna()].shape[0] > 0:
      log.error("There were unmatched students.  Please correct and re-run.")
      exit(2)
    
    # Now we have a list of graded submissions
    log.info(f"We have graded {len(graded_submissions)} submissions!")
    assignment.submissions = graded_submissions
    self.ready_to_finalize = True
  
  def grade_assignment(self, assignment: Assignment, *args, **kwargs) -> None:
    if self.is_grading_complete():
      self.finalize(assignment, args, **kwargs)
    else:
      self.prepare(assignment, *args, **kwargs)
  
  def assignment_needs_preparation(self):
    return not self.is_grading_complete()


class Grader__docker(Grader, abc.ABC):
  client = None
  
  def __init__(self, image=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # Set up docker client class-wide if it hasn't been set up yet
    if self.__class__.client is None:
      self.__class__.client = docker.from_env()
    
    # Default to using ubuntu image
    self.image = image if image is not None else "ubuntu"
    self.container: Optional[docker.models.containers.Container] = None
    
  def __del__(self):
    # Try to remove image, and if it hasn't been set up properly delete
    try:
      self.image.remove(force=True)
    except AttributeError:
      pass
  
  @classmethod
  def build_docker_image(cls, dockerfile_str):
    """
    Given a dockerfile as a string, creates and returns this image
    :param dockerfile_str: dockerfile as a single string
    :return: a docker image
    """
    log.info("Building docker image for grading...")
    
    image, logs = cls.client.images.build(
      fileobj=io.BytesIO(dockerfile_str.encode()),
      pull=True,
      nocache=True,
      tag=f"grading:{cls.__name__.lower()}",
      rm=True,
      forcerm=True
    )
    
    log.debug(f"Successfully build docker image {image.tags}")
    return image
  
  def start_container(self, image : docker.models.images):
    self.container = self.client.containers.run(
      image=image,
      detach=True,
      tty=True,
      remove=True
    )
    
  def stop_container(self):
    self.container.stop(timeout=1)
    self.container = None
  
  def add_files_to_docker(self, files_to_copy : List[Tuple[str,str]] = None):
    """
    
    :param files_to_copy: Format is [(src, target), ...]):
    :return:
    """
  
    def add_file_to_container(src_file, target_dir, container):
      # Create a TarInfo object
      tar_info = tarfile.TarInfo(name=src_file.name if hasattr(src_file, 'name') else 'file')
      
      # Get file size
      src_file.seek(0, io.SEEK_END)
      tar_info.size = src_file.tell()
      src_file.seek(0)  # Reset to beginning
      
      # Set modification time
      tar_info.mtime = int(time.time())
      
      # Prepare the tarball
      tarstream = io.BytesIO()
      with tarfile.open(fileobj=tarstream, mode="w") as tarhandle:
        tarhandle.addfile(tar_info, src_file)
      tarstream.seek(0)
      
      # Push to container
      container.put_archive(f"{target_dir}", tarstream)
    
    for src_file, target_dir in files_to_copy:
      add_file_to_container(src_file, target_dir, self.container)
  
  def execute_command_in_container(self, command="", container=None, workdir=None) -> Tuple[int, str, str]:
    log.debug(f"executing: {command}")
    if container is None:
      container = self.container
    
    extra_args = {}
    if workdir is not None:
      extra_args["workdir"] = workdir
    
    rc, (stdout, stderr) = container.exec_run(
      cmd=f"bash -c \"{command}\"",
      demux=True,
      tty=True,
      **extra_args
    )
    
    return rc, stdout, stderr
  
  def read_file_from_container(self, path_to_file) -> Optional[str]:
    
    try:
      # Try to find the file on the system
      bits, stats = self.container.get_archive(path_to_file)
    except docker.errors.APIError as e:
      log.error(f"Get archive failed: {e}")
      return None
    
    # Read file from docker
    f = io.BytesIO()
    for chunk in bits:
      f.write(chunk)
    f.seek(0)
    
    # Open the tarball we just pulled and read the contents to a string buffer
    with tarfile.open(fileobj=f, mode="r") as tarhandle:
      results_f = tarhandle.getmember("results.json")
      f = tarhandle.extractfile(results_f)
      f.seek(0)
      return f.read().decode()
  
  def __enter__(self):
    log.info(f"Starting docker image {self.image} context")
    self.start_container(self.image)
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    log.info(f"Exiting docker image context")
    self.stop_container()
    if exc_type is not None:
      log.error(f"An exception occured: {exc_val}")
      log.error(exc_tb)
    return False
  
  @abc.abstractmethod
  def execute_grading(self, *args, **kwargs):
    pass
  
  @abc.abstractmethod
  def score_grading(self, execution_results, *args, **kwargs) -> Feedback:
    pass
  
  def grade_in_docker(self, files_to_copy=None, *args, **kwargs) -> Feedback:
    with self:
      if files_to_copy is not None:
        self.add_files_to_docker(files_to_copy)
      execution_results = self.execute_grading(*args, **kwargs)
      return self.score_grading(execution_results,*args,  **kwargs)


@GraderRegistry.register("CST334")
class Grader__CST334(Grader__docker):
  
  dockerfile_str = """
  FROM samogden/cst334
  RUN git clone https://www.github.com/samogden/CST334-assignments.git /tmp/grading/
  WORKDIR /tmp/grading
  CMD ["/bin/bash"]
  """
  
  def __init__(self, assignment_path, use_online_repo=False):
    super().__init__()
    if use_online_repo:
      github_repo = "https://github.com/samogden/CST334-assignments-online.git"
    else:
      github_repo = "https://github.com/samogden/CST334-assignments.git"
    self.assignment_path = assignment_path
    self.image = self.build_docker_image(self.dockerfile_str)
  
  def check_for_trickery(self, submission) -> bool:
    def contains_string(str, f) -> bool:
      try:
        if str.encode() in f.read():
          return True
        else:
          return False
      finally:
        f.seek(0)
      
    for f in submission.files:
      if contains_string("exit(0)", f):
        return True
    return False
  
  @staticmethod
  def build_feedback(results_dict, score=None) -> str:
    feedback_strs = [
      "##############",
      "## FEEDBACK ##",
      "##############",
      "",
    ]
    
    if score is not None:
      feedback_strs.extend([
        f"Score reported: {score} points",
        ""
      ])
    
    if "overall_feedback" in results_dict:
      feedback_strs.extend([
        "## Overall Feedback ##",
        results_dict["overall_feedback"],
        "\n\n"
      ])
    
    feedback_strs.extend([
      "## Unit Tests ##",
    ])
    if "suites" in results_dict:
      for suite_name in results_dict["suites"].keys():
        
        if len(results_dict["suites"][suite_name]["PASSED"]) > 0:
          feedback_strs.extend([
            f"SUITE: {suite_name}",
            "  * passed:",
          ])
          feedback_strs.extend([
            textwrap.indent('\n'.join(results_dict["suites"][suite_name]["PASSED"]), '    '),
            ""
          ])
        
        if len(results_dict["suites"][suite_name]["FAILED"]) > 0:
          feedback_strs.extend([
            f"SUITE: {suite_name}",
            "  * failed:",
          ])
          feedback_strs.extend([
            textwrap.indent('\n'.join(results_dict["suites"][suite_name]["FAILED"]), '    '),
            ""
          ])
      feedback_strs.extend([
        "################",
        "",
      ])
    
    if "build_logs" in results_dict:
      feedback_strs.extend([
        "## Build Logs ##",
      ])
      feedback_strs.extend([
        "Build Logs:",
        ''.join(results_dict["build_logs"])[1:-1].encode('utf-8').decode('unicode_escape')
      ])
      feedback_strs.extend([
        "################",
      ])
    
    
    if "lint_logs" in results_dict:
      feedback_strs.extend([
        "## Lint Logs ##",
        f"Lint success: {results_dict['lint_success']}\n"
      ])
      feedback_strs.extend([
        "Lint Logs:",
        ''.join(results_dict["lint_logs"])[1:-1].encode('utf-8').decode('unicode_escape')
      ])
      feedback_strs.extend([
        "################",
      ])
    
    return '\n'.join(feedback_strs)
  
  def execute_grading(self, path_to_programming_assignment, *args, **kwargs) -> Tuple[int, str, str]:
    rc, stdout, stderr = self.execute_command_in_container(
      command="timeout 120 python ../../helpers/grader.py --output /tmp/results.json",
      # command="make",
      workdir=f"/tmp/grading/{path_to_programming_assignment}/"
    )
    return rc, stdout, stderr
  
  def score_grading(self, *args, **kwargs) -> Feedback:
    results = self.read_file_from_container("/tmp/results.json")
    if results is None:
      # Then something went awry in reading back feedback file
      return Feedback(
        score=0,
        comments="Something went wrong during grading, likely a timeout.  Please check your assignment for infinite loops and/or contact your professor."
      )
    results_dict = json.loads(results)
    if "lint_success" in results_dict and results_dict["lint_success"] and "lint_bonus" in kwargs:
      results_dict["score"] += kwargs["lint_bonus"]
    
    return Feedback(
      score=results_dict["score"],
      comments=self.build_feedback(results_dict, results_dict["score"])
    )
  
  def grade_submission(self, submission, **kwargs) -> Feedback:
    # todo: make it flexible for programming assignments
    path_to_programming_assignment = kwargs.get("path_to_programming_assignment", "programming-assignments/PA1")
    
    # Gather submission files in a format to copy over
    submission_files = []
    for f in submission.files:
      log.debug(f"f: {f.__class__} {f.name}")
      submission_files.append(
        (f, f"/tmp/grading/{path_to_programming_assignment}/{'src' if f.name.endswith('.c') else 'include'}")
      )
    
    # Check for trickery, per Elijah's trials (so far)
    if self.check_for_trickery(submission):
      return Feedback(
        score=0.0,
        comments="It was detected that you might have been trying to game the scoring via exiting early from a unit test.  Please contact your professor if you think this was in error."
      )
    
    # Grade as many times as we're requested to, gathering results for later
    all_feedback : List[Feedback] = []
    
    for i in range(kwargs.get("num_repeats", 3)):
      # Grade results in docker
      all_feedback.append(
        self.grade_in_docker(
          files_to_copy=submission_files,
          path_to_programming_assignment=path_to_programming_assignment
        )
      )
      
    # Select feedback and return
    feedback = min(all_feedback)
    
    full_feedback =  "##################\n"
    full_feedback += "## All results: ##\n"
    for i, result in enumerate(all_feedback):
      full_feedback += f"test {i}: {result.comments} points\n"
    full_feedback += "##################\n"

    feedback.comments += f"\n\n\n{full_feedback}"
    return feedback

@GraderRegistry.register("CST334online")
class Grader__CST334online(Grader__CST334):
  dockerfile_str = """
  FROM samogden/cst334
  RUN git clone https://www.github.com/samogden/CST334-assignments-online.git /tmp/grading/
  WORKDIR /tmp/grading
  CMD ["/bin/bash"]
  """


@GraderRegistry.register("Step-by-step")
class Grader_stepbystep(Grader__docker):
  
  def __init__(self, rubric_file, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.rubric = self.parse_rubric(rubric_file)
    self.golden_container : docker.models.containers.Container = None
    self.student_container : docker.models.containers.Container = None
  
  def parse_rubric(self, rubric_file):
    with open(rubric_file) as fid:
      rubric = yaml.safe_load(fid)
    if not isinstance(rubric["steps"], list):
      rubric["steps"] = rubric["steps"].split('\n')
    return rubric
  
  def parse_student_file(self, student_file):
    with open(student_file) as fid:
      return [l.strip() for l in fid.readlines()]
  
  def rollback(self):
    # Stop and delete student container
    self.student_container.stop(timeout=1)
    self.student_container.remove(force=True)
    self.student_container = None
    
    # Make a copy of the golden_container
    rollback_image = self.golden_container.commit(repository="rollback", tag="latest")
    
    # Start student from the copy we just made
    self.student_container = self.client.containers.run(
      image=rollback_image.id,
      detach=True,
      tty=True
    )
  
  def start(self, image : docker.models.images,):
    self.golden_container = self.client.containers.run(
      image=image,
      detach=True,
      tty=True
    )
    self.student_container = self.client.containers.run(
      image=image,
      detach=True,
      tty=True
    )
  
  def stop_container(self):
    self.golden_container.stop(timeout=1)
    self.golden_container.remove(force=True)
    self.golden_container = None
    self.student_container.stop(timeout=1)
    self.student_container.remove(force=True)
    self.student_container = None
  
  
  def execute_grading(self, golden_lines=[], student_lines=[], rollback=True, *args, **kwargs):
    golden_results = defaultdict(list)
    student_results = defaultdict(list)
    def add_results(results_dict, rc, stdout, stderr):
      results_dict["rc"].append(rc)
      results_dict["stdout"].append(stdout)
      results_dict["stderr"].append(stderr)
    
    for i, (golden, student) in enumerate(zip(golden_lines, student_lines)):
      log.debug(f"commands: '{golden}' <-> '{student}'")
      rc_g, stdout_g, stderr_g = self.execute_command_in_container(container=self.golden_container, command=golden)
      rc_s, stdout_s, stderr_s = self.execute_command_in_container(container=self.student_container, command=student)
      add_results(golden_results, rc_g, stdout_g, stderr_g)
      add_results(student_results, rc_s, stdout_s, stderr_s)
      if (not self.outputs_match(stdout_g, stdout_s, stderr_g, stderr_s, rc_g, rc_s) ) and rollback:
        # Bring the student container up to date with our container
        self.rollback()
    
    return golden_results, student_results
  
  @staticmethod
  def outputs_match(stdout_g, stdout_s, stderr_g, stderr_s, rc_g, rc_s) -> bool:
    if stdout_g != stdout_s:
      return False
    if stderr_g != stderr_s:
      return False
    if rc_g != rc_s:
      return False
    return True
  
  def score_grading(self, execution_results, *args, **kwargs) -> Feedback:
    log.debug(f"execution_results: {execution_results}")
    golden_results, student_results = execution_results
    num_lines = len(golden_results["stdout"])
    num_matches = 0
    for i in range(num_lines):
      if not self.outputs_match(
          golden_results["stdout"][i], student_results["stdout"][i],
          golden_results["stderr"][i], student_results["stderr"][i],
          golden_results["rc"][i], student_results["rc"][i]
      ):
        continue
      num_matches += 1
    
    return Feedback(
      score=(100.0 * num_matches / len(golden_results["stdout"])),
      comments=f"Matched {num_matches} out of {len(golden_results['stdout'])}"
    )
  
  
  def grade_assignment(self, input_files: List[str], *args, **kwargs) -> Feedback:
    
    golden_lines = self.rubric["steps"]
    student_lines = self.parse_student_file(input_files[0])
    
    results = self.grade_in_docker(golden_lines=golden_lines, student_lines=student_lines, *args, **kwargs)
    
    log.debug(f"final results: {results}")
    return results

