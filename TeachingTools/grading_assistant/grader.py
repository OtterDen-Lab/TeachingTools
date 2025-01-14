#!env python
from __future__ import annotations

import abc
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

from typing import List, Tuple, Optional

from TeachingTools.grading_assistant.assignment import Assignment
from TeachingTools.lms_interface.classes import Feedback

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Grader(abc.ABC):
  pass

  @abc.abstractmethod
  def grade(self, assignment: Assignment, *args, **kwargs) -> None:
    """
    Takes an assignment and walks through its submissions and grades each.
    :param assignment: Takes in an assignment.Assignment object to grade
    :return:
    """
    pass


class Grader__Dummy(Grader):
  def grade(self, assignment: Assignment, *args, **kwargs) -> None:
    for submission in assignment.submissions:
      log.info(f"Grading {submission}...")
      submission.feedback = Feedback(43.0, "Excellent job")


class Grader__docker(Grader, abc.ABC):
  client = docker.from_env()
  
  def __init__(self, image=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.image = image if image is not None else "ubuntu"
    self.container: Optional[docker.models.containers.Container] = None
  
  @classmethod
  def build_docker_image(cls, base_image, github_repo):
    log.info("Building docker image for grading...")
    
    docker_file = io.BytesIO(f"""
    FROM samogden/cst334
    RUN git clone {github_repo} /tmp/grading/
    WORKDIR /tmp/grading
    CMD ["/bin/bash"]
    """.encode())
    
    image, logs = cls.client.images.build(
      fileobj=docker_file,
      tag="grading",
      pull=True,
      nocache=True
    )
    # log.debug(logs)
    log.debug("Docker image built successfully")
    return image
  
  def start(self, image : docker.models.images,):
    self.container = self.client.containers.run(
      image=image,
      detach=True,
      tty=True
    )
  
  def add_files_to_docker(self, files_to_copy : List[Tuple[str,str]] = None):
    """
    
    :param files_to_copy: Format is [(src, target), ...]):
    :return:
    """
    
    def add_file_to_container(src_file, target_dir, container):
      # Prepare the files as a tarball to push into container
      tarstream = io.BytesIO()
      with tarfile.open(fileobj=tarstream, mode="w") as tarhandle:
        tarhandle.add(src_file, arcname=os.path.basename(src_file))
      tarstream.seek(0)
      
      # Push student files to image
      container.put_archive(f"{target_dir}", tarstream)
    
    for src_file, target_dir in files_to_copy:
      add_file_to_container(src_file, target_dir, self.container)
  
  def execute(self, command="", container=None, workdir=None) -> Tuple[int, str, str]:
    log.debug(f"execute: {command}")
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
  
  def read_file(self, path_to_file) -> Optional[str]:
    
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
  
  def stop(self):
    self.container.stop(timeout=1)
    self.container.remove()
    self.container = None
  
  def __enter__(self):
    log.info(f"Starting docker image {self.image} context")
    self.start(self.image)
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    log.info(f"Exiting docker image context")
    self.stop()
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
  
  def grade(self, assignment: Assignment, *args, **kwargs) -> None:
    for submission in assignment.submissions:
      if submission.files is None or len(submission.files) == 0:
        submission.feedback = Feedback(0.0, "Assignment submission files missing")
        continue
      submission.feedback = self.grade_assignment(submission.files, **kwargs)

class Grader__CST334(Grader__docker):
  
  def __init__(self, assignment_path, use_online_repo=False):
    super().__init__()
    if use_online_repo:
      github_repo = "https://github.com/samogden/CST334-assignments-online.git"
    else:
      github_repo = "https://github.com/samogden/CST334-assignments.git"
    self.assignment_path = assignment_path
    self.image = self.build_docker_image(base_image="samogden/cst334", github_repo=github_repo)
  
  def check_for_trickery(self, files_submitted) -> bool:
    for input_file in files_submitted:
      try:
        with open(input_file) as f:
          if "exit(0)" in f.read():
            return True
      except IsADirectoryError:
        pass
    if not any(map(lambda f: f.endswith(".c") and "student_code" in f, files_submitted)):
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
  
  def execute_grading(self, programming_assignment, *args, **kwargs) -> Tuple[int, str, str]:
    rc, stdout, stderr = self.execute(
      command="timeout 120 python ../../helpers/grader.py --output /tmp/results.json",
      workdir=f"/tmp/grading/programming-assignments/{programming_assignment}/"
    )
    return rc, stdout, stderr
  
  def score_grading(self, *args, **kwargs) -> Feedback:
    results = self.read_file("/tmp/results.json")
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
  
  def grade_in_docker(self, source_dir, programming_assignment, lint_bonus) -> Feedback:
    files_to_copy = [
      (
        f,
        f"/tmp/grading/programming-assignments/{programming_assignment}/{'src' if f.endswith('.c') else 'include'}"
      )
      for f in [os.path.join(source_dir, f_wo_path) for f_wo_path in os.listdir(source_dir)]
    ]
    return super().grade_in_docker(files_to_copy, programming_assignment=programming_assignment, lint_bonus=lint_bonus)
  
  def grade_assignment(self, input_files: List[str], *args, **kwargs) -> Feedback:
    
    # Legacy settings
    use_max = "use_max" in kwargs and kwargs["use_name"]
    tags = ["main"] if "tags" not in kwargs else kwargs["tags"]
    num_repeats = 10 if "num_repeats" not in kwargs else kwargs["num_repeats"]
    
    # Setup input files
    # todo: convert to using a temp file since I currently have to manually delete later on
    if os.path.exists("student_code"):
      shutil.rmtree("student_code")
    os.mkdir("student_code")
    
    # Copy the student code to the staging directory
    files_copied = []
    for file_extension in [".c", ".h"]:
      try:
        file_to_copy = list(filter(lambda f: "student_code" in f and f.endswith(file_extension), input_files))[0]
        files_copied.append(file_to_copy)
        shutil.copy(
          file_to_copy,
          f"./student_code/student_code{file_extension}"
        )
      except IndexError:
        log.warning("Single file submitted")
    
    # Check for trickery, per Elijah's trials (so far)
    if self.check_for_trickery(files_copied):
      return Feedback(
        score=0.0,
        comments="It was detected that you might have been trying to game the scoring via exiting early from a unit test.  Please contact your professor if you think this was in error."
      )
    
    # Set up to be able to run multiple times
    # todo: I should probably move to the results format for this
    
    list_of_results : List[Feedback] = []
    
    for i in range(num_repeats):
      result = self.grade_in_docker(
        os.path.abspath("./student_code"),
        self.assignment_path,
        1
      )
      log.debug(result)
      list_of_results.append(result)
    shutil.rmtree("student_code")
    
    # Select best feedback and add a little bit on
    final_feedback = min(list_of_results, key=(lambda f: f.score))
    final_feedback.comments += "\n\n"
    final_feedback.comments += "###################\n"
    final_feedback.comments += "## Full results: ##\n"
    for i, result in enumerate(list_of_results):
      final_feedback.comments += f"test {i}: {result.comments} points\n"
    final_feedback.comments += "###################\n"
    
    return final_feedback