#!env python
from __future__ import annotations

import argparse
import ast
import base64
import collections
import dataclasses
import importlib
import math
import pathlib
import pkgutil
import random
import shutil
import sys
import time
import urllib
from typing import List, Tuple, Dict, Optional
import io
import abc
import enum
import functools
import fitz
import fuzzywuzzy.fuzz
import numpy as np
import os

import pandas as pd

import TeachingTools.grading_assistant.ai_helper as ai_helper
from TeachingTools.lms_interface.canvas_interface import CanvasCourse, CanvasAssignment
from TeachingTools.lms_interface.classes import Student, Submission, Feedback

import logging
import colorama


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

NAME_SIMILARITY_THRESHOLD = 95


class Assignment(abc.ABC):
  """
  Class to represent an assignment and act as an abstract base class for other classes.  Will be passed to a grader.
  Two functions need to be overriden for child classes:
  1. prepare : prepares files for grading by downloading, anonymizing, etc.
  2. finalize : combines parts of grading as necessary
  """
  def __init__(self, lms_assignment : CanvasAssignment, grading_root_dir, *args, **kwargs):
    self.lms_assignment = lms_assignment
    self.grading_root_dir = grading_root_dir
    self.submissions : List[Submission] = []
    self.original_dir = None
  
  def __enter__(self) -> Assignment:
    """Enables use as a context manager (e.g. `with [Assignment]`) by managing working directory"""
    # todo: Enable use of anonymous temp directories
    self.original_dir = os.getcwd()
    os.chdir(self.grading_root_dir)
    return self
    
  def __exit__(self, exc_type, exc_value, traceback):
    """Enables use as a context manager (e.g. `with [Assignment]`) by managing working directory"""
    os.chdir(self.original_dir)
  
  @abc.abstractmethod
  def prepare(self, *args, **kwargs):
    """
    This function is intended to set up any directories or files as appropriate for grading.
    It should take in some sort of input and prepare Submissions to be passed to a grader object.
    :return: None
    """
    pass
  
  def finalize(self, *args, **kwargs):
    """
    This function is intended to finalize any grading.  This could be reloading the grading CSV and matching names,
    or could just be a noop.
    :param args:
    :param kwargs:
    :return:
    """
    if kwargs.get("push", False):
      log.debug("Pushing")
      for submission in self.submissions:
        log.info(f"Pushing feedback for: {submission}")
        self.lms_assignment.push_feedback(
          score=submission.feedback.score,
          comments=submission.feedback.comments,
          attachments=submission.feedback.attachments,
          user_id=submission.student.user_id,
          keep_previous_best=True,
          clobber_feedback=False
        )


class AssignmentRegistry:
  _registry = {}
  _scanned = False
  
  @classmethod
  def register(cls, assignment_type=None):
    log.debug("Registering...")
    
    def decorator(subclass):
      # Use the provided name or fall back to the class name
      name = assignment_type.lower() if assignment_type else subclass.__name__.lower()
      cls._registry[name] = subclass
      return subclass
    
    return decorator
  
  @classmethod
  def create(cls, assignment_type, **kwargs):
    """Instantiate a registered subclass."""
    
    # If we haven't already loaded our premades, do so now
    if not cls._scanned:
      cls.load_premade_questions()
    # Check to see if it's in the registry
    if assignment_type.lower() not in cls._registry:
      raise ValueError(f"Unknown assignment type: {assignment_type}")
    
    return cls._registry[assignment_type.lower()](**kwargs)
  
  
  @classmethod
  def load_premade_questions(cls):
    package_name = "TeachingTools.grading_assistant"  # Fully qualified package name
    package_path = pathlib.Path(__file__).parent / "grader"
    log.debug(f"package_path: {package_path}")
    
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
      # Import the module
      module = importlib.import_module(f"{package_name}.{module_name}")
      log.debug(f"Loaded module: {module}")


@AssignmentRegistry.register("ProgrammingAssignment")
class Assignment__ProgrammingAssignment(Assignment):
  """
  Assignment for programming assignment grading, where prepare will download files and finalize will upload feedback.
  Will hopefully be run automatically.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def prepare(self, *args, limit=None, regrade=False, only_include_latest=True, **kwargs):
    
    # Steps:
    #  1. Get the submissions
    #  2. Filter out submissions we don't want
    #  3. possibly download proactively
    self.submissions = self.lms_assignment.get_submissions(limit=(None if not regrade else limit))
    if not regrade:
      self.submissions = list(filter(lambda s: s.status == Submission.Status.UNGRADED, self.submissions))
    
    log.info(f"Total students to grade: {len(self.submissions)}")
    if limit is not None:
      log.warning(f"Limiting to {limit} students")
      self.submissions = self.submissions[:limit]
    for i, submission in enumerate(self.submissions):
      log.debug(f"{i+1 : 0{math.ceil(math.log10(len(self.submissions)))}} : {submission.student.name} -> files: {submission.files}")
    
  def finalize(self, *args, **kwargs):
    super().finalize(*args, **kwargs)


@AssignmentRegistry.register("Exam")
class Assignment__Exam(Assignment):
  NAME_RECT =  {
    "x" : 360,
    "y" : 180,
    "width" : 600,
    "height" : 250
  }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    self.fitz_name_rect = fitz.Rect([
      self.NAME_RECT["x"],
      self.NAME_RECT["y"],
      self.NAME_RECT["x"] + self.NAME_RECT["width"],
      self.NAME_RECT["y"] + self.NAME_RECT["height"],
    ])

  class Submission__pdf(Submission):
    def __init__(self, document_id, *args, student=None, approximate_name=None, feedback:Optional[Feedback]=None, question_scores=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.document_id = document_id
      self.approximate_name = approximate_name
      if student is not None:
        self.approximate_name = student.name
        Submission.student.fset(self, student)
      self.feedback : Optional[Feedback] = feedback
      self.question_scores : Dict[int,float] = question_scores
    
    @Submission.student.setter
    def student(self, student):
      new_name_ratio = fuzzywuzzy.fuzz.ratio(student.name, self.approximate_name)
      old_name_ratio = 0 if self.student is None else fuzzywuzzy.fuzz.ratio(self.student.name, self.approximate_name)
      log.info(f'Setting student to "{student}" ({new_name_ratio}%) ({self.approximate_name})')
      if (fuzzywuzzy.fuzz.ratio(student.name, self.approximate_name) / 100.0) <= NAME_SIMILARITY_THRESHOLD:
        log.warning(
          colorama.Back.RED
          + colorama.Fore.LIGHTGREEN_EX
          + colorama.Style.BRIGHT
          + "Similarity below threshold"
          + colorama.Style.RESET_ALL
        )
      
      if new_name_ratio < old_name_ratio:
        log.warning(
          colorama.Back.RED
          + colorama.Fore.LIGHTGREEN_EX
          + colorama.Style.BRIGHT
          + "New name worse than old name"
          + colorama.Style.RESET_ALL
        )
      
      # Call the parent class's student setter explicitly
      Submission.student.fset(self, student)
    
  def prepare(self, input_directory, limit=None, *args, **kwargs):
    
    # Get students from canvas to try to match
    canvas_students : List[Student] = self.lms_assignment.get_students()
    unmatched_canvas_students : List[Student] = canvas_students
    log.debug(canvas_students)
    
    # Read in all the PDFs
    input_pdfs = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".pdf")]
    log.debug(f"input_pdfs: {input_pdfs}")
    
    # Clean and create output folders
    for path_name in ["01-shuffled", "02-redacted"]:
      if os.path.exists(path_name):
        shutil.rmtree(path_name)
      os.mkdir(path_name)
    
    # Shuffle inputs
    random.shuffle(input_pdfs)
    
    # Prepare page ranges
    num_pages_in_pdf = fitz.open(input_pdfs[0]).page_count
    pages_to_merge = [num for start, end in kwargs.get("pages_to_merge", []) for num in range(start, end + 1)]
    page_ranges = [
      (p, p) for p in range(num_pages_in_pdf) if p not in pages_to_merge
    ]
    page_ranges.extend([tuple(r) for r in kwargs.get("pages_to_merge", [])])
    page_ranges.sort()
    
    # Pre-allocate page mappings
    # It feels weird to pre-allocate, but it makes it clearer than doing it in flow I think
    pdfs_to_process = input_pdfs[:(limit if limit is not None else len(input_pdfs))]
    num_users = len(pdfs_to_process)
    num_pages_to_expect = len(page_ranges)
    page_mappings_by_user = collections.defaultdict(list)
    
    # For each page, shuffle the order so we can handle them in different orders
    for page in range(num_pages_to_expect):
      for user_id, random_page_id in enumerate(random.sample(range(num_users), k=num_users)):
        page_mappings_by_user[user_id].append(random_page_id)
    
    # Walk through student submissions, shuffle and redact, and get approximate name
    assignment_submissions : List[Assignment__Exam.Submission__pdf] = []
    for document_id, pdf_filepath in enumerate(pdfs_to_process):
      log.debug(f"Processing {document_id+1}th document: {pdf_filepath}")
      
      approximate_student_name = self.get_approximate_student_name(
        path_to_pdf=pdf_filepath,
        use_ai=kwargs.get("use_ai", True),
        all_student_names=[s.name for s in unmatched_canvas_students]
      )
      log.debug(f"Suggested name: {approximate_student_name}")
      
      # Find best match of the unmatched canvas
      (score, best_match) = max(
        (
          (fuzzywuzzy.fuzz.ratio(s.name, approximate_student_name), s)
          for s in unmatched_canvas_students
        ),
        key=lambda x: x[0]
      )
      if score > NAME_SIMILARITY_THRESHOLD:
        submission = Assignment__Exam.Submission__pdf(
          document_id,
          student=best_match
        )
        unmatched_canvas_students.remove(best_match)
      else:
        log.warning(f"Rejecting proposed match for \"{approximate_student_name}\": \"{best_match.name}\" ({score})")
        submission = Assignment__Exam.Submission__pdf(
          document_id,
          approximate_name=approximate_student_name
        )
      
      # Add in the page numbers
      submission.set_extra({"document_id" : document_id})
      submission.set_extra({
        f"P{page_number}" : page
        for page_number, page in enumerate(page_mappings_by_user[document_id])
      })
      submission.set_extra({"page_mappings": page_mappings_by_user[document_id]})
      
      # Add to submissions
      assignment_submissions.append(submission)
      
      
      # Save aside a copy that's been shuffled for later reference and easy confirmation
      shuffled_document = fitz.open(pdf_filepath)
      shuffled_document.save(os.path.join("01-shuffled", f"{document_id:03}.pdf"))
      
      # Break up each submission into pages
      page_docs : List[fitz.Document] = self.redact_and_split(pdf_filepath, page_ranges=page_ranges)
      for page_number, page in enumerate(page_docs):
        
        # Determine the output directory
        page_directory = os.path.join("02-redacted", f"{page_number:03}")
        
        # Make the output directroy if it doesn't exist
        if not os.path.exists(page_directory): os.mkdir(page_directory)
        
        # Save the page to the appropriate directory, with the number connected to it.
        try:
          page.save(os.path.join(page_directory, f"{page_mappings_by_user[document_id][page_number]:03}.pdf"))
          page.close()
        except IndexError:
          log.warning(f"No page {page_number} found for {document_id}")
      
    log.debug(f"assignment_submissions: {assignment_submissions}")
    
    self.submissions = assignment_submissions
    
    return
  
  def finalize(self, *args, **kwargs):
    log.debug("Finalizing grades")
    
    for submission in self.submissions:
      graded_exam = self.merge_pages(
        "02-redacted",
        submission.extra_info.get('page_mappings', []),
        num_documents=len(self.submissions)
      )
      graded_exam.name = f"exam.pdf"
      submission.feedback.attachments.append(
        graded_exam
      )
      pass
    
    super().finalize(*args, **kwargs)
    
  @staticmethod
  def match_students_to_submissions(students : List[Student], submissions : List[Submission__pdf]) -> Tuple[List[Submission__pdf], List[Submission__pdf], List[Student], List[Student]]:
    # Modified from https://chatgpt.com/share/6743c2aa-477c-8001-9eb6-e5800c3f44da
    # todo: this function has become a mess of what it's doing and returning.
    submissions_w_names : List[Assignment__Exam.Submission__pdf] = []
    submissions_wo_names : List[Assignment__Exam.Submission__pdf] = []
    matched_students : List[Student] = []
    unmatched_students : List[Student] = []
    
    while students and submissions:
      # Find the pair with the maximum comparison value
      best_pair : Tuple[Assignment__Exam.Submission__pdf, Student]|None = None
      best_value = float('-inf')
      
      # todo: update this to go through by using itertools, so we can make sure that the 2nd best is significantly worse than the best
      
      # Go through all the submissions and compare for best match
      for submission in submissions:
        for student in students:
          log.debug(f"submission.approximate_name: {submission.approximate_name}")
          log.debug(f"student.name: {student.name}\n")
          value = fuzzywuzzy.fuzz.ratio(submission.approximate_name, student.name)
          if value > best_value:
            best_value = value
            best_pair = (submission, student)
      
      # Once we've figured out the best current match, assign the Student to the submission, or add it to the unmatched list.
      if (best_value / 100.0) > NAME_SIMILARITY_THRESHOLD:
        best_pair[0].student = best_pair[1]
        submissions_w_names.append(best_pair[0])
        matched_students.append(best_pair[1])
      else:
        submissions_wo_names.append(best_pair[0])
        unmatched_students.append(best_pair[1])
        log.warning("Threshold not met, skipping")
        
      # Remove the matches, even if it wasn't the best match
      submissions.remove(best_pair[0])
      students.remove(best_pair[1])
    # log.debug(f"submissions_w_names: {len(submissions_w_names)}")
    # log.debug(f"submissions_wo_names: {len(submissions_wo_names)}")
    try:
      log.debug(f"Matched {100*(len(submissions_w_names) / len(submissions_w_names + submissions_wo_names)):0.2f}% of submissions")
    except ZeroDivisionError:
      log.warning("No possible submissions to match passed in")
    return submissions_w_names, submissions_wo_names, matched_students, unmatched_students
  
  def redact_and_split(self, path_to_pdf: str, page_ranges : Optional[List[Tuple[int,int]]] = None, *args, **kwargs) -> List[fitz.Document]:
    pdf_document = fitz.open(path_to_pdf)
    
    # First, we redact the first page
    pdf_document[0].draw_rect(self.fitz_name_rect, color=(0,0,0), fill=(0,0,0))
    
    # Next, we break the PDF up into individual pages:
    pdf_pages = []
    
    # If no ranges are specified, simply make groups of single pages
    if page_ranges is None:
      num_pages_per_group = 3
      num_total_pages = len(pdf_document)
      page_ranges = [
        (start, min(start + num_pages_per_group - 1, num_total_pages))
                    for start in range(0, num_total_pages + 1, num_pages_per_group)
      ]
    log.debug(f"page_ranges: {page_ranges}")
    
    # Loop through all pages
    for (start_page, end_page) in page_ranges:
      # Create a new document in memory
      single_page_pdf = fitz.open()
      
      # Insert the current page into the new document
      single_page_pdf.insert_pdf(pdf_document, from_page=start_page, to_page=end_page)
      
      # Append the single-page document to the list
      pdf_pages.append(single_page_pdf)
    
    return pdf_pages
  
  def get_approximate_student_name(self, path_to_pdf, use_ai=True, all_student_names=None):
    
    if use_ai:
      document = fitz.open(path_to_pdf)
      page = document[0]
      pix = page.get_pixmap(clip=list(self.fitz_name_rect))
      image_bytes = pix.tobytes("png")
      base64_str = base64.b64encode(image_bytes).decode("utf-8")
      document.close()
      
      query_string = "What name is written in this picture?  Please respond with only the name."
      if all_student_names is not None:
        query_string += "Some possible names are listed below, but use them as a guide rather than definitive list."
        query_string += "\n - ".join(sorted(all_student_names))
      response = ai_helper.AI_Helper__Anthropic().query_ai(query_string, attachments=[("png", base64_str)])
      return response
      
      
      # response = ai_helper.AI_Helper().query_ai("Can you summarize this for me?  Please return a json object with the key \"contents\" that elides the summary.", [("png", base64_str)], max_response_tokens=100)
      # student_name = response.get("contents", None)
      # if student_name is not None and ':' in student_name:
      #   student_name = ''.join(student_name.split(':')[1:])
      # return student_name
    else:
      return None

  @classmethod
  def merge_pages(cls, input_directory, page_mappings, num_documents) -> io.BytesIO:
    exam_pdf = fitz.open()
    
    for page_number, page_map in enumerate(page_mappings):
      # log.debug(f"Adding {page_number} from {page_mappings}")
      pdf_path = os.path.join(
        input_directory,
        f"{page_number:03}",
        f"{page_map:03}.pdf"
      )
      try:
        exam_pdf.insert_pdf(fitz.open(pdf_path))
      except RuntimeError as e:
        log.error("Page error")
        log.error(e)
        continue
    
    output_bytes = io.BytesIO()
    exam_pdf.save(output_bytes)
    exam_pdf.save("temp.pdf")
    output_bytes.seek(0)
    return output_bytes
  
  def check_student_names(self, submissions: List[Submission__pdf], threshold=0.8):
    
    id_width = max(map(lambda s: len(str(s.student.user_id)), submissions))
    local_width = max(map(lambda s: len(s.student.name), submissions))
    
    comparisons = []
    log.debug("Checking user IDs")
    for submission in submissions:
      sys.stderr.write('.')
      sys.stderr.flush()
      # canvas_name = self.canvas_course.get_user(int(user_id)).name
      ratio = (fuzzywuzzy.fuzz.ratio(submission.approximate_name, submission.student.name) / 100.0)
      comparisons.append((
        ratio,
        submission.student.user_id,
        submission.approximate_name,
        submission.student.name
      ))
    sys.stderr.write('\n')
    
    for (ratio, user_id, student_name, canvas_name) in sorted(comparisons):
      compare_str = f"{user_id:{id_width}} : {100*ratio:3}% : {student_name:{local_width}} ?? {canvas_name}"
      if (fuzzywuzzy.fuzz.ratio(student_name, canvas_name) / 100.0) <= threshold:
        compare_str = colorama.Back.RED + colorama.Fore.LIGHTGREEN_EX + colorama.Style.BRIGHT + compare_str + colorama.Style.RESET_ALL
      
      log.debug(compare_str)

  @staticmethod
  def generate_feedback_comments(df_row : pd.DataFrame):
    total_score = df_row["total"]
    by_question_scores = {}
    for key in df_row.keys():
      if key.startswith("Q"):
        by_question_scores[int(key.replace('Q', ''))] = df_row[key]
    
    feedback_comments_lines = []
    for key in sorted(by_question_scores.keys()):
      if by_question_scores[key] == "-":
        feedback_comments_lines.extend([
          f"Q{key:<{1+int(math.log10(len(by_question_scores)))}} : 0 (unanswered)"
        ])
      else:
        feedback_comments_lines.extend([
          f"Q{key:<{1+int(math.log10(len(by_question_scores)))}} : {int(by_question_scores[key])}"
        ])
    feedback_comments_lines.extend([
      f"Total: {total_score} points"
    ])
    
    return '\n'.join(feedback_comments_lines)


@AssignmentRegistry.register("ExamCST231")
class Assignment__JoshExam(Assignment__Exam):
  NAME_RECT = {
    "x" : 210,
    "y" : 200,
    "width" : 350,
    "height" : 125
  }
