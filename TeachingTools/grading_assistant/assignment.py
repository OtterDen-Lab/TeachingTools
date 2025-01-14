#!env python
from __future__ import annotations

import argparse
import ast
import base64
import collections
import dataclasses
import math
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

SIMILARITY_THRESHOLD = 95


class Assignment(abc.ABC):
  """
  Class to represent an assignment and act as an abstract base class for other classes.  Will be passed to a grader.
  Two functions need to be overriden for child classes:
  1. prepare : prepares files for grading by downloading, anonymizing, etc.
  2. finalize : combines parts of grading as necessary
  """
  def __init__(self, grading_root_dir, *args, **kwargs):
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
  
  @abc.abstractmethod
  def finalize(self, *args, **kwargs):
    """
    This function is intended to finalize any grading.  This could be reloading the grading CSV and matching names,
    or could just be a noop.
    :param args:
    :param kwargs:
    :return:
    """
    pass


class Assignment__ProgrammingAssignment(Assignment):
  """
  Assignment for programming assignment grading, where prepare will download files and finalize will upload feedback.
  Will hopefully be run automatically.
  """
  def __init__(self, *args, lms_assignment : CanvasAssignment, **kwargs):
    super().__init__(*args, **kwargs)
    self.lms_assignment = lms_assignment
  
  def prepare(self, *args, limit=None, regrade=False, only_include_latest=True, **kwargs):
    
    # Steps:
    #  1. Get the submissions
    #  2. Filter out submissions we don't want
    #  3. possibly download proactively
    self.submissions = self.lms_assignment.get_submissions()
    if not regrade:
      self.submissions = list(filter(lambda s: s.status == Submission.Status.UNGRADED, self.submissions))
    
    log.info(f"Total students to grade: {len(self.submissions)}")
    if limit is not None:
      log.warning(f"Limiting to {limit} students")
      self.submissions = self.submissions[:limit]
    for i, submission in enumerate(self.submissions):
      log.debug(f"{i+1 : 0{math.ceil(math.log10(len(self.submissions)))}} : {submission.student.name} -> files: {submission.files}")
    
  def finalize(self, *args, **kwargs):
    pass
  
  
  def old__download_submission_files(self, submissions: List[Submission], download_all_variations=False, download_dir=None, overwrite=False, user_id=None) \
      -> Dict[Tuple[int, int, str],List[str]]:
    log.debug(f"download_submission_files(self, {len(submissions)} submissions)")
    
    # Set up the attachments directory if not passed in as an argument
    if download_dir is None:
      download_dir = self.working_dir
    
    if overwrite:
      if os.path.exists(download_dir): shutil.rmtree(download_dir)
    if not os.path.exists(download_dir):
      os.mkdir(download_dir)
    
    submission_files = collections.defaultdict(list)
    
    for student_submission in submissions:
      if student_submission.missing:
        # skip missing assignments
        continue
      if user_id is not None and student_submission.user_id != user_id:
        continue
      
      # Get student name for posterity
      student_name = self.canvas_course.get_user(student_submission.user_id)
      log.debug(f"For {student_submission.user_id} there are {len(student_submission.submission_history)} submissions")
      
      # Cycle through each attempt, but walk the list backwards so we grab the latest first, in case that's the only one we end up grading
      for attempt_number, submission_attempt in enumerate(student_submission.submission_history[::-1]):
        
        # todo: this might have to be improved to grab each combination of files separately in case a resubmission didn't have a full set of files for some reason
        
        # If there are no attachments then the student never submitted anything and this submission was automatically closed
        if "attachments" not in submission_attempt:
          continue
        log.debug(f"Submission #{attempt_number+1} has {len(submission_attempt['attachments'])} variations")
        
        # Download each attachment
        for attachment in submission_attempt['attachments']:
          
          # Generate a local file name with a number of options
          local_file_name = f"{student_name.name.replace(' ', '-')}_{attempt_number}_{student_submission.user_id}_{attachment['filename']}"
          local_path = os.path.join(download_dir, local_file_name)
          
          # Download the file, overwriting if appropriate.
          if overwrite or not os.path.exists(local_path):
            log.debug(f"Downloading {attachment['url']} to {local_path}")
            urllib.request.urlretrieve(attachment['url'], local_path)
            time.sleep(0.1)
          else:
            log.debug(f"{local_path} already exists")
          
          # Store the local filenames on a per-(student,attempt) basis
          submission_files[(student_submission.user_id, attempt_number, student_name)].append(local_path)
        
        # Break if we were only supposed to download a single variation
        if not download_all_variations:
          break
    return dict(submission_files)

class Assignment__Exam(Assignment):
  NAME_RECT = [360,50,600,130]

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
      if (fuzzywuzzy.fuzz.ratio(student.name, self.approximate_name) / 100.0) <= SIMILARITY_THRESHOLD:
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
    
  
  def prepare(self, input_directory, canvas_interface, limit=None, *args, **kwargs):
    
    # Get students from canvas to try to match
    canvas_students : List[Student] = canvas_interface.get_students()
    unmatched_canvas_students : List[Student] = canvas_students # canvas_interface.get_students()
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
    
    # Pre-allocate page mappings
    # It feels weird to pre-allocate, but it makes it clearer than doing it in flow I think
    pdfs_to_process = input_pdfs[:(limit if limit is not None else len(input_pdfs))]
    num_users = len(pdfs_to_process)
    num_pages_to_expect = fitz.open(input_pdfs[0]).page_count
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
      if score > SIMILARITY_THRESHOLD:
        assignment_submissions.append(Assignment__Exam.Submission__pdf(document_id,  student=best_match))
        unmatched_canvas_students.remove(best_match)
      else:
        log.warning(f"Rejecting proposed match for \"{approximate_student_name}\": \"{best_match.name}\" ({score})")
        assignment_submissions.append(Assignment__Exam.Submission__pdf(document_id, approximate_name=approximate_student_name))
      
      # Save aside a copy that's been shuffled for later reference and easy confirmation
      shuffled_document = fitz.open(pdf_filepath)
      shuffled_document.save(os.path.join("01-shuffled", f"{document_id:0{int(math.log10(len(input_pdfs))+1)}}.pdf"))
      
      # Break up each submission into pages
      page_docs : List[fitz.Document] = self.redact_and_split(pdf_filepath)
      for page_number, page in enumerate(page_docs):
        
        # Determine the output directory
        page_directory = os.path.join("02-redacted", f"{page_number:0{int(math.log10(len(page_docs))+1)}}")
        
        # Make the output directroy if it doesn't exist
        if not os.path.exists(page_directory): os.mkdir(page_directory)
        
        # Save the page to the appropriate directory, with the number connected to it.
        try:
          page.save(os.path.join(page_directory, f"{page_mappings_by_user[document_id][page_number]:0{int(math.log10(len(input_pdfs))+1)}}.pdf"))
          page.close()
        except IndexError:
          log.warning(f"No page {page_number} found for {document_id}")
      
    log.debug(f"assignment_submissions: {assignment_submissions}")
    
    log.debug(assignment_submissions)
    # Make a dataframe
    df = pd.DataFrame([
      {
        "document_id" : submission.document_id,
        "page_mappings" : page_mappings_by_user[submission.document_id],
        "name" : submission.student.name if submission.student is not None else "",
        "user_id" : submission.student.user_id if submission.student is not None else "",
        "total" : 0.0
      }
      for submission in assignment_submissions
    ])
    print(df.head())
    df = df.sort_values(by="document_id")
    
    df.to_csv("grades.intermediate.csv", index=False)
    
  def finalize(self, path_to_grading_csv, canvas_assignment : CanvasAssignment, *args, **kwargs):
    log.debug("Finalizing grades")
    
    grades_df = pd.read_csv(path_to_grading_csv)
    canvas_students_by_id = { s.user_id : s for s in canvas_assignment.get_students()}
    graded_submissions : List[Assignment__Exam.Submission__pdf] = []
    
    # todo: Steps are going to be:
    # 0. Recombine PDFs
    # 1. For any entry that doesn't have an user_id associated with it load the name column as approximate name
    # 2. Go through the matching process again to try to associate names
    # 3. confirm names match using the previous process
    # 4. upload and save out final grades
    
    # Clean out the extra columns not associated with any submission
    grades_df = grades_df[grades_df["document_id"].notna()]
    # grades_df["user_id"] = grades_df["user_id"].astype(int)
    
    page_mappings_by_document_id = grades_df.set_index("document_id")["page_mappings"].apply(ast.literal_eval).to_dict()
    
    log.debug(page_mappings_by_document_id)
    
    
    # Merge the PDFs back together
    # todo: this directory is hardcoded and used throughout, but should it be?
    #  future is probably to move away from having the directory at all and go straight to file buffer
    shutil.rmtree("03-final", ignore_errors=True)
    os.mkdir("03-final")
    self.merge_pages("02-redacted", "03-final", page_mappings_by_document_id)
    documents_by_id : Dict[str,io.BytesIO] = {}
    for document_id in grades_df["document_id"].values:
      pdf_name = f"{document_id:0{math.ceil(math.log10(len(grades_df["document_id"].values)))}}.pdf"
      with open(os.path.join("03-final", pdf_name), 'rb') as fid:
        file_buffer = io.BytesIO(fid.read())
        file_buffer.name = '999.pdf'
        documents_by_id[document_id] = file_buffer
    
    # Putting this into a function for easy reuse
    def get_Submission__pdf(row) -> Assignment__Exam.Submission__pdf:
      by_question_scores = {}
      for key in row.keys():
        if key.startswith("Q"):
          try:
            by_question_scores[int(key.replace('Q', ''))] = float(row[key])
          except ValueError:
            by_question_scores[int(key.replace('Q', ''))] = '-'
      
      log.debug(f'row["user_id"]: {row["user_id"]}')
      if pd.notna(row["user_id"]):
        submission = Assignment__Exam.Submission__pdf(
          document_id=row["document_id"],
          student=canvas_students_by_id[int(row["user_id"])],
          feedback=Feedback(
            score=row["total"],
            comments=self.generate_feedback_comments(row),
            attachments=[documents_by_id[row["document_id"]]]
          ),
          question_scores=by_question_scores
        )
      else:
        log.debug(f"approximate name: {row}")
        submission = Assignment__Exam.Submission__pdf(
          document_id=row["document_id"],
          approximate_name=row["name"],
          feedback=Feedback(
            score=row["total"],
            comments=self.generate_feedback_comments(row),
            attachments=[documents_by_id[row["document_id"]]]
          ),
          question_scores=by_question_scores
        )
      return submission
    
    # Make submission objects for students who have already been matched
    for _, row in grades_df[grades_df["user_id"].notna()].iterrows():
      graded_submissions.append(get_Submission__pdf(row))
      del canvas_students_by_id[int(row["user_id"])]
    
    log.info(f"There are {len(canvas_students_by_id)} unmatched canvas users")
    log.debug(canvas_students_by_id)
    
    # Get the students who have yet to be matched
    manually_named_submissions : List[Assignment__Exam.Submission__pdf] = []
    log.info(f"There are {len(grades_df[grades_df['user_id'].isna()])} manually named students.")
    for _, row in grades_df[grades_df["user_id"].isna()].iterrows():
      manually_named_submissions.append(get_Submission__pdf(row))
    log.debug([s.approximate_name for s in manually_named_submissions])
    
    # Pass through the matching algorithm
    matched_submissions, _, _, unmatched_students = self.match_students_to_submissions(list(canvas_students_by_id.values()), manually_named_submissions)
    graded_submissions.extend(matched_submissions)
    
    log.debug(f"Unmatched canvas students: {len(unmatched_students)}")
    if len(unmatched_students) > 0:
      log.error(f"There are {len(unmatched_students)} unmatched students remaining in the CSV.  Please correct and rerun.")
      for student in unmatched_students:
        log.warning(f"Found no match for student: {student.name} ({student.user_id})")
      # todo: Should I really have an exit here?
      exit(4)
    
    # Now we have a list of graded submissions
    log.info(f"We have graded {len(graded_submissions)} submissions!")
    # todo: This check doesn't do much, so maybe rematch original submissions via Claude again to see real predictions?
    self.check_student_names(graded_submissions)
  
    if kwargs.get("push", False):
      for submission in graded_submissions:
        log.info(f"Adding in feedback for: {submission}")
        canvas_assignment.push_feedback(
          score=submission.feedback.score,
          comments=submission.feedback.comments,
          attachments=submission.feedback.attachments,
          user_id=submission.student.user_id,
          keep_previous_best=False,
          clobber_feedback=True
        )
      
    # Make a dataframe
    df = pd.DataFrame([
      {
        "document_id" : submission.document_id,
        "name" : submission.student.name,
        "user_id" : submission.student.user_id if submission.student is not None else "",
        "total" : submission.feedback.score,
        **{f"Q{key}" : score for key, score in submission.question_scores.items()}
      }
      for submission in graded_submissions
    ])
    df = df.sort_values(by="document_id")
    
    df.to_csv("grades.final.csv", index=False)
    
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
      if (best_value / 100.0) > SIMILARITY_THRESHOLD:
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
  
  @classmethod
  def redact_and_split(cls, path_to_pdf: str) -> List[fitz.Document]:
    pdf_document = fitz.open(path_to_pdf)
    
    # First, we redact the first page
    pdf_document[0].draw_rect(fitz.Rect(cls.NAME_RECT), color=(0,0,0), fill=(0,0,0))
    
    # Next, we break the PDF up into individual pages:
    pdf_pages = []
    
    # Loop through all pages
    for page_number in range(len(pdf_document)):
      # Create a new document in memory
      single_page_pdf = fitz.open()
      
      # Insert the current page into the new document
      single_page_pdf.insert_pdf(pdf_document, from_page=page_number, to_page=page_number)
      
      # Append the single-page document to the list
      pdf_pages.append(single_page_pdf)
    
    return pdf_pages
  
  @classmethod
  def get_approximate_student_name(cls, path_to_pdf, use_ai=True, all_student_names=None):
    
    if use_ai:
      document = fitz.open(path_to_pdf)
      page = document[0]
      pix = page.get_pixmap(clip=cls.NAME_RECT)
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
  def merge_pages(cls, input_directory, output_directory, page_mappings_by_document_id):
    # todo: refactor this so it is for a single PDF, returns a fitz object, and can take in a list of page number-index values
    
    # exam_pdfs = collections.defaultdict(lambda : fitz.open())
    exam_pdfs_by_document_id = collections.defaultdict(lambda : fitz.open())
    
    for document_id, page_mappings in page_mappings_by_document_id.items():
      for page_number, page_map in enumerate(page_mappings):
        pdf_path = os.path.join(
          input_directory,
          f"{page_number:0{math.ceil(math.log10(len(page_mappings)))}}",
          f"{page_map:0{math.ceil(math.log10(len(page_mappings_by_document_id)))}}.pdf"
        )
        try:
          exam_pdfs_by_document_id[document_id].insert_pdf(fitz.open(pdf_path))
        except RuntimeError:
          continue
    
    for document_id, pdf_document in exam_pdfs_by_document_id.items():
      output_pdf_name = os.path.join(
        output_directory,
        f"{document_id:0{math.ceil(math.log10(len(page_mappings_by_document_id)))}}.pdf"
      )
      exam_pdfs_by_document_id[document_id].save(output_pdf_name)
    
    return
  
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
