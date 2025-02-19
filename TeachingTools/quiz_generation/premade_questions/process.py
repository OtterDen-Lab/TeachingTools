#!env python
from __future__ import annotations

import collections
import dataclasses
import enum
import logging
import os
import random
import uuid
from typing import List, Optional

import canvasapi.course, canvasapi.quiz
import matplotlib.pyplot as plt

from TeachingTools.quiz_generation.misc import OutputFormat
from TeachingTools.quiz_generation.question import Question, Answer, QuestionRegistry

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



class ProcessQuestion(Question):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.PROCESS)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class SchedulingQuestion(ProcessQuestion):
  class Kind(enum.Enum):
    FIFO = enum.auto()
    # LIFO = enum.auto()
    ShortestDuration = enum.auto()
    ShortestTimeRemaining = enum.auto()
    RoundRobin = enum.auto()
  
  @staticmethod
  def get_kind_from_string(kind_str: str) -> Kind:
    try:
      return SchedulingQuestion.Kind[kind_str]
    except KeyError:
      return SchedulingQuestion.Kind.FIFO  # or raise ValueError(f"Invalid Kind: {kind_str}")


  MAX_JOBS = 4
  MAX_ARRIVAL_TIME = 10
  MIN_JOB_DURATION = 2
  MAX_JOB_DURATION = 10
  
  ANSWER_EPSILON = 1.0
  
  SCHEDULER_KIND = None
  SCHEDULER_NAME = None
  SELECTOR = None
  PREEMPTABLE = False
  TIME_QUANTUM = None
  
  ROUNDING_DIGITS = 2
  
  @dataclasses.dataclass
  class Job():
    job_id: int
    arrival: float
    duration: float
    elapsed_time: float = 0
    response_time: float = None
    turnaround_time: float = None
    unpause_time: float | None = None
    last_run: float = 0               # When were we last scheduled
    
    state_change_times : List[float] = dataclasses.field(default_factory=lambda : [])
    
    SCHEDULER_EPSILON = 1e-09
    
    def run(self, curr_time, is_rr=False) -> None:
      if self.response_time is None:
        # Then this is the first time running
        self.mark_start(curr_time)
      self.unpause_time = curr_time
      if not is_rr:
        self.state_change_times.append(curr_time)
    
    
    def stop(self, curr_time, is_rr=False) -> None:
      self.elapsed_time += (curr_time - self.unpause_time)
      if self.is_complete(curr_time):
        self.mark_end(curr_time)
      self.unpause_time = None
      self.last_run = curr_time
      if not is_rr:
        self.state_change_times.append(curr_time)
    
    def mark_start(self, curr_time) -> None:
      self.start_time = curr_time
      self.response_time = curr_time - self.arrival + self.SCHEDULER_EPSILON
    def mark_end(self, curr_time) -> None:
      self.end_time = curr_time
      self.turnaround_time = curr_time - self.arrival + self.SCHEDULER_EPSILON
    
    def time_remaining(self, curr_time) -> float:
      time_remaining = self.duration
      time_remaining -= self.elapsed_time
      if self.unpause_time is not None:
        time_remaining -= (curr_time - self.unpause_time)
      return time_remaining
    
    def is_complete(self, curr_time) -> bool:
      # log.debug(f"is complete: {self.duration} <= {self.elapsed_time} : {self.duration <= self.elapsed_time}")
      return self.duration <= self.elapsed_time + self.SCHEDULER_EPSILON # self.time_remaining(curr_time) <= 0
    
    def has_started(self) -> bool:
      return self.response_time is None
  
  def simulation(self, jobs_to_run: List[SchedulingQuestion.Job], selector, preemptable, time_quantum=None):
    curr_time = 0
    selected_job : SchedulingQuestion.Job | None = None
    
    self.timeline = collections.defaultdict(list)
    self.timeline[curr_time].append("Simulation Start")
    for job in jobs_to_run:
      self.timeline[job.arrival].append(f"Job{job.job_id} arrived")
    
    while len(jobs_to_run) > 0:
      possible_time_slices = []
      
      # Get the jobs currently in the system
      available_jobs = list(filter(
        (lambda j: j.arrival <= curr_time),
        jobs_to_run
      ))
      
      # Get the jobs that will enter the system in the future
      future_jobs : List[SchedulingQuestion.Job] = list(filter(
        (lambda j: j.arrival > curr_time),
        jobs_to_run
      ))
      
      # Check whether there are jobs in the system already
      if len(available_jobs) > 0:
        # Use the selector to identify what job we are going to run
        selected_job : SchedulingQuestion.Job = min(
          available_jobs,
          key=(lambda j: selector(j, curr_time))
        )
        if selected_job.has_started():
          self.timeline[curr_time].append(f"Starting Job{selected_job.job_id} (resp = {curr_time - selected_job.arrival:0.{self.ROUNDING_DIGITS}f}s)")
        # We start the job that we selected
        selected_job.run(curr_time, (self.SCHEDULER_KIND == self.Kind.RoundRobin))
        
        # We could run to the end of the job
        possible_time_slices.append(selected_job.time_remaining(curr_time))
      
      # Check if we are preemptable or if we haven't found any time slices yet
      if preemptable or len(possible_time_slices) == 0:
        # Then when a job enters we could stop the current task
        if len(future_jobs) != 0:
          next_arrival : SchedulingQuestion.Job = min(
            future_jobs,
            key=(lambda j: j.arrival)
          )
          possible_time_slices.append( (next_arrival.arrival - curr_time))
      
      if time_quantum is not None:
        possible_time_slices.append(time_quantum)
      
      ## Now we pick the minimum
      try:
        next_time_slice = min(possible_time_slices)
      except ValueError:
        log.error("No jobs available to schedule")
        break
      if self.SCHEDULER_KIND != SchedulingQuestion.Kind.RoundRobin:
        if selected_job is not None:
          self.timeline[curr_time].append(f"Running Job{selected_job.job_id} for {next_time_slice:0.{self.ROUNDING_DIGITS}f}s")
        else:
          self.timeline[curr_time].append(f"(No job running)")
      curr_time += next_time_slice
      
      # We stop the job we selected, and potentially mark it as complete
      if selected_job is not None:
        selected_job.stop(curr_time, (self.SCHEDULER_KIND == self.Kind.RoundRobin))
        if selected_job.is_complete(curr_time):
          self.timeline[curr_time].append(f"Completed Job{selected_job.job_id} (TAT = {selected_job.turnaround_time:0.{self.ROUNDING_DIGITS}f}s)")
      selected_job = None
      
      # Filter out completed jobs
      jobs_to_run : List[SchedulingQuestion.Job] = list(filter(
        (lambda j: not j.is_complete(curr_time)),
        jobs_to_run
      ))
      if len(jobs_to_run) == 0:
        break
  
  def __init__(self, num_jobs=3, scheduler_kind=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    log.debug(f"scheduler_kind: {scheduler_kind}")
    self.num_jobs = num_jobs
    
    if scheduler_kind is None:
      self.scheduler_kind_generator = lambda : random.choice(list(SchedulingQuestion.Kind))
    else:
      self.scheduler_kind_generator = lambda : SchedulingQuestion.get_kind_from_string(scheduler_kind)
    self.instantiate(scheduler_kind=scheduler_kind)
    
  def instantiate(self, rng_seed=None, previous : Optional[SchedulingQuestion]=None, *args, **kwargs):
    super().instantiate(rng_seed=rng_seed, *args, **kwargs)
    
    self.job_stats = {}
    self.SCHEDULER_KIND = self.scheduler_kind_generator()
    
    log.debug(f"Using a {self.SCHEDULER_KIND} scheduler")
    
    # Default to FIFO an then change as necessary
    # This is the default case
    self.SCHEDULER_NAME = "FIFO"
    self.SELECTOR = (lambda j, curr_time: (j.arrival, j.job_id))
    self.PREEMPTABLE = False
    self.TIME_QUANTUM = None
    if self.SCHEDULER_KIND == SchedulingQuestion.Kind.ShortestDuration:
      self.SCHEDULER_NAME = "Shortest Job First"
      self.SELECTOR = (lambda j, curr_time: (j.duration, j.job_id))
    elif self.SCHEDULER_KIND == SchedulingQuestion.Kind.ShortestTimeRemaining:
      self.SCHEDULER_NAME = "Shortest Remaining Time to Completion"
      self.SELECTOR = (lambda j, curr_time: (j.time_remaining(curr_time), j.job_id))
      self.PREEMPTABLE = True
    # elif self.SCHEDULER_KIND == SchedulingQuestion.Kind.LIFO:
    #   self.SCHEDULER_NAME = "LIFO"
    #   self.SELECTOR = (lambda j, curr_time: (-j.arrival, j.job_id))
    #   self.PREEMPTABLE = True
    elif self.SCHEDULER_KIND == SchedulingQuestion.Kind.RoundRobin:
      self.SCHEDULER_NAME = "Round Robin"
      self.SELECTOR = (lambda j, curr_time: (j.last_run, j.job_id))
      self.PREEMPTABLE = True
      self.TIME_QUANTUM = 1e-04
    else:
      # then we default to FIFO
      pass
    
    jobs = [
      SchedulingQuestion.Job(
        job_id,
        random.randint(0, self.MAX_ARRIVAL_TIME),
        random.randint(self.MIN_JOB_DURATION, self.MAX_JOB_DURATION)
      )
      for job_id in range(self.num_jobs)
    ]
    
    self.simulation(jobs, self.SELECTOR, self.PREEMPTABLE, self.TIME_QUANTUM)
    
    self.job_stats = {
      i : {
        "arrival" : job.arrival,            # input
        "duration" : job.duration,          # input
        "Response" : job.response_time,     # output
        "TAT" : job.turnaround_time,         # output
        "state_changes" : [job.arrival] + job.state_change_times + [job.arrival + job.turnaround_time],
      }
      for (i, job) in enumerate(jobs)
    }
    self.overall_stats = {
      "Response" : sum([job.response_time for job in jobs]) / len(jobs),
      "TAT" : sum([job.turnaround_time for job in jobs]) / len(jobs)
    }
    
    # todo: make this less convoluted
    self.average_response = self.overall_stats["Response"]
    self.average_tat = self.overall_stats["TAT"]
    
    for job_id in sorted(self.job_stats.keys()):
      self.answers.extend([
        Answer(
          f"answer__response_time_job{job_id}",
          self.job_stats[job_id]["Response"],
          variable_kind=Answer.VariableKind.AUTOFLOAT
        ),
        Answer(
          f"answer__turnaround_time_job{job_id}",
          self.job_stats[job_id]["TAT"],
          variable_kind=Answer.VariableKind.AUTOFLOAT
        ),
      ])
    self.answers.extend([
      Answer("answer__average_response_time", sum([job.response_time for job in jobs]) / len(jobs), variable_kind=Answer.VariableKind.AUTOFLOAT),
      Answer("answer__average_turnaround_time", sum([job.turnaround_time for job in jobs]) / len(jobs), variable_kind=Answer.VariableKind.AUTOFLOAT)
    ])
  
  def get_body_lines(self, output_format: OutputFormat|None = None, *args, **kwargs) -> List[str]:
    
    lines = []
    
    lines.extend([
      f"Given the below information, compute the required values if using <b>{self.SCHEDULER_NAME}</b> scheduling.  Break any ties using the job number.",
    ])
    
    lines.extend([
      "",
      "Please format answer as fractions, mixed numbers, or numbers rounded to a maximum of 4 digits."
      "Examples of appropriately formatted answers would be `0`, `3/2`, `1 1/3`, `1.6667`, and `1.25`."
    ])
    
    log.debug(f"get_body_lines: {output_format}")
    if output_format is not None and output_format == OutputFormat.CANVAS:
      lines.extend([
        
        "Please round all answers to 2 decimal places (even if they are whole numbers)."
      ])
    
    lines.extend(
      self.get_table_generator(
        headers=["Arrival", "Duration", "Response Time", "TAT"],
        table_data={
          f"Job{job_id}" : [
            self.job_stats[job_id]["arrival"],
            self.job_stats[job_id]["duration"],
            f"[answer__response_time_job{job_id}]",
            f"[answer__turnaround_time_job{job_id}]",
          ]
          for job_id in sorted(self.job_stats.keys())
        },
        add_header_space=True
      )
    )
    
    lines.extend([
      f"Overall average response time: [answer__average_response_time]",
      "",
      f"Overall average TAT: [answer__average_turnaround_time]"
    ])
    
    return lines
  
  def get_explanation_lines(
      self,
      course: canvasapi.course.Course,
      quiz: canvasapi.quiz.Quiz,
      image_dir="imgs",
      *args, **kwargs
  ) -> List[str]:
    
    explanation_lines = []
    
    explanation_lines.extend([
      f"To calculate the overall Turnaround and Response times using {self.SCHEDULER_KIND} we want to first start by calculating the respective target and response times of all of our individual jobs."
    ])



    # Give the general formula
    explanation_lines.extend([
      "We do this by subtracting arrival time from either the completion time or the start time.  That is:"
      "",
      f"Job_TAT = Job_completion - Job_arrival\n",
      ""
      f"Job_response = Job_start - Job_arrival\n",
      "",
    ])

    # Individual job explanation
    explanation_lines.extend([
      f"For each of our {len(self.job_stats.keys())} jobs, we can make these calculations.",
      ""
    ])

    ## Add in TAT
    explanation_lines.extend([
      "For turnaround time (TAT) this would be:"
    ])
    explanation_lines.extend([
      f"Job{job_id}_TAT = {self.job_stats[job_id]['arrival'] + self.job_stats[job_id]['TAT']:0.{self.ROUNDING_DIGITS}f} - {self.job_stats[job_id]['arrival']:0.{self.ROUNDING_DIGITS}f} = {self.job_stats[job_id]['TAT']:0.{self.ROUNDING_DIGITS}f}"
      for job_id in sorted(self.job_stats.keys())
    ])
    explanation_lines.extend(["\n"])
    summation_line = ' + '.join([
      f"{self.job_stats[job_id]['TAT']:0.{self.ROUNDING_DIGITS}f}" for job_id in sorted(self.job_stats.keys())
    ])
    explanation_lines.extend([
      f"We then calculate the average of these to find the average TAT time",
      f"Avg(TAT) = ({summation_line}) / ({len(self.job_stats.keys())}) = {self.overall_stats['TAT']:0.{self.ROUNDING_DIGITS}f}",
      "\n",
    ])


    ## Add in Response
    explanation_lines.extend([
      "For response time this would be:"
    ])
    explanation_lines.extend([
      f"Job{job_id}_response = {self.job_stats[job_id]['arrival'] + self.job_stats[job_id]['Response']:0.{self.ROUNDING_DIGITS}f} - {self.job_stats[job_id]['arrival']:0.{self.ROUNDING_DIGITS}f} = {self.job_stats[job_id]['Response']:0.{self.ROUNDING_DIGITS}f}"
      for job_id in sorted(self.job_stats.keys())
    ])

    explanation_lines.extend(["\n"])
    summation_line = ' + '.join([
      f"{self.job_stats[job_id]['Response']:0.{self.ROUNDING_DIGITS}f}" for job_id in sorted(self.job_stats.keys())
    ])
    explanation_lines.extend([
      f"We then calculate the average of these to find the average Response time",
      f"Avg(Response) = ({summation_line}) / ({len(self.job_stats.keys())}) = {self.overall_stats['Response']:0.{self.ROUNDING_DIGITS}f}",
      "\n",
    ])

    explanation_lines.extend([
      "We can track these events either in a table or by drawing a diagram.  Note that in the diagram color corresponds to how many jobs are running at once, and events that happen at the same time may be merged together (i.e. there might not be 2N vertical lines for N jobs).",
      ""
    ])

    ## Add in table
    explanation_lines.extend(
      self.get_table_generator(
        headers=["Time", "Events"],
        table_data={
          f"{t:02.{self.ROUNDING_DIGITS}f}s" : ['\n'.join(self.timeline[t])]
          for t in sorted(self.timeline.keys())
        },
        sorted_keys=[f"{t:02.{self.ROUNDING_DIGITS}f}s" for t in sorted(self.timeline.keys())],
      )
    )
    
    # todo: see if I can move this out of this file, or somehow figure out the looping to make it not silly
    # it will probably be something along the lines of
    # 1. generate question
    # 2. Generate image from question information
    # 3. generate explanation with image generated
    image_path = self.make_image(image_dir)
    course.create_folder(f"{quiz.id}", parent_folder_path="Quiz Files")
    upload_success, f = course.upload(self.img, parent_folder_path=f"Quiz Files/{quiz.id}")

    explanation_lines.extend(
      [f"![Process Scheduling Overview](/courses/{course.id}/files/{f['id']}/preview)"]
    )
    
    return explanation_lines
  
  def is_interesting(self) -> bool:
    duration_sum = sum([self.job_stats[job_id]['duration'] for job_id in self.job_stats.keys()])
    tat_sum = sum([self.job_stats[job_id]['TAT'] for job_id in self.job_stats.keys()])
    return (tat_sum >= duration_sum * 1.1)
  
  
  def make_image(self, image_dir="imgs"):
    
    fig, ax = plt.subplots(1, 1)
    
    for x_loc in set([t for job_id in self.job_stats.keys() for t in self.job_stats[job_id]["state_changes"] ]):
      ax.axvline(x_loc, zorder=0)
      plt.text(x_loc + 0,len(self.job_stats.keys())-0.3,f'{x_loc:0.{self.ROUNDING_DIGITS}f}s',rotation=90)
    
    if self.SCHEDULER_KIND != self.Kind.RoundRobin:
      for y_loc, job_id in enumerate(sorted(self.job_stats.keys(), reverse=True)):
        for i, (start, stop) in enumerate(zip(self.job_stats[job_id]["state_changes"], self.job_stats[job_id]["state_changes"][1:])):
          ax.barh(
            y = [y_loc],
            left = [start],
            width = [stop - start],
            # color = 'white',
            edgecolor='black',
            linewidth = 2,
            color = 'white' if (i % 2 == 1) else 'black'
          )
    else:
      pass
      job_deltas = collections.defaultdict(int)
      for job_id in self.job_stats.keys():
        job_deltas[self.job_stats[job_id]["state_changes"][0]] += 1
        job_deltas[self.job_stats[job_id]["state_changes"][1]] -= 1
      # log.debug(f"job_deltas: {job_deltas}")
      
      regimes_ranges = zip(sorted(job_deltas.keys()), sorted(job_deltas.keys())[1:])
      
      for (low, high) in regimes_ranges:
        # log.debug(f"(low, high): {(low, high)}")
        jobs_in_range = [
          i for i, job_id in enumerate(list(self.job_stats.keys())[::-1])
          if
          (self.job_stats[job_id]["state_changes"][0] <= low)
          and
          (self.job_stats[job_id]["state_changes"][1] >= high)
        ]
        
        # log.debug(f"jobs_in_range: {jobs_in_range}")
        if len(jobs_in_range) == 0: continue
        # continue
        ax.barh(
          y = jobs_in_range,
          left = [low for _ in jobs_in_range],
          width = [high - low for _ in jobs_in_range],
          color=f"{ 1 - ((len(jobs_in_range) - 1) / (len(self.job_stats.keys())))}",
          # edgecolor='blue',
          # linewidth=2,
        )
        # The core idea is that we want to track how many jobs are running in parallel and color the bars based on that
    
    
    # Plot the overall TAT
    ax.barh(
      y = [i for i in range(len(self.job_stats))][::-1],
      left = [self.job_stats[job_id]["arrival"] for job_id in sorted(self.job_stats.keys())],
      width = [self.job_stats[job_id]["TAT"] for job_id in sorted(self.job_stats.keys())],
      tick_label = [f"Job{job_id}" for job_id in sorted(self.job_stats.keys())],
      color=(0,0,0,0),
      edgecolor='black',
      linewidth=2,
      # hatch='/'
    )
    
    
    ax.set_xlim(xmin=0)
    # plt.show()
    
    if not os.path.exists(image_dir): os.mkdir(image_dir)
    image_path = os.path.join(image_dir, f"{self.SCHEDULER_NAME.replace(' ', '_')}-{uuid.uuid4()}.png")
    plt.savefig(image_path)
    plt.close(fig)
    self.img = image_path
    return image_path
  