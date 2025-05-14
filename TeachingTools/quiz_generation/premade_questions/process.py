#!env python
from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import io
import logging
import math
import os
import queue
import random
import uuid
from typing import List, Optional

import matplotlib.pyplot as plt

from TeachingTools.quiz_generation.misc import OutputFormat, ContentAST
from TeachingTools.quiz_generation.question import Question, Answer, QuestionRegistry

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ProcessQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.PROCESS)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class SchedulingQuestion(ProcessQuestion):
  class Kind(enum.Enum):
    FIFO = enum.auto()
    ShortestDuration = enum.auto()
    ShortestTimeRemaining = enum.auto()
    RoundRobin = enum.auto()
    
    def __str__(self):
      return self.name
  
  @staticmethod
  def get_kind_from_string(kind_str: str) -> SchedulingQuestion.Kind:
    try:
      return SchedulingQuestion.Kind[kind_str]
    except KeyError:
      return SchedulingQuestion.Kind.FIFO

  MAX_JOBS = 4
  MAX_ARRIVAL_TIME = 20
  MIN_JOB_DURATION = 2
  MAX_JOB_DURATION = 10
  
  ANSWER_EPSILON = 1.0
  
  scheduler_algorithm = None
  SELECTOR = None
  PREEMPTABLE = False
  TIME_QUANTUM = None
  
  ROUNDING_DIGITS = 2
  
  @dataclasses.dataclass
  class Job:
    job_id: int
    arrival: float
    duration: float
    elapsed_time: float = 0
    response_time: float = None
    turnaround_time: float = None
    unpause_time: float | None = None
    last_run: float = 0               # When were we last scheduled
    
    state_change_times: List[float] = dataclasses.field(default_factory=lambda: [])
    
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
  
  def get_workload(self, num_jobs, *args, **kwargs) -> List[SchedulingQuestion.Job]:
    """Makes a guaranteed interesting workload by following rules
    1. First job to arrive is the longest
    2. At least 2 other jobs arrive in its runtime
    3. Those jobs arrive in reverse length order, with the smaller arriving 2nd
    
    This will clearly show when jobs arrive how they are handled, since FIFO will be different than SJF, and STCF will cause interruptions
    """
    
    workload = []
    
    # First create a job that is relatively long-running and arrives first.
    # Set arrival time to something fairly low
    job0_arrival = self.rng.randint(0, int(0.25 * self.MAX_ARRIVAL_TIME))
    # Set duration to something fairly long
    job0_duration = self.rng.randint(int(self.MAX_JOB_DURATION * 0.75), self.MAX_JOB_DURATION)
    
    # Next, let's create a job that will test whether we are preemptive or not.
    #  The core characteristics of this job are that it:
    #  1) would also finish _before_ the end of job0 if selected to run immediately.  This tests STCF
    # The bounds for arrival and completion will be:
    #  arrival:
    #   lower: (job0_arrival + 1) so we have a definite first job
    #   upper: (job0_arrival + job0_duration - self.MIN_JOB_DURATION) so we have enough time for a job to run
    #  duration:
    #   lower: self.MIN_JOB_DURATION
    #   upper:
    job1_arrival = self.rng.randint(
      job0_arrival + 1, # Make sure we start _after_ job0
      job0_arrival + job0_duration - self.MIN_JOB_DURATION - 2 # Make sure we always have enough time for job1 & job2
    )
    job1_duration = self.rng.randint(
      self.MIN_JOB_DURATION + 1, # default minimum and leave room for job2
      job0_arrival + job0_duration - job1_arrival - 1 # Make sure our job ends _at least_ before job0 would end
    )
    
    # Finally, we want to differentiate between STCF and SJF
    #  So, if we don't preempt job0 we want to make it be a tough choice between the next 2 jobs when it completes.
    #  This means we want a job that arrives _before_ job0 finishes, after job1 enters, and is shorter than job1
    job2_arrival = self.rng.randint(
      job1_arrival + 1, # Make sure we arrive after job1 so we subvert FIFO
      job0_arrival + job0_duration - 1 # ...but before job0 would exit the system
    )
    job2_duration = self.rng.randint(
      self.MIN_JOB_DURATION, # Make sure it's at least the minimum.
      job1_duration - 1, # Make sure it's shorter than job1
    )
    
    # Package them up so we can add more jobs as necessary
    job_tuples = [
      (job0_arrival, job0_duration),
      (job1_arrival, job1_duration),
      (job2_arrival, job2_duration),
    ]
    
    if num_jobs > 3:
      job_tuples.extend([
        (self.rng.randint(0, self.MAX_ARRIVAL_TIME), self.rng.randint(self.MIN_JOB_DURATION, self.MAX_JOB_DURATION))
        for _ in range(num_jobs - 3)
      ])
    
    # Shuffle jobs so they are in a random order
    self.rng.shuffle(job_tuples)
    
    # Make workload from job tuples
    workload = []
    for i, (arr, dur) in enumerate(job_tuples):
      workload.append(
        SchedulingQuestion.Job(
          job_id=i,
          arrival=arr,
          duration=dur
        )
      )
    
    return workload
    
  
  def run_simulation(self, jobs_to_run: List[SchedulingQuestion.Job], selector, preemptable, time_quantum=None):
    curr_time = 0
    selected_job: SchedulingQuestion.Job | None = None
    
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
        selected_job.run(curr_time, (self.scheduler_algorithm == self.Kind.RoundRobin))
        
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
      if self.scheduler_algorithm != SchedulingQuestion.Kind.RoundRobin:
        if selected_job is not None:
          self.timeline[curr_time].append(f"Running Job{selected_job.job_id} for {next_time_slice:0.{self.ROUNDING_DIGITS}f}s")
        else:
          self.timeline[curr_time].append(f"(No job running)")
      curr_time += next_time_slice
      
      # We stop the job we selected, and potentially mark it as complete
      if selected_job is not None:
        selected_job.stop(curr_time, (self.scheduler_algorithm == self.Kind.RoundRobin))
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
    self.num_jobs = num_jobs
    
    self.scheduler_algorithm = self.rng.choice(list(SchedulingQuestion.Kind))
    
  def refresh(self, *args, **kwargs):
    if not kwargs.get("hard_refresh", True):
      self.rng_seed_offset += 1
    else:
      self.scheduler_algorithm = self.rng.choice(list(SchedulingQuestion.Kind))
    
    log.debug(f"self.scheduler_algorithm: {self.scheduler_algorithm}")
      
    super().refresh(*args, **kwargs)
    
    self.job_stats = {}
    
    # Get workload jobs
    jobs = self.get_workload(self.num_jobs)
    
    # Run simulations different depending on which algorithm we chose
    match self.scheduler_algorithm:
      case SchedulingQuestion.Kind.ShortestDuration:
        self.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.duration, j.job_id)),
          preemptable=False,
          time_quantum=None
        )
      case SchedulingQuestion.Kind.ShortestTimeRemaining:
        self.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.time_remaining(curr_time), j.job_id)),
          preemptable=True,
          time_quantum=None
        )
      case SchedulingQuestion.Kind.RoundRobin:
        self.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.last_run, j.job_id)),
          preemptable=True,
          time_quantum=1e-04
        )
      case _:
        self.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.arrival, j.job_id)),
          preemptable=False,
          time_quantum=None
        )
      
    # Collate stats
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
      self.answers.update({
        f"answer__response_time_job{job_id}": Answer(
          f"answer__response_time_job{job_id}",
          self.job_stats[job_id]["Response"],
          variable_kind=Answer.VariableKind.AUTOFLOAT
        ),
        f"answer__turnaround_time_job{job_id}": Answer(
          f"answer__turnaround_time_job{job_id}",
          self.job_stats[job_id]["TAT"],
          variable_kind=Answer.VariableKind.AUTOFLOAT
        ),
      })
    self.answers.update({
      "answer__average_response_time": Answer(
        "answer__average_response_time",
        sum([job.response_time for job in jobs]) / len(jobs),
        variable_kind=Answer.VariableKind.AUTOFLOAT
      ),
      "answer__average_turnaround_time": Answer("answer__average_turnaround_time",
        sum([job.turnaround_time for job in jobs]) / len(jobs),
        variable_kind=Answer.VariableKind.AUTOFLOAT
      )
    })
  
  def get_body(self, output_format: OutputFormat|None = None, *args, **kwargs) -> ContentAST.Section:
    
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        f"Given the below information, compute the required values if using <b>{self.scheduler_algorithm}</b> scheduling.  "
        f"Break any ties using the job number.",
      ])
    )
    
    body.add_element(
      ContentAST.Text(
        "Please format answer as fractions, mixed numbers, or numbers rounded to a maximum of 4 digits.  "
        "Examples of appropriately formatted answers would be `0`, `3/2`, `1 1/3`, `1.6667`, and `1.25`.",
        hide_from_latex=True
      )
    )
    
    body.add_element(
      ContentAST.Table(
        headers=["Job ID", "Arrival", "Duration", "Response Time", "TAT"],
        data=[
          [
            f"Job{job_id}",
            self.job_stats[job_id]["arrival"],
            self.job_stats[job_id]["duration"],
            ContentAST.Answer(self.answers[f"answer__response_time_job{job_id}"]),
            ContentAST.Answer(self.answers[f"answer__turnaround_time_job{job_id}"])
          ]
          for job_id in sorted(self.job_stats.keys())
        ]
      )
    )
    
    body.add_element(
      ContentAST.AnswerBlock([
        ContentAST.Answer(self.answers["answer__average_response_time"], label="Overall avereage response time"),
        ContentAST.Answer(self.answers["answer__average_turnaround_time"], label="Overall average TAT")
      ])
    )
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        f"To calculate the overall Turnaround and Response times using {self.scheduler_algorithm} "
        f"we want to first start by calculating the respective target and response times of all of our individual jobs."
      ])
    )
    
    explanation.add_elements([
      ContentAST.Paragraph([
        "We do this by subtracting arrival time from either the completion time or the start time.  That is:"
        ]),
      ContentAST.Equation("Job_{TAT} = Job_{completion} - Job_{arrival}"),
      ContentAST.Equation("Job_{response} = Job_{start} - Job_{arrival}"),
    ])
    
    explanation.add_element(
      ContentAST.Paragraph([
        f"For each of our {len(self.job_stats.keys())} jobs, we can make these calculations.",
      ])
    )
    
    ## Add in TAT
    explanation.add_element(
      ContentAST.Paragraph([
        "For turnaround time (TAT) this would be:"
      ] + [
        f"Job{job_id}_TAT "
        f"= {self.job_stats[job_id]['arrival'] + self.job_stats[job_id]['TAT']:0.{self.ROUNDING_DIGITS}f} "
        f"- {self.job_stats[job_id]['arrival']:0.{self.ROUNDING_DIGITS}f} "
        f"= {self.job_stats[job_id]['TAT']:0.{self.ROUNDING_DIGITS}f}"
        for job_id in sorted(self.job_stats.keys())
      ])
    )
    
    summation_line = ' + '.join([
      f"{self.job_stats[job_id]['TAT']:0.{self.ROUNDING_DIGITS}f}" for job_id in sorted(self.job_stats.keys())
    ])
    explanation.add_element(
      ContentAST.Paragraph([
        f"We then calculate the average of these to find the average TAT time",
        f"Avg(TAT) = ({summation_line}) / ({len(self.job_stats.keys())}) "
        f"= {self.overall_stats['TAT']:0.{self.ROUNDING_DIGITS}f}",
      ])
    )
    
    
    ## Add in Response
    explanation.add_element(
      ContentAST.Paragraph([
        "For response time this would be:"
      ] + [
      f"Job{job_id}_response "
      f"= {self.job_stats[job_id]['arrival'] + self.job_stats[job_id]['Response']:0.{self.ROUNDING_DIGITS}f} "
      f"- {self.job_stats[job_id]['arrival']:0.{self.ROUNDING_DIGITS}f} "
      f"= {self.job_stats[job_id]['Response']:0.{self.ROUNDING_DIGITS}f}"
      for job_id in sorted(self.job_stats.keys())
    ])
    )
    
    summation_line = ' + '.join([
      f"{self.job_stats[job_id]['Response']:0.{self.ROUNDING_DIGITS}f}" for job_id in sorted(self.job_stats.keys())
    ])
    explanation.add_element(
      ContentAST.Paragraph([
        f"We then calculate the average of these to find the average Response time",
        f"Avg(Response) "
        f"= ({summation_line}) / ({len(self.job_stats.keys())}) "
        f"= {self.overall_stats['Response']:0.{self.ROUNDING_DIGITS}f}",
        "\n",
      ])
    )
    
    explanation.add_element(
      ContentAST.Table(
        headers=["Time", "Events"],
        data=[
          [f"{t:02.{self.ROUNDING_DIGITS}f}s"] + ['\n'.join(self.timeline[t])]
          for t in sorted(self.timeline.keys())
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Picture(
        img_data=self.make_image(),
        caption="Process Scheduling Overview"
      )
    )
    
    return explanation
  
  def is_interesting(self) -> bool:
    duration_sum = sum([self.job_stats[job_id]['duration'] for job_id in self.job_stats.keys()])
    tat_sum = sum([self.job_stats[job_id]['TAT'] for job_id in self.job_stats.keys()])
    return (tat_sum >= duration_sum * 1.1)
  
  def make_image(self):
    
    fig, ax = plt.subplots(1, 1)
    
    for x_loc in set([t for job_id in self.job_stats.keys() for t in self.job_stats[job_id]["state_changes"] ]):
      ax.axvline(x_loc, zorder=0)
      plt.text(x_loc + 0, len(self.job_stats.keys())-0.3, f'{x_loc:0.{self.ROUNDING_DIGITS}f}s', rotation=90)
    
    if self.scheduler_algorithm != self.Kind.RoundRobin:
      for y_loc, job_id in enumerate(sorted(self.job_stats.keys(), reverse=True)):
        for i, (start, stop) in enumerate(zip(self.job_stats[job_id]["state_changes"], self.job_stats[job_id]["state_changes"][1:])):
          ax.barh(
            y = [y_loc],
            left = [start],
            width = [stop - start],
            edgecolor='black',
            linewidth = 2,
            color = 'white' if (i % 2 == 1) else 'black'
          )
    else:
      job_deltas = collections.defaultdict(int)
      for job_id in self.job_stats.keys():
        job_deltas[self.job_stats[job_id]["state_changes"][0]] += 1
        job_deltas[self.job_stats[job_id]["state_changes"][1]] -= 1
      
      regimes_ranges = zip(sorted(job_deltas.keys()), sorted(job_deltas.keys())[1:])
      
      for (low, high) in regimes_ranges:
        jobs_in_range = [
          i for i, job_id in enumerate(list(self.job_stats.keys())[::-1])
          if
          (self.job_stats[job_id]["state_changes"][0] <= low)
          and
          (self.job_stats[job_id]["state_changes"][1] >= high)
        ]
        
        if len(jobs_in_range) == 0: continue
        
        ax.barh(
          y = jobs_in_range,
          left = [low for _ in jobs_in_range],
          width = [high - low for _ in jobs_in_range],
          color=f"{ 1 - ((len(jobs_in_range) - 1) / (len(self.job_stats.keys())))}",
        )
    
    # Plot the overall TAT
    ax.barh(
      y = [i for i in range(len(self.job_stats))][::-1],
      left = [self.job_stats[job_id]["arrival"] for job_id in sorted(self.job_stats.keys())],
      width = [self.job_stats[job_id]["TAT"] for job_id in sorted(self.job_stats.keys())],
      tick_label = [f"Job{job_id}" for job_id in sorted(self.job_stats.keys())],
      color=(0,0,0,0),
      edgecolor='black',
      linewidth=2,
    )
    
    ax.set_xlim(xmin=0)
    
    # Save to BytesIO object instead of a file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    
    # Reset buffer position to the beginning
    buffer.seek(0)
    return buffer
    
  def make_image_file(self, image_dir="imgs"):
    
    image_buffer = self.make_image()
    
    # Original file-saving logic
    if not os.path.exists(image_dir): os.mkdir(image_dir)
    image_path = os.path.join(image_dir, f"{str(self.scheduler_algorithm).replace(' ', '_')}-{uuid.uuid4()}.png")

    with open(image_path, 'wb') as fid:
      fid.write(image_buffer.getvalue())
    return image_path
    

class MLFQ_Question(ProcessQuestion):
  
  MIN_DURATION = 10
  MAX_DURATION = 100
  MIN_ARRIVAL = 0
  MAX_ARRIVAL = 100
  
  @dataclasses.dataclass
  class Job():
    arrival: float
    duration: float
    elapsed_time: float = 0.0
    response_time: float = None
    turnaround_time: float = None
    
    def run_for_slice(self, slice_duration):
      self.elapsed_time += slice_duration
    
    def is_complete(self):
      return math.isclose(self.duration, self.elapsed_time)
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Set up defaults
    # todo: allow for per-queue specification of durations, likely through dicts
    num_queues = kwargs.get("num_queues", 2)
    num_jobs = kwargs.get("num_jobs", 2)
    
    # Set up queues that we will be using
    mlfq_queues = {
      priority : queue.Queue()
      for priority in range(num_queues)
    }
    
    # Set up jobs that we'll be using
    jobs = [
      MLFQ_Question.Job(
        arrival=random.randint(self.MIN_ARRIVAL, self.MAX_ARRIVAL),
        duration=random.randint(self.MIN_DURATION, self.MAX_DURATION),
        
      )
    ]
    
    curr_time = -1.0
    while True:
      pass