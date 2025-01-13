# Teaching Tools

A set of tools that are used to help make writing and grading assignments easier, currently with a focus on Canvas-based grading.

## Pieces

### Interface Scripts

There are two direct scripts currently enabled:
1. [`generate_quiz.py`](generate_quiz.py): Generates quiz PDFs and canvas variations based on a range of input types and pre-defined problems
2. [`grader.py`](grader.py: Helps to streamline the grading flow.  Currently only for exams but I'll be migrating code over shortly

### Code Libraries

There are three main modules currently available in `TeachingTools`:
1. [QuizGeneration](TeachingTools/quiz_generation): Helps generate quizzes both for canvas and paper and generate variations.
2. [GradingAssistant](TeachingTools/grading_assistant): Helps streamline grading, currently just for exams but with more code transfer coming soon.
3. [LMSInterface](TeachingTools/lms_interface): Interfaces with LMSes (currently only canvas since it's what we use at CSUMB), and has some helpers for marking assignments missing.

