#!env python
from __future__ import annotations

import decimal
import fractions
from decimal import Decimal
from typing import List, Dict

import enum

class OutputFormat(enum.Enum):
  LATEX = enum.auto(),
  CANVAS = enum.auto()


class Answer():
  class AnswerKind(enum.Enum):
    BLANK = "fill_in_multiple_blanks_question"
    MULTIPLE_ANSWER = "multiple_answers_question" # todo: have baffles?
    ESSAY = "essay_question"
  class VariableKind(enum.Enum): # todo: use these for generate variations?
    STR = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()
    BINARY = enum.auto()
    HEX = enum.auto()
    BINARY_OR_HEX = enum.auto()
    AUTOFLOAT = enum.auto()
  def __init__(
      self, key:str,
      value,
      kind : Answer.AnswerKind = AnswerKind.BLANK,
      variable_kind : Answer.VariableKind = VariableKind.STR,
      display=None,
      length=None,
      correct=True
  ):
    self.key = key
    self.value = value
    self.kind = kind
    self.variable_kind = variable_kind
    self.display = display if display is not None else value
    self.length = length # Used for bits and hex to be printed appropriately
    self.correct = correct
  
  def get_for_canvas(self) -> List[Dict]:
    if self.variable_kind == Answer.VariableKind.FLOAT:
      return [{
        "blank_id": self.key,
        "answer_text": f"{self.value:0.2f}",
        "answer_weight": 100,
      }]
    elif self.variable_kind == Answer.VariableKind.BINARY:
      return [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.HEX:
      return [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{(self.length // 8) + 1 if self.length is not None else 0}X}",
          "answer_weight": 100,
        },{
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{(self.length // 8) + 1 if self.length is not None else 0}X}",
          "answer_weight": 100,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.BINARY_OR_HEX:
      return [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        },{
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}X}",
          "answer_weight": 100,
        },{
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{self.length if self.length is not None else 0}X}",
          "answer_weight": 100,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.AUTOFLOAT:
      value_fraction = fractions.Fraction(self.value).limit_denominator(3*4*5) # For process questions, these are the numbers of jobs we'd have
      
      possible_answers = [{
        "blank_id": self.key,
        "answer_text": value_fraction,
        "answer_weight": 100,
      }]
      if not value_fraction.is_integer():
        possible_answers.extend([
          {
            "blank_id": self.key,
            "answer_text": f"{value_fraction.numerator / value_fraction.denominator:0.6}",
            "answer_weight": 100,
          },
          {
            "blank_id": self.key,
            "answer_text":
              f"{value_fraction.numerator // value_fraction.denominator} {value_fraction.numerator % value_fraction.denominator}/{value_fraction.denominator}",
            "answer_weight": 100,
          },
        ])
      
      return possible_answers
    canvas_answer = {
      "blank_id": self.key,
      "answer_text": self.value,
      "answer_weight": 100 if self.correct else 0,
    }
    return [canvas_answer]
