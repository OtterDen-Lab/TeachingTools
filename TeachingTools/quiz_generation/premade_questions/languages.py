#!env python
from __future__ import annotations

import abc
import enum
import itertools
from typing import List, Dict, Optional, Tuple, Any
import random

from TeachingTools.quiz_generation.question import QuestionRegistry, Question, Answer

import logging

from TeachingTools.quiz_generation.misc import ContentAST

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LanguageQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.LANGUAGES)
    super().__init__(*args, **kwargs)

class BNF:
  
  class Grammar:
    def __init__(self, symbols, start_symbol=None):
      self.start_symbol = start_symbol if start_symbol is not None else symbols[0]
      self.symbols = symbols
  
    def generate(self, include_spaces=False, early_exit=False, early_exit_min_iterations=5):
      curr_symbols : List[BNF.Symbol] = [self.start_symbol]
      prev_symbols: List[BNF.Symbol] = curr_symbols
      
      iteration_count = 0
      # Check to see if we have any non-terminals left
      while any(map(lambda s: s.kind == BNF.Symbol.Kind.NonTerminal, curr_symbols)):
        # Grab the previous symbols in case we are early exitting
        prev_symbols = curr_symbols
        
        # Walk through the current symbols and build a new list of symbols from it
        next_symbols : List[BNF.Symbol] = []
        for symbol in curr_symbols:
          next_symbols.extend(symbol.expand())
        curr_symbols = next_symbols
        
        iteration_count += 1
        
        if early_exit and iteration_count > early_exit_min_iterations:
          break
        
      if early_exit:
        # If we are doing an early exit then we are going to return things with non-terminals
        curr_symbols = prev_symbols
      
      # Take all the current symbols and combine them
      return ('' if not include_spaces else ' ').join([str(s) for s in curr_symbols])
    
    def print(self):
      for symbol in self.symbols:
        print(symbol.get_full_str())
        
    def get_grammar_string(self):
      lines = []
      lines.append('```')
      for symbol in self.symbols:
        lines.append(f"{symbol.get_full_str()}")
      
      lines.append('```')
      return '\n'.join(lines)
      
  class Symbol:
    
    class Kind(enum.Enum):
      NonTerminal = enum.auto()
      Terminal = enum.auto()
      
    def __init__(self, symbol : str, kind : Kind):
      self.symbol = symbol
      self.kind = kind
      self.productions : List[BNF.Production] = [] # productions
    
    def __str__(self):
      # if self.kind == BNF.Symbol.Kind.NonTerminal:
      #   return f"`{self.symbol}`"
      # else:
      #   return f"{self.symbol}"
      return f"{self.symbol}"
    
    def get_full_str(self):
      return f"{self.symbol} ::= {' | '.join([str(p) for p in self.productions])}"
    
    def add_production(self, production: BNF.Production):
      self.productions.append(production)
    
    def expand(self) -> List[BNF.Symbol]:
      if self.kind == BNF.Symbol.Kind.Terminal:
        return [self]
      return random.choice(self.productions).production
  
  class Production:
    def __init__(self, production_line, nonterminal_symbols: Dict[str, BNF.Symbol]):
      if len(production_line.strip()) == 0:
        self.production = []
      else:
        self.production = [
          (nonterminal_symbols.get(symbol, BNF.Symbol(symbol, BNF.Symbol.Kind.Terminal)))
          for symbol in production_line.split(' ')
        ]
      
    def __str__(self):
      if len(self.production) == 0:
        return '""'
      return f"{' '.join([str(s) for s in self.production])}"
  
  
  @staticmethod
  def parse_bnf(grammar_str) -> BNF.Grammar:
    
    # Figure out all the nonterminals and create a Token for them
    terminal_symbols = {}
    start_symbol = None
    for line in grammar_str.strip().splitlines():
      if "::=" in line:
        non_terminal_str, _ = line.split("::=", 1)
        non_terminal_str = non_terminal_str.strip()
        
        terminal_symbols[non_terminal_str] = BNF.Symbol(non_terminal_str, BNF.Symbol.Kind.NonTerminal)
        if start_symbol is None:
          start_symbol = terminal_symbols[non_terminal_str]
    
    # Parse the grammar statement
    for line in grammar_str.strip().splitlines():
      if "::=" in line:
        # Split the line into non-terminal and its expansions
        non_terminal_str, expansions = line.split("::=", 1)
        non_terminal_str = non_terminal_str.strip()
        
        non_terminal = terminal_symbols[non_terminal_str]
        
        for production_str in expansions.split('|'):
          production_str = production_str.strip()
          non_terminal.add_production(BNF.Production(production_str, terminal_symbols))
    bnf_grammar = BNF.Grammar(list(terminal_symbols.values()), start_symbol)
    return bnf_grammar


@QuestionRegistry.register("LanguageQuestion")
class ValidStringsInLanguageQuestion(LanguageQuestion):
  MAX_TRIES = 1000
  
  def __init__(self, grammar_str_good: Optional[str] = None, grammar_str_bad: Optional[str] = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    if grammar_str_good is not None and grammar_str_bad is not None:
      self.grammar_str_good = grammar_str_good
      self.grammar_str_bad = grammar_str_bad
      self.include_spaces = kwargs.get("include_spaces", False)
      self.MAX_LENGTH = kwargs.get("max_length", 30)
    else:
      which_grammar = random.choice(range(4))
      
      if which_grammar == 0:
        # todo: make a few different kinds of grammars that could be picked
        self.grammar_str_good = """
          <expression> ::= <term> | <expression> + <term> | <expression> - <term>
          <term>       ::= <factor> | <term> * <factor> | <term> / <factor>
          <factor>     ::= <number>
          <number>     ::= <digit> | <number> <digit>
          <digit>      ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        """
        # Adding in a plus to number
        self.grammar_str_bad = """
          <expression> ::= <term> | <expression> + <term> | <expression> - <term>
          <term>       ::= <factor> | <term> * <factor> | <term> / <factor>
          <factor>     ::= <number>
          <number>     ::= <digit> + | <digit> <number>
          <digit>      ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        """
        self.include_spaces = False
        self.MAX_LENGTH = 30
      elif which_grammar == 1:
        self.grammar_str_good = """
          <sentence> ::= <subject> <verb> <object>
          <subject> ::= The cat | A dog | The bird | A child | <adjective> <animal>
          <animal> ::= cat | dog | bird | child
          <adjective> ::= happy | sad | angry | playful
          <verb> ::= chases | sees | hates | loves
          <object> ::= the ball | the toy | the tree | <adjective> <object>
        """
        self.grammar_str_bad = """
          <sentence> ::= <subject> <verb> <object>
          <subject> ::= The human | The dog | A bird | Some child | A <adjective> <animal>
          <animal> ::= cat | dog | bird | child
          <adjective> ::= happy | sad | angry | playful
          <verb> ::= chases | sees | hates | loves
          <object> ::= the ball | the toy | the tree | <adjective> <object>
        """
        self.include_spaces = True
        self.MAX_LENGTH = 100
      elif which_grammar == 2:
        self.grammar_str_good = """
          <poem> ::= <line> | <line> <poem>
          <line> ::= <subject> <verb> <object> <modifier>
          <subject> ::= whispers | shadows | dreams | echoes | <compound-subject>
          <compound-subject> ::= <subject> and <subject>
          <verb> ::= dance | dissolve | shimmer | collapse | <compound-verb>
          <compound-verb> ::= <verb> then <verb>
          <object> ::= beneath | between | inside | around | <compound-object>
          <compound-object> ::= <object> through <object>
          <modifier> ::= silently | violently | mysteriously | endlessly | <recursive-modifier>
          <recursive-modifier> ::= <modifier> and <modifier>
        """
        self.grammar_str_bad = """
          <bad-poem> ::= <almost-valid-line> | <bad-poem> <bad-poem>
          <almost-valid-line> ::= <tricky-subject> <tricky-verb> <tricky-object> <tricky-modifier>
          <tricky-subject> ::= whispers | shadows and and | <duplicate-subject>
          <duplicate-subject> ::= whispers whispers
          <tricky-verb> ::= dance | <incorrect-verb-placement> | <verb-verb>
          <incorrect-verb-placement> ::= dance dance
          <verb-verb> ::= dance whispers
          <tricky-object> ::= beneath | <object-verb-swap> | <duplicate-object>
          <object-verb-swap> ::= dance beneath
          <duplicate-object> ::= beneath beneath
          <tricky-modifier> ::= silently | <modifier-subject-swap> | <duplicate-modifier>
          <modifier-subject-swap> ::= whispers silently
          <duplicate-modifier> ::= silently silently
        """
        self.include_spaces = True
        self.MAX_LENGTH = 100
      elif which_grammar == 3:
        self.grammar_str_good = """
          <A> ::= a <B> a |
          <B> ::= b <C> b |
          <C> ::= c <A> c |
        """
        self.grammar_str_bad = """
          <A> ::= a <B> c
          <B> ::= b <C> a |
          <C> ::= c <A> b |
        """
        self.include_spaces = False
        self.MAX_LENGTH = 100
    
    self.grammar_good = BNF.parse_bnf(self.grammar_str_good)
    self.grammar_bad = BNF.parse_bnf(self.grammar_str_bad)
    
    self.num_answer_blanks = kwargs.get("num_answer_blanks", 4)
    
    self.refresh()
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    self.answers = {}
    
    self.answers.update(
      {
        "answer_good" : Answer(
          f"answer_good",
          self.grammar_good.generate(self.include_spaces),
          Answer.AnswerKind.MULTIPLE_ANSWER,
          correct=True
        )
      }
    )
    
    self.answers.update(
      {
        "answer_bad":
          Answer(
            f"answer_bad",
            self.grammar_bad.generate(self.include_spaces),
            Answer.AnswerKind.MULTIPLE_ANSWER,
            correct=False
          )
      })
    self.answers.update({
      "answer_bad_early":
        Answer(
          f"answer_bad_early",
          self.grammar_bad.generate(self.include_spaces, early_exit=True),
          Answer.AnswerKind.MULTIPLE_ANSWER,
          correct=False
        )
    })
    
    answer_text_set = {a.value for a in self.answers.values()}
    num_tries = 0
    while len(self.answers) < 10 and num_tries < self.MAX_TRIES:
      
      correct = random.choice([True, False])
      if not correct:
        early_exit = random.choice([True, False])
      else:
        early_exit = False
      new_answer = Answer(
        f"answer_{num_tries}",
        (
          self.grammar_good
          if correct or early_exit
          else self.grammar_bad
        ).generate(self.include_spaces, early_exit=early_exit),
        Answer.AnswerKind.MULTIPLE_ANSWER,
        correct= correct and not early_exit
      )
      if len(new_answer.value) < self.MAX_LENGTH and new_answer.value not in answer_text_set:
        self.answers.update({new_answer.key : new_answer})
        answer_text_set.add(new_answer.value)
      num_tries += 1
  
  def get_body(self, *args, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        ContentAST.TextHTML("Given the following grammar, which of the below strings are part of the language?"),
        ContentAST.TextLatex(
          "Given the following grammar "
          "please circle any provided strings that are part of the language (or indicate clearly if there are none), "
          "and on each blank line provide generate a new, unique string that is part of the language."
        )
      ])
    )
    
    body.add_element(
      ContentAST.Paragraph([
        self.grammar_good.get_grammar_string()
      ])
    )
    
    body.add_element(
      ContentAST.OnlyLatex([
        ContentAST.TextLatex(f"- `{answer.value}`")
        for answer in self.answers.values()
      ])
    )
    
    body.add_element(
      ContentAST.OnlyLatex([
        ContentAST.AnswerBlock([ContentAST.Answer() for _ in range(self.num_answer_blanks)])
      ])
    )
    
    return body
  
  def get_explanation(self, *args, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    explanation.add_element(
      ContentAST.Paragraph([
        "Remember, for a string to be part of our language, we need to be able to derive it from our grammar.",
        "Unfortunately, there isn't space here to demonstrate the derivation so please work through them on your own!"
      ])
    )
    return explanation

  def get_answers(self, *args, **kwargs) -> Tuple[Answer.AnswerKind, List[Dict[str,Any]]]:
    
    return Answer.AnswerKind.MULTIPLE_ANSWER, list(itertools.chain(*[a.get_for_canvas() for a in self.answers.values()]))
