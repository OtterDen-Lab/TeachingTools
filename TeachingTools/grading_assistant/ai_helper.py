import abc
import json
import os
import random
from typing import Tuple, Dict, List

import dotenv
import openai.types.chat.completion_create_params
from openai import OpenAI
from anthropic import Anthropic

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class AI_Helper(abc.ABC):
  _client = None
  
  def __init__(self):
    if self._client is None:
      log.debug("Loading dotenv")# Load the .env file
      dotenv.load_dotenv(os.path.expanduser('~/.env'))
  
  @classmethod
  @abc.abstractmethod
  def query_ai(cls, message : str, attachments : List[Tuple[str, str]], *args, **kwargs):
    pass


class AI_Helper__Anthropic(AI_Helper):
  def __init__(self):
    super().__init__()
    self.__class__._client = Anthropic()
  
  @classmethod
  def query_ai(cls, message : str, attachments : List[Tuple[str, str]], max_response_tokens=1000, max_retries=3):
    messages = []
    
    attachment_messages = []
    for file_type, b64_file_contents in attachments:
      if file_type == "png":
        attachment_messages.append({
          "type": "image",
          "source": {
            "type" : "base64",
            "media_type" : "image/png",
            "data" : b64_file_contents
          }
        })
    
    messages.append(
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text":
              f"{message}"
          },
          *attachment_messages
        ]
      }
    )
    
    message = cls._client.messages.create(
      model="claude-3-5-sonnet-20241022",
      max_tokens=10-0,
      messages=messages
    )
    log.debug(message.content)
    return message.content[0].text


class AI_Helper__OpenAI(AI_Helper):
  def __init__(self):
    super().__init__()
    self.__class__._client = OpenAI()
  
  @classmethod
  def query_ai(cls, message : str, attachments : List[Tuple[str, str]], max_response_tokens=1000, max_retries=3):
    messages = []
    
    attachment_messages = []
    for file_type, b64_file_contents in attachments:
      if file_type == "png":
        attachment_messages.append({
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{b64_file_contents}"
          }
        })
        
    messages.append(
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text":
              f"{message}"
          },
          *attachment_messages
        ]
      }
    )
    
    response = cls._client.chat.completions.create(
      model="gpt-4o",
      response_format=openai.types.chat.completion_create_params.ResponseFormat(type="json_object"), #{ type: "json_object" },
      messages=messages,
      temperature=1,
      max_tokens=max_response_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    log.debug(response.choices[0])
    try:
      return json.loads(response.choices[0].message.content)
    except TypeError:
      if max_retries > 0:
        return cls.query_ai(message, attachments, max_response_tokens, max_retries-1)
      else:
        return {}
