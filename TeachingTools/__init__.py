
import logging.config
import yaml
import os
import re

def setup_logging():
  config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
  if os.path.exists(config_path):
    with open(config_path, 'r') as f:
      config_text = f.read()
    
    # Process environment variables in the format ${VAR:-default}
    def replace_env_vars(match):
      var_name = match.group(1)
      default_value = match.group(2)
      return os.environ.get(var_name, default_value)
    
    config_text = re.sub(r'\$\{([^}:]+):-([^}]+)\}', replace_env_vars, config_text)
    config = yaml.safe_load(config_text)
    logging.config.dictConfig(config)
  else:
    # Fallback to basic configuration if logging.yaml is not found
    logging.basicConfig(level=logging.INFO)

# Call this once when your application starts
setup_logging()