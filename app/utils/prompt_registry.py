"""Load a jinja2 template from the prompt registry."""
from jinja2 import Environment, FileSystemLoader

from app.utils.logger import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    """A registry for managing prompt templates."""
    
    def __init__(self, registry_path='app/prompts'):
        self.registry_path = registry_path

    def load_prompt_template(self, template_name: str, template_version: int):
        """Load a prompt template from the registry."""
        try:
            env = Environment(loader=FileSystemLoader(self.registry_path))
            template = env.get_template(f"{template_name}_v{template_version}.jinja2")
            logger.info(f"Loaded template: {template_name}, version: {template_version}")
            return template
        except Exception as e:
            logger.error("Error loading template %s: %s", template_name, repr(e), exc_info=True)
            raise