from typing import List
from app.utils.logger import get_logger
from app.utils.prompt_registry import PromptRegistry
from app.utils.openai_client import get_openai_client

logger = get_logger(__name__)


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def generate_cluster_name(titles: List[str], genres: List[str], movie_count: int, tv_count: int) -> str:
    """Generate a descriptive name for a cluster using an LLM.

    Args:
        titles: List of movie/TV titles in the cluster
        genres: List of top genres in the cluster
        movie_count: Number of movies in the cluster
        tv_count: Number of TV shows in the cluster
    Returns:
        A 2-4 word descriptive cluster name
    """
    try:
        # build media description
        media_desc = []
        if movie_count > 0:
            media_desc.append(f"{movie_count} movie{'s' if movie_count != 1 else ''}")
        if tv_count > 0:
            media_desc.append(f"{tv_count} TV show{'s' if tv_count != 1 else ''}")
        media_str = " and ".join(media_desc)

        titles_str = ", ".join(titles[:8])
        genres_str = ", ".join(genres) if genres else "mixed genres"

        registry = PromptRegistry()
        template = registry.load_prompt_template("cluster/name_cluster", 1)
        prompt = template.render(
            media_str=media_str,
            genres_str=genres_str,
            titles_str=titles_str
        )

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates short, descriptive names for content clusters. Respond with just the name, nothing else."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=20
        )

        name = response.choices[0].message.content.strip()
        # remove quotes if LLM added them
        name = name.strip('"').strip("'")
        return name

    except Exception as e:
        logger.warning(f"Failed to generate cluster name with LLM: {e}")
        # fallback to genre-based name
        if genres:
            return f"{genres[0]} Collection"
        return "Mixed Content"

