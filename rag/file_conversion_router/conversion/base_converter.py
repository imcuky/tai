"""Base classes for all file type converters."""
import string
import logging
import yaml
import re
from abc import ABC, abstractmethod
from pathlib import Path
# from concurrent.futures import Future, ThreadPoolExecutor
from shutil import copy2
# from threading import Lock
from typing import Dict, List, Union, Tuple, Any

from file_conversion_router.classes.chunk import Chunk
from file_conversion_router.classes.new_page import Page
from file_conversion_router.embedding_optimization.src.pipeline.optimizer import (
    EmbeddingOptimizer,
)
from file_conversion_router.utils.logger import (
    conversion_logger,
    logger,
    content_logger,
)
from file_conversion_router.utils.utils import ensure_path
from file_conversion_router.utils.title_handle import *


class BaseConverter(ABC):
    """Base classes for all file type converters.

    This base classes defines the interface for all type converters, with a standardized workflow, which outputs 3 files:
    - Markdown
    - Tree txt
    - Pickle

    All child classes need to implement the abstract methods:
    - _to_markdown

    As long as a child classes can convert a file to Markdown,
    the base classes will handle the rest of the conversion process.
    """

    DEFAULT_EMBEDDING_OPTIMIZATION_CONFIG_PATH = str(
        (
            Path(__file__).parent
            / ".."
            / "embedding_optimization"
            / "src"
            / "configs"
            / "default_config.yaml"
        ).resolve()
    )

    def __init__(
        self, course_name, course_code, file_uuid, optimizer_config_path: Union[str, Path] = None
    ):
        self.index_helper = None
        self.course_name = course_name
        self.course_code = course_code
        self._md_parser = None

        self._md_path = None
        self._logger = logger
        self._content_logger = content_logger
        self.file_name = ""
        self.file_uuid = file_uuid
        # if optimizer_config_path is None:
        #     optimizer_config_path = self.DEFAULT_EMBEDDING_OPTIMIZATION_CONFIG_PATH
        # self.optimizer_config_path = optimizer_config_path

        if optimizer_config_path:
            config_path = Path(optimizer_config_path)
            if not config_path.is_file():
                self._logger.error(
                    f"Optimizer config file does not exist at: {config_path}"
                )
                raise FileNotFoundError(
                    f"Optimizer config file does not exist at: {config_path}"
                )

            self.optimizer = EmbeddingOptimizer(config_path=str(config_path))
            self._logger.info(
                f"EmbeddingOptimizer initialized with config: {config_path}"
            )
        else:
            self.optimizer = None
            self._logger.info("Embedding optimization is disabled.")

    @conversion_logger
    def convert(
        self, input_path: Union[str, Path], output_folder: Union[str, Path], input_root: Union[str, Path] = None,
    ) -> Tuple[List[Chunk], dict]:
        """Convert an input file to mark down under the output folder.

        Args:
            input_path: The path for a single file to be converted. e.g. 'path/to/file.txt'
            output_folder: The folder where the output files will be saved. e.g. 'path/to/output_folder'
                other files will be saved in the output folder, e.g.:
                - 'path/to/output_folder/file.md'
            input_root: The root folder of the input file, used to calculate the relative path of the input file.
        """
        self.file_name = input_path.name
        self.relative_path = input_path.relative_to(input_root)
        input_path, output_folder = ensure_path(input_path), ensure_path(output_folder)
        if not input_path.exists():
            self._logger.error(f"The file {input_path} does not exist.")
            raise FileNotFoundError(f"The file {input_path} does not exist.")
        chunks, metadata = self._convert(input_path, output_folder)
        # future = self.cache.get_future(file_hash)
        # if not future or not future.running():
        #     with ThreadPoolExecutor() as executor:
        #         future = executor.submit(
        #             self._convert_and_cache, input_path, output_folder, file_hash
        #         )
        #         self.cache.store_future(file_hash, future)
        #         self._logger.info(
        #             "New conversion task started or previous task completed."
        #         )
        #         return
        #
        # self._logger.info(
        #     "Conversion is already in progress, waiting for it to complete."
        # )
        # # This will block until the future is completed
        # future.result()
        return chunks, metadata

    @conversion_logger
    def _convert_to_markdown(self, input_path: Path, output_path: Path) -> None:
        """Convert the input file to Expected Markdown format."""
        self._to_markdown(input_path, output_path)

    def _setup_output_paths(
        self, input_path: Union[str, Path], output_folder: Union[str, Path]
    ) -> None:
        """Set up the output paths for the Markdown, tree txt, and pkl files."""
        input_path = ensure_path(input_path)
        output_folder = ensure_path(output_folder)
        self.file_name = input_path.name
        self._md_path = ensure_path(output_folder / f"{self.file_name}.md")

    def _convert(self, input_path: Path, output_folder: Path,) -> Tuple[List[Chunk], dict]:
        """Convert the input file to Mark down and return the chunks."""
        self._setup_output_paths(input_path, output_folder)
        (chunks, metadata), conversion_time = self._perform_conversion(input_path, output_folder)
        return chunks, metadata

    def _read_metadata(self, metadata_path: Path) -> dict:
        """Read metadata from file or return mocked data if file doesn't exist."""
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as metadata_file:
                    return yaml.safe_load(metadata_file)
            except Exception as e:
                self._logger.error(f"Error reading metadata file: {str(e)}")
                return {"URL": "", }
        else:
            self._logger.warning(
                f"Metadata file not found: {metadata_path}. Using mocked metadata."
            )
            return {"URL": "", }

    @conversion_logger
    def _perform_conversion(self, input_path: Path, output_folder: Path) -> Tuple[List[Chunk], dict]:
        """Perform the file conversion process."""
        logging.getLogger().setLevel(logging.INFO)
        logger.info(f"🚀 Starting conversion for {input_path}")
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Output folder did not exist, it's now created: {output_folder}"
            )
        logger.info(f"📄 Expected Markdown Path: {self._md_path}")
        page, metadata = self._to_page(input_path, self._md_path)
        logger.info(f"✅ Page conversion completed for {input_path}.")
        chunks = page.to_chunk()
        logger.info("✅ Successfully converted page content to chunks.")
        return chunks, metadata

    def _to_page(self, input_path: Path, output_path: Path) -> Tuple[Page, dict]:
        """Convert the input file to a Page object and return it along with metadata."""
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_type = input_path.suffix.lstrip(".")
        md_path = self._to_markdown(input_path, output_path)
        metadata_path = input_path.with_name(f"{self.file_name}_metadata.yaml")
        if md_path:
            structured_md, content_dict = self.apply_markdown_structure(input_md_path=md_path, file_type=self.file_type)
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(structured_md)
        else:
            structured_md = ""
            content_dict = {}
        metadata_content = self._read_metadata(metadata_path)
        metadata_content = {'URL': ''} if not metadata_content else {'URL': metadata_content.get('URL', '')}
        metadata = self._put_content_dict_to_metadata(
            content_dict=content_dict,
            metadata_content=metadata_content,
        )
        with open(metadata_path, "w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(metadata, yaml_file, allow_unicode=True)
        url = metadata.get("URL")
        content = {"text": structured_md}
        return Page(
            course_name=self.course_name,
            course_code=self.course_code,
            filetype=self.file_type,
            content=content,
            page_name=self.file_name,
            page_url=url,
            index_helper=self.index_helper,
            file_path=self.relative_path,
            file_uuid=self.file_uuid,
        ), metadata

    def _put_content_dict_to_metadata(self, content_dict: dict, metadata_content: dict) -> dict:
        url = metadata_content.get('URL', '')
        metadata_content['file_uuid'] = self.file_uuid
        metadata_content["file_name"] = str(self.file_name)
        metadata_content['file_path'] = str(self.relative_path)
        metadata_content["course_name"] = self.course_name
        metadata_content["course_code"] = self.course_code
        metadata_content['URL'] = url
        if content_dict.get('speakers'):
            metadata_content["speakers"] = content_dict['speakers']
        if not content_dict:
            return metadata_content
        metadata_content["sections"] = content_dict['key_concepts']
        for section in metadata_content["sections"]:
            section["name"] = section.pop('source_section_title')
            section["index"] = section.pop('source_section_index')
            section["key_concept"] = section.pop('concepts')
            section["aspects"] = section.pop('content_coverage')
            for aspect in section["aspects"]:
                aspect["type"] = aspect.pop('aspect')
        if content_dict.get('recap_questions'):
            metadata_content["recap_questions"] = content_dict['recap_questions']
        if self.file_type == "ipynb":
            metadata_content["problems"] = self.process_problems(content_dict)
        return metadata_content

    def match_a_title_and_b_title(self, a_titles: str, b_titles: str, operator):
        """
        Match the helper titles with the levels titles.
        This method is used to fix the index_helper with titles and their levels.
        Handles quote variations, markdown formatting, and semantic similarities.
        """
        def normalize_title(title: str) -> str:
            """Normalize a title for matching by removing common variations."""
            # Remove markdown formatting (**, *, etc.)
            title = re.sub(r'\*+', '', title)
            # Remove punctuation and convert to lowercase
            title = title.translate(str.maketrans('', '', string.punctuation)).lower().strip()
            # Remove backslashes
            title = title.replace('\\', '').strip()
            # Remove common words that might vary (options, section, etc.)
            common_variations = ['options', 'section', 'part', 'chapter']
            words = title.split()
            filtered_words = [w for w in words if w not in common_variations]
            return ' '.join(filtered_words)

        normalized_a = normalize_title(a_titles)
        normalized_b = normalize_title(b_titles)

        # First try exact match after normalization
        if operator(normalized_a, normalized_b):
            return True

        # Try reverse order for contains operation
        if operator == str.__contains__:
            if normalized_b in normalized_a:
                return True

        # Try with individual words for better semantic matching
        words_a = set(normalized_a.split())
        words_b = set(normalized_b.split())

        # Check if significant overlap exists (at least 70% of words match)
        if words_a and words_b:
            intersection = words_a.intersection(words_b)
            min_words = min(len(words_a), len(words_b))
            if len(intersection) / min_words >= 0.7:
                return True

        return False

    def process_problems(self, content_dict):
        # Return just the list of problems, not a dictionary
        problems_list = []
        for problem in content_dict['problems']:
            processed_problem = {}
            for title in self.index_helper:
                if self.match_a_title_and_b_title(title[-1],problem['ID'], str.__contains__):
                    processed_problem['problem_index'] = self.index_helper[title][0]
                    break
            else:
                processed_problem['problem_index'] = None
            processed_problem['problem_id'] = problem['ID']
            processed_problem['problem_content'] = problem['content']

            # Create questions structure
            processed_problem['questions'] = {}
            for i in range(1, 3):
                question_key = f'sub_problem_{i}'
                sub_prob=problem[question_key]
                processed_problem['questions'][f'question_{i}'] = {
                    'question': sub_prob.get('description_of_problem', ''),
                    'choices': sub_prob.get('options', []),
                    'answer': sub_prob.get('answers_options', []),
                    'explanation': sub_prob.get('explanation_of_answer', '')
                }

            problems_list.append(processed_problem)
        return problems_list

    def count_header_levels(self, content_text: str) -> int:
        """
        Count the number of unique header levels in the markdown content.
        """
        header_levels = set()
        for line in content_text.splitlines():
            if line.startswith("#"):
                level = line.count("#")
                header_levels.add(level)
        return len(header_levels)
    def update_content_dict_titles_with_levels(self, content_dict: dict,content_text: str) -> dict:
        """
        Update the content_dict with titles and their levels from the markdown content.
        Only includes titles that exist in index_helper if available.
        """
        titles_with_levels = []

        # Get valid titles from index_helper if available
        valid_titles = set()
        if hasattr(self, 'index_helper') and self.index_helper:
            for item in self.index_helper:
                for title in item.keys():
                    valid_titles.add(title)

        for line in content_text.splitlines():
            if line.startswith("#"):
                level = line.count("#")
                title = line.lstrip("#").strip()
                if title == "":
                    continue
                # Remove * characters from title to match NotebookConverter logic
                title = title.replace('*', '')
                if title.strip():
                    title = title.strip()
                    # Only include titles that are in index_helper (if index_helper exists)
                    if valid_titles and title not in valid_titles:
                        print(f"Skipping title not in index_helper: '{title}'")
                        continue
                    titles_with_levels.append({"title": title, "level_of_title": level})
        content_dict["titles_with_levels"] = titles_with_levels
        return content_dict

    def fix_index_helper_with_titles_with_level(self, content_dict: dict):
        title_with_levels = content_dict.get("titles_with_levels", [])
        index_helper = []

        for twl_item in title_with_levels:
            found = False
            for item in self.index_helper:
                if self.match_a_title_and_b_title(list(item.keys())[0], twl_item["title"], str.__contains__):
                    index_helper.append(item)
                    twl_item["title"] = list(item.keys())[0]
                    found = True
                    break

            if not found:
                raise AssertionError(f"No match found for: {twl_item['title']}")
        self.index_helper = index_helper


    def apply_markdown_structure(
        self, input_md_path: Path | None, file_type: str):
        """
        Apply the markdown structure based on the file type and content.
        input_md_path: Path to the input markdown file.
        file_type: Type of the file, e.g., "mp4", "pdf", "ipynb".
        returns: md content and a dictionary with structured content.
        """
        file_name = input_md_path.stem
        with open(input_md_path, "r", encoding="UTF-8") as input_file:
            content_text = input_file.read()
            if not content_text.strip():
                logging.warning(f"The file {input_md_path} is empty.")
                return "", {}
            pattern = r'^\s*#\s*ROAR ACADEMY EXERCISES\s*$'
            content_text = re.sub(pattern, '', content_text, flags=re.MULTILINE)
            content_text = re.sub(r'\n{3,}', '\n\n', content_text)
        header_levels = self.count_header_levels(content_text)
        if header_levels == 0 and file_type in ["mp4", "mkv", "webm", "mov"]:
            json_path = input_md_path.with_suffix(".json")
            content_dict = get_structured_content_without_title(
                md_content=content_text,
                file_name=file_name,
                course_name=self.course_name,
            )
            new_md = apply_structure_for_no_title(
                md_content=content_text, content_dict=content_dict
            )
            # Apply speaker role assignment
            new_md = extract_and_assign_speakers(content_dict, new_md, str(json_path))
            # Update index helper and add titles BEFORE grouping
            self.update_index_helper(content_dict,new_md)
            add_titles_to_json(index_helper=self.index_helper, json_file_path=json_path)
            # Group sentences in transcript to reduce list length (after adding titles)
            group_sentences_in_transcript(str(json_path), max_time_gap=5.0, max_words=200)
        elif file_type == "ipynb":
            content_dict = get_strutured_content_for_ipynb(
                md_content=content_text,
                file_name=file_name,
                course_name=self.course_name,
                index_helper=self.index_helper,
            )
            content_dict=self.update_content_dict_titles_with_levels(
                content_dict=content_dict, content_text=content_text
            )
            self.fix_index_helper_with_titles_with_level(content_dict)
            new_md =content_text

        elif file_type == "pdf":
            content_dict = get_structured_content_with_one_title_level(
                md_content=content_text,
                file_name=file_name,
                course_name=self.course_name,
                index_helper=self.index_helper,
            )
            new_md = apply_structure_for_one_title(
                md_content=content_text, content_dict=content_dict
            )
        else:
            content_dict = get_only_key_concepts(
                md_content=content_text,
                index_helper =self.index_helper)
            content_dict=self.update_content_dict_titles_with_levels(content_dict=content_dict, content_text=content_text)
            self.fix_index_helper_with_titles_with_level(content_dict)
            new_md=content_text

        content_dict = self.add_source_section_index(content_dict= content_dict, md_content=new_md)
        return new_md, content_dict

    def generate_index_helper(self, md: str, data = None):
        """ Generate an index helper from the Markdown content.
        """
        self.index_helper = []
        lines = md.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("#"):
                title = line.strip().lstrip("#").strip()
                if title == "":
                    continue
                self.index_helper.append({title: i + 1})

    def add_source_section_index(self, content_dict: dict, md_content: str = None) -> dict:
        if self.file_type not in ["mp4", "mkv", "webm", "mov"]:
            self.update_index_helper(content_dict, md_content=md_content)
        # If there are key concepts, update their source_section_title and source_section_index
        if 'key_concepts' in content_dict:
            for concept in content_dict['key_concepts']:
                source_title = concept['source_section_title']
                found = False
                for titles in self.index_helper.keys():
                    real_title = titles[-1] if isinstance(titles, tuple) else titles
                    if self.match_a_title_and_b_title(real_title, source_title, str.__contains__):
                        concept['source_section_title'] = real_title
                        concept['source_section_index'] = self.index_helper[titles][0] # page index
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"Source section title '{source_title}' not found in index_helper: {self.index_helper}"
                    )
        return content_dict

    def update_index_helper(self, content_dict, md_content=None):
        """create a helper for titles with their levels, including all sub-paths"""
        titles_with_levels = content_dict.get("titles_with_levels")
        result = {}
        path_stack = []
        index_helper_iter = iter(self.index_helper)
        for level_info in titles_with_levels:
            level_info["title"] = level_info["title"].strip()
            target_title = level_info["title"]
            if not target_title:
                continue
            level = level_info["level_of_title"]
            target_index = level - 1
            path_stack = path_stack[:target_index]
            path_stack.append(target_title)
            path = tuple(path_stack)
            found_match = False
            while True:
                try:
                    item = next(index_helper_iter)
                    title = list(item.keys())[0]
                    index = list(item.values())[0]
                    if not title:
                        continue
                    if self.match_a_title_and_b_title(title, target_title, str.__contains__):
                        result[path] = index
                        found_match = True
                        break
                except StopIteration:
                    break
            if not found_match:
                raise AssertionError(f"No matching index found for title: {target_title}")
        self.index_helper = result
        self.add_line_number_to_index_helper(md_content=md_content)

    def add_line_number_to_index_helper(self, md_content: str) -> dict:
        """
        Update self.index_helper so each value becomes (page_index, line_number).
        - page_index is the original value stored in index_helper
        - line_number is 1-based, counted in md_content
        """

        # Build a quick lookup: header-text → first line number (1-based)
        header_lines = {}
        for ln, raw in enumerate(md_content.splitlines(), start=1):
            stripped = raw.lstrip()# ignore leading spaces
            if stripped.startswith("#"):
                header_text = stripped.lstrip("#").strip()
                header_text = header_text.lstrip("*").rstrip("*").strip()  # Remove leading/trailing asterisks
                header_lines[header_text] = ln
        # Walk the existing helper and attach line numbers
        for path, page_idx in list(self.index_helper.items()):
            title = path[-1]
            line_num = header_lines.get(title)
            if line_num is None:
                title = title.replace("'", '"').strip()
                line_num = header_lines.get(title)
            self.index_helper[path] = (page_idx, line_num)
        
        return self.index_helper



    @abstractmethod
    def _to_markdown(self, input_path: Path, output_path: Path) -> None:
        """Convert the input file to Expected Markdown format. To be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")
