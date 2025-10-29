from pathlib import Path
import re
import nbformat
from nbconvert import MarkdownExporter
from file_conversion_router.conversion.base_converter import BaseConverter
from nbformat.validator import normalize
import uuid

class NotebookConverter(BaseConverter):
    def __init__(self, course_name, course_code, file_uuid: str = None):
        super().__init__(course_name, course_code, file_uuid)
        self.index_helper = None

    def extract_all_markdown_titles(self, content):
        """
        Extract titles from markdown content that start with #
        Returns a list of all found titles
        """
        if not content.strip():
            return []
        titles = []
        # Process line by line to match BaseConverter logic
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                if title == "":
                    continue
                # Remove * characters from title
                title = title.replace('*', '')
                if title.strip():
                    titles.append(title.strip())
        return titles


    def generate_index_helper(self, notebook_content, markdown_content):
        self.index_helper = []
        for i, cell in enumerate(notebook_content.cells):
            titles = self.extract_all_markdown_titles(cell.source)
            for title in titles:
                self.index_helper.append({title: i + 1})

    # Override
    def _to_markdown(self, input_path: Path, output_path: Path) -> Path:
        output_path = output_path.with_suffix(".md")

        with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
            # Read and normalize notebook in memory (don't modify source file)
            content = nbformat.read(input_file, as_version=4)
            content = self.pre_process_notebook(content)
            normalize(content)
            for cell in getattr(content, "cells", []):
                cell.setdefault("id", uuid.uuid4().hex)
            markdown_converter = MarkdownExporter()
            (markdown_content, resources) = markdown_converter.from_notebook_node(
                content
            )
            self.generate_index_helper(content,markdown_content)
            output_file.write(self._post_process_markdown(markdown_content))
        return output_path

    def _post_process_markdown(self, markdown_content: str) -> str:
        lines = markdown_content.split("\n")[
            1:
        ]  # first line is the title of the course section

        processed_lines = []
        for i, line in enumerate(lines):
            if i == 1:  # convert lecture title to h1
                processed_lines.append(f"# {line.lstrip('#').strip()}")
            elif line.startswith("#"):  # convert all other heading down one level
                processed_lines.append(f"#{line.strip()}")
            else:
                processed_lines.append(line.strip())
        # Fix title levels in markdown content
        md_content = self.fix_markdown_title_levels("\n".join(processed_lines))
        return md_content

    def pre_process_notebook(self, content):
        """
        Pre-process the notebook in memory to clean up cell IDs.
        Works on notebook object without modifying the source file.
        Returns the cleaned notebook object.
        """
        for cell in content.cells:
            if 'id' in cell:
                del cell['id']
        return content

    def fix_markdown_title_levels(self, md_content):
        """
        Fix title levels in markdown content to ensure they are sequential.
        Takes markdown content as string and returns the fixed content.
        """
        lines = md_content.split('\n')
        title_info = []

        # Extract all title information with line numbers
        for i, line in enumerate(lines):
            # Match markdown headers (# ## ### etc.)
            header_match = re.match(r'^(#+)\s*(.*)', line.strip())
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                title_info.append({
                    'line_index': i,
                    'level_of_title': level,
                    'title': title,
                    'original_line': line
                })

        # Apply your fixing logic
        last_level = 0
        for i in range(len(title_info)):
            current_level = title_info[i]["level_of_title"]

            if current_level > last_level + 1:
                diff = current_level - (last_level + 1)
                j = i
                while j < len(title_info) and title_info[j]["level_of_title"] >= current_level:
                    title_info[j]["level_of_title"] -= diff
                    j += 1

            last_level = title_info[i]["level_of_title"]

        # Reconstruct the markdown content with fixed levels
        result_lines = lines.copy()
        for info in title_info:
            new_header = '#' * info['level_of_title'] + ' ' + info['title']
            result_lines[info['line_index']] = new_header
        return '\n'.join(result_lines)

