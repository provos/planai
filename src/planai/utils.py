# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional
from xml.dom.minidom import parseString

import dicttoxml
from pydantic import BaseModel

# Suppress dicttoxml INFO logs at module initialization
dicttoxml_logger = logging.getLogger("dicttoxml")
dicttoxml_logger.setLevel(logging.WARNING)


def setup_logging(
    logs_dir: str = "logs",
    logs_prefix="general_log",
    logger_name: Optional[str] = None,
    level: Optional[int] = logging.INFO,
) -> logging.Logger:
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{logs_prefix}_{timestamp}.log")

    logger = logging.getLogger(name=logger_name)
    logger.handlers = []
    logger.setLevel(level)

    # Create file handler but don't create the file until the first emit
    file_handler = logging.FileHandler(log_file, mode="w", delay=True)
    file_handler.setLevel(level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


@contextmanager
def measure_time():
    start_time = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        end_time = time.perf_counter()
        result["elapsed_time"] = (end_time - start_time) * 1000


class PydanticDictWrapper(BaseModel):
    """This class creates a pydantic model from a dict object that can be used in the pre_process method of LLMTaskWorker."""

    data: Dict[str, Any]

    def model_dump_json(self, **kwargs):
        return json.dumps(self.data, **kwargs)

    def model_dump_xml(self, root: str = "root"):
        return dict_dump_xml(self.data, root=root)


def _sanitize_for_xml(obj: Any) -> Any:
    """Recursively sanitize values in a dictionary or list for XML compatibility."""
    if isinstance(obj, dict):
        return {_sanitize_for_xml(k): _sanitize_for_xml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_xml(item) for item in obj]
    elif isinstance(obj, str):
        # Remove surrogate characters and other invalid Unicode first
        try:
            # Try encoding/decoding to handle surrogate pairs
            cleaned = obj.encode("utf-8", errors="ignore").decode("utf-8")
        except UnicodeError:
            cleaned = "".join(c for c in obj if ord(c) < 0xD800 or ord(c) > 0xDFFF)

        # Then filter for XML-valid characters
        return "".join(
            char
            for char in cleaned
            if (
                ord(char) in (0x9, 0xA, 0xD)
                or (0x20 <= ord(char) <= 0xD7FF)
                or (0xE000 <= ord(char) <= 0xFFFD)
                or (0x10000 <= ord(char) <= 0x10FFFF)
            )
        )
    return obj


def dict_dump_xml(dict: Dict[Any, Any], root: str = "root") -> str:
    """Formats the task as XML with sanitization and error handling."""
    # Sanitize the dictionary before conversion
    sanitized_dict = _sanitize_for_xml(dict)
    xml = dicttoxml.dicttoxml(sanitized_dict, custom_root=root, attr_type=False)
    # Decode bytes to string with utf-8 encoding
    xml_str = xml.decode("utf-8")

    xml_string = parseString(xml_str).toprettyxml(indent="  ")

    # Remove the XML declaration efficiently
    if xml_string.startswith("<?xml"):
        newline_index = xml_string.find("\n")
        if newline_index != -1:
            xml_string = xml_string[(newline_index + 1) :]
    return xml_string
