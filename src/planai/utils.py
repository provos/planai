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
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional
from xml.dom.minidom import parseString

import dicttoxml

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


def _is_valid_xml_char(char):
    """Checks if a character is valid in XML 1.0 (content)."""
    ord_val = ord(char)
    return (
        ord_val == 0x9  # Tab
        or ord_val == 0xA  # Line Feed
        or ord_val == 0xD  # Carriage Return
        or (0x20 <= ord_val <= 0xD7FF)
        or (0xE000 <= ord_val <= 0xFFFD)
        or (0x10000 <= ord_val <= 0x10FFFF)
    )


def _sanitize_key_for_xml(key: str) -> str:
    """Sanitizes dictionary keys to be valid XML tag names.
    Replaces invalid characters with '_'. Ensures key starts with a letter or underscore.
    """
    sanitized_key = ""
    for char in key:
        if char.isalnum() or char == "_":  # Allow alphanumeric and underscore
            sanitized_key += char
        else:
            sanitized_key += "_"  # Replace invalid key chars with underscore

    # Ensure key starts with a letter or underscore (XML tag requirement)
    if not (sanitized_key and (sanitized_key[0].isalpha() or sanitized_key[0] == "_")):
        sanitized_key = "_" + sanitized_key  # Prepend underscore if needed

    if not sanitized_key:
        return "_"  # Fallback if key becomes empty after sanitization

    return sanitized_key


def _sanitize_value_for_xml(value: Any) -> Any:
    """Sanitizes XML values while preserving type information."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    elif isinstance(value, (dict, list, tuple, set)):
        return _sanitize_for_xml(value)
    elif isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            value = str(value)
        return _sanitize_value_for_xml(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    else:
        # Convert to string and sanitize only string values
        value = str(value)
        cleaned_string = ""
        for char in value:
            if _is_valid_xml_char(char):
                cleaned_string += char
            else:
                cleaned_string += "?"
        return cleaned_string


def _sanitize_for_xml(obj: Any) -> Any:
    """Recursively sanitize keys and values in a dictionary or list for XML."""
    if isinstance(obj, dict):
        return {
            _sanitize_key_for_xml(k): _sanitize_value_for_xml(v) for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_xml(item) for item in obj]
    else:
        return _sanitize_value_for_xml(obj)


def dict_dump_xml(dict: Dict[Any, Any], root: str = "root") -> str:
    """Formats the task as XML with sanitization and error handling."""
    # Sanitize the dictionary before conversion
    sanitized_dict = _sanitize_for_xml(dict)
    xml = dicttoxml.dicttoxml(
        sanitized_dict,
        custom_root=root,
        attr_type=False,
        item_func=lambda _: "item",  # Use "item" for all list items
    )
    # Decode bytes to string with utf-8 encoding
    xml_str = xml.decode("utf-8")

    xml_string = parseString(xml_str).toprettyxml(indent="  ")

    # Remove the XML declaration efficiently
    if xml_string.startswith("<?xml"):
        newline_index = xml_string.find("\n")
        if newline_index != -1:
            xml_string = xml_string[(newline_index + 1) :]
    return xml_string
