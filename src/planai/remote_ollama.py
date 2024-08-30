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
import re
import time
from typing import Any, Callable

import requests
from ollama import Client

from .utils import setup_logging

logger = setup_logging(logs_prefix="remote_ollama", logger_name=__name__)


class SSHCommandError(Exception):
    """Exception raised for errors in the execution of SSH commands."""

    def __init__(self, command, message):
        self.command = command
        self.message = message
        super().__init__(f"Error executing command '{command}': {message}")
        logger.error(f"SSHCommandError: {self.command} - {self.message}")


class ServerValidationError(Exception):
    """Exception raised for errors in server validation."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)
        logger.error(f"ServerValidationError: {self.message}")


class InvalidModelNameError(Exception):
    """Exception raised for invalid model name."""

    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(f"Invalid model name: {model_name}")
        logger.error(f"InvalidModelNameError: {self.model_name}")


class RemoteOllama:
    OLLAMA_PORT = 11434

    def __init__(self, ssh_connection, model_name, ollama_port=11434):
        self.ssh_connection = ssh_connection
        self.ollama_port = ollama_port
        self.set_model_name(model_name)
        self.ollama_path = "/usr/local/bin/ollama"
        self.ollama_client = Client(host=f"http://localhost:{self.ollama_port}")
        self.generate = self._wrap_with_reconnect(self.ollama_client.generate)
        self.chat = self._wrap_with_reconnect(self.ollama_client.chat)

        if not ssh_connection.is_active():
            ssh_connection.connect()
        logger.info("SSH connection established")

    def _wrap_with_reconnect(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except requests.RequestException as e:
                logger.warning(
                    "Request failed: %s. Attempting to re-establish SSH connection.", e
                )
                if not self.ssh_connection.is_active():
                    self.ssh_connection.connect()
                    self.ssh_connection.start_port_forward(
                        self.ollama_port, "localhost", self.OLLAMA_PORT
                    )
                    logger.info("SSH connection re-established.")
                else:
                    logger.info("SSH connection was still active.")
                # Retry the function call
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    logger.error("Request failed again after reconnecting: %s", e)
                    raise

        return wrapper

    def set_model_name(self, model_name):
        if not re.match(r"^[a-zA-Z0-9_.\-:/]+$", model_name):
            raise InvalidModelNameError(model_name)
        self.model_name = model_name
        logger.info(f"Model name set to {self.model_name}")

    def find_ollama(self) -> bool:
        for cmd_prefix in ["", "/usr/local/bin/", "/usr/bin/", "/bin/"]:
            cmd = f"{cmd_prefix}ollama --version"
            result = self.ssh_connection.execute_command(cmd)
            logger.debug("Executed command: %s, Result: %s", cmd, result)
            if result["exit_status"] == 0:
                self.ollama_path = f"{cmd_prefix}ollama"
                logger.info("Ollama found at %s", self.ollama_path)
                return True
        logger.warning("Ollama not found")
        return False

    def install_ollama(self):
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y curl",
            "curl https://ollama.ai/install.sh | sh",
        ]
        for cmd in commands:
            result = self.ssh_connection.execute_command(cmd)
            logger.debug("Executed command: %s, Result: %s", cmd, result)
            if result["exit_status"] != 0:
                logger.error(
                    f"Failed to execute command: {cmd}, Error: {result['stderr']}"
                )
                raise SSHCommandError(cmd, result["stderr"])
        logger.info("Ollama installed successfully")

    def pull_model(self):
        cmd = f"{self.ollama_path} pull {self.model_name}"
        result = self.ssh_connection.execute_command(cmd)
        logger.debug("Executed command: %s, Result: %s", cmd, result)
        if result["exit_status"] != 0:
            logger.error(f"Failed to execute command: {cmd}, Error: {result['stderr']}")
            raise SSHCommandError(cmd, result["stderr"])
        logger.info(f"Model {self.model_name} pulled successfully")

    def start_server(self):
        cmd = f"nohup {self.ollama_path} serve > /dev/null 2>&1 &"
        result = self.ssh_connection.execute_command(cmd)
        logger.debug("Executed command: %s, Result: %s", cmd, result)
        if result["exit_status"] != 0:
            logger.error(f"Failed to execute command: {cmd}, Error: {result['stderr']}")
            raise SSHCommandError(cmd, result["stderr"])
        self.ssh_connection.start_port_forward(
            self.ollama_port, "localhost", self.OLLAMA_PORT
        )
        logger.info("Ollama server started and port forwarding configured")

    def validate_server(self) -> bool:
        max_retries = 5
        retry_delay = 2
        for _ in range(max_retries):
            try:
                response = requests.get(
                    f"http://localhost:{self.ollama_port}/api/tags", timeout=5
                )
                if response.status_code == 200:
                    logger.info("Server validation successful")
                    return True
            except requests.RequestException as e:
                logger.warning("Server validation failed with exception: %s", e)
            time.sleep(retry_delay)
        logger.error("Server validation failed after maximum retries")
        return False

    def setup_ollama(self) -> bool:
        if not self.find_ollama():
            self.install_ollama()
        if not self.find_ollama():
            raise ServerValidationError("Failed to install ollama")
        self.pull_model()
        self.start_server()
        if not self.validate_server():
            raise ServerValidationError("Failed to validate server")
        logger.info("Ollama setup completed successfully")
        return True

    def stop_server(self):
        cmd = "pkill ollama"
        result = self.ssh_connection.execute_command(cmd)
        logger.debug(f"Executed command: {cmd}, Result: {result}")
        if result["exit_status"] != 0:
            logger.error(f"Failed to execute command: {cmd}, Error: {result['stderr']}")
            raise SSHCommandError(cmd, result["stderr"])
        self.ssh_connection.stop_port_forward()
        logger.info("Ollama server stopped and port forwarding terminated")
