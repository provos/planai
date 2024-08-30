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
import select
import threading

import paramiko

from .utils import setup_logging

try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

logger = setup_logging(logs_prefix="ssh_connection", logger_name=__name__)


class SSHConnection:
    def __init__(self, hostname, username, port=22):
        self.hostname = hostname
        self.username = username
        self.port = port
        self.client = None
        self.forward_thread = None

    def is_active(self):
        return (
            self.client is not None
            and self.client.get_transport() is not None
            and self.client.get_transport().is_active()
        )

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Get keys from the SSH agent
        agent = paramiko.Agent()
        keys = agent.get_keys()
        if not keys:
            logger.error("No keys loaded into SSH agent")
            raise Exception("No keys loaded into SSH agent")

        # Try connecting with each key
        for key in keys:
            try:
                self.client.connect(
                    self.hostname, port=self.port, username=self.username, pkey=key
                )
                logger.info(
                    "Successfully connected to {} as {}".format(
                        self.hostname, self.username
                    )
                )
                return  # Exit the method if connection is successful
            except paramiko.AuthenticationException as auth_error:
                logger.warning(
                    "Authentication with key {} failed: {}".format(
                        key.get_base64(), auth_error
                    )
                )

        logger.error("All agent keys failed to authenticate")
        raise Exception("All agent keys failed to authenticate")

    def start_port_forward(self, local_port, remote_host, remote_port):
        class ForwardServer(SocketServer.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        class Handler(SocketServer.BaseRequestHandler):
            def handle(self):
                try:
                    chan = self.ssh_transport.open_channel(
                        "direct-tcpip",
                        (self.chain_host, self.chain_port),
                        self.request.getpeername(),
                    )
                except Exception as e:
                    logger.error(
                        "Incoming request to {}:{} failed: {}".format(
                            self.chain_host, self.chain_port, repr(e)
                        )
                    )
                    return
                if chan is None:
                    logger.error(
                        "Incoming request to {}:{} was rejected by the SSH server.".format(
                            self.chain_host, self.chain_port
                        )
                    )
                    return

                logger.info(
                    "Connected! Tunnel open {} -> {} -> {}".format(
                        self.request.getpeername(),
                        chan.getpeername(),
                        (self.chain_host, self.chain_port),
                    )
                )
                while True:
                    r, w, x = select.select([self.request, chan], [], [])
                    if self.request in r:
                        data = self.request.recv(1024)
                        if len(data) == 0:
                            break
                        chan.send(data)
                    if chan in r:
                        data = chan.recv(1024)
                        if len(data) == 0:
                            break
                        self.request.send(data)

                peername = self.request.getpeername()
                chan.close()
                self.request.close()
                logger.info("Tunnel closed from {}".format(peername))

        def forward_tunnel():
            ForwardServer(("", local_port), Handler).serve_forever()

        Handler.chain_host = remote_host
        Handler.chain_port = remote_port
        Handler.ssh_transport = self.client.get_transport()

        self.forward_thread = threading.Thread(target=forward_tunnel)
        self.forward_thread.daemon = True
        self.forward_thread.start()

    def execute_command(self, command):
        if not self.client:
            logger.error("Not connected. Call connect() first.")
            raise Exception("Not connected. Call connect() first.")
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        return {
            "exit_status": exit_status,
            "stdout": stdout.read().decode("utf-8"),
            "stderr": stderr.read().decode("utf-8"),
        }

    def stop_port_forward(self):
        if self.forward_thread:
            transport = self.client.get_transport()
            if transport:
                transport.cancel_port_forward("", 0)
            self.forward_thread.join()
            self.forward_thread = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
