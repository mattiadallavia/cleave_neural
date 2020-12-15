#  Copyright (c) 2020 KTH Royal Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import multiprocessing as mp
import os
import signal
import uuid
from pathlib import Path
from typing import Any, Mapping, Type

from klein import Klein
from klein.resource import KleinResource
from twisted.internet import endpoints
from twisted.internet.posixbase import PosixReactorBase
from twisted.python.failure import Failure
from twisted.web import server
from twisted.web.http import Request
from twisted.web.server import Site

from ..network.backend import UDPControllerService
from ...api.controller import Controller


class MalformedRequest(Exception):
    pass


def write_json_response(request: Request,
                        status_code: int,
                        response: Mapping,
                        finish: bool = True) -> None:
    request.setResponseCode(status_code)
    request.setHeader('content-type', 'application/json')
    request.write(json.dumps(response).encode('utf8'))
    if finish:
        request.finish()


def malformed_request(request: Request,
                      failure: Failure) -> Any:
    # TODO: parameterize content type
    write_json_response(
        request=request,
        response={'error': 'Malformed request.'},
        status_code=400
    )
    return server.NOT_DONE_YET


def json_decode_error(request: Request,
                      failure: Failure) -> Any:
    write_json_response(
        request=request,
        response={'error': 'Could not decode JSON payload.'},
        status_code=400
    )
    return server.NOT_DONE_YET


def ensure_headers(req: Request, headers: Mapping[str, str]) -> None:
    for header, exp_value in headers.items():
        req_value = req.getHeader(header)
        if req_value is None or req_value.lower() != exp_value.lower():
            raise MalformedRequest()


class ControllerProcessResource:
    def __init__(self,
                 control_cls: Type[Controller],
                 params: Mapping[str, Any]):
        super(ControllerProcessResource, self).__init__()
        self._id = uuid.uuid4()
        self._app = Klein()
        self._app.route('/status', methods=['GET'])(self.info)

        out_q = mp.Queue()
        self._process = mp.Process(
            target=ControllerProcessResource._control_process,
            args=(control_cls, params, str(self._id), out_q))
        self._process.start()
        self._interface, self._port, self._out_path = out_q.get()

    @property
    def resource(self) -> KleinResource:
        return self._app.resource()

    @property
    def uuid(self) -> uuid.UUID:
        return self._id

    def info(self, req: Request) -> Any:
        # ensure_headers(req, {'content-type': 'application/json'})
        write_json_response(
            request=req,
            response={
                'id'        : f'{self._id}',
                'process_id': self._process.pid,
                'interface' : self._interface,
                'port'      : self._port
            },
            status_code=200
        )
        return server.NOT_DONE_YET

    def shut_down(self, timeout: float = 5) -> None:
        os.kill(self._process.pid, signal.SIGINT)
        try:
            self._process.join(timeout=timeout)
        except TimeoutError:
            self._process.terminate()
            self._process.join()

    @staticmethod
    def _control_process(control_cls: Type[Controller],
                         params: Mapping[str, Any],
                         uid: str,
                         out_q: mp.Queue) -> None:
        from twisted.internet import reactor
        reactor: PosixReactorBase = reactor

        controller = control_cls(**params)
        path = Path(f'./controllers/{uid}').resolve()
        service = UDPControllerService(
            controller=controller,
            output_dir=path)

        port = reactor.listenUDP(0, service.protocol)
        out_q.put((port.interface, port.port, path))
        reactor.run()


class Dispatcher:
    def __init__(self, controllers: Mapping[str, Type[Controller]]):
        super(Dispatcher, self).__init__()
        self._app = Klein()
        self._controllers: Mapping[str, ControllerResource] = dict()
        self._controller_cls = controllers

        # set up routes
        self._app.route('/', methods=['POST', 'GET', 'DELETE']) \
            (self.controller_collection)
        self._app.route('/<string:ctrl_id>', branch=True) \
            (self.controller_resource)

        self._app.handle_errors(MalformedRequest)(malformed_request)
        self._app.handle_errors(json.JSONDecodeError)(json_decode_error)

    def run(self, host: str, port: int) -> None:
        from twisted.internet import reactor
        reactor: PosixReactorBase = reactor

        # Create desired endpoint
        host = host.replace(':', '\:')
        endpoint_description = f'tcp:port={port}:interface={host}'
        endpoint = endpoints.serverFromString(reactor, endpoint_description)

        # This actually starts listening on the endpoint with the Klein app
        endpoint.listen(Site(self._app.resource()))
        reactor.run()

    def _spawn_controller(self,
                          request: Request,
                          req_dict: Mapping[str, Any]) -> Any:
        try:
            controller_cls = req_dict['controller']
            parameters = req_dict['parameters']
            assert type(parameters) == dict
        except (KeyError, AssertionError):
            raise MalformedRequest()

        # TODO: make it deferred?
        # TODO: actually spawn shit
        ctrl_resource = ControllerResource()
        self._controllers[ctrl_resource.uuid] = ctrl_resource

        resp_dict = {
            'controller': str(ctrl_resource.uuid)
        }

        write_json_response(
            request=request,
            response=resp_dict,
            status_code=200
        )
        return server.NOT_DONE_YET

    def _list_controllers(self, request: Request) -> Any:
        # TODO: return wrapped controller class, not ControllerResource
        write_json_response(request=request,
                            response={
                                str(c_id): ctrl.__class__.__name__
                                for c_id, ctrl in self._controllers.items()
                            },
                            status_code=200)
        return server.NOT_DONE_YET

    def controller_collection(self, request: Request) -> Any:
        method = request.method.decode('utf-8').upper()
        if method == 'GET':
            return self._list_controllers(request)

        ensure_headers(request, {'content-type': 'application/json'})
        req_dict = json.load(request.content)
        if method == 'POST':
            return self._spawn_controller(request, req_dict)
        elif method == 'DELETE':
            pass  # TODO
        else:
            raise RuntimeError(f'Request with unexpected '
                               f'method {method}!')

    def controller_resource(self,
                            request: Request,
                            ctrl_id: str) -> Any:
        # ensure_headers(request, {'content-type': 'application/json'})
        try:
            ctrl_id = uuid.UUID(ctrl_id)
            return self._controllers[ctrl_id].resource
        except ValueError:
            write_json_response(
                request=request,
                response={'error': f'Malformed uuid {ctrl_id}'},
                status_code=400
            )
        except KeyError:
            write_json_response(
                request=request,
                response={'error': f'No such controller: {ctrl_id}'},
                status_code=404
            )
        return server.NOT_DONE_YET