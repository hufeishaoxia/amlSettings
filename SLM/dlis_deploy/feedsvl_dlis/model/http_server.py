#####################
#### DO NOT EDIT ####
#####################

"""
Owner: isst

This file will start an HTTP server

Run following command
curl http://localhost:8888 --data <the post content>
"""
import os
import platform
import threading
import tornado.ioloop
import tornado.web
import utils
import protobuf_helper
import json
import traceback
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from model import ModelImp

def is_binary_content(headers):
    content_type = headers.get("Content-Type", "text/plain")
    return (content_type == "application/binary")

def is_protobuf(headers):
    is_protobuf = headers.get("IsProtobuf", "false")
    return (is_protobuf.lower() == "true")

def post_core(request_handler):
    if (not request_handler.request.body):
        request_handler.set_status(500)
        request_handler.write("empty request")
        return

    try:
        if is_protobuf(request_handler.request.headers):
            response = protobuf_helper.Execute(request_handler.request.body, model_imp)
            request_handler.add_header("IsProtobuf", "true")
            request_handler.set_header("Content-Type", "application/binary")
            request_handler.write(response)
        elif is_binary_content(request_handler.request.headers):
            start_time = time.time()
            response = model_imp.EvalBinary(request_handler.request.body)
            end_time = time.time()
            request_handler.set_header("UnderlyingModelLatencyInUs", str(int((end_time - start_time) * 1000000)))
            request_handler.set_header("Content-Type", "application/binary")
            request_handler.write(response)
        else:
            start_time = time.time()
            response = model_imp.Eval(request_handler.request.body.decode("utf-8"))
            end_time = time.time()
            request_handler.set_header("UnderlyingModelLatencyInUs", str(int((end_time - start_time) * 1000000)))
            request_handler.set_header("Content-Type", "text/plain")
            request_handler.write(response)
    
        return
    except Exception as e:
        formatted_exc = traceback.format_exc()
        str_exc = str(formatted_exc).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(str_exc, file=sys.stderr)
        request_handler.set_status(500)
        request_handler.write("internal server error: %s" % formatted_exc)

    return

model_imp = None
class ThreadPoolHandler(tornado.web.RequestHandler):
    executor = None
    
    def __init__(self, application, request, **kwargs):
        super(ThreadPoolHandler, self).__init__(application, request, **kwargs)

    @tornado.gen.coroutine
    def post(self):
        yield self.executor.submit(post_core, self)
        self.finish()

class MainHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, **kwargs):
        super(MainHandler, self).__init__(application, request, **kwargs)

    def post(self):
        post_core(self)
        self.finish()
    
def make_app(parallelism):
    print("Using ThreadPoolHandler")
    ThreadPoolHandler.executor = ThreadPoolExecutor(parallelism)
    return tornado.web.Application([(r"/", ThreadPoolHandler)])

def start(model):
    global model_imp
    model_imp = model
    listeningPort = utils.get_listening_port(8888)
    parallelism = utils.get_parallelism(1)

    app = make_app(parallelism)
    app.listen(listeningPort)

    print("Parallelism is {0}. Expecting {0} threads to call this server concurrently".format(parallelism))
    print("Will listen on port "+str(listeningPort)+".   To invoke manually, run: curl http://localhost:"+str(listeningPort)+" --data <the post content>")
    print("running \n")

    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    model = ModelImp()
    start(model)
