import socketserver
from socket import *
import os
import errno
import sys
import time
import platform
import model_serving_client_request_response_pb2
import utils
import protobuf_helper
from model import ModelImp

MAX_REQUEST_SIZE = 65507
MAX_RESPONSE_SIZE = 65507 - 140 # Accounts for protobuf overhead

def get_response_too_large_message(data_id):
    temp_response = model_serving_client_request_response_pb2.ModelServingClientResponse()
    temp_response.Code = model_serving_client_request_response_pb2.Fail
    temp_response.Response = "Response is too large"
    temp_response.Id = data_id
    return temp_response.SerializeToString()

class SocketWrapper:
    def __init__(self, model_imp):
        self.model = model_imp

    def receive_request(self, sock):
        return sock.recvfrom(MAX_REQUEST_SIZE)

    def process(self, data):
        response_bytes = protobuf_helper.Execute(data, self.model)
        if len(response_bytes) > MAX_RESPONSE_SIZE:
            return get_response_too_large_message(data_id)
        else:
            return response_bytes

    def send_response(self, message, sock, address):
        # TODO: timeout for send but not receive
        sock.sendto(message, address)

    def start_server(self, address, listening_port):
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.bind((address, listening_port))
        print("Will listen on UDP port " + str(listening_port))
        print("running \n")

        while True:
            try:
                message, address = self.receive_request(sock)
                response_message = self.process(message)
                self.send_response(response_message, sock, address)
            except Exception as e:
                print(e)

def start(model):
    listeningPort = utils.get_listening_port(7777)
    address = "127.0.0.1" if (platform.system() == "Windows") else "0.0.0.0"
    socket_wrapper = SocketWrapper(model)
    socket_wrapper.start_server(address, listeningPort)

if __name__ == "__main__":
    model = ModelImp()
    start(model)
