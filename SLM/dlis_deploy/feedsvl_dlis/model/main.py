import google.protobuf
import tornado.ioloop
import tornado.web
import sys
import http_server
import udp_server
import offline_process
import utils
from model import ModelImp
import logging
# don't load gRPC yet because not all base images support it

def main(action):
    model = ModelImp()
    utils.set_up_data_updating(model)
    logging.getLogger().setLevel(logging.WARNING)
    if action == "http":
        http_server.start(model)
    elif action == "grpc":
        grpc_server.start(model)
    elif action == "udp":
        udp_server.start(model)
    elif action == "offline":
        offline_process.process(sys.argv[2], sys.argv[3], model)
    else:
        print("Action {type} is unknown. Options are http, grpc, udp, or offline".format(type = action))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <action>")
        sys.exit(1)

    main(sys.argv[1])