import sys
import http_server
import utils
from model import ModelImp

def main(action):
    model = ModelImp()
    utils.set_up_data_updating(model)
    if action == "http":
        http_server.start(model)
    else:
        print(f"Action {action} is unknown. Options are: http")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <action>")
        sys.exit(1)
    main(sys.argv[1])
