import time
import model_serving_client_request_response_pb2
import traceback

"""
Reads binary-serialized Protobuf request,
executes model based on request, and returns
binary-serialized Protobuf response

Data: Binary data representing Protobuf request
Model: Implementation of ModelImp
"""
def Execute(data, model):
    try:
        request = model_serving_client_request_response_pb2.ModelServingClientRequest()
        response = model_serving_client_request_response_pb2.ModelServingClientResponse()

        request.ParseFromString(data)
        data_id = request.Id

        if request.Action is model_serving_client_request_response_pb2.Ping:
            response.Responses.append("Success")
            response.Code = model_serving_client_request_response_pb2.Success
        else:
            start_time = time.time()

            # only one of Requests, or RequestBlobs should be set
            if request.Requests and len(request.Requests) == 1:
                response.Responses.append(model.Eval(request.Requests[0]))
            elif request.Requests and len(request.Requests) > 1:
                response.Responses.extend(model.EvalBatch(request.Requests))
            elif request.RequestBlobs and len(request.RequestBlobs) == 1:
                response.ResponseBlobs.append(model.EvalBinary(request.RequestBlobs[0]))
            elif request.RequestBlobs and len(request.RequestBlobs) > 1:
                response.ResponseBlobs.extend(model.EvalBatchBinary(request.RequestBlobs))
            else:
                raise Exception("Unable to find valid request in protobuf")

            end_time = time.time()
            response.ModelLatencyInUs = int((end_time - start_time) * 1000000)
            response.Code = model_serving_client_request_response_pb2.Success
    except Exception as ex:
        formatted_exc = traceback.format_exc()
        print(formatted_exc)
        response.Code = model_serving_client_request_response_pb2.Fail
        response.Responses.append("internal server error: {e}".format(e = formatted_exc))

    response.Id = data_id
    response_bytes = response.SerializeToString()
    return response_bytes