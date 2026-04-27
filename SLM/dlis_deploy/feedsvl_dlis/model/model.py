"""
Owner: isst
Please implement your own ModelImp
"""
import os
import json
import utils
import logging
from qwen_vl_inference.vlm_class import VLM_Inference

# During Docker runtime using multi gpu and vLLM engine, DLIS the 'CUDA_VISIBLE_DEVICES' to the physical UUID.
# In the Ray framework(vLLM), get_gpu_ids() function need to obtain the integer GPU ID.
# Here, you can convert the physical UUID to GPU index id.
# Default CUDA_VISIBLE_DEVICES=GPU-84eded04-9db2-f942-f978-4aec528160a3,GPU-38c11551-0dda-da40-817c-705c2c1c6848"
# Result CUDA_VISIBLE_DEVICES="0,1"
#
# cuda_devices_env = os.getenv("CUDA_VISIBLE_DEVICES","") 
# if cuda_devices_env:
#     devices_uuids = cuda_devices_env.split(',')
#     cuda_devices_index_ids = [str(index) for index in range(len(devices_uuids))]
#     cuda_devices_index_ids_env = ','.join(cuda_devices_index_ids)
#     os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_index_ids_env

class ModelImp:
    # self.model_path is "Model Path" specified when deploying the model (i.e. <path1>/run.sh)
    #
    # self.model_dir is directory where model is located under "Model Path" (i.e. <path1>/model/)
    #
    # self.data_path is the "Model Data Path" specified when deploying the model (i.e. <path2>/data.txt)
    #   If it was not specified during deployment, it will be none
    #
    # self.data_dir is the directory containing the file specified in data_path (i.e. <path2>)
    #
    # To access files in /model, use self.model_dir
    # To access files in the data folder, use self.data_dir
    def  __init__(self):
        self.model_path = utils.get_model_path()
        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(self.model_path)), "model")

        self.data_path = utils.get_data_path()
        self.data_dir = None
        if self.data_path is not None:
            self.data_dir = os.path.dirname(os.path.realpath(self.data_path))
        self.initial_dyanmic_data_paths = utils.get_initial_dynamic_data_paths()
        self.initial_dynamic_data_dirs = utils.get_named_directories(self.initial_dyanmic_data_paths)           

        print("Model Path: {model_path}".format(model_path=self.model_path))
        print("Model Dir: {model_dir}".format(model_dir=self.model_dir))
        print("Data Path: {data_path}".format(data_path=self.data_path))
        print("Data Dir: {data_dir}".format(data_dir=self.data_dir))
        if self.initial_dyanmic_data_paths is not None:
           for namedpath in self.initial_dynamic_data_dirs:
              print("Updatable data labled {name} is initially in {directory}".format(name= namedpath.name, directory=namedpath.path))

        print('loading model...')
        # Load your model here
        print('model loaded.')
        base_model_path = os.path.join(self.model_dir, 'qwen_vl_model/Qwen3-VL-2B-Instruct')
        lora_model_path = os.path.join(self.model_dir, 'qwen_vl_model/Lora_Model')
        self.vlm_model = VLM_Inference(model_path=lora_model_path, model_base=base_model_path)
        print('model loaded.')
 
    # string version of Eval
    # data is a string
    def Eval(self, data):
        # Implement your string evaluation here
        try:
            input_json_dict = json.loads(data)
        except Exception as e:
            logging.error(f'{e} {data}')
            return json.dumps([f"Invalid Format {e}"])
        try:
            result = self.vlm_model.infer_sample(input_json_dict)
            return result
        except Exception as e:
            logging.error(f'{e} {data}')
            return json.dumps([f'Bad Request: {e}'])

    # batch string version of Eval
    # data_list is a list of strings
    # response must be a list of strings, where the order is the same as the input,
    # example:
    #   input: ["request1", "request2"]
    #   output: ["response1", "response2"]
    def EvalBatch(self, data_list):
        responses = []
        for i in range(0, len(data_list)):
            responses.append("EvalBatch: {0}".format(data_list[i]))
        return responses
        
    # binary version of Eval
    # data is python class "bytes"
    def EvalBinary(self, data):
        # Implement your binary evaluation here
        return data

    # batch binary version of Eval
    # data is a list of python class "bytes"
    # responses must be a list of bytes, where the order is the same as the input
    def EvalBatchBinary(self, data_list):
        responses = []
        for i in range(0, len(data_list)):
            responses.append(data_list[i])
        return responses
        
    # If your model expects data updates, this method will be called when a fresh data
    # set has been downloaded to the machine.  Each path represents a whole directory
    # worth of downloaded data (same convention as the static ModelDataPath fetched in the Init()).
    def OnDataUpdate(self, updated_paths):
        print('Got a fresh set of updated data')
        updated_dirs = utils.get_named_directories(updated_paths)
        for namedpath in updated_dirs:
              print("Updated data labled {name} is loaded in {directory}".format(name= namedpath.name, directory=namedpath.path))

        # Load and swap in your updated data here.
