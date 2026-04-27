import os
import sys
import utils
from abc import abstractmethod
from dlis_model_opt.optimization.runner_factory import create

"""
Use opt.zip to Create an oaas runner when it exists in oaas_models, otherwise use the model passed into the constructor.
"""
TEST_ORIGINAL_MODEL_VARIABLE = "TEST_ORIGINAL_MODEL"


class OaasWrapper:
    # model is the model name (i.e. dummy.onnx)
    #
    # self.oaas_model_dir is directory where oaas model is located (i.e. <path1>/oaas_models/)
    #
    # self.oaas_model_path is the model path formed by join oaas_model_dir and model name (i.e. <path1>/oaas_models/dummy.onnx)
    #
    # self.opt_zip_path is an optimized zip file for the loaded model (i.e. <path1>/oaas_models/dummy.opt.zip)
    #
    # self.oaas_runner is an OaaS runner for the loaded model
    def __init__(self, model):
        if model is None:
            raise Exception("Model must not be None.")

        """
        To create a runner, we need to check if it is test model in BertTrt optimization.
        If it is, we will parse the model path file to get the model name.
        If TEST_ORIGINAL_MODEL para in os.environ and it is set to True, we will create a runner with the original model.
        Otherwise, a runner will be created with the optimized model.
        """

        self.oaas_runner = None
        self.oaas_model_dir = utils.get_oaas_model_dir()

        if model.endswith(".onnx"):
            self.wrapper = OnnxruntimeWrapper()
        else:
            self.wrapper = BertTrtWrapper()

        self.model = self.wrapper.parse_model_file(os.path.join(self.oaas_model_dir, model))
        self.oaas_model_path = os.path.join(self.oaas_model_dir, self.model)

        # use <model name>.opt.zip as optimized zip file name (i.e. dummy.opt.zip)
        self.opt_zip_path = os.path.join(self.oaas_model_dir, os.path.splitext(self.model)[0] + ".opt.zip")

        self.oaas_runner = self.wrapper.create_runner(self.is_test_original_model(), self.oaas_model_path, self.opt_zip_path)

    def is_test_original_model(self):
        if TEST_ORIGINAL_MODEL_VARIABLE not in os.environ:
            return False
        return os.environ[TEST_ORIGINAL_MODEL_VARIABLE] == "True"  # return True if TEST_ORIGINAL_MODEL is set to True        

    """
    Runs the loaded model.

    Params
    ------
    feed_dict: model input.
    """

    def run(self, feed_dict):
        # run the loaded model by runner
        if not self.oaas_runner:
            raise OSError("The runner is invalid, maybe it could not get a valid model to create")
        return self.oaas_runner.run(feed_dict)


class BaseWrapper:
    @abstractmethod
    def parse_model_file(self, model_path):
        pass

    @abstractmethod
    def create_runner(self, test_original_model, oaas_model_path, opt_zip_path):
        pass


class OnnxruntimeWrapper(BaseWrapper):
    def parse_model_file(self, model_path):
        return os.path.basename(model_path)

    def create_runner(self, test_original_model, oaas_model_path, opt_zip_path):
        print("input model is a file, will create a runner for the onnxruntime model")
        if not test_original_model and os.path.exists(opt_zip_path):
            return create(opt_zip_path)
        if os.path.exists(oaas_model_path):
            return create(oaas_model_path)
        raise OSError("Unable to find either {0} or {1}".format(opt_zip_path, oaas_model_path))


class BertTrtWrapper(BaseWrapper):
    def parse_model_file(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception("The input model is not a existing file")
        with open(model_path, "r") as f:
            weight_file_path = f.readline().strip()
        weight_file = os.path.basename(weight_file_path)
        return weight_file

    def create_runner(self, test_original_model, oaas_model_path, opt_zip_path):
        print("input model is a existing folder, will create a runner for the BertTrt model")

        if test_original_model:
            return None
        if os.path.exists(opt_zip_path):
            return create(opt_zip_path)
        sys.stdout.write("The optimized engine is not found, please check the input model path")
        return None
