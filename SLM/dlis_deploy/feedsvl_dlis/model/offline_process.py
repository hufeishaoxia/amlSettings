#####################
#### DO NOT EDIT ####
#####################

"""
owner: isst.

This file will do offline processing
The 1st parameter should be input file path and the 2nd parameter should be output file path
"""

import sys
from model import ModelImp

def process(inputFilePath, outputFilePath, model):
  try:  
    if hasattr(model, 'EvalFile'):
      model.EvalFile(inputFilePath, outputFilePath)
    else:
      with  open(inputFilePath, 'r', encoding = 'utf-8', errors='ignore') as inFileHandle:
        with open(outputFilePath, "w", encoding = 'utf-8', errors='ignore') as outFileHandle:
          line = inFileHandle.readline()
          line = line.strip()
          while(line):
            result = model.Eval(line)
            outFileHandle.write(result)
            outFileHandle.write("\n")
            line = inFileHandle.readline()
            line = line.strip()
  except Exception as e:            
    errMessage = str(e)
    sys.stderr.write(errMessage)

if __name__ == "__main__":
  if(len(sys.argv) < 3):
    raise Exception("Must contains input Path and output Path as first two argumentations")
  model = ModelImp()
  Process(sys.argv[1], sys.argv[2], model)

