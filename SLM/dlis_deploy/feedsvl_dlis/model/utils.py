import platform
import os
import json
import watchdog.events
import threading
import time
import asyncio
import concurrent.futures
import tornado.httpclient
import traceback

global lastReadManifest

def get_model_path():
    current_path = os.getenv("CURRENT_PATH", os.getcwd())
    run_name = "run.cmd" if platform.system() == "Windows" else "run.sh"
    model_path = os.getenv("_ModelPath_", os.path.join(current_path, run_name))
    return model_path

def get_data_path():
    return os.getenv("_ModelDataPath_", None)

def get_listening_port(default_port):
    listeningPort = default_port
    stringPort = os.getenv('_ListeningPort_')
    if not stringPort:
        print('The environment variable _ListeningPort_ is not set. Falling back on default port.')
        stringPort = listeningPort
    try:
        listeningPort = int(stringPort)
    except ValueError:
        print('The environment variable _ListeningPort_ must be set to an integer. It was: {port}'.format(port=stringPort))
        exit()

    return listeningPort

def get_parallelism(default_parallelism):
    parallelism = default_parallelism
    stringParallelism = os.getenv('_Parallelism_')
    if not stringParallelism:
        print('The environment variable _Parallelism_ is not set. Falling back on default parallelism.')
        stringParallelism = default_parallelism

    try:
        parallelism = int(stringParallelism)
    except ValueError:
        print('The environment variable _Parallelism_ must be set to an integer. It was: {parallelism}'.format(parallelism=stringParallelism))
        exit()

    return parallelism

def get_initial_dynamic_data_paths():
    manifest_path = os.getenv('_DlisDynamicDataJsonPath_', None)
    return parse_manifest(manifest_path)

def get_named_directories(named_paths):
    if named_paths is None:
        return None
    named_directories = []
    for named_path in named_paths:
         named_directories.append(NamedPath(named_path.name, os.path.dirname(os.path.realpath(named_path.path))))
    return named_directories

def set_up_data_updating(model):
    manifest_path = os.getenv('_DlisDynamicDataJsonPath_', None)
    if manifest_path is None or not os.path.exists(manifest_path):
         print('Dynamic data updates not set up, bypassing')
         return None
    print('Dynamic Data manifest path is set {manifest_path}'.format(manifest_path = manifest_path))

    successCallbackUrl = os.getenv('_DlisReportUpdateCompleteUrl_', None)
    executionUnitName = os.getenv('_ExecutionUnitName_', None)
    monitor = ManifestMonitor(manifest_path, model, successCallbackUrl, executionUnitName)
    monitor.start_monitor()

def parse_manifest(manifest_path):
    if manifest_path == None:
         return None

    if not os.path.exists(manifest_path):
         print('No manifest file found at {mp}'.format(mp = manifest_path))
         return None

    named_paths = []
    try:
        with open(manifest_path, 'r') as manifest:
            contents = manifest.read()
        parsed = json.loads(contents)
        global lastReadManifest
        lastReadManifest = contents

        for namedPath in parsed:
            named_paths.append(NamedPath(namedPath['Name'], namedPath['Path']))
    except Exception as ex:
        print('Failed to parse any dynamic paths from the manifest at {mp}.'.format(mp = manifest_path))
        print(ex)

    return named_paths

'''
Class that constantly monitors the manifest to check for updates
'''
class ManifestMonitor():
    def __init__(self, manifest_path, model, successCallbackUrl, executionUnitName):
        self.manifest_path = manifest_path
        self.model = model
        self.successCallbackUrl = successCallbackUrl
        self.executionUnitName = executionUnitName
        parse_manifest(self.manifest_path)

    def start_monitor(self):
        self.thread = threading.Thread(target = self.watch_forever, args=())
        self.thread.daemon = True
        self.thread.start()

    def watch_forever(self):
        print ('Starting Manifest Monitoring.')
        while True:
            try:
                # Check once every minute
                time.sleep(60)
                self.update_manifest_if_changed()
            except:
                print ('The ManifestMonitor hit an unexpected failure')
                traceback.print_exc()

    def update_manifest_if_changed(self):
        try:
            with open(self.manifest_path, 'r') as manifest:
                contents = manifest.read()
        except:
            print('Manifest Update failed to read current manifest')
            traceback.print_exc()
            return
        global lastReadManifest
        if (contents != lastReadManifest):
            try:
                print('Got signal that data has been updated.')
                updated_paths = parse_manifest(self.manifest_path)
                self.model.OnDataUpdate(updated_paths)
                print('Successfully processed data update.')
                self.report_success()
            except:
                print ('Manifest Update failed with unexpected exception ')
                traceback.print_exc()

    def report_success(self):
        request = tornado.httpclient.HTTPRequest(url=self.successCallbackUrl, method='POST', body=self.executionUnitName)
        httpClient = tornado.httpclient.HTTPClient()
        result = httpClient.fetch(request)
        httpClient.close()

class NamedPath(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path

"""
Get oaas_models folder path.

Params
------
oaas_model_dir: the directory where oaas model is located.
"""
def get_oaas_model_dir():
    current_path = os.getcwd()
    oaas_model_dir = os.path.join(os.path.realpath(current_path), "oaas_models")
    return oaas_model_dir