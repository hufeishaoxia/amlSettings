import os
import jwt
import requests

verify_endpoint = "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/oauth2/v2.0/token"
# DLIS Scope
verify_scope = "e65e832b-d26e-4d59-be94-d261cd10435c/.default"

verify_client_id = "e2d5da44-f8aa-4cce-b9eb-729b3402fc62"
verify_client_secret = "emj8Q~_4kmkwPaFxfbYWhEiC3KFT4L-obUP4dcXt"

verify_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
verify_request = {
    "client_id": verify_client_id,
    "client_secret": verify_client_secret,
    "scope": verify_scope,
    "grant_type": "client_credentials"
}
verify_response = requests.post(verify_endpoint, data=verify_request, headers=verify_headers)
print(verify_response)
verify_token = verify_response.json()["access_token"]

dlis_endpoint = "https://EastUS2.bing.prod.dlis.binginternal.com/route/coreranker.Imagen_flux_schnell"

json_dict = {"prompt": "A scale with weights shifting back and forth"}
response = requests.post(dlis_endpoint, headers={'Content-Type': 'application/json', "Authorization": "Bearer " + verify_token}, json=json_dict)
print(response)
