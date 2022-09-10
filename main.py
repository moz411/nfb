import os
from google.cloud import storage
from flask import Flask, request

from report import report
from jinja2 import Environment, FileSystemLoader

jinja_env = Environment(loader=FileSystemLoader('.'))

app = Flask(__name__)


# Start a new analysis and return
@app.route('/<string:uuid>/<string:sessionid>', methods = ['POST'])
def generate(uuid, sessionid):
    board_id = int(request.form.get('board_id'))
    report(uuid,sessionid,board_id)
    return('Report %s generated' % sessionid)
  
# Request existing results.json
@app.route('/<string:uuid>/<string:sessionid>/<string:filename>', methods = ['GET'])
def retreive(uuid, sessionid, filename):
    bucket = os.environ.get("BUCKET_NAME")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    destination_file_name = '/tmp/%s.%s.%s' % (filename, uuid, sessionid)
    source_blob_name = '%s/%s/%s' % (uuid, sessionid, filename)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    with open(destination_file_name, 'r') as file:
        results = file.read()
    os.unlink(destination_file_name)
    return(results)

# Start script
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
