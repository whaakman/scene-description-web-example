import requests
import json
import prompty
from dataclasses import dataclass, asdict
from typing import List
import prompty.azure
from prompty.tracer import trace, Tracer, console_tracer, PromptyTracer
import os
import sys
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from tqdm import tqdm
import shutil

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Tracer.add("console", console_tracer)
json_tracer = PromptyTracer()
Tracer.add(PromptyTracer, json_tracer.tracer)

@dataclass
class Metadata:
    width: int
    height: int

@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class Value:
    text: str
    confidence: float
    boundingBox: BoundingBox

@dataclass
class DenseCaptionResult:
    values: List[Value]

@dataclass
class AnalyzeResult:
    modelVersion: str
    metadata: Metadata
    denseCaptionsResult: DenseCaptionResult

@dataclass
class AnalyzeRequest:
    uri: str

CognitiveServicesEndpoint = os.getenv("AZURE_COGNITIVESERVICES_ENDPOINT")
CognitiveServiceEndpointKey = os.getenv("AZURE_OPENAI_KEY")
DefaultImageURL = "https://saimagesaiwh.blob.core.windows.net/images/image0.jpeg"

class DenseCaption:
    def generate_dense_caption(self, image_url: str):
        endpoint = CognitiveServicesEndpoint
        url = f"{endpoint}computervision/imageanalysis:analyze?features=denseCaptions&gender-neutral-caption=false&api-version=2023-10-01"
        key = CognitiveServiceEndpointKey 

        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Content-Type': 'application/json; charset=utf-8'
        }

        analyze_request = AnalyzeRequest(uri=image_url)
        json_data = asdict(analyze_request)

        response = requests.post(url, headers=headers, json=json_data)
        response_content = response.text

        data = json.loads(response_content)
        try:
            deserialized_object = self.from_dict(AnalyzeResult, data)
            captions = [value.text for value in deserialized_object.denseCaptionsResult.values]
            return captions
        except KeyError as e:
            print(f"KeyError: {e}. Please check the JSON response structure.")
            return []

    def from_dict(self, data_class, data):
        if isinstance(data, list):
            return [self.from_dict(data_class.__args__[0], item) for item in data]
        if isinstance(data, dict):
            fieldtypes = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
            return data_class(**{k: self.from_dict(fieldtypes[k], v) for k, v in data.items()})
        return data

class SceneDescriptionAssistant:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint

    @trace
    def run(
        question: any
    ) -> str:
        result = prompty.execute(
            "imagecaption.prompty",
            inputs={"question": question}
        )
        return result
    
# Initialize Flask app
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("BLOB_STORAGE_CONNECTION_STRING"))
container_name = "uploads"
container_client = blob_service_client.get_container_client(container_name)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + "_" + filename

            # Upload to Azure Blob Storage directly with progress bar
            blob_client = container_client.get_blob_client(unique_filename)
            file_size = file.content_length
            progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=filename)

            def upload_progress(response):
                progress.update(int(response.http_response.headers['Content-Length']))

            blob_client.upload_blob(file.stream, raw_response_hook=upload_progress)
            progress.close()

            # Empty the /uploads folder
            upload_folder = os.path.join(app.root_path, 'uploads')
            if os.path.exists(upload_folder):
                shutil.rmtree(upload_folder)
                os.makedirs(upload_folder)

            image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{unique_filename}"
            denseCaption = DenseCaption()
            captions = denseCaption.generate_dense_caption(image_url)
            results = SceneDescriptionAssistant.run(captions)
            return render_template('result.html', image_url=image_url, captions=captions, results=results)
    return render_template('index.html')

@app.route('/upload_async', methods=['POST'])
def upload_file_async():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename

        # Upload to Azure Blob Storage directly with progress bar
        blob_client = container_client.get_blob_client(unique_filename)
        file_size = file.content_length
        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=filename)

        def upload_progress(response):
            progress.update(int(response.http_response.headers['Content-Length']))

        blob_client.upload_blob(file.stream, raw_response_hook=upload_progress)
        progress.close()

        image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{unique_filename}"
        denseCaption = DenseCaption()
        captions = denseCaption.generate_dense_caption(image_url)
        results = SceneDescriptionAssistant.run(captions)
        return jsonify({"image_url": image_url, "captions": captions, "results": results})
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('upload_file'))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
