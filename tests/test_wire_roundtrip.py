"""Real loopback upload and Python-to-JavaScript binary compatibility."""
import json
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from werkzeug.serving import make_server

from brainelecqc import create_app
from tests.edf_fixtures import write_edf_bdf


def test_http_upload_to_javascript_samples(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js required for cross-language wire acceptance")
    source = write_edf_bdf(tmp_path / "input.edf", [
        {"label": "same", "records": [[-2, -1, 0, 1]]},
        {"label": "same", "records": [[12, 11, 10, 9]], "unit": "mV"}])
    app = create_app({"STORAGE_ROOT": str(tmp_path / "store")})
    server = make_server("127.0.0.1", 0, app)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}/api/v1/ephys"
    boundary = "brain-elec-qc-test-boundary"
    payload = json.dumps({"modality": "eeg", "label": "roundtrip"}).encode()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n'.encode()+payload+
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="input.edf"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+source.read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
    try:
        req = Request(base+"/recordings", data=body, headers={"Content-Type": "multipart/form-data; boundary="+boundary})
        with urlopen(req, timeout=10) as response:
            assert response.status == 201
            record = json.load(response)["data"]
        aid = record["input_artifact_id"]
        with urlopen(base+f"/artifacts/{aid}/signals", timeout=10) as response:
            stream = json.load(response)["data"]["streams"][0]
        ordered = [c["channel_id"] for c in reversed(stream["channels"])]
        expected = {"artifact_id": aid, "stream_id": stream["stream_id"],
                    "segment_id": stream["segments"][0]["segment_id"], "channel_ids": ordered,
                    "start_sample": 0, "stop_sample": 4, "representation": "physical"}
        query = [(k, v) for k, v in expected.items() if k not in ("artifact_id", "channel_ids")]
        query += [("channel_id", c) for c in ordered]
        with urlopen(base+f"/artifacts/{aid}/samples?"+urlencode(query), timeout=10) as response:
            frame = response.read()
            assert response.headers.get_content_type() == "application/octet-stream"
        wire = tmp_path / "wire.bin"
        wire.write_bytes(frame)
        decoder = (Path(__file__).parents[1] / "web/ephysBinary.js").resolve().as_uri()
        js = """
import {readFileSync} from 'node:fs';
const {decodeEphysSamples}=await import(process.argv[1]);
const result=decodeEphysSamples(readFileSync(process.argv[2]), JSON.parse(process.argv[3]));
console.log(JSON.stringify({samples:[...result.samples],units:result.metadata.units,shape:result.metadata.shape}));
"""
        result = subprocess.run(["node", "--input-type=module", "-e", js, decoder, str(wire), json.dumps(expected)],
                                capture_output=True, text=True, check=True)
        decoded = json.loads(result.stdout)
        assert decoded["samples"] == pytest.approx([.012,.011,.010,.009,-2e-6,-1e-6,0,1e-6], abs=1e-15)
        assert decoded["units"] == ["V", "V"] and decoded["shape"] == [2, 4]
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
