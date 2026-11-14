"""Small HTTP adapter; binds to localhost unless explicitly configured otherwise."""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import traceback
from pathlib import Path
from .model import Extractor


def validate_call(call):
    if not isinstance(call, dict): raise ValueError('Expected a JSON object')
    if not isinstance(call.get('call_id'), str) or not call['call_id']: raise ValueError('call_id must be a nonempty string')
    if not isinstance(call.get('started_at'),str): raise ValueError('started_at must be an ISO datetime')
    datetime.fromisoformat(call['started_at'])
    if not isinstance(call.get('turns'), list): raise ValueError('turns must be a list')
    seen=set()
    for turn in call['turns']:
        if not isinstance(turn,dict) or not isinstance(turn.get('text'),str): raise ValueError('Each turn needs text')
        if turn.get('speaker') not in ['client','operator']: raise ValueError('speaker must be client or operator')
        if type(turn.get('idx')) is not int or turn['idx']<0 or turn['idx'] in seen: raise ValueError('Turn idx must be unique nonnegative integers')
        seen.add(turn['idx'])
    # Explicitly exclude reference labels before entering the model.
    return {k:v for k,v in call.items() if k!='facts'}


def serve(model_path, host='127.0.0.1', port=8000, engine='baseline', device='cpu', normalizer='artifacts/cpu_baseline.joblib', generator=None,
          date_ranker=None):
    if engine=='hybrid':
        from .hybrid import HybridExtractor
        model=HybridExtractor(model_path,normalizer,device,generator,date_ranker=date_ranker)
    elif engine=='neural':
        from .neural import NeuralExtractor
        model=NeuralExtractor(model_path,normalizer,device)
    elif engine=='temporal':
        from .pipeline import TemporalExtractor
        model=TemporalExtractor(model_path)
    else:model=Extractor.load(model_path)
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data):
            body=json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path=='/':
                body=Path(__file__).with_name('demo.html').read_bytes()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            elif self.path=='/health': self.respond(200,{'status':'ok','model':engine,'device':device})
            else: self.respond(404,{'error':'Use POST /extract or GET /health'})

        def do_POST(self):
            if self.path!='/extract': return self.respond(404,{'error':'Unknown endpoint'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 2_000_000: return self.respond(413,{'error':'Body must be 1..2000000 bytes'})
                payload=json.loads(self.rfile.read(length))
                call=validate_call(payload)
            except (ValueError,TypeError,UnicodeDecodeError) as exc:
                return self.respond(400,{'error':str(exc)})
            try: self.respond(200,model.predict(call))
            except Exception:
                traceback.print_exc()
                self.respond(500,{'error':'Inference failed'})
    print(f'Listening on http://{host}:{port}',flush=True)
    HTTPServer((host,port),Handler).serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='artifacts/cpu_baseline.joblib')
    parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--engine',choices=['baseline','temporal','neural','hybrid'],default='baseline')
    parser.add_argument('--device',default='cpu');parser.add_argument('--normalizer',default='artifacts/cpu_baseline.joblib')
    parser.add_argument('--generator',help='Optional saved ruT5 value normalizer (hybrid engine)')
    parser.add_argument('--date-ranker',help='Optional fit-trained date ranker (hybrid engine)')
    args=parser.parse_args()
    serve(args.model,args.host,args.port,args.engine,args.device,args.normalizer,args.generator,args.date_ranker)
