#!/usr/bin/env python3
"""
Simple HTTP server with CORS support for serving report files.
Run from the reports directory: python3 server.py
"""
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/reports':
            # Return list of JSON report files
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Get all .json files in current directory
            json_files = [f for f in os.listdir('.') if f.endswith('.json')]
            json_files.sort(reverse=True)  # Most recent first
            
            response = {'files': json_files}
            self.wfile.write(json.dumps(response).encode())
        else:
            # Serve files normally
            super().do_GET()


if __name__ == '__main__':
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f'Server running on http://localhost:{port}')
    print(f'Serving files from: {os.getcwd()}')
    print('Press Ctrl+C to stop')
    httpd.serve_forever()
