#!/usr/bin/env python3
"""
Servidor HTTP simple para servir el frontend.
"""

import http.server
import socketserver
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        pass  # silenciar logs de cada request

    def handle_error(self, request, client_address):
        pass  # silenciar BrokenPipeError del navegador

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), Handler) as httpd:
        print("="*60)
        print(f"🚀 Servidor iniciado en http://localhost:{PORT}")
        print("="*60)
        print()
        print("Abre tu navegador en: http://localhost:8000")
        print()
        print("Presiona Ctrl+C para detener el servidor")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Servidor detenido")
