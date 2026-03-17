import asyncio
import json
import socket

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

latest_node_data = {}
node_ips = set()
# Per-node debug state, keyed by node IP
node_debug_state = {}


class UDPRelayServer:
    def __init__(self, listen_port=5005, cmd_port=5006):
        self.listen_port = listen_port
        self.cmd_port = cmd_port
        self.sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_in.bind(("0.0.0.0", self.listen_port))
        self.sock_in.setblocking(False)
        self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def listen(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                data, addr = await loop.sock_recvfrom(self.sock_in, 65507)  # Max UDP size
                ip = addr[0]
                node_ips.add(ip)
                latest_node_data[ip] = json.loads(data.decode())
            except Exception:
                await asyncio.sleep(0.001)

    def send_command(self, ip, cmd):
        self.sock_out.sendto(json.dumps(cmd).encode(), (ip, self.cmd_port))


relay = UDPRelayServer()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(relay.listen())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive setup/commands from browser
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)

                # Toggle debug for specific node only
                if "debugNode" in msg and "debug" in msg:
                    ip = str(msg["debugNode"])
                    val = bool(msg["debug"])
                    node_debug_state[ip] = val
                    relay.send_command(ip, {"debug": val})

                if "resetNode" in msg:
                    ip = str(msg["resetNode"])
                    latest_node_data.pop(ip, None)
                    node_ips.discard(ip)
                    node_debug_state.pop(ip, None)
            except asyncio.TimeoutError:
                pass

            # Enforce per-node debug sync (for reconnect resilience)
            for ip in node_ips:
                desired = bool(node_debug_state.get(ip, False))
                relay.send_command(ip, {"debug": desired})

            await websocket.send_json(latest_node_data)
            await asyncio.sleep(0.25)
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
