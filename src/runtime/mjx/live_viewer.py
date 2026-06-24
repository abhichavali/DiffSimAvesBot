"""Interactive live viewer: MJX steps on the GPU, frames stream to the browser.

A self-contained aiohttp server. A background task advances the composed MJX
rollout (drive + pitch + aero + ``mjx.step``) on the GPU, renders the current
state headlessly (EGL) through a mutable free camera, and pushes JPEG frames over
a WebSocket. The browser canvas draws the frames and sends mouse input
(orbit / zoom / pan) plus pause/reset back over the socket.

Run on the server (binds localhost by default)::

    cd /workspace/Tensegrity-Playground && . .venv/bin/activate
    MUJOCO_GL=egl PYTHONPATH=src python -m runtime.mjx.live_viewer --port 17070

Then view it from your machine via an SSH tunnel::

    ssh -L 17070:localhost:17070 HanatServer   # then open http://localhost:17070

or embed it in Jupyter with an IFrame to the same URL (see notebooks/).

This is a *pixel* stream (server renders), so it works through plain TCP/HTTP —
no WebRTC/UDP, no TURN. Latency is ~video-call class, not native-viewer class.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU render; must precede mujoco import
# Don't let JAX grab ~75% of VRAM up front — it OOM-probes and starves the EGL
# renderer (which shares the GPU). Allocate on demand instead.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
import mujoco
import numpy as np
from aiohttp import WSMsgType, web
from PIL import Image

import mujoco.mjx as mjx
from runtime.mjx import aero_runtime, rollout, wing_pitch


class _Sim:
    """Holds the jitted advance fn, render state, and the shared free camera."""

    def __init__(self, *, steps_per_frame: int, height: int, width: int):
        from runtime.model_loader import load_model

        self.model, data = load_model()  # CPU model, calibrated
        self.height, self.width = height, width
        self.steps_per_frame = steps_per_frame

        mx = mjx.put_model(self.model)
        self._init = (
            mjx.put_data(self.model, data),
            aero_runtime.init_aero_state(aero_runtime.build_aero_config(self.model)),
        )
        drive = rollout.build_drive_config(self.model)
        pitch = wing_pitch.config_from_runtime(
            self.model, centers=rollout._pitch_centers(self.model, data)
        )
        aero = aero_runtime.build_aero_config(self.model)
        step = rollout.make_step_fn(mx, drive, pitch, aero, lambda d: 0.0, self._init)

        @jax.jit
        def advance(carry):
            carry, _ = jax.lax.scan(step, carry, xs=None, length=steps_per_frame)
            return carry

        self._advance = advance
        # Warm both trace signatures (init carry, then steady carry) up front so the
        # live loop hits the compile cache and runs at full speed from frame 1.
        warm = self._advance(self._init)
        warm = self._advance(warm)
        warm[0].qpos.block_until_ready()
        self.carry = self._init
        self.paused = False

        # CPU render state + a free camera the browser drives.
        self._render_data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

    def reset(self):
        self.carry = self._init

    def frame(self) -> bytes:
        """Advance (unless paused), render the current state, return JPEG bytes."""
        if not self.paused:
            self.carry = self._advance(self.carry)
        qpos = np.asarray(self.carry[0].qpos)  # blocks on the device
        self._render_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._render_data)
        self._renderer.update_scene(self._render_data, camera=self.cam)
        pixels = self._renderer.render()
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    # --- camera controls (called from the websocket input handler) ---
    def orbit(self, dx: float, dy: float):
        self.cam.azimuth = (self.cam.azimuth - 0.4 * dx) % 360.0
        self.cam.elevation = float(np.clip(self.cam.elevation - 0.4 * dy, -89.0, 89.0))

    def zoom(self, delta: float):
        self.cam.distance = float(np.clip(self.cam.distance * (1.0 + 0.001 * delta), 1e-3, 50.0))

    def pan(self, dx: float, dy: float):
        # Move the lookat in the camera's screen plane, scaled by distance.
        az = np.deg2rad(self.cam.azimuth)
        s = 0.0015 * self.cam.distance
        right = np.array([np.cos(az), np.sin(az), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        self.cam.lookat[:] = np.asarray(self.cam.lookat) - s * dx * right + s * dy * up


_INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>MJX live viewer</title>
<style>
  html,body{margin:0;background:#111;color:#ccc;font:13px monospace;height:100%}
  #wrap{display:flex;flex-direction:column;height:100%}
  #bar{padding:6px 10px;background:#1b1b1b}
  #bar button{font:12px monospace;margin-right:6px}
  #c{flex:1;display:block;width:100%;background:#000;cursor:grab;touch-action:none}
  #c:active{cursor:grabbing}
</style></head><body><div id=wrap>
<div id=bar>
  <button id=pause>pause</button><button id=reset>reset</button>
  <span id=stat>connecting…</span>
  <span style="float:right">drag=orbit · wheel=zoom · shift+drag=pan</span>
</div>
<canvas id=c></canvas></div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),stat=document.getElementById('stat');
const proto=location.protocol==='https:'?'wss':'ws';
const ws=new WebSocket(proto+'://'+location.host+location.pathname.replace(/\\/$/,'')+'/ws');
ws.binaryType='arraybuffer';
let n=0,t0=performance.now();
ws.onopen=()=>stat.textContent='connected';
ws.onclose=()=>stat.textContent='disconnected';
ws.onmessage=ev=>{
  if(typeof ev.data==='string')return;
  createImageBitmap(new Blob([ev.data],{type:'image/jpeg'})).then(bm=>{
    cv.width=bm.width;cv.height=bm.height;ctx.drawImage(bm,0,0);bm.close();
    if(++n%15===0){const fps=15000/(performance.now()-t0);t0=performance.now();
      stat.textContent='live · '+fps.toFixed(0)+' fps';}
  });
};
const send=o=>{if(ws.readyState===1)ws.send(JSON.stringify(o));};
let drag=false,px=0,py=0,shift=false;
cv.addEventListener('mousedown',e=>{drag=true;px=e.clientX;py=e.clientY;shift=e.shiftKey;});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;
  send({t:shift?'pan':'orbit',dx,dy});});
cv.addEventListener('wheel',e=>{e.preventDefault();send({t:'zoom',d:e.deltaY});},{passive:false});
document.getElementById('pause').onclick=()=>send({t:'pause'});
document.getElementById('reset').onclick=()=>send({t:'reset'});
</script></body></html>"""


async def _index(request):
    return web.Response(
        text=_INDEX_HTML,
        content_type="text/html",
        headers={"Content-Security-Policy": "frame-ancestors *"},  # allow Jupyter IFrame
    )


async def _ws(request):
    sim: _Sim = request.app["sim"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["clients"].add(ws)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            import json

            m = json.loads(msg.data)
            t = m.get("t")
            if t == "orbit":
                sim.orbit(m["dx"], m["dy"])
            elif t == "pan":
                sim.pan(m["dx"], m["dy"])
            elif t == "zoom":
                sim.zoom(m["d"])
            elif t == "pause":
                sim.paused = not sim.paused
            elif t == "reset":
                sim.reset()
    finally:
        request.app["clients"].discard(ws)
    return ws


async def _broadcast_loop(app):
    sim: _Sim = app["sim"]
    target_dt = 1.0 / 30.0
    while True:
        if app["clients"]:
            # Render on the event-loop thread: the EGL/GL context is thread-affine
            # to where the Renderer was created. ~70 ms/frame; WS input coroutines
            # run during the sleep below.
            jpeg = sim.frame()
            dead = []
            for ws in app["clients"]:
                try:
                    await ws.send_bytes(jpeg)
                except ConnectionResetError:
                    dead.append(ws)
            for ws in dead:
                app["clients"].discard(ws)
        await asyncio.sleep(target_dt)


def build_app(*, steps_per_frame: int, height: int, width: int) -> web.Application:
    app = web.Application()
    app["sim"] = _Sim(steps_per_frame=steps_per_frame, height=height, width=width)
    app["clients"] = set()
    app.router.add_get("/", _index)
    app.router.add_get("/ws", _ws)

    async def _on_start(app):
        app["task"] = asyncio.create_task(_broadcast_loop(app))

    async def _on_stop(app):
        app["task"].cancel()

    app.on_startup.append(_on_start)
    app.on_cleanup.append(_on_stop)
    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=17070)
    p.add_argument("--steps-per-frame", type=int, default=20, help="MJX steps advanced per rendered frame")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--certfile", default=None, help="TLS cert (serve https; needed to embed in an https Jupyter)")
    p.add_argument("--keyfile", default=None, help="TLS key")
    args = p.parse_args(argv)

    ssl_context = None
    if args.certfile and args.keyfile:
        import ssl

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.certfile, args.keyfile)

    print(f"JAX devices: {jax.devices()}")
    app = build_app(steps_per_frame=args.steps_per_frame, height=args.height, width=args.width)
    scheme = "https" if ssl_context else "http"
    print(f"live viewer on {scheme}://{args.host}:{args.port}  (steps/frame={args.steps_per_frame})")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
