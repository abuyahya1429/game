"""
Neon Hand Hockey 3D - Ultimate Edition
Features: Hologram Scoreboard, Power Shot, Slow Motion, Tron Arena,
Lightning Trail, Multiplayer, Voice Command, AI Prediction, Dynamic Shaders,
Realtime Reflections, 3D Particles
"""

import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders
import mediapipe as mp
import math
import random
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import threading
import queue
import socket
import json
import struct

# Try importing voice recognition
try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    print("SpeechRecognition not installed. Voice commands disabled.")

# ============== CONSTANTS ==============
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS_TARGET = 60

# Arena dimensions
ARENA_WIDTH = 7.0
ARENA_HEIGHT = 9.0
WALL_HEIGHT = 0.6
GOAL_WIDTH = 2.2

# Physics
PUCK_RADIUS = 0.28
PADDLE_RADIUS = 0.38
FRICTION = 0.992
RESTITUTION = 0.88
MAX_PUCK_SPEED = 18.0
POWER_SHOT_MULTIPLIER = 2.5

# Colors (Tron Legacy Style)
NEON_CYAN = (0.0, 0.9, 1.0)
NEON_ORANGE = (1.0, 0.4, 0.0)
NEON_WHITE = (0.9, 0.95, 1.0)
NEON_BLUE = (0.1, 0.4, 1.0)
NEON_PURPLE = (0.7, 0.2, 1.0)
NEON_YELLOW = (1.0, 0.9, 0.2)
NEON_GREEN = (0.2, 1.0, 0.5)
NEON_RED = (1.0, 0.2, 0.2)
TRON_BLACK = (0.01, 0.01, 0.02)

# ============== SHADERS ==============
VERTEX_SHADER = """
#version 120
varying vec3 vertex_position;
varying vec3 vertex_normal;
varying vec2 tex_coord;

void main() {
    vertex_position = vec3(gl_ModelViewMatrix * gl_Vertex);
    vertex_normal = normalize(gl_NormalMatrix * gl_Normal);
    tex_coord = gl_MultiTexCoord0.xy;
    gl_Position = ftransform();
}
"""

FRAGMENT_SHADER_GLOW = """
#version 120
varying vec3 vertex_position;
varying vec3 vertex_normal;
uniform vec3 base_color;
uniform float glow_intensity;
uniform float time;

void main() {
    vec3 light_dir = normalize(vec3(0.0, 0.0, 1.0));
    float diff = max(dot(vertex_normal, light_dir), 0.0);
    
    // Fresnel effect for neon glow
    vec3 view_dir = normalize(-vertex_position);
    float fresnel = pow(1.0 - max(dot(vertex_normal, view_dir), 0.0), 3.0);
    
    vec3 color = base_color * (0.3 + diff * 0.7);
    color += base_color * fresnel * glow_intensity;
    
    // Pulsing effect
    float pulse = 0.8 + 0.2 * sin(time * 3.0);
    color *= pulse;
    
    gl_FragColor = vec4(color, 1.0);
}
"""

FRAGMENT_SHADER_LIGHTNING = """
#version 120
varying vec2 tex_coord;
uniform float time;
uniform vec3 color1;
uniform vec3 color2;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = tex_coord;
    float noise = random(uv + time) * 0.3;
    
    // Electric arcs
    float arc = sin(uv.x * 50.0 + time * 20.0) * 0.1;
    arc += sin(uv.x * 100.0 - time * 30.0) * 0.05;
    
    float dist = abs(uv.y - 0.5 - arc);
    float intensity = 1.0 - smoothstep(0.0, 0.2, dist);
    intensity += noise * intensity;
    
    vec3 color = mix(color1, color2, intensity);
    gl_FragColor = vec4(color * intensity * 2.0, intensity);
}
"""

# ============== DATA CLASSES ==============
@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        if scalar == 0:
            return Vector3()
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector3()
    
    def lerp(self, target, t):
        return self + (target - self) * t
    
    def to_tuple(self):
        return (self.x, self.y, self.z)

@dataclass
class Particle3D:
    position: Vector3
    velocity: Vector3
    color: Tuple[float, float, float]
    size: float = 0.1
    life: float = 1.0
    max_life: float = 1.0
    gravity: float = 0.0
    rotation: float = 0.0
    rotation_speed: float = 0.0
    particle_type: str = "sphere"
    
    def update(self, dt):
        self.position = self.position + self.velocity * dt
        self.velocity.y -= self.gravity * dt
        self.velocity = self.velocity * 0.98
        self.life -= dt
        self.rotation += self.rotation_speed * dt
        self.size *= 0.995
    
    def is_alive(self):
        return self.life > 0 and self.size > 0.01

# ============== SHADER MANAGER ==============
class ShaderManager:
    def __init__(self):
        self.programs = {}
        self.time = 0.0
        self._compile_shaders()
    
    def _compile_shaders(self):
        try:
            # Glow shader
            vs = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
            fs = shaders.compileShader(FRAGMENT_SHADER_GLOW, GL_FRAGMENT_SHADER)
            self.programs['glow'] = shaders.compileProgram(vs, fs)
            
            # Lightning shader
            fs_lightning = shaders.compileShader(FRAGMENT_SHADER_LIGHTNING, GL_FRAGMENT_SHADER)
            self.programs['lightning'] = shaders.compileProgram(vs, fs_lightning)
        except Exception as e:
            print(f"Shader compilation failed: {e}")
            self.programs = {}
    
    def use(self, name):
        if name in self.programs:
            glUseProgram(self.programs[name])
    
    def stop(self):
        glUseProgram(0)
    
    def set_uniform(self, name, value):
        for prog_name, prog in self.programs.items():
            try:
                loc = glGetUniformLocation(prog, name)
                if loc != -1:
                    if isinstance(value, (int, float)):
                        glUniform1f(loc, value)
                    elif isinstance(value, tuple) and len(value) == 3:
                        glUniform3f(loc, *value)
            except:
                pass
    
    def update(self, dt):
        self.time += dt

# ============== VOICE CONTROLLER ==============
class VoiceController:
    def __init__(self):
        self.enabled = VOICE_ENABLED
        self.command = None
        self.running = False
        self.thread = None
        
        if self.enabled:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
    
    def _listen_loop(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=1)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    if "boost" in text:
                        self.command = "boost"
                    elif "slow" in text:
                        self.command = "slow"
                    elif "reset" in text:
                        self.command = "reset"
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
            except:
                time.sleep(0.1)
    
    def get_command(self):
        cmd = self.command
        self.command = None
        return cmd
    
    def release(self):
        self.running = False

# ============== NETWORK MANAGER ==============
class NetworkManager:
    def __init__(self):
        self.socket = None
        self.is_host = False
        self.connected = False
        self.peer_address = None
        self.running = False
        self.receive_thread = None
        self.data_queue = queue.Queue()
        
    def host_game(self, port=5555):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', port))
            self.socket.settimeout(0.1)
            self.is_host = True
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            return True
        except Exception as e:
            print(f"Host error: {e}")
            return False
    
    def join_game(self, host_ip, port=5555):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(0.1)
            self.peer_address = (host_ip, port)
            self.is_host = False
            self.running = True
            
            # Send handshake
            self.send_data({'type': 'connect'})
            
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            return True
        except Exception as e:
            print(f"Join error: {e}")
            return False
    
    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                if not self.peer_address and self.is_host:
                    self.peer_address = addr
                    self.connected = True
                parsed = json.loads(data.decode())
                self.data_queue.put(parsed)
            except socket.timeout:
                pass
            except:
                pass
    
    def send_data(self, data: dict):
        if self.socket and self.peer_address:
            try:
                self.socket.sendto(json.dumps(data).encode(), self.peer_address)
            except:
                pass
    
    def get_received_data(self):
        data_list = []
        while not self.data_queue.empty():
            data_list.append(self.data_queue.get())
        return data_list
    
    def close(self):
        self.running = False
        if self.socket:
            self.socket.close()

# ============== HAND TRACKING ==============
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        self.detected = False
        self.landmarks = None
        self.gesture = "open"
        self.position = (0.5, 0.5)
        self.fist_hold_time = 0.0
        self.power_charged = 0.0
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True
        
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
    
    def _process_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    self.detected = True
                    self.landmarks = results.multi_hand_landmarks[0]
                    self._analyze_gesture()
                    self._get_palm_position()
                else:
                    self.detected = False
                    self.gesture = "none"
                
                self.result_queue.put((self.detected, self.gesture, self.position, self.power_charged))
            except:
                pass
    
    def update(self, frame, dt):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(frame)
        
        # Update power charge
        if self.gesture == "fist":
            self.fist_hold_time += dt
            self.power_charged = min(1.0, self.fist_hold_time / 1.5)
        else:
            self.fist_hold_time = 0.0
            self.power_charged = max(0, self.power_charged - dt * 2)
        
        try:
            detected, gesture, position, _ = self.result_queue.get_nowait()
            self.detected = detected
            self.gesture = gesture
            self.position = position
        except:
            pass
    
    def _analyze_gesture(self):
        if not self.landmarks:
            return
        
        # Finger landmarks
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        
        fingers_up = 0
        
        # Thumb
        if self.landmarks.landmark[4].x < self.landmarks.landmark[3].x:
            fingers_up += 1
        
        # Other fingers
        for i in range(1, 5):
            if self.landmarks.landmark[tips[i]].y < self.landmarks.landmark[pips[i]].y:
                fingers_up += 1
        
        if fingers_up >= 4:
            self.gesture = "open"
        elif fingers_up <= 1:
            self.gesture = "fist"
        elif self.landmarks.landmark[8].y < self.landmarks.landmark[6].y:
            self.gesture = "pointing"
        else:
            self.gesture = "partial"
    
    def _get_palm_position(self):
        if not self.landmarks:
            return
        
        wrist = self.landmarks.landmark[0]
        middle_mcp = self.landmarks.landmark[9]
        
        palm_x = (wrist.x + middle_mcp.x) / 2
        palm_y = (wrist.y + middle_mcp.y) / 2
        
        self.position = (1 - palm_x, palm_y)
    
    def get_arena_position(self):
        x = (self.position[0] - 0.5) * ARENA_WIDTH
        y = (0.5 - self.position[1]) * ARENA_HEIGHT * 0.45 + ARENA_HEIGHT * 0.25
        return (x, y)
    
    def release(self):
        self.running = False
        self.hands.close()

# ============== PARTICLE SYSTEM ==============
class ParticleSystem3D:
    def __init__(self):
        self.particles: List[Particle3D] = []
        self.max_particles = 800
    
    def emit_explosion(self, position: Vector3, color: Tuple, count: int = 50):
        for _ in range(count):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            speed = random.uniform(3, 10)
            
            velocity = Vector3(
                speed * math.sin(phi) * math.cos(theta),
                speed * math.sin(phi) * math.sin(theta),
                speed * math.cos(phi)
            )
            
            particle = Particle3D(
                position=Vector3(position.x, position.y, position.z),
                velocity=velocity,
                color=color,
                size=random.uniform(0.05, 0.2),
                life=random.uniform(0.5, 1.5),
                max_life=1.5,
                gravity=random.uniform(0, 3),
                rotation=random.uniform(0, 360),
                rotation_speed=random.uniform(-360, 360),
                particle_type=random.choice(["sphere", "spark", "ring"])
            )
            self.particles.append(particle)
        
        self._limit_particles()
    
    def emit_collision_sparks(self, position: Vector3, normal: Vector3, color: Tuple):
        for _ in range(30):
            direction = Vector3(
                normal.x + random.uniform(-0.5, 0.5),
                normal.y + random.uniform(-0.5, 0.5),
                random.uniform(0.3, 1.0)
            ).normalize()
            
            particle = Particle3D(
                position=Vector3(position.x, position.y, position.z),
                velocity=direction * random.uniform(2, 6),
                color=color,
                size=random.uniform(0.03, 0.1),
                life=random.uniform(0.2, 0.6),
                max_life=0.6,
                particle_type="spark"
            )
            self.particles.append(particle)
    
    def emit_lightning_bolt(self, start: Vector3, end: Vector3):
        mid = (start + end) * 0.5
        mid.x += random.uniform(-0.3, 0.3)
        mid.z += random.uniform(0.1, 0.3)
        
        particle = Particle3D(
            position=mid,
            velocity=Vector3(0, 0, 0),
            color=NEON_CYAN,
            size=0.3,
            life=0.1,
            max_life=0.1,
            particle_type="lightning"
        )
        self.particles.append(particle)
    
    def _limit_particles(self):
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]
    
    def update(self, dt):
        for particle in self.particles[:]:
            particle.update(dt)
            if not particle.is_alive():
                self.particles.remove(particle)
    
    def render(self):
        if not self.particles:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        for particle in self.particles:
            if particle.particle_type == "spark":
                self._render_spark(particle)
            elif particle.particle_type == "ring":
                self._render_ring(particle)
            else:
                self._render_sphere(particle)
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def _render_sphere(self, particle):
        alpha = particle.life / particle.max_life
        glColor4f(particle.color[0], particle.color[1], particle.color[2], alpha)
        glPushMatrix()
        glTranslatef(particle.position.x, particle.position.y, particle.position.z)
        glRotatef(particle.rotation, 0, 0, 1)
        
        quadric = gluNewQuadric()
        gluSphere(quadric, particle.size, 8, 8)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def _render_spark(self, particle):
        alpha = particle.life / particle.max_life
        glColor4f(particle.color[0], particle.color[1], particle.color[2], alpha)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex3f(particle.position.x, particle.position.y, particle.position.z)
        end_pos = particle.position + particle.velocity.normalize() * particle.size * 2
        glVertex3f(end_pos.x, end_pos.y, end_pos.z)
        glEnd()
    
    def _render_ring(self, particle):
        alpha = particle.life / particle.max_life
        glColor4f(particle.color[0], particle.color[1], particle.color[2], alpha)
        glPushMatrix()
        glTranslatef(particle.position.x, particle.position.y, particle.position.z)
        glRotatef(particle.rotation, 0, 0, 1)
        
        glBegin(GL_LINE_LOOP)
        for i in range(16):
            angle = 2 * math.pi * i / 16
            glVertex3f(math.cos(angle) * particle.size, math.sin(angle) * particle.size, 0)
        glEnd()
        
        glPopMatrix()

# ============== GAME OBJECTS ==============
@dataclass
class Puck:
    position: Vector3
    velocity: Vector3
    radius: float = PUCK_RADIUS
    trail: List[Vector3] = field(default_factory=list)
    lightning_points: List[Vector3] = field(default_factory=list)
    glow_intensity: float = 1.0
    is_powered: bool = False
    
    def update(self, dt, slow_motion_factor=1.0):
        effective_dt = dt * slow_motion_factor
        self.position = self.position + self.velocity * effective_dt
        self.velocity = self.velocity * FRICTION
        
        speed = self.velocity.magnitude()
        if speed > MAX_PUCK_SPEED:
            self.velocity = self.velocity.normalize() * MAX_PUCK_SPEED
        
        # Trail
        self.trail.append(Vector3(self.position.x, self.position.y, self.position.z))
        if len(self.trail) > 25:
            self.trail.pop(0)
        
        # Lightning effect points
        if speed > 8:
            self._generate_lightning()
    
    def _generate_lightning(self):
        self.lightning_points.clear()
        if len(self.trail) < 2:
            return
        
        for i in range(len(self.trail) - 1):
            point = self.trail[i]
            jitter = Vector3(
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(0, 0.15)
            )
            self.lightning_points.append(point + jitter)
    
    def reset(self, towards_player=True):
        self.position = Vector3(0, 0, PUCK_RADIUS + 0.1)
        direction = 1 if towards_player else -1
        angle = random.uniform(-0.3, 0.3)
        speed = random.uniform(4, 6)
        self.velocity = Vector3(math.sin(angle) * speed, direction * math.cos(angle) * speed, 0)
        self.trail.clear()
        self.lightning_points.clear()
        self.is_powered = False

@dataclass
class Paddle:
    position: Vector3
    target_position: Vector3
    velocity: Vector3 = field(default_factory=Vector3)
    radius: float = PADDLE_RADIUS
    color: Tuple = NEON_CYAN
    is_boosted: bool = False
    is_player: bool = True
    power_charge: float = 0.0
    prev_position: Vector3 = field(default_factory=Vector3)
    
    def update(self, dt, slow_motion_factor=1.0):
        self.prev_position = Vector3(self.position.x, self.position.y, self.position.z)
        
        speed = 18.0 if self.is_boosted else 12.0
        diff = self.target_position - self.position
        self.velocity = diff * speed
        self.position = self.position + self.velocity * dt * slow_motion_factor
        
        # Constrain to arena half
        half_height = ARENA_HEIGHT / 2
        if self.is_player:
            self.position.y = max(0.5, min(half_height - 0.4, self.position.y))
        else:
            self.position.y = max(-half_height + 0.4, min(-0.5, self.position.y))
        
        self.position.x = max(-ARENA_WIDTH/2 + self.radius + 0.3,
                              min(ARENA_WIDTH/2 - self.radius - 0.3, self.position.x))

# ============== AI SYSTEM ==============
class AdvancedAI:
    def __init__(self, difficulty: int = 3):
        self.difficulty = difficulty
        self.prediction_accuracy = 0.6 + difficulty * 0.08
        self.reaction_delay = 0.15 - difficulty * 0.02
        self.last_update = 0
        self.predicted_positions: List[Vector3] = []
        self.target = Vector3()
    
    def update(self, puck: Puck, paddle: Paddle, dt: float):
        self.last_update += dt
        
        if self.last_update < self.reaction_delay:
            paddle.update(dt)
            return
        
        self.last_update = 0
        self.predicted_positions.clear()
        
        if puck.velocity.y < 0:  # Puck moving towards AI
            target_x, target_y = self._predict_intercept(puck)
            self.target = Vector3(target_x, target_y, paddle.position.z)
        else:
            # Return to defense position
            self.target = Vector3(puck.position.x * 0.5, -ARENA_HEIGHT * 0.38, paddle.position.z)
        
        paddle.target_position = self.target
        paddle.update(dt)
    
    def _predict_intercept(self, puck: Puck) -> Tuple[float, float]:
        if puck.velocity.y >= 0:
            return (0, -ARENA_HEIGHT * 0.35)
        
        # Simulate puck trajectory
        sim_pos = Vector3(puck.position.x, puck.position.y, puck.position.z)
        sim_vel = Vector3(puck.velocity.x, puck.velocity.y, puck.velocity.z)
        
        target_y = -ARENA_HEIGHT * 0.35
        dt = 0.016
        
        for step in range(120):
            sim_pos = sim_pos + sim_vel * dt
            sim_vel = sim_vel * FRICTION
            
            # Wall bounces
            if abs(sim_pos.x) > ARENA_WIDTH / 2 - PUCK_RADIUS:
                sim_vel.x = -sim_vel.x
                sim_pos.x = max(-ARENA_WIDTH/2 + PUCK_RADIUS, min(ARENA_WIDTH/2 - PUCK_RADIUS, sim_pos.x))
            
            if sim_pos.y <= target_y:
                predicted_x = sim_pos.x + random.gauss(0, 0.2 * (6 - self.difficulty))
                
                # Store prediction for visualization
                self.predicted_positions.append(Vector3(predicted_x, target_y, 0.1))
                
                return (predicted_x, target_y)
        
        return (0, target_y)
    
    def render_prediction(self):
        if not self.predicted_positions:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        
        # Draw prediction path
        glColor4f(*NEON_ORANGE, 0.5)
        glBegin(GL_LINE_STRIP)
        for pos in self.predicted_positions:
            glVertex3f(pos.x, pos.y, pos.z)
        glEnd()
        
        # Draw target marker
        if self.predicted_positions:
            target = self.predicted_positions[-1]
            self._draw_target_marker(target)
        
        glEnable(GL_LIGHTING)
    
    def _draw_target_marker(self, pos: Vector3):
        glColor4f(*NEON_ORANGE, 0.8)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = 2 * math.pi * i / 32 + time.time() * 2
            r = 0.3 + 0.1 * math.sin(time.time() * 5)
            glVertex3f(pos.x + math.cos(angle) * r, pos.y + math.sin(angle) * r, 0.1)
        glEnd()

# ============== RENDERER ==============
class AdvancedRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shader_manager = ShaderManager()
        self.camera_distance = 14.0
        self.camera_shake = Vector3()
        self.quadric = None
        self.time = 0.0
        
    def init_opengl(self):
        glClearColor(*TRON_BLACK, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)
        
        # Main light
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 15, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.25, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.6, 0.6, 0.7, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1))
        
        # Rim light
        glLightfv(GL_LIGHT1, GL_POSITION, (5, 5, 10, 1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.3, 0.3, 0.5, 1))
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
    
    def update(self, dt):
        self.time += dt
        self.shader_manager.update(dt)
        
        # Decay camera shake
        self.camera_shake = self.camera_shake * 0.9
    
    def add_shake(self, intensity: float):
        self.camera_shake = Vector3(
            random.uniform(-1, 1) * intensity,
            random.uniform(-1, 1) * intensity,
            random.uniform(-1, 1) * intensity * 0.5
        )
    
    def setup_camera(self, game_state):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Dynamic camera with slow motion effect
        angle = self.time * 0.1
        shake = self.camera_shake
        
        cam_x = math.sin(angle) * 2 + shake.x
        cam_y = 10 + shake.y
        cam_z = self.camera_distance + math.cos(angle) * 1.5 + shake.z
        
        gluLookAt(cam_x, cam_y, cam_z,
                  0, 0, 0,
                  0, 0, 1)
    
    def render_arena_with_reflections(self, puck: Puck, player_paddle: Paddle, ai_paddle: Paddle):
        # Render floor reflection first
        self._render_floor_reflection(puck, player_paddle, ai_paddle)
        
        # Render main scene
        self._render_tron_arena()
        
        # Render game objects
        self._render_puck(puck)
        self._render_paddle(player_paddle)
        self._render_paddle(ai_paddle)
    
    def _render_floor_reflection(self, puck: Puck, player_paddle: Paddle, ai_paddle: Paddle):
        glEnable(GL_STENCIL_TEST)
        
        # Draw floor to stencil
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
        glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        
        glBegin(GL_QUADS)
        glVertex3f(-ARENA_WIDTH/2 - 1, -ARENA_HEIGHT/2 - 1, 0)
        glVertex3f(ARENA_WIDTH/2 + 1, -ARENA_HEIGHT/2 - 1, 0)
        glVertex3f(ARENA_WIDTH/2 + 1, ARENA_HEIGHT/2 + 1, 0)
        glVertex3f(-ARENA_WIDTH/2 - 1, ARENA_HEIGHT/2 + 1, 0)
        glEnd()
        
        # Render reflections
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)
        glStencilFunc(GL_EQUAL, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        
        glPushMatrix()
        glScalef(1, 1, -1)
        glTranslatef(0, 0, -0.05)
        
        # Draw reflected objects with reduced alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glColor4f(0.3, 0.3, 0.3, 0.4)
        self._render_puck_geometry(puck, alpha=0.3)
        self._render_paddle_geometry(player_paddle, alpha=0.3)
        self._render_paddle_geometry(ai_paddle, alpha=0.3)
        
        glPopMatrix()
        
        glDisable(GL_STENCIL_TEST)
        
        # Render transparent floor
        glEnable(GL_BLEND)
        glColor4f(0.02, 0.02, 0.05, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(-ARENA_WIDTH/2 - 1, -ARENA_HEIGHT/2 - 1, 0.001)
        glVertex3f(ARENA_WIDTH/2 + 1, -ARENA_HEIGHT/2 - 1, 0.001)
        glVertex3f(ARENA_WIDTH/2 + 1, ARENA_HEIGHT/2 + 1, 0.001)
        glVertex3f(-ARENA_WIDTH/2 - 1, ARENA_HEIGHT/2 + 1, 0.001)
        glEnd()
    
    def _render_tron_arena(self):
        glDisable(GL_LIGHTING)
        
        # Grid floor - Tron style
        glColor4f(*NEON_CYAN, 0.15)
        glLineWidth(1)
        glBegin(GL_LINES)
        for i in range(-4, 5):
            glVertex3f(i, -ARENA_HEIGHT/2, 0.01)
            glVertex3f(i, ARENA_HEIGHT/2, 0.01)
        for i in range(-5, 6):
            glVertex3f(-ARENA_WIDTH/2, i, 0.01)
            glVertex3f(ARENA_WIDTH/2, i, 0.01)
        glEnd()
        
        # Main arena lines
        glColor4f(*NEON_CYAN, 0.9)
        glLineWidth(3)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-ARENA_WIDTH/2, -ARENA_HEIGHT/2, 0.02)
        glVertex3f(ARENA_WIDTH/2, -ARENA_HEIGHT/2, 0.02)
        glVertex3f(ARENA_WIDTH/2, ARENA_HEIGHT/2, 0.02)
        glVertex3f(-ARENA_WIDTH/2, ARENA_HEIGHT/2, 0.02)
        glEnd()
        
        # Center line
        glColor4f(*NEON_ORANGE, 0.7)
        glBegin(GL_LINES)
        glVertex3f(-ARENA_WIDTH/2, 0, 0.02)
        glVertex3f(ARENA_WIDTH/2, 0, 0.02)
        glEnd()
        
        # Center circle
        self._draw_glowing_circle(0, 0, 1.2, NEON_ORANGE, 32)
        
        # Goals with glow
        goal_half = GOAL_WIDTH / 2
        
        # Player goal (cyan)
        glColor4f(*NEON_CYAN, 0.6)
        glLineWidth(4)
        glBegin(GL_LINES)
        glVertex3f(-goal_half, -ARENA_HEIGHT/2, 0.02)
        glVertex3f(goal_half, -ARENA_HEIGHT/2, 0.02)
        glEnd()
        
        # AI goal (orange)
        glColor4f(*NEON_ORANGE, 0.6)
        glBegin(GL_LINES)
        glVertex3f(-goal_half, ARENA_HEIGHT/2, 0.02)
        glVertex3f(goal_half, ARENA_HEIGHT/2, 0.02)
        glEnd()
        
        # Animated walls
        self._render_neon_walls()
        
        glEnable(GL_LIGHTING)
    
    def _render_neon_walls(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Wall segments with pulse
        pulse = 0.7 + 0.3 * math.sin(self.time * 3)
        
        # Side walls
        for x in [-ARENA_WIDTH/2, ARENA_WIDTH/2]:
            glPushMatrix()
            glTranslatef(x, 0, WALL_HEIGHT/2)
            glColor4f(*NEON_CYAN, 0.7 * pulse)
            self._draw_glowing_box(0.15, ARENA_HEIGHT/2, WALL_HEIGHT/2)
            glPopMatrix()
        
        # End walls with goal gaps
        goal_half = GOAL_WIDTH / 2
        segments = [
            (-ARENA_WIDTH/4 - goal_half/2, ARENA_WIDTH/4 - goal_half/2),
            (ARENA_WIDTH/4 + goal_half/2, ARENA_WIDTH/4 - goal_half/2)
        ]
        
        for y in [-ARENA_HEIGHT/2, ARENA_HEIGHT/2]:
            color = NEON_CYAN if y < 0 else NEON_ORANGE
            for x, width in segments:
                glPushMatrix()
                glTranslatef(x, y, WALL_HEIGHT/2)
                glColor4f(*color, 0.7 * pulse)
                self._draw_glowing_box(width, 0.15, WALL_HEIGHT/2)
                glPopMatrix()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def _draw_glowing_box(self, width, height, depth):
        glBegin(GL_QUADS)
        # All 6 faces
        for normal, vertices in [
            ((0,0,1), [(-width,-height,depth), (width,-height,depth), (width,height,depth), (-width,height,depth)]),
            ((0,0,-1), [(-width,-height,-depth), (-width,height,-depth), (width,height,-depth), (width,-height,-depth)]),
            ((0,1,0), [(-width,height,-depth), (-width,height,depth), (width,height,depth), (width,height,-depth)]),
            ((0,-1,0), [(-width,-height,-depth), (width,-height,-depth), (width,-height,depth), (-width,-height,depth)]),
            ((-1,0,0), [(-width,-height,-depth), (-width,-height,depth), (-width,height,depth), (-width,height,-depth)]),
            ((1,0,0), [(width,-height,-depth), (width,height,-depth), (width,height,depth), (width,-height,depth)])
        ]:
            glNormal3f(*normal)
            for v in vertices:
                glVertex3f(*v)
        glEnd()
    
    def _draw_glowing_circle(self, x, y, radius, color, segments):
        glColor4f(*color, 0.8)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments + self.time
            glVertex3f(x + math.cos(angle) * radius, y + math.sin(angle) * radius, 0.02)
        glEnd()
    
    def _render_puck(self, puck: Puck):
        self._render_lightning_trail(puck)
        self._render_puck_geometry(puck)
    
    def _render_lightning_trail(self, puck: Puck):
        if len(puck.trail) < 3:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glLineWidth(3)
        
        # Main trail
        glBegin(GL_LINE_STRIP)
        for i, pos in enumerate(puck.trail):
            alpha = i / len(puck.trail)
            glColor4f(*NEON_ORANGE, alpha * 0.8)
            glVertex3f(pos.x, pos.y, pos.z)
        glEnd()
        
        # Lightning effect
        if puck.velocity.magnitude() > 8:
            glColor4f(*NEON_YELLOW, 0.9)
            glLineWidth(2)
            
            for _ in range(3):
                glBegin(GL_LINE_STRIP)
                for i, pos in enumerate(puck.trail):
                    jitter = Vector3(
                        random.uniform(-0.15, 0.15),
                        random.uniform(-0.15, 0.15),
                        random.uniform(-0.05, 0.1)
                    )
                    p = pos + jitter
                    glVertex3f(p.x, p.y, p.z)
                glEnd()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
    
    def _render_puck_geometry(self, puck: Puck, alpha=1.0):
        glPushMatrix()
        glTranslatef(puck.position.x, puck.position.y, puck.position.z)
        
        # Glow effect
        if alpha >= 1.0:
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            
            glow_size = PUCK_RADIUS * (1.3 + 0.2 * math.sin(self.time * 8))
            glColor4f(*NEON_ORANGE, 0.4 * alpha)
            
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            for i in range(33):
                angle = 2 * math.pi * i / 32
                glVertex3f(math.cos(angle) * glow_size, math.sin(angle) * glow_size, 0)
            glEnd()
            
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LIGHTING)
        
        # Main puck body
        color = NEON_YELLOW if puck.is_powered else NEON_ORANGE
        glColor4f(*color, alpha)
        glRotatef(-90, 1, 0, 0)
        gluCylinder(self.quadric, PUCK_RADIUS, PUCK_RADIUS, 0.12, 32, 1)
        gluDisk(self.quadric, 0, PUCK_RADIUS, 32, 1)
        glTranslatef(0, 0, 0.12)
        gluDisk(self.quadric, 0, PUCK_RADIUS, 32, 1)
        
        glPopMatrix()
    
    def _render_paddle(self, paddle: Paddle):
        self._render_paddle_geometry(paddle)
    
    def _render_paddle_geometry(self, paddle: Paddle, alpha=1.0):
        glPushMatrix()
        glTranslatef(paddle.position.x, paddle.position.y, paddle.position.z)
        
        # Power charge glow
        if paddle.power_charge > 0 and alpha >= 1.0:
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            
            charge = paddle.power_charge
            glow_radius = PADDLE_RADIUS * (1.2 + charge * 0.5)
            glColor4f(*NEON_YELLOW, 0.5 * charge * alpha)
            
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            for i in range(33):
                angle = 2 * math.pi * i / 32
                glVertex3f(math.cos(angle) * glow_radius, math.sin(angle) * glow_radius, 0)
            glEnd()
            
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LIGHTING)
        
        # Main paddle
        color = NEON_YELLOW if paddle.is_boosted else paddle.color
        glColor4f(*color, alpha)
        glRotatef(-90, 1, 0, 0)
        
        # Base cylinder
        gluCylinder(self.quadric, PADDLE_RADIUS, PADDLE_RADIUS, 0.15, 32, 1)
        gluDisk(self.quadric, 0, PADDLE_RADIUS, 32, 1)
        
        # Handle
        glTranslatef(0, 0, 0.15)
        gluDisk(self.quadric, 0, PADDLE_RADIUS, 32, 1)
        gluCylinder(self.quadric, PADDLE_RADIUS * 0.5, PADDLE_RADIUS * 0.5, 0.2, 16, 1)
        glTranslatef(0, 0, 0.2)
        gluDisk(self.quadric, 0, PADDLE_RADIUS * 0.5, 16, 1)
        
        glPopMatrix()
    
    def render_hologram_scoreboard(self, player_score: int, ai_score: int):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Hologram position
        glPushMatrix()
        glTranslatef(0, 0, 2.5)
        glRotatef(self.time * 20, 0, 1, 0)
        
        # Base plate
        glColor4f(*NEON_CYAN, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(-2, -0.5, 0)
        glVertex3f(2, -0.5, 0)
        glVertex3f(2, 0.5, 0)
        glVertex3f(-2, 0.5, 0)
        glEnd()
        
        # Score text (rendered as 3D boxes)
        self._render_hologram_number(player_score, -0.8, 0, NEON_CYAN)
        self._render_hologram_number(ai_score, 0.8, 0, NEON_ORANGE)
        
        # Divider
        glColor4f(*NEON_WHITE, 0.8)
        glLineWidth(3)
        glBegin(GL_LINES)
        glVertex3f(0, -0.4, 0.01)
        glVertex3f(0, 0.4, 0.01)
        glEnd()
        
        # Scan lines
        for i in range(10):
            y = -0.5 + i * 0.1 + (self.time * 0.5) % 0.1
            glColor4f(*NEON_CYAN, 0.1)
            glBegin(GL_LINES)
            glVertex3f(-2, y, 0.02)
            glVertex3f(2, y, 0.02)
            glEnd()
        
        glPopMatrix()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def _render_hologram_number(self, number, x, y, color):
        segments = {
            0: [(-0.2,0.3), (0.2,0.3), (0.2,-0.3), (-0.2,-0.3), (-0.2,0.3)],
            1: [(0,0.3), (0,-0.3)],
            2: [(-0.2,0.3), (0.2,0.3), (0.2,0), (-0.2,0), (-0.2,-0.3), (0.2,-0.3)],
            3: [(-0.2,0.3), (0.2,0.3), (0.2,-0.3), (-0.2,-0.3), (0.2,-0.3), (0.2,0), (-0.2,0)],
            4: [(-0.2,0.3), (-0.2,0), (0.2,0), (0.2,0.3), (0.2,-0.3)],
            5: [(0.2,0.3), (-0.2,0.3), (-0.2,0), (0.2,0), (0.2,-0.3), (-0.2,-0.3)]
        }
        
        glColor4f(*color, 0.9)
        glLineWidth(3)
        
        if number in segments:
            glBegin(GL_LINE_STRIP)
            for px, py in segments[number]:
                glVertex3f(x + px, y + py, 0.05)
            glEnd()
    
    def render_background_city(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Deep space gradient
        glBegin(GL_QUADS)
        glColor3f(0.0, 0.0, 0.02)
        glVertex3f(-1, -1, 0.999)
        glVertex3f(1, -1, 0.999)
        glColor3f(0.02, 0.0, 0.05)
        glVertex3f(1, 1, 0.999)
        glVertex3f(-1, 1, 0.999)
        glEnd()
        
        # Tron city silhouette
        self._render_tron_city()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def _render_tron_city(self):
        # Animated buildings
        t = self.time
        
        for i in range(-15, 16):
            height = 0.15 + 0.25 * abs(math.sin(i * 0.5 + t * 0.3))
            width = 0.04 + 0.02 * math.sin(i * 0.8)
            x = i * 0.06
            
            # Building outline
            glColor4f(0.1, 0.1, 0.15, 0.9)
            glBegin(GL_LINE_LOOP)
            glVertex3f(x - width, -1, 0.998)
            glVertex3f(x + width, -1, 0.998)
            glVertex3f(x + width, -1 + height, 0.998)
            glVertex3f(x - width, -1 + height, 0.998)
            glEnd()
            
            # Glowing windows
            if i % 2 == 0:
                window_alpha = 0.5 + 0.3 * math.sin(t * 2 + i)
                glColor4f(*NEON_CYAN, window_alpha)
                glBegin(GL_QUADS)
                for j in range(3):
                    wy = -1 + 0.02 + j * 0.04
                    glVertex3f(x - width * 0.5, wy, 0.998)
                    glVertex3f(x + width * 0.5, wy, 0.998)
                    glVertex3f(x + width * 0.5, wy + 0.02, 0.998)
                    glVertex3f(x - width * 0.5, wy + 0.02, 0.998)
                glEnd()

# ============== MAIN GAME ==============
class NeonHandHockey3DUltimate:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Neon Hand Hockey 3D - Ultimate Edition")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        
        try:
            glutInit()
        except:
            pass
        
        # Initialize systems
        self.renderer = AdvancedRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.renderer.init_opengl()
        
        self.hand_tracker = HandTracker()
        self.particle_system = ParticleSystem3D()
        self.voice_controller = VoiceController()
        self.network_manager = NetworkManager()
        self.ai = AdvancedAI(difficulty=3)
        
        # Game objects
        self.puck = Puck(position=Vector3(0, 0, PUCK_RADIUS + 0.1), velocity=Vector3())
        self.puck.reset(True)
        
        self.player_paddle = Paddle(
            position=Vector3(0, ARENA_HEIGHT * 0.35, PADDLE_RADIUS),
            target_position=Vector3(0, ARENA_HEIGHT * 0.35, PADDLE_RADIUS),
            color=NEON_CYAN,
            is_player=True
        )
        
        self.ai_paddle = Paddle(
            position=Vector3(0, -ARENA_HEIGHT * 0.35, PADDLE_RADIUS),
            target_position=Vector3(0, -ARENA_HEIGHT * 0.35, PADDLE_RADIUS),
            color=NEON_ORANGE,
            is_player=False
        )
        
        # Game state
        self.player_score = 0
        self.ai_score = 0
        self.game_mode = "menu"  # menu, playing, multiplayer, paused
        self.slow_motion_factor = 1.0
        self.goal_effect_timer = 0.0
        self.power_shot_ready = False
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = 60.0
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                self._handle_key(event.key)
    
    def _handle_key(self, key):
        if key == K_ESCAPE:
            if self.game_mode in ["playing", "multiplayer"]:
                self.game_mode = "menu"
            else:
                self.running = False
        elif key == K_1:
            self.ai = AdvancedAI(difficulty=1)
        elif key == K_2:
            self.ai = AdvancedAI(difficulty=2)
        elif key == K_3:
            self.ai = AdvancedAI(difficulty=3)
        elif key == K_4:
            self.ai = AdvancedAI(difficulty=4)
        elif key == K_5:
            self.ai = AdvancedAI(difficulty=5)
        elif key == K_SPACE:
            if self.game_mode == "menu":
                self.game_mode = "playing"
                self._reset_game()
        elif key == K_m:
            # Toggle multiplayer
            if self.game_mode == "menu":
                self.game_mode = "multiplayer_setup"
        elif key == K_r:
            self._reset_game()
    
    def update(self, dt):
        # Update webcam
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.hand_tracker.update(frame, dt)
        
        # Voice commands
        if self.voice_controller.enabled:
            cmd = self.voice_controller.get_command()
            if cmd == "boost":
                self.player_paddle.is_boosted = True
            elif cmd == "slow":
                self.slow_motion_factor = 0.3
        
        if self.game_mode != "playing":
            return
        
        # Slow motion recovery
        self.slow_motion_factor = min(1.0, self.slow_motion_factor + dt * 0.5)
        
        # Update hand control
        if self.hand_tracker.detected:
            arena_pos = self.hand_tracker.get_arena_position()
            self.player_paddle.target_position.x = arena_pos[0]
            self.player_paddle.target_position.y = arena_pos[1]
            self.player_paddle.is_boosted = self.hand_tracker.gesture == "fist"
            self.player_paddle.power_charge = self.hand_tracker.power_charged
            
            # Power shot release
            if self.hand_tracker.gesture != "fist" and self.player_paddle.power_charge > 0.8:
                self.power_shot_ready = True
        
        # Update paddles
        self.player_paddle.update(dt, self.slow_motion_factor)
        self.ai_paddle.update(dt, self.slow_motion_factor)
        
        # AI prediction and update
        self.ai.update(self.puck, self.ai_paddle, dt)
        
        # Update puck
        self.puck.update(dt, self.slow_motion_factor)
        
        # Physics collisions
        self._handle_collisions()
        
        # Goal detection
        goal = self._check_goal()
        if goal:
            self._handle_goal(goal)
        
        # Update particles
        self.particle_system.update(dt)
        
        # Goal effect timer
        if self.goal_effect_timer > 0:
            self.goal_effect_timer -= dt
            self.slow_motion_factor = 0.2
        
        # Update renderer
        self.renderer.update(dt)
    
    def _handle_collisions(self):
        # Puck-Paddle collision
        for paddle in [self.player_paddle, self.ai_paddle]:
            dx = self.puck.position.x - paddle.position.x
            dy = self.puck.position.y - paddle.position.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < PUCK_RADIUS + PADDLE_RADIUS:
                # Collision response
                nx = dx / dist if dist > 0 else 0
                ny = dy / dist if dist > 0 else 1
                
                # Separate
                overlap = PUCK_RADIUS + PADDLE_RADIUS - dist
                self.puck.position.x += nx * overlap
                self.puck.position.y += ny * overlap
                
                # Velocity transfer
                rel_vel = self.puck.velocity - paddle.velocity
                vel_normal = rel_vel.x * nx + rel_vel.y * ny
                
                if vel_normal < 0:
                    restitution = RESTITUTION
                    
                    # Power shot
                    if paddle == self.player_paddle and self.player_paddle.power_charge > 0.8:
                        restitution *= POWER_SHOT_MULTIPLIER
                        self.puck.is_powered = True
                        self.particle_system.emit_explosion(self.puck.position, NEON_YELLOW, 40)
                        self.player_paddle.power_charge = 0
                        self.renderer.add_shake(0.8)
                    
                    impulse = -(1 + restitution) * vel_normal
                    self.puck.velocity.x += impulse * nx
                    self.puck.velocity.y += impulse * ny
                    
                    # Add paddle velocity
                    self.puck.velocity.x += paddle.velocity.x * 0.5
                    self.puck.velocity.y += paddle.velocity.y * 0.5
                    
                    # Particles
                    color = NEON_CYAN if paddle.is_player else NEON_ORANGE
                    self.particle_system.emit_collision_sparks(
                        self.puck.position,
                        Vector3(nx, ny, 0.2),
                        color
                    )
        
        # Wall collisions
        half_width = ARENA_WIDTH / 2
        half_height = ARENA_HEIGHT / 2
        
        if abs(self.puck.position.x) > half_width - PUCK_RADIUS:
            self.puck.position.x = math.copysign(half_width - PUCK_RADIUS, self.puck.position.x)
            self.puck.velocity.x = -self.puck.velocity.x * RESTITUTION
            self.particle_system.emit_collision_sparks(
                self.puck.position,
                Vector3(-1 if self.puck.position.x > 0 else 1, 0, 0.3),
                NEON_CYAN
            )
        
        goal_half = GOAL_WIDTH / 2
        
        if self.puck.position.y > half_height - PUCK_RADIUS:
            if abs(self.puck.position.x) > goal_half:
                self.puck.position.y = half_height - PUCK_RADIUS
                self.puck.velocity.y = -self.puck.velocity.y * RESTITUTION
                self.particle_system.emit_collision_sparks(
                    self.puck.position,
                    Vector3(0, -1, 0.3),
                    NEON_ORANGE
                )
        
        if self.puck.position.y < -half_height + PUCK_RADIUS:
            if abs(self.puck.position.x) > goal_half:
                self.puck.position.y = -half_height + PUCK_RADIUS
                self.puck.velocity.y = -self.puck.velocity.y * RESTITUTION
                self.particle_system.emit_collision_sparks(
                    self.puck.position,
                    Vector3(0, 1, 0.3),
                    NEON_CYAN
                )
    
    def _check_goal(self):
        half_height = ARENA_HEIGHT / 2
        goal_half = GOAL_WIDTH / 2
        
        if self.puck.position.y > half_height and abs(self.puck.position.x) < goal_half:
            return "player"
        elif self.puck.position.y < -half_height and abs(self.puck.position.x) < goal_half:
            return "ai"
        return None
    
    def _handle_goal(self, scorer):
        if scorer == "player":
            self.player_score += 1
            self.particle_system.emit_explosion(Vector3(0, ARENA_HEIGHT/2, 0.5), NEON_CYAN, 100)
        else:
            self.ai_score += 1
            self.particle_system.emit_explosion(Vector3(0, -ARENA_HEIGHT/2, 0.5), NEON_ORANGE, 100)
        
        self.goal_effect_timer = 1.5
        self.slow_motion_factor = 0.1
        self.renderer.add_shake(1.2)
        self.puck.is_powered = False
        
        # Reset after delay
        threading.Timer(2.0, lambda: self.puck.reset(scorer == "ai")).start()
    
    def _reset_game(self):
        self.player_score = 0
        self.ai_score = 0
        self.puck.reset(True)
        self.player_paddle.position = Vector3(0, ARENA_HEIGHT * 0.35, PADDLE_RADIUS)
        self.player_paddle.target_position = Vector3(0, ARENA_HEIGHT * 0.35, PADDLE_RADIUS)
        self.ai_paddle.position = Vector3(0, -ARENA_HEIGHT * 0.35, PADDLE_RADIUS)
        self.ai_paddle.target_position = Vector3(0, -ARENA_HEIGHT * 0.35, PADDLE_RADIUS)
        self.particle_system.particles.clear()
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Background
        self.renderer.render_background_city()
        
        # Camera
        self.renderer.setup_camera(None)
        
        # Arena with reflections
        self.renderer.render_arena_with_reflections(self.puck, self.player_paddle, self.ai_paddle)
        
        # AI prediction visualization
        self.ai.render_prediction()
        
        # Particles
        self.particle_system.render()
        
        # Hologram scoreboard
        self.renderer.render_hologram_scoreboard(self.player_score, self.ai_score)
        
        # HUD
        self._render_hud()
        
        pygame.display.flip()
    
    def _render_hud(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # FPS
        glColor3f(*NEON_GREEN)
        glRasterPos2f(10, 20)
        fps_text = f"FPS: {int(self.fps)}"
        try:
            for c in fps_text:
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(c))
        except:
            pass
        
        # Hand tracking status
        status = "HAND TRACKING ACTIVE" if self.hand_tracker.detected else "NO HAND DETECTED"
        glColor3f(*NEON_CYAN if self.hand_tracker.detected else NEON_RED)
        glRasterPos2f(WINDOW_WIDTH - 250, 20)
        try:
            for c in status:
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(c))
        except:
            pass
        
        # Power charge indicator
        if self.player_paddle.power_charge > 0:
            charge_text = f"POWER: {int(self.player_paddle.power_charge * 100)}%"
            glColor3f(*NEON_YELLOW)
            glRasterPos2f(WINDOW_WIDTH - 250, 45)
            try:
                for c in charge_text:
                    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(c))
            except:
                pass
        
        # Voice command indicator
        if self.voice_controller.enabled:
            glColor3f(*NEON_PURPLE)
            glRasterPos2f(10, 45)
            voice_text = "VOICE: ENABLED (Say 'BOOST')"
            try:
                for c in voice_text:
                    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(c))
            except:
                pass
        
        # Slow motion indicator
        if self.slow_motion_factor < 0.8:
            glColor3f(*NEON_ORANGE)
            glRasterPos2f(WINDOW_WIDTH // 2 - 50, 30)
            slow_text = "SLOW MOTION"
            try:
                for c in slow_text:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(c))
            except:
                pass
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def run(self):
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = min(current_time - last_time, 0.05)
            last_time = current_time
            
            self.fps = 1.0 / dt if dt > 0 else 60
            
            self.handle_events()
            self.update(dt)
            self.render()
            
            self.clock.tick(FPS_TARGET)
        
        self.cleanup()
    
    def cleanup(self):
        self.hand_tracker.release()
        self.voice_controller.release()
        self.network_manager.close()
        self.cap.release()
        pygame.quit()

# ============== ENTRY POINT ==============
def main():
    print("=" * 60)
    print("   NEON HAND HOCKEY 3D - ULTIMATE EDITION")
    print("   Tron Legacy Style with AI Hand Tracking")
    print("=" * 60)
    print("\n🎮 CONTROLS:")
    print("   • Hand Movement: Control paddle position")
    print("   • Open Hand: Normal mode")
    print("   • Fist (hold 1s): Charge POWER SHOT")
    print("   • Release Fist: Execute power shot")
    print("\n🗣️ VOICE COMMANDS (if enabled):")
    print("   • Say 'BOOST' for speed boost")
    print("   • Say 'SLOW' for slow motion")
    print("\n⌨️ KEYBOARD:")
    print("   • SPACE: Start game")
    print("   • 1-5: AI Difficulty")
    print("   • R: Reset game")
    print("   • ESC: Exit")
    print("\n" + "=" * 60)
    
    try:
        game = NeonHandHockey3DUltimate()
        game.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()