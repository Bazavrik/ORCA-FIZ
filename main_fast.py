import math
import time

import numpy as np
import rvo2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import fuzzy_sd_cpp
fis_cpp = fuzzy_sd_cpp.FuzzySD(301)

Y = 500.0
WORLD_X = (0.0, 100.0)
WORLD_Y = (0.0, Y)

DT = 0.1
STEPS_PER_FRAME = 1

NEIGHBOR_DIST = Y
MAX_NEIGHBORS = 100
TIME_HORIZON = 3
TIME_HORIZON_OBST = 3

ROBOT_RADIUS = 1.0
ROBOT_MAX_SPEED = 1.6

OBS_RADIUS_BASE = 0.35
OBS_RADIUS = OBS_RADIUS_BASE * 3.0
HUMAN_MAX_SPEED = 2.1
ANIMAL_MAX_SPEED = 4.1
CYCLIST_MAX_SPEED = HUMAN_MAX_SPEED * 2.0

TRAIL_MAX_POINTS = 1500
WAYPOINT_EPS = 0.4

C_BICYCLE = 1.0
C_HUMAN = 2.0
C_ANIMAL = 3.0
C_UNDEFINED = 4.0

TEXT_UPDATE_EVERY = 2


def clamp_point(p):
    x = min(max(p[0], WORLD_X[0]), WORLD_X[1])
    y = min(max(p[1], WORLD_Y[0]), WORLD_Y[1])
    return (x, y)


def pref_velocity(pos, goal, vmax):
    dx, dy = (goal[0] - pos[0], goal[1] - pos[1])
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return (0.0, 0.0)
    s = min(vmax, d) / d
    return (dx * s, dy * s)


def bounce_goal_if_reached(pos, goal, alt_goal, eps=0.8):
    if math.hypot(pos[0] - goal[0], pos[1] - goal[1]) < eps:
        return alt_goal
    return goal


def rect_poly(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def move_out_of_rects(p, rects, margin):
    x, y = p
    changed = True
    it = 0
    while changed and it < 25:
        changed = False
        it += 1
        for (rx, ry, rw, rh) in rects:
            inside = (rx <= x <= rx + rw) and (ry <= y <= ry + rh)
            if not inside:
                continue
            dl = x - rx
            dr = (rx + rw) - x
            db = y - ry
            dt = (ry + rh) - y
            m = min(dl, dr, db, dt)
            if m == dl:
                x = rx - margin
            elif m == dr:
                x = rx + rw + margin
            elif m == db:
                y = ry - margin
            else:
                y = ry + rh + margin
            x, y = clamp_point((x, y))
            changed = True
    return (x, y)


sim = rvo2.PyRVOSimulator(
    DT,
    NEIGHBOR_DIST,
    MAX_NEIGHBORS,
    TIME_HORIZON,
    TIME_HORIZON_OBST,
    ROBOT_RADIUS,
    ROBOT_MAX_SPEED,
)

static_rects = [
    (1.0, 1.0, 1.0, 1.0)
]
for (x, y, w, h) in static_rects:
    sim.addObstacle(rect_poly(x, y, w, h))
sim.processObstacles()

robot_start = move_out_of_rects((50.0, 3.0), static_rects, margin=ROBOT_RADIUS + 0.2)
robot_id = sim.addAgent(robot_start)
waypoints = [(50.0, 3.0), (50.0, 97.0)]
wp_index = 0

agents_meta = []

def add_agent(name, init_pos, goal_a, goal_b, vmax, color, class_id, radius=OBS_RADIUS):
    safe_init = move_out_of_rects(init_pos, static_rects, margin=radius + 0.2)
    aid = sim.addAgent(safe_init)
    sim.setAgentRadius(aid, radius)
    sim.setAgentMaxSpeed(aid, vmax)
    agents_meta.append({
        "id": aid,
        "name": name,
        "vmax": vmax,
        "goal": goal_a,
        "alt_goal": goal_b,
        "color": color,
        "class_id": float(class_id),
        "radius": float(radius),
    })

human_inits = [(10.0, i * 5.0) for i in range(int(Y/5))]
human_g1 = [(90.0, i * 5.0) for i in range(int(Y/5))]
human_g2 = [(10.0, i * 5.0) for i in range(int(Y/5))]
for i in range(len(human_inits)):
    add_agent(
        name=f"Человек {i+1}",
        init_pos=human_inits[i],
        goal_a=human_g1[i],
        goal_b=human_g2[i],
        vmax=HUMAN_MAX_SPEED,
        color=None,
        class_id=C_HUMAN,
    )

agent_ids = [m["id"] for m in agents_meta]
agent_radii = np.array([m["radius"] for m in agents_meta], dtype=np.float64)
agent_classes = np.array([m["class_id"] for m in agents_meta], dtype=np.float64)

fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(WORLD_X)
ax.set_ylim(WORLD_Y)
ax.set_title("ORCA (rvo2) + Fuzzy_SD (C++ batch)")
fig.subplots_adjust(right=0.78)

ax.plot(
    [WORLD_X[0], WORLD_X[1], WORLD_X[1], WORLD_X[0], WORLD_X[0]],
    [WORLD_Y[0], WORLD_Y[0], WORLD_Y[1], WORLD_Y[1], WORLD_Y[0]],
    linewidth=1.5
)

for (x, y, w, h) in static_rects:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="black", edgecolor="black", alpha=0.9))

wp_x = [p[0] for p in waypoints]
wp_y = [p[1] for p in waypoints]
wp_plot, = ax.plot(
    wp_x, wp_y,
    linestyle="None",
    marker="o",
    markersize=5,
    markerfacecolor="none",
    markeredgewidth=1.2
)
ax.plot(wp_x, wp_y, linewidth=1.0, alpha=0.5)

trail_x, trail_y = [], []
robot_trail_line, = ax.plot([], [], linewidth=1.0, color="yellow", alpha=0.9)
fps_text = ax.text(0.02, 0.98, "FPS: --", transform=ax.transAxes, va="top", ha="left")
robot_circle = plt.Circle(sim.getAgentPosition(robot_id), ROBOT_RADIUS, fill=True, color="green")
ax.add_patch(robot_circle)
robot_circle.set_label("Робот")

agent_circles = []
score_texts = []
agent_goal_plots = []

for meta in agents_meta:
    c = plt.Circle(sim.getAgentPosition(meta["id"]), meta["radius"], fill=True)
    if meta["color"] is not None:
        c.set_facecolor(meta["color"])
        c.set_edgecolor(meta["color"])
    ax.add_patch(c)
    c.set_label(meta["name"])
    agent_circles.append(c)

    p = sim.getAgentPosition(meta["id"])
    t = ax.text(p[0], p[1] + meta["radius"] + 0.4, "0.00", ha="center", va="bottom", fontsize=8)
    score_texts.append(t)

    gp, = ax.plot([meta["goal"][0]], [meta["goal"][1]], marker="x", markersize=7, linewidth=0)
    agent_goal_plots.append(gp)

robot_vel_line, = ax.plot([], [], linewidth=2)
agent_vel_lines = [ax.plot([], [], linewidth=1.5)[0] for _ in agents_meta]

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    fontsize=8,
    framealpha=0.9,
    borderaxespad=0.6,
    handlelength=1.2,
    labelspacing=0.4,
)

last_t = time.perf_counter()
fps_ema = None
FPS_ALPHA = 0.15
frame_idx = 0
last_scores = np.full(len(agents_meta), 0.5, dtype=np.float64)


def step_sim():
    global wp_index

    for i, meta in enumerate(agents_meta):
        pos = sim.getAgentPosition(meta["id"])
        meta["goal"] = bounce_goal_if_reached(pos, meta["goal"], meta["alt_goal"])
        agent_goal_plots[i].set_data([meta["goal"][0]], [meta["goal"][1]])

    rpos = sim.getAgentPosition(robot_id)
    if math.hypot(rpos[0] - waypoints[wp_index][0], rpos[1] - waypoints[wp_index][1]) < WAYPOINT_EPS:
        wp_index = min(wp_index + 1, len(waypoints) - 1)

    r_goal = waypoints[wp_index]
    sim.setAgentPrefVelocity(robot_id, pref_velocity(rpos, r_goal, ROBOT_MAX_SPEED))

    for meta in agents_meta:
        pos = sim.getAgentPosition(meta["id"])
        sim.setAgentPrefVelocity(meta["id"], pref_velocity(pos, meta["goal"], meta["vmax"]))

    sim.doStep()

    for aid in [robot_id] + agent_ids:
        p = sim.getAgentPosition(aid)
        cp = clamp_point(p)
        if cp != p:
            sim.setAgentPosition(aid, cp)


def update(_frame):
    global last_t, fps_ema, frame_idx, last_scores

    for _ in range(STEPS_PER_FRAME):
        step_sim()

    rpos = np.array(sim.getAgentPosition(robot_id), dtype=np.float64)
    rvel = np.array(sim.getAgentVelocity(robot_id), dtype=np.float64)

    robot_circle.center = tuple(rpos)
    trail_x.append(rpos[0])
    trail_y.append(rpos[1])
    if len(trail_x) > TRAIL_MAX_POINTS:
        trail_x.pop(0)
        trail_y.pop(0)
    robot_trail_line.set_data(trail_x, trail_y)
    robot_vel_line.set_data([rpos[0], rpos[0] + rvel[0]], [rpos[1], rpos[1] + rvel[1]])

    positions = np.array([sim.getAgentPosition(pid) for pid in agent_ids], dtype=np.float64)
    velocities = np.array([sim.getAgentVelocity(pid) for pid in agent_ids], dtype=np.float64)

    rel_vel = velocities - rvel
    v_rel = np.linalg.norm(rel_vel, axis=1)

    to_obs = positions - rpos
    center_dist = np.linalg.norm(to_obs, axis=1)
    d_gap = np.maximum(0.0, center_dist - (agent_radii + ROBOT_RADIUS))

    rvel_norm = np.linalg.norm(rvel)
    if rvel_norm < 1e-9:
        fi_deg = np.zeros(len(agent_ids), dtype=np.float64)
    else:
        to_obs_norm = np.linalg.norm(to_obs, axis=1)
        dot = to_obs @ rvel
        den = np.maximum(rvel_norm * to_obs_norm, 1e-9)
        cosang = np.clip(dot / den, -1.0, 1.0)
        fi_deg = np.degrees(np.arccos(cosang))

    X = np.column_stack((v_rel, d_gap, fi_deg, agent_classes)).astype(np.float64, copy=False)
    last_scores = fis_cpp.eval_batch(X)

    for i, meta in enumerate(agents_meta):
        pos = positions[i]
        vel = velocities[i]
        agent_circles[i].center = tuple(pos)
        agent_vel_lines[i].set_data([pos[0], pos[0] + vel[0]], [pos[1], pos[1] + vel[1]])

    if frame_idx % TEXT_UPDATE_EVERY == 0:
        for i, meta in enumerate(agents_meta):
            pos = positions[i]
            score_texts[i].set_position((pos[0], pos[1] + meta["radius"] + 0.4))
            score_texts[i].set_text(f"{float(last_scores[i]):.2f}")

    now = time.perf_counter()
    dt = now - last_t
    last_t = now
    inst_fps = (1.0 / dt) if dt > 1e-9 else 0.0
    fps_ema = inst_fps if fps_ema is None else (FPS_ALPHA * inst_fps + (1.0 - FPS_ALPHA) * fps_ema)
    fps_text.set_text(f"FPS: {fps_ema:5.1f}")
    frame_idx += 1

    return [
        robot_circle, robot_trail_line, robot_vel_line, wp_plot, fps_text,
        *agent_circles, *agent_goal_plots, *agent_vel_lines, *score_texts
    ]


ani = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()
