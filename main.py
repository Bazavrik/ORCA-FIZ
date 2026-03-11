import math
import time

import numpy as np
import rvo2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from fuzzy_sd import FIS


# =========================
# Настройки "карты"
# =========================
WORLD_X = (0.0, 30.0)
WORLD_Y = (0.0, 30.0)

DT = 0.1
STEPS_PER_FRAME = 1  # можно 2-4 для ускорения

# =========================
# ORCA / RVO2 параметры
# =========================
NEIGHBOR_DIST = 7.5
MAX_NEIGHBORS = 12
TIME_HORIZON = 2.5
TIME_HORIZON_OBST = 2.5

ROBOT_RADIUS = 1.0
ROBOT_MAX_SPEED = 1.6

OBS_RADIUS_BASE = 0.35
OBS_RADIUS = OBS_RADIUS_BASE * 3.0  # люди в 3 раза больше
HUMAN_MAX_SPEED = 1.1
ANIMAL_MAX_SPEED = 1.1
CYCLIST_MAX_SPEED = HUMAN_MAX_SPEED * 2.0

# След робота
TRAIL_MAX_POINTS = 1500

# Порог "достиг контрольной точки"
WAYPOINT_EPS = 0.4

# Классы для FIS (Input4 C: 0..4, пики MF около 1/2/3/4)
C_BICYCLE = 1.0
C_HUMAN = 2.0
C_ANIMAL = 3.0
C_UNDEFINED = 4.0


# =========================
# Утилиты
# =========================
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
    """Прямоугольник как полигон (CCW) для addObstacle."""
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def move_out_of_rects(p, rects, margin):
    """
    Если точка внутри прямоугольника, выталкиваем её к ближайшей стороне + margin.
    """
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


def safe_norm(v):
    return math.hypot(v[0], v[1])


def angle_abs_deg(v, w):
    """
    |угол| между векторами v и w в градусах [0..180]
    Если один из векторов ~0 => 0
    """
    nv = safe_norm(v)
    nw = safe_norm(w)
    if nv < 1e-9 or nw < 1e-9:
        return 0.0
    dot = v[0] * w[0] + v[1] * w[1]
    c = max(-1.0, min(1.0, dot / (nv * nw)))
    return abs(math.degrees(math.acos(c)))


# =========================
# Загружаем FIS (Python)
# =========================
fis = FIS.from_fis("Fuzzy_SD.fis", grid_n=301)  # 301 быстрее, чем 501


# =========================
# Создаём симулятор
# =========================
sim = rvo2.PyRVOSimulator(
    DT,
    NEIGHBOR_DIST,
    MAX_NEIGHBORS,
    TIME_HORIZON,
    TIME_HORIZON_OBST,
    ROBOT_RADIUS,
    ROBOT_MAX_SPEED,
)

# =========================
# Статичные препятствия (прямоугольники)
# =========================
static_rects = [
    (6.0,  6.0,  3.0, 10.0),
    (18.0, 9.0,  8.0,  3.0),
    (5.0, 20.0,  6.0,  2.2),
]

# Добавляем в ORCA как препятствия
for (x, y, w, h) in static_rects:
    sim.addObstacle(rect_poly(x, y, w, h))
sim.processObstacles()

# =========================
# Робот и траектория по контрольным точкам
# =========================
robot_start = move_out_of_rects((3.0, 3.0), static_rects, margin=ROBOT_RADIUS + 0.2)
robot_id = sim.addAgent(robot_start)

waypoints = [(3.0, 3.0), (15.0, 2.0), (18.0, 28.0), (29.0, 29.0)]
wp_index = 0

# =========================
# Подвижные агенты: 5 людей + животное + велосипедист
# =========================
agents_meta = []  # {id, name, vmax, goal, alt_goal, color, class_id}

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

# 5 людей
human_inits = [(16.0, 24.0), (24.0, 18.0), (8.0, 14.0), (12.0, 26.0), (26.0, 6.0)]
human_g1   = [(16.0,  8.0), ( 8.0, 18.0), (26.0, 14.0), (12.0, 10.0), (10.0, 6.0)]
human_g2   = [(16.0, 24.0), (26.0, 18.0), ( 8.0, 24.0), (20.0, 26.0), (26.0, 20.0)]

animals_inits = [(28.0, 2.0), (6.0, 10.0), (15.0, 15.0)]
animals_g1   = [(2.0,  28.0), (22.0, 4.0), (15.0, 2.0)]
animals_g2   = [(1.0, 1.0), (6.0, 22.0), (15.0, 28.0)]

for i in range(5):
    add_agent(
        name=f"Человек {i+1}",
        init_pos=human_inits[i],
        goal_a=human_g1[i],
        goal_b=human_g2[i],
        vmax=HUMAN_MAX_SPEED,
        color=None,
        class_id=C_HUMAN,
    )

for i in range(len(animals_inits)):
    add_agent(
        name="Животное",
        init_pos=animals_inits[i],
        goal_a=animals_g1[i],
        goal_b=animals_g2[i],
        vmax=ANIMAL_MAX_SPEED,
        color="red",
        class_id=C_ANIMAL,
)

# Велосипедист (жёлтый, vmax x2)
add_agent(
    name="Велосипедист",
    init_pos=(28.0, 14.0),
    goal_a=(4.0, 28.0),
    goal_b=(28.0, 2.0),
    vmax=CYCLIST_MAX_SPEED,
    color="yellow",
    class_id=C_BICYCLE,
)

# =========================
# Визуализация
# =========================
fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(WORLD_X)
ax.set_ylim(WORLD_Y)
ax.set_title("ORCA (rvo2) + Fuzzy_SD (Python): оценка 0..1 над препятствиями")

# чтобы легенда не перекрывала карту
fig.subplots_adjust(right=0.78)

# границы карты
ax.plot(
    [WORLD_X[0], WORLD_X[1], WORLD_X[1], WORLD_X[0], WORLD_X[0]],
    [WORLD_Y[0], WORLD_Y[0], WORLD_Y[1], WORLD_Y[1], WORLD_Y[0]],
    linewidth=1.5
)

# статичные прямоугольники на карте
for (x, y, w, h) in static_rects:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="black", edgecolor="black", alpha=0.9))

# контрольные точки
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

# след робота
trail_x, trail_y = [], []
robot_trail_line, = ax.plot([], [], linewidth=1.0, color="yellow", alpha=0.9)

# FPS
fps_text = ax.text(0.02, 0.98, "FPS: --", transform=ax.transAxes, va="top", ha="left")

# robot circle
robot_circle = plt.Circle(sim.getAgentPosition(robot_id), ROBOT_RADIUS, fill=True, color="green")
ax.add_patch(robot_circle)
robot_circle.set_label("Робот")

# препятствия (круги) + подписи fuzzy
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

    # текст над агентом (score)
    p = sim.getAgentPosition(meta["id"])
    t = ax.text(p[0], p[1] + meta["radius"] + 0.4, "0.00", ha="center", va="bottom", fontsize=8)
    score_texts.append(t)

    # крестик цели
    gp, = ax.plot([meta["goal"][0]], [meta["goal"][1]], marker="x", markersize=7, linewidth=0)
    agent_goal_plots.append(gp)

# вектор скорости робота и агентов (опционально)
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

# FPS smoothing
last_t = time.perf_counter()
fps_ema = None
FPS_ALPHA = 0.15


# =========================
# Шаг симуляции
# =========================
def step_sim():
    global wp_index

    # 1) Люди/животное/велосипедист: цели туда-сюда
    for i, meta in enumerate(agents_meta):
        pos = sim.getAgentPosition(meta["id"])
        meta["goal"] = bounce_goal_if_reached(pos, meta["goal"], meta["alt_goal"])
        agent_goal_plots[i].set_data([meta["goal"][0]], [meta["goal"][1]])

    # 2) Робот: контрольные точки
    rpos = sim.getAgentPosition(robot_id)
    if math.hypot(rpos[0] - waypoints[wp_index][0], rpos[1] - waypoints[wp_index][1]) < WAYPOINT_EPS:
        wp_index = min(wp_index + 1, len(waypoints) - 1)

    r_goal = waypoints[wp_index]
    sim.setAgentPrefVelocity(robot_id, pref_velocity(rpos, r_goal, ROBOT_MAX_SPEED))

    # 3) Агенты: предпочтительные скорости к своим целям
    for meta in agents_meta:
        pos = sim.getAgentPosition(meta["id"])
        sim.setAgentPrefVelocity(meta["id"], pref_velocity(pos, meta["goal"], meta["vmax"]))

    # 4) ORCA шаг
    sim.doStep()

    # 5) Удерживаем внутри границ (на всякий)
    for aid in [robot_id] + [m["id"] for m in agents_meta]:
        p = sim.getAgentPosition(aid)
        cp = clamp_point(p)
        if cp != p:
            sim.setAgentPosition(aid, cp)


def update(_frame):
    global last_t, fps_ema

    for _ in range(STEPS_PER_FRAME):
        step_sim()

    # --- robot visuals
    rpos = sim.getAgentPosition(robot_id)
    robot_circle.center = rpos

    trail_x.append(rpos[0])
    trail_y.append(rpos[1])
    if len(trail_x) > TRAIL_MAX_POINTS:
        trail_x.pop(0)
        trail_y.pop(0)
    robot_trail_line.set_data(trail_x, trail_y)

    rvel = sim.getAgentVelocity(robot_id)
    robot_vel_line.set_data([rpos[0], rpos[0] + rvel[0]],
                            [rpos[1], rpos[1] + rvel[1]])

    # --- agents + fuzzy score
    scores = []
    for meta in agents_meta:
        pid = meta["id"]
        pos = sim.getAgentPosition(pid)
        vel = sim.getAgentVelocity(pid)

        # Входы FIS:
        # V: относительная скорость (0..30)
        v_rel = safe_norm((vel[0] - rvel[0], vel[1] - rvel[1]))

        # d: расстояние между границами (0..20)
        center_dist = safe_norm((pos[0] - rpos[0], pos[1] - rpos[1]))
        d_gap = max(0.0, center_dist - (meta["radius"] + ROBOT_RADIUS))

        # fi: |угол| между скоростью робота и направлением на препятствие (в градусах 0..180)
        to_obs = (pos[0] - rpos[0], pos[1] - rpos[1])
        fi_deg = angle_abs_deg(rvel, to_obs)

        # C: класс препятствия
        C = meta["class_id"]

        # Вычисляем score 0..1
        score = fis.eval([v_rel, d_gap, fi_deg, C])
        scores.append(score)

    # --- update agent visuals
    for i, meta in enumerate(agents_meta):
        pid = meta["id"]
        pos = sim.getAgentPosition(pid)

        agent_circles[i].center = pos

        # текст score
        score_texts[i].set_position((pos[0], pos[1] + meta["radius"] + 0.4))
        score_texts[i].set_text(f"{scores[i]:.2f}")

        # velocity line
        vel = sim.getAgentVelocity(pid)
        agent_vel_lines[i].set_data([pos[0], pos[0] + vel[0]],
                                    [pos[1], pos[1] + vel[1]])

    # --- FPS
    now = time.perf_counter()
    dt = now - last_t
    last_t = now
    inst_fps = (1.0 / dt) if dt > 1e-9 else 0.0
    fps_ema = inst_fps if fps_ema is None else (FPS_ALPHA * inst_fps + (1.0 - FPS_ALPHA) * fps_ema)
    fps_text.set_text(f"FPS: {fps_ema:5.1f}")

    return [
        robot_circle, robot_trail_line, robot_vel_line, wp_plot, fps_text,
        *agent_circles, *agent_goal_plots, *agent_vel_lines, *score_texts
    ]


ani = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()