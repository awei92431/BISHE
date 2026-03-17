import argparse
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
URDF_PATH = ROOT / "RM65-B" / "urdf" / "RM65-B.urdf"
ROBOT_MESH_DIR = ROOT / "RM65-B" / "meshes"
ENDOSCOPE_MESH_PATH = ROOT / "en.STL"
CAMERA_MESH_PATH = ROOT / "camera.STL"
PACKAGE_PREFIX = "package://RM65-B/meshes/"
CAPTURE_DIR = ROOT / "captures"

BENT_QPOS = (0.0, -1.1, 1.8, 0.0, 0.9, 0.0)
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
JOINT_LIMITS_DEG = (
    (-178.0, 178.0),
    (-130.0, 130.0),
    (-135.0, 135.0),
    (-178.0, 178.0),
    (-128.0, 128.0),
    (-360.0, 360.0),
)

TOOL_BODY_NAME = "endoscope_tool"
TARGET_BODY_NAME = "inspection_target"
CAMERA_NAME = "endoscope_camera"

CAMERA_RESOLUTION = (640, 480)  # width, height
OVERLAY_RESOLUTION = (320, 240)  # width, height
TOOL_MESH_SCALE = (0.001, 0.001, 0.001)
ADAPTER_POS = np.array([0.0, 0.0, 0.012], dtype=float)
ADAPTER_SIZE = (0.018, 0.012, 0.0)
ENDOSCOPE_MESH_POS = np.array([-0.017076289, -0.019918535, 0.024], dtype=float)
CAMERA_MESH_POS = np.array([-0.018953737, -0.018953737, 0.024], dtype=float)
TARGET_BOARD_OFFSET_TO_SPHERE = np.array([0.05, 0.0, 0.0], dtype=float)

BASE_CAMERA_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


@dataclass
class EndoscopeCameraConfig:
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0
    camera_z_m: float = 0.50
    fovy_deg: float = 55.0
    sensor_width_m: float = 0.0064
    sensor_height_m: float = 0.0048
    yaw_deg: float = 0.0
    pitch_deg: float = -20.0
    roll_deg: float = 0.0
    target_x_m: float = -0.61
    target_y_m: float = 0.0
    target_z_m: float = 0.497


DEFAULT_CONFIG = EndoscopeCameraConfig()


def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ],
            dtype=float,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )
    quat /= np.linalg.norm(quat)
    return quat


def camera_quaternion(config: EndoscopeCameraConfig) -> np.ndarray:
    yaw_matrix = rotation_y(math.radians(config.yaw_deg))
    pitch_matrix = rotation_x(math.radians(config.pitch_deg))
    roll_matrix = rotation_z(math.radians(config.roll_deg))
    rotation = BASE_CAMERA_ROTATION @ yaw_matrix @ pitch_matrix @ roll_matrix
    return quat_from_matrix(rotation)


def camera_local_position(config: EndoscopeCameraConfig) -> np.ndarray:
    return np.array([config.camera_x_m, config.camera_y_m, config.camera_z_m], dtype=float)


def camera_pixel_intrinsics(config: EndoscopeCameraConfig) -> tuple[float, float, float, float]:
    width, height = CAMERA_RESOLUTION
    fy = 0.5 * height / math.tan(math.radians(config.fovy_deg) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def camera_metric_intrinsics(config: EndoscopeCameraConfig) -> tuple[float, float, float, float]:
    width, height = CAMERA_RESOLUTION
    fx, fy, cx, cy = camera_pixel_intrinsics(config)
    return (
        fx * config.sensor_width_m / width,
        fy * config.sensor_height_m / height,
        cx * config.sensor_width_m / width,
        cy * config.sensor_height_m / height,
    )


def target_position(config: EndoscopeCameraConfig) -> np.ndarray:
    return np.array([config.target_x_m, config.target_y_m, config.target_z_m], dtype=float)


def target_sphere_center(config: EndoscopeCameraConfig) -> np.ndarray:
    return target_position(config) + TARGET_BOARD_OFFSET_TO_SPHERE


def copy_import_assets(temp_root: Path) -> None:
    for mesh_path in ROBOT_MESH_DIR.glob("*.STL"):
        shutil.copy2(mesh_path, temp_root / mesh_path.name)
    for mesh_path in ROBOT_MESH_DIR.glob("*.stl"):
        shutil.copy2(mesh_path, temp_root / mesh_path.name)
    shutil.copy2(ENDOSCOPE_MESH_PATH, temp_root / ENDOSCOPE_MESH_PATH.name)
    shutil.copy2(CAMERA_MESH_PATH, temp_root / CAMERA_MESH_PATH.name)


def add_tool_mesh_assets(spec: mujoco.MjSpec) -> None:
    endoscope_mesh = spec.add_mesh(name="endoscope_mesh")
    endoscope_mesh.file = ENDOSCOPE_MESH_PATH.name
    endoscope_mesh.scale = TOOL_MESH_SCALE

    camera_mesh = spec.add_mesh(name="camera_mesh")
    camera_mesh.file = CAMERA_MESH_PATH.name
    camera_mesh.scale = TOOL_MESH_SCALE


def add_visual_mesh_geom(
    body,
    *,
    name: str,
    mesh_name: str,
    pos: np.ndarray,
    rgba: tuple[float, float, float, float],
) -> None:
    geom = body.add_geom(name=name)
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh_name
    geom.pos = pos
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0


def add_endoscope_tool(spec: mujoco.MjSpec) -> None:
    link6 = spec.body("link_6")
    if link6 is None:
        raise ValueError("Failed to find link_6 in generated MJCF model.")

    tool_body = link6.add_body(name=TOOL_BODY_NAME)

    adapter = tool_body.add_geom(name="tool_adapter")
    adapter.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    adapter.pos = ADAPTER_POS
    adapter.size = ADAPTER_SIZE
    adapter.rgba = (0.68, 0.70, 0.74, 1.0)
    adapter.contype = 0
    adapter.conaffinity = 0

    add_visual_mesh_geom(
        tool_body,
        name="endoscope_body_geom",
        mesh_name="endoscope_mesh",
        pos=ENDOSCOPE_MESH_POS,
        rgba=(0.78, 0.80, 0.84, 1.0),
    )
    add_visual_mesh_geom(
        tool_body,
        name="camera_body_geom",
        mesh_name="camera_mesh",
        pos=CAMERA_MESH_POS,
        rgba=(0.92, 0.94, 0.98, 1.0),
    )

    camera = tool_body.add_camera(name=CAMERA_NAME)
    camera.pos = camera_local_position(DEFAULT_CONFIG)
    camera.quat = camera_quaternion(DEFAULT_CONFIG)
    camera.resolution = CAMERA_RESOLUTION
    camera.sensor_size = [DEFAULT_CONFIG.sensor_width_m, DEFAULT_CONFIG.sensor_height_m]
    camera.focal_pixel = camera_pixel_intrinsics(DEFAULT_CONFIG)[:2]
    camera.principal_pixel = camera_pixel_intrinsics(DEFAULT_CONFIG)[2:]


def add_target_body(spec: mujoco.MjSpec) -> None:
    target_body = spec.worldbody.add_body(name=TARGET_BODY_NAME)
    target_body.pos = target_position(DEFAULT_CONFIG)

    board = target_body.add_geom(name="target_board")
    board.type = mujoco.mjtGeom.mjGEOM_BOX
    board.size = [0.015, 0.04, 0.04]
    board.rgba = (0.18, 0.72, 0.44, 1.0)
    board.contype = 0
    board.conaffinity = 0

    cyl = target_body.add_geom(name="target_feature_cyl")
    cyl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cyl.pos = [0.021, 0.0, 0.0]
    cyl.quat = quat_from_matrix(rotation_y(math.pi / 2.0))
    cyl.size = [0.012, 0.009, 0.0]
    cyl.rgba = (0.92, 0.96, 0.28, 1.0)
    cyl.contype = 0
    cyl.conaffinity = 0

    sphere = target_body.add_geom(name="target_feature_sphere")
    sphere.type = mujoco.mjtGeom.mjGEOM_SPHERE
    sphere.pos = [0.018, -0.018, 0.02]
    sphere.size = [0.01, 0.0, 0.0]
    sphere.rgba = (0.94, 0.30, 0.22, 1.0)
    sphere.contype = 0
    sphere.conaffinity = 0

    box = target_body.add_geom(name="target_feature_box")
    box.type = mujoco.mjtGeom.mjGEOM_BOX
    box.pos = [0.018, 0.02, -0.018]
    box.size = [0.006, 0.01, 0.01]
    box.rgba = (0.22, 0.72, 0.96, 1.0)
    box.contype = 0
    box.conaffinity = 0

    front_sphere = target_body.add_geom(name="target_front_sphere")
    front_sphere.type = mujoco.mjtGeom.mjGEOM_SPHERE
    front_sphere.pos = TARGET_BOARD_OFFSET_TO_SPHERE
    front_sphere.size = [0.018, 0.0, 0.0]
    front_sphere.rgba = (0.96, 0.20, 0.20, 1.0)
    front_sphere.contype = 0
    front_sphere.conaffinity = 0


def add_scene_lights(spec: mujoco.MjSpec) -> None:
    key_light = spec.worldbody.add_light(name="key_light")
    key_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    key_light.pos = [-0.15, -0.8, 1.2]
    key_light.dir = [-0.2, 0.8, -0.8]
    key_light.ambient = [0.30, 0.30, 0.30]
    key_light.diffuse = [0.85, 0.85, 0.85]
    key_light.specular = [0.25, 0.25, 0.25]
    key_light.castshadow = False

    fill_light = spec.worldbody.add_light(name="fill_light")
    fill_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    fill_light.pos = [-0.9, 0.6, 0.9]
    fill_light.dir = [0.8, -0.2, -0.4]
    fill_light.ambient = [0.12, 0.12, 0.12]
    fill_light.diffuse = [0.45, 0.45, 0.45]
    fill_light.specular = [0.10, 0.10, 0.10]
    fill_light.castshadow = False


def load_model() -> mujoco.MjModel:
    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    if not ROBOT_MESH_DIR.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {ROBOT_MESH_DIR}")
    if not ENDOSCOPE_MESH_PATH.is_file():
        raise FileNotFoundError(f"Missing endoscope mesh: {ENDOSCOPE_MESH_PATH}")
    if not CAMERA_MESH_PATH.is_file():
        raise FileNotFoundError(f"Missing camera mesh: {CAMERA_MESH_PATH}")

    urdf_text = URDF_PATH.read_text(encoding="utf-8")
    resolved_urdf = urdf_text.replace(PACKAGE_PREFIX, "")

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        temp_urdf = temp_root / URDF_PATH.name
        temp_urdf.write_text(resolved_urdf, encoding="utf-8")
        copy_import_assets(temp_root)

        imported_model = mujoco.MjModel.from_xml_path(str(temp_urdf))
        temp_mjcf = temp_root / "fang_endoscope_scene.xml"
        mujoco.mj_saveLastXML(str(temp_mjcf), imported_model)
        spec = mujoco.MjSpec.from_file(str(temp_mjcf))
        add_tool_mesh_assets(spec)
        add_endoscope_tool(spec)
        add_target_body(spec)
        add_scene_lights(spec)
        return spec.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RM65-B endoscope scene in MuJoCo.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Only load the model and print a summary.")
    mode.add_argument("--capture-only", action="store_true", help="Save a camera image and exit.")
    parser.add_argument("--no-ui", action="store_true", help="Disable the control panel.")
    parser.add_argument(
        "--joint-deg",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Set the six arm joint angles in degrees.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Sleep duration between viewer sync calls.")
    return parser.parse_args()


def joint_degrees_from_qpos(qpos: np.ndarray) -> list[float]:
    return np.degrees(qpos[: len(BENT_QPOS)]).tolist()


def qpos_from_joint_degrees(base_qpos: np.ndarray, joint_deg: list[float] | tuple[float, ...] | None) -> np.ndarray:
    qpos = base_qpos.copy()
    if joint_deg is None:
        qpos[: len(BENT_QPOS)] = BENT_QPOS
    else:
        qpos[: len(BENT_QPOS)] = np.radians(np.array(joint_deg, dtype=float))
    return qpos


def freeze_state(data: mujoco.MjData, qpos: np.ndarray) -> None:
    data.qpos[:] = qpos
    data.qvel[:] = 0
    data.qacc[:] = 0


def apply_runtime_config(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    config: EndoscopeCameraConfig,
) -> None:
    freeze_state(data, qpos)

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
    model.body_pos[target_body_id] = target_position(config)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    focal_mx, focal_my, principal_mx, principal_my = camera_metric_intrinsics(config)
    model.cam_pos[camera_id] = camera_local_position(config)
    model.cam_quat[camera_id] = camera_quaternion(config)
    model.cam_fovy[camera_id] = config.fovy_deg
    model.cam_sensorsize[camera_id] = [config.sensor_width_m, config.sensor_height_m]
    model.cam_intrinsic[camera_id] = [focal_mx, focal_my, principal_mx, principal_my]

    mujoco.mj_forward(model, data)


def endoscope_camera_parameters(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EndoscopeCameraConfig,
) -> dict[str, float | list[float] | str]:
    fx, fy, cx, cy = camera_pixel_intrinsics(config)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    return {
        "name": CAMERA_NAME,
        "local_position": np.array(model.cam_pos[camera_id], dtype=float).round(6).tolist(),
        "world_position": np.array(data.cam_xpos[camera_id], dtype=float).round(6).tolist(),
        "quat": np.array(model.cam_quat[camera_id], dtype=float).round(6).tolist(),
        "resolution": np.array(model.cam_resolution[camera_id], dtype=int).tolist(),
        "sensor_size_m": np.array(model.cam_sensorsize[camera_id], dtype=float).round(6).tolist(),
        "focal_length_m": np.array(model.cam_intrinsic[camera_id][:2], dtype=float).round(6).tolist(),
        "principal_point_m": np.array(model.cam_intrinsic[camera_id][2:], dtype=float).round(6).tolist(),
        "focal_pixel": [round(fx, 6), round(fy, 6)],
        "principal_pixel": [round(cx, 6), round(cy, 6)],
        "fovy_deg": round(float(model.cam_fovy[camera_id]), 6),
    }


def render_endoscope_view(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    width, height = CAMERA_RESOLUTION
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=CAMERA_NAME)
    image = renderer.render().copy()
    renderer.close()
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(path)


def save_endoscope_view(model: mujoco.MjModel, data: mujoco.MjData) -> Path:
    output_path = CAPTURE_DIR / "endoscope.png"
    save_png(output_path, render_endoscope_view(model, data))
    return output_path


def overlay_endoscope_view(handle, model: mujoco.MjModel, data: mujoco.MjData, config: EndoscopeCameraConfig) -> None:
    image = render_endoscope_view(model, data)
    overlay_width, overlay_height = OVERLAY_RESOLUTION
    display_image = np.array(
        Image.fromarray(image, mode="RGB").resize(
            (overlay_width, overlay_height),
            Image.Resampling.BILINEAR,
        )
    )
    viewport = handle.viewport
    rect = mujoco.MjrRect(viewport.width - overlay_width - 20, 20, overlay_width, overlay_height)
    handle.set_images([(rect, display_image)])
    handle.set_texts(
        [
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Camera FOV", f"{config.fovy_deg:.1f} deg"),
            (
                None,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "Camera Offset",
                f"{config.camera_x_m:.3f}, {config.camera_y_m:.3f}, {config.camera_z_m:.3f} m",
            ),
        ]
    )


def print_runtime_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    config: EndoscopeCameraConfig,
) -> None:
    fx, fy, cx, cy = camera_pixel_intrinsics(config)
    print(f"Loaded model from {URDF_PATH}")
    print(
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"nbody={model.nbody}, ngeom={model.ngeom}, nmesh={model.nmesh}"
    )
    print(f"qpos={qpos.tolist()}")
    print(f"camera_local_position_m={camera_local_position(config).round(6).tolist()}")
    print(f"camera_fovy_deg={config.fovy_deg}")
    print(f"camera_sensor_size_m={[config.sensor_width_m, config.sensor_height_m]}")
    print(
        "camera_intrinsics_pixels="
        f"{{'fx': {fx:.6f}, 'fy': {fy:.6f}, 'cx': {cx:.6f}, 'cy': {cy:.6f}}}"
    )
    print(f"camera_yaw_pitch_roll_deg={[config.yaw_deg, config.pitch_deg, config.roll_deg]}")
    print(f"target_position_m={target_position(config).round(6).tolist()}")
    print(f"target_sphere_center_m={target_sphere_center(config).round(6).tolist()}")
    print(f"camera={endoscope_camera_parameters(model, data, config)}")


class EndoscopeControlPanel:
    def __init__(self, config: EndoscopeCameraConfig, qpos: np.ndarray):
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("Endoscope Camera Controls")
        self._root.geometry("560x1040")
        self._root.resizable(False, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._root.after(200, lambda: self._root.attributes("-topmost", False))

        self._dirty = True
        self._closed = False
        self._capture_requested = False
        self._status_var = tk.StringVar()
        self._vars: dict[str, tk.DoubleVar] = {}
        self._defaults = config
        self._default_joint_deg = joint_degrees_from_qpos(qpos)
        self._base_qpos = qpos.copy()

        self._build_controls(config, qpos)
        self._refresh_status()

    def _on_close(self) -> None:
        self._closed = True
        self._root.destroy()

    def close(self) -> None:
        if not self._closed:
            self._on_close()

    def _mark_dirty(self, *_args) -> None:
        self._dirty = True
        self._refresh_status()

    def _build_controls(self, config: EndoscopeCameraConfig, qpos: np.ndarray) -> None:
        tk = self._tk
        frame = tk.Frame(self._root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        joint_controls = [
            (
                f"joint_{joint_index + 1}_deg",
                f"{JOINT_NAMES[joint_index]} (deg)",
                JOINT_LIMITS_DEG[joint_index][0],
                JOINT_LIMITS_DEG[joint_index][1],
                0.5,
                joint_degrees_from_qpos(qpos)[joint_index],
            )
            for joint_index in range(len(BENT_QPOS))
        ]
        camera_controls = [
            ("camera_x_mm", "Camera X (mm)", -80.0, 80.0, 1.0, config.camera_x_m * 1000.0),
            ("camera_y_mm", "Camera Y (mm)", -80.0, 80.0, 1.0, config.camera_y_m * 1000.0),
            ("camera_z_mm", "Camera Z (mm)", 200.0, 600.0, 1.0, config.camera_z_m * 1000.0),
            ("fovy_deg", "Vertical FOV (deg)", 20.0, 110.0, 0.5, config.fovy_deg),
            ("yaw_deg", "Camera Yaw (deg)", -45.0, 45.0, 0.5, config.yaw_deg),
            ("pitch_deg", "Camera Pitch (deg)", -45.0, 45.0, 0.5, config.pitch_deg),
            ("roll_deg", "Camera Roll (deg)", -180.0, 180.0, 1.0, config.roll_deg),
            ("sensor_w_mm", "Sensor Width (mm)", 2.0, 12.0, 0.1, config.sensor_width_m * 1000.0),
            ("sensor_h_mm", "Sensor Height (mm)", 2.0, 12.0, 0.1, config.sensor_height_m * 1000.0),
            ("target_x_mm", "Target X (mm)", -800.0, -300.0, 1.0, config.target_x_m * 1000.0),
            ("target_y_mm", "Target Y (mm)", -120.0, 120.0, 1.0, config.target_y_m * 1000.0),
            ("target_z_mm", "Target Z (mm)", 300.0, 700.0, 1.0, config.target_z_m * 1000.0),
        ]
        controls = joint_controls + camera_controls

        for row, (key, label, start, stop, resolution, value) in enumerate(controls):
            tk.Label(frame, text=label, anchor="w").grid(row=row * 2, column=0, sticky="w", pady=(0, 2))
            var = tk.DoubleVar(value=value)
            self._vars[key] = var
            var.trace_add("write", self._mark_dirty)
            tk.Scale(
                frame,
                from_=start,
                to=stop,
                resolution=resolution,
                orient="horizontal",
                length=480,
                variable=var,
            ).grid(row=row * 2 + 1, column=0, sticky="we", pady=(0, 8))

        button_frame = tk.Frame(frame, pady=8)
        button_frame.grid(row=len(controls) * 2, column=0, sticky="we")

        tk.Button(button_frame, text="Save PNG", command=self.request_capture, width=14).pack(side="left")
        tk.Button(button_frame, text="Reset Defaults", command=self.reset, width=16).pack(side="left", padx=8)
        tk.Button(button_frame, text="Close", command=self._on_close, width=10).pack(side="right")

        tk.Label(
            frame,
            text=(
                "The endoscope meshes are fixed to link_6 and the virtual camera is mounted at the tip.\n"
                "Adjust joints, camera pose, intrinsics, and target position, then use Save PNG."
            ),
            justify="left",
            anchor="w",
        ).grid(row=len(controls) * 2 + 1, column=0, sticky="we", pady=(4, 8))

        tk.Label(frame, textvariable=self._status_var, justify="left", anchor="w", font=("Consolas", 10)).grid(
            row=len(controls) * 2 + 2, column=0, sticky="we"
        )

    def _refresh_status(self) -> None:
        config = self.current_config()
        joint_deg = self.current_joint_deg()
        fx, fy, cx, cy = camera_pixel_intrinsics(config)
        camera_pos = camera_local_position(config)
        sphere = target_sphere_center(config)
        self._status_var.set(
            "\n".join(
                [
                    "Joints: " + ", ".join(f"{name}={deg:.1f}" for name, deg in zip(JOINT_NAMES, joint_deg)),
                    f"Resolution: {CAMERA_RESOLUTION[0]} x {CAMERA_RESOLUTION[1]}",
                    f"fx={fx:.2f}px  fy={fy:.2f}px",
                    f"cx={cx:.1f}px  cy={cy:.1f}px",
                    f"Camera local pos: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}] m",
                    f"Target sphere: [{sphere[0]:.3f}, {sphere[1]:.3f}, {sphere[2]:.3f}] m",
                ]
            )
        )

    def current_config(self) -> EndoscopeCameraConfig:
        return EndoscopeCameraConfig(
            camera_x_m=self._vars["camera_x_mm"].get() / 1000.0,
            camera_y_m=self._vars["camera_y_mm"].get() / 1000.0,
            camera_z_m=self._vars["camera_z_mm"].get() / 1000.0,
            fovy_deg=self._vars["fovy_deg"].get(),
            yaw_deg=self._vars["yaw_deg"].get(),
            pitch_deg=self._vars["pitch_deg"].get(),
            roll_deg=self._vars["roll_deg"].get(),
            sensor_width_m=self._vars["sensor_w_mm"].get() / 1000.0,
            sensor_height_m=self._vars["sensor_h_mm"].get() / 1000.0,
            target_x_m=self._vars["target_x_mm"].get() / 1000.0,
            target_y_m=self._vars["target_y_mm"].get() / 1000.0,
            target_z_m=self._vars["target_z_mm"].get() / 1000.0,
        )

    def current_joint_deg(self) -> list[float]:
        return [self._vars[f"joint_{joint_index + 1}_deg"].get() for joint_index in range(len(BENT_QPOS))]

    def current_qpos(self) -> np.ndarray:
        return qpos_from_joint_degrees(self._base_qpos, self.current_joint_deg())

    def consume_dirty(self) -> bool:
        was_dirty = self._dirty
        self._dirty = False
        return was_dirty

    def request_capture(self) -> None:
        self._capture_requested = True

    def consume_capture_request(self) -> bool:
        requested = self._capture_requested
        self._capture_requested = False
        return requested

    def reset(self) -> None:
        defaults = self._defaults
        for joint_index, joint_deg in enumerate(self._default_joint_deg, start=1):
            self._vars[f"joint_{joint_index}_deg"].set(joint_deg)
        self._vars["camera_x_mm"].set(defaults.camera_x_m * 1000.0)
        self._vars["camera_y_mm"].set(defaults.camera_y_m * 1000.0)
        self._vars["camera_z_mm"].set(defaults.camera_z_m * 1000.0)
        self._vars["fovy_deg"].set(defaults.fovy_deg)
        self._vars["yaw_deg"].set(defaults.yaw_deg)
        self._vars["pitch_deg"].set(defaults.pitch_deg)
        self._vars["roll_deg"].set(defaults.roll_deg)
        self._vars["sensor_w_mm"].set(defaults.sensor_width_m * 1000.0)
        self._vars["sensor_h_mm"].set(defaults.sensor_height_m * 1000.0)
        self._vars["target_x_mm"].set(defaults.target_x_m * 1000.0)
        self._vars["target_y_mm"].set(defaults.target_y_m * 1000.0)
        self._vars["target_z_mm"].set(defaults.target_z_m * 1000.0)
        self._mark_dirty()

    def update(self) -> bool:
        if self._closed:
            return False
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
            return False
        return True


def main() -> None:
    args = parse_args()
    model = load_model()
    data = mujoco.MjData(model)
    base_qpos = data.qpos.copy()
    current_qpos = qpos_from_joint_degrees(base_qpos, args.joint_deg)

    config = DEFAULT_CONFIG
    apply_runtime_config(model, data, current_qpos, config)

    if args.check_only:
        print_runtime_summary(model, data, current_qpos, config)
        return

    if args.capture_only:
        output_path = save_endoscope_view(model, data)
        print(f"endoscope image saved to {output_path}")
        print_runtime_summary(model, data, current_qpos, config)
        return

    from mujoco import viewer

    control_panel = None if args.no_ui else EndoscopeControlPanel(config, current_qpos)

    with viewer.launch_passive(model, data) as handle:
        overlay_endoscope_view(handle, model, data, config)
        print_runtime_summary(model, data, current_qpos, config)

        while handle.is_running():
            if control_panel is not None:
                if not control_panel.update():
                    handle.close()
                    break
                if control_panel.consume_dirty():
                    config = control_panel.current_config()
                    current_qpos = control_panel.current_qpos()
                    apply_runtime_config(model, data, current_qpos, config)
                    overlay_endoscope_view(handle, model, data, config)
                if control_panel.consume_capture_request():
                    apply_runtime_config(model, data, current_qpos, config)
                    output_path = save_endoscope_view(model, data)
                    print(f"endoscope image saved to {output_path}")
                    print_runtime_summary(model, data, current_qpos, config)

            freeze_state(data, current_qpos)
            mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(args.dt)

    if control_panel is not None:
        control_panel.close()


if __name__ == "__main__":
    main()
