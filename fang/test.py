import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
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
DEFAULT_PROJECTOR_PATTERN_PATH = ROOT.parent / "structured_light_patterns.png"

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

TOOL_RIG_BODY_NAME = "dual_endoscope_rig"
TARGET_BODY_NAME = "inspection_target"
HEAD_BODY_NAMES = {
    "primary": "primary_endoscope_head",
    "secondary": "secondary_endoscope_head",
}
CAMERA_NAMES = {
    "primary": "primary_endoscope_camera",
    "secondary": "secondary_endoscope_camera",
}
CAMERA_LABELS = {
    "primary": "Left Camera",
    "secondary": "Right Camera",
}
CAMERA_OUTPUT_STEMS = {
    "primary": "left_camera",
    "secondary": "right_camera",
}
DEFAULT_ACTIVE_CAMERA_KEY = "primary"
PROJECTOR_BODY_NAME = "structured_light_projector"
PROJECTOR_TEXTURE_NAME = "structured_light_projector_texture"
PROJECTOR_MATERIAL_NAME = "structured_light_projector_material"

CAMERA_RESOLUTION = (640, 480)  # width, height
OVERLAY_RESOLUTION = (320, 240)  # width, height
TOOL_MESH_SCALE = (0.001, 0.001, 0.001)
PROJECTOR_TEXTURE_SIZE = (512, 512)  # width, height
TARGET_BOARD_HALF_SIZE = np.array([0.015, 0.04, 0.04], dtype=float)
BRACKET_MOUNT_POS = np.array([0.0, 0.0, 0.005], dtype=float)
BRACKET_MOUNT_SIZE = (0.026, 0.005, 0.0)
BRACKET_SPINE_POS = np.array([0.0, 0.0, 0.012], dtype=float)
BRACKET_SPINE_SIZE = (0.008, 0.010, 0.012)
BRACKET_CROSSBAR_POS = np.array([0.0, 0.0, 0.024], dtype=float)
BRACKET_CROSSBAR_HALF_THICKNESS = 0.007
BRACKET_CROSSBAR_HALF_HEIGHT = 0.006
BRACKET_CROSSBAR_MARGIN_M = 0.014
ADAPTER_POS = np.array([0.0, 0.0, 0.012], dtype=float)
ADAPTER_SIZE = (0.018, 0.012, 0.0)
# The endoscope rod axis is offset inside the STL, so align to that axis rather
# than to the overall mesh bounding-box center. Its local z is placed so the
# endoscope starts after the connector ring instead of overlapping the camera.
ENDOSCOPE_MESH_POS = np.array([-0.017095, -0.025616, 0.003], dtype=float)
# The camera STL points backwards in its file coordinates. Rotate it 180 degrees
# around the local x axis so the visible front matches the virtual camera front.
# Its z is pulled slightly backward so the connector starts in front of it.
CAMERA_MESH_POS = np.array([-0.018954, 0.018954, 0.010], dtype=float)
CAMERA_MESH_QUAT = (0.0, 1.0, 0.0, 0.0)
TARGET_BOARD_OFFSET_TO_SPHERE = np.array([0.05, 0.0, 0.0], dtype=float)
# Move the whole camera module forward so the housing starts after the bracket
# instead of occupying the same volume as the crossbar.
HEAD_CAMERA_MODULE_OFFSET = np.array([-0.008, 0.0, 0.0455], dtype=float)
HEAD_CAMERA_SUPPORT_START = np.array([0.0, 0.0, 0.012], dtype=float)
HEAD_CAMERA_SUPPORT_END = HEAD_CAMERA_MODULE_OFFSET + np.array([0.001, 0.0, -0.0155], dtype=float)
CAMERA_TO_RING_CONNECTOR_OFFSET = np.array([0.0, 0.0, 0.0175], dtype=float)
RING_TO_ENDOSCOPE_OFFSET = np.array([0.0, 0.0, 0.004], dtype=float)
RING_CONNECTOR_CORE_SIZE = (0.0082, 0.0045, 0.0)
RING_CONNECTOR_FLANGE_SIZE = (0.0108, 0.0018, 0.0)
RING_CONNECTOR_FLANGE_REAR_POS = np.array([0.0, 0.0, -0.0027], dtype=float)
RING_CONNECTOR_FLANGE_FRONT_POS = np.array([0.0, 0.0, 0.0027], dtype=float)
PROJECTOR_SUPPORT_START_POS = np.array([-0.010, 0.0, 0.040], dtype=float)
PROJECTOR_SUPPORT_HALF_SIZE_XY = (0.0030, 0.0030)
PROJECTOR_HOUSING_SIZE = (0.012, 0.010, 0.012)
PROJECTOR_LENS_POS = np.array([0.0, 0.0, 0.012], dtype=float)
PROJECTOR_LENS_SIZE = (0.006, 0.0045, 0.0)
PROJECTOR_MOUNT_CAP_POS = np.array([-0.010, 0.0, 0.041], dtype=float)
PROJECTOR_MOUNT_CAP_SIZE = (0.014, 0.018, 0.0035)

ROBOT_ARM_RGBA = (0.30, 0.36, 0.46, 1.0)
BRACKET_RGBA = (0.75, 0.59, 0.22, 1.0)
CONNECTOR_RGBA = (0.42, 0.45, 0.49, 1.0)
PROJECTOR_RGBA = (0.56, 0.28, 0.76, 1.0)
PROJECTOR_LENS_RGBA = (0.82, 0.90, 1.00, 1.0)
TARGET_BASE_RGBA = (0.18, 0.72, 0.44, 1.0)
HEAD_COLORS = {
    "primary": {
        "endoscope": (0.07, 0.67, 0.60, 1.0),
        "camera": (0.63, 0.96, 0.90, 1.0),
    },
    "secondary": {
        "endoscope": (0.88, 0.39, 0.17, 1.0),
        "camera": (0.98, 0.81, 0.32, 1.0),
    },
}

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
    head_spacing_m: float = 0.055
    head_toe_in_deg: float = 30.0
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0
    camera_z_m: float = 0.50
    fovy_deg: float = 55.0
    sensor_width_m: float = 0.0064
    sensor_height_m: float = 0.0048
    yaw_deg: float = 40.0
    pitch_deg: float = -20.0
    roll_deg: float = 0.0
    projector_x_m: float = -0.020
    projector_y_m: float = 0.0
    projector_z_m: float = 0.072
    projector_yaw_deg: float = 0.0
    projector_pitch_deg: float = -24.0
    projector_roll_deg: float = 0.0
    projector_fovy_deg: float = 62.0
    projector_pattern_path: str = str(DEFAULT_PROJECTOR_PATTERN_PATH)
    projector_enable: bool = False
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


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return vector.copy()
    return vector / norm


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


def quat_align_z_to_vector(direction: np.ndarray) -> np.ndarray:
    direction = normalize(direction)
    if float(np.linalg.norm(direction)) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(direction, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    x_axis = normalize(np.cross(reference, direction))
    y_axis = normalize(np.cross(direction, x_axis))
    rotation = np.column_stack((x_axis, y_axis, direction))
    return quat_from_matrix(rotation)


def camera_yaw_for_head(config: EndoscopeCameraConfig, camera_key: str) -> float:
    direction = 1.0 if camera_key == "primary" else -1.0
    return config.yaw_deg + direction * config.head_toe_in_deg


def camera_quaternion(config: EndoscopeCameraConfig, camera_key: str) -> np.ndarray:
    yaw_matrix = rotation_y(math.radians(camera_yaw_for_head(config, camera_key)))
    pitch_matrix = rotation_x(math.radians(config.pitch_deg))
    roll_matrix = rotation_z(math.radians(config.roll_deg))
    rotation = BASE_CAMERA_ROTATION @ yaw_matrix @ pitch_matrix @ roll_matrix
    return quat_from_matrix(rotation)


def camera_local_position(config: EndoscopeCameraConfig) -> np.ndarray:
    return np.array([config.camera_x_m, config.camera_y_m, config.camera_z_m], dtype=float)


def projector_rotation_matrix(config: EndoscopeCameraConfig) -> np.ndarray:
    yaw_matrix = rotation_y(math.radians(config.projector_yaw_deg))
    pitch_matrix = rotation_x(math.radians(config.projector_pitch_deg))
    roll_matrix = rotation_z(math.radians(config.projector_roll_deg))
    return BASE_CAMERA_ROTATION @ yaw_matrix @ pitch_matrix @ roll_matrix


def projector_quaternion(config: EndoscopeCameraConfig) -> np.ndarray:
    return quat_from_matrix(projector_rotation_matrix(config))


def projector_local_position(config: EndoscopeCameraConfig) -> np.ndarray:
    return np.array([config.projector_x_m, config.projector_y_m, config.projector_z_m], dtype=float)


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


def head_body_offsets(config: EndoscopeCameraConfig) -> dict[str, np.ndarray]:
    half_spacing = config.head_spacing_m / 2.0
    return {
        "primary": np.array([0.0, -half_spacing, 0.0], dtype=float),
        "secondary": np.array([0.0, half_spacing, 0.0], dtype=float),
    }


def crossbar_half_span(config: EndoscopeCameraConfig) -> float:
    return config.head_spacing_m / 2.0 + BRACKET_CROSSBAR_MARGIN_M


def resolve_projector_pattern_path(pattern_path: str) -> Path | None:
    cleaned = pattern_path.strip()
    if not cleaned:
        return None

    candidate = Path(cleaned).expanduser()
    search_paths = [candidate]
    if not candidate.is_absolute():
        search_paths.extend([ROOT / candidate, ROOT.parent / candidate])

    for path in search_paths:
        if path.is_file():
            return path.resolve()
    return None


def load_projector_pattern_image(pattern_path: str) -> tuple[np.ndarray, str | None]:
    resolved_path = resolve_projector_pattern_path(pattern_path)
    if resolved_path is None:
        return np.zeros((*PROJECTOR_TEXTURE_SIZE[::-1], 3), dtype=np.uint8), None

    pattern = (
        Image.open(resolved_path)
        .convert("RGB")
        .resize(PROJECTOR_TEXTURE_SIZE, Image.Resampling.BILINEAR)
    )
    return np.array(pattern, dtype=np.uint8), str(resolved_path)


def base_target_texture() -> np.ndarray:
    texture = np.empty((*PROJECTOR_TEXTURE_SIZE[::-1], 3), dtype=np.uint8)
    texture[:, :] = np.round(np.array(TARGET_BASE_RGBA[:3]) * 255.0).astype(np.uint8)
    return texture


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


def add_projector_texture_assets(spec: mujoco.MjSpec) -> None:
    # Pre-allocate a runtime-updatable texture. At runtime we overwrite its pixels
    # with a structured-light projector approximation built from the external PNG.
    texture = spec.add_texture(name=PROJECTOR_TEXTURE_NAME)
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    texture.width = PROJECTOR_TEXTURE_SIZE[0]
    texture.height = PROJECTOR_TEXTURE_SIZE[1]
    texture.nchannel = 3
    texture.rgb1 = list(TARGET_BASE_RGBA[:3])
    texture.rgb2 = list(TARGET_BASE_RGBA[:3])

    material = spec.add_material(name=PROJECTOR_MATERIAL_NAME)
    material.rgba = [1.0, 1.0, 1.0, 1.0]
    material.emission = 0.25
    material.specular = 0.02
    material.texrepeat = [1.0, 1.0]
    material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = PROJECTOR_TEXTURE_NAME


def add_visual_mesh_geom(
    body,
    *,
    name: str,
    mesh_name: str,
    pos: np.ndarray,
    quat: tuple[float, float, float, float] | None = None,
    rgba: tuple[float, float, float, float],
) -> None:
    geom = body.add_geom(name=name)
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh_name
    geom.pos = pos
    if quat is not None:
        geom.quat = quat
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0


def add_bracket_geom(
    body,
    *,
    name: str,
    geom_type,
    pos: np.ndarray | list[float],
    size: tuple[float, float, float] | list[float],
) -> None:
    geom = body.add_geom(name=name)
    geom.type = geom_type
    geom.pos = pos
    geom.size = size
    geom.rgba = BRACKET_RGBA
    geom.contype = 0
    geom.conaffinity = 0


def add_capsule_between_points(
    body,
    *,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    vector = np.array(end, dtype=float) - np.array(start, dtype=float)
    length = max(float(np.linalg.norm(vector)), 1e-5)

    geom = body.add_geom(name=name)
    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom.pos = ((np.array(start, dtype=float) + np.array(end, dtype=float)) / 2.0).tolist()
    geom.quat = quat_align_z_to_vector(vector)
    geom.size = [radius, length / 2.0, 0.0]
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0


def add_sensor_head(body, camera_key: str) -> None:
    adapter = body.add_geom(name=f"{camera_key}_adapter")
    adapter.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    adapter.pos = ADAPTER_POS
    adapter.size = ADAPTER_SIZE
    adapter.rgba = BRACKET_RGBA
    adapter.contype = 0
    adapter.conaffinity = 0

    add_capsule_between_points(
        body,
        name=f"{camera_key}_camera_support",
        start=HEAD_CAMERA_SUPPORT_START,
        end=HEAD_CAMERA_SUPPORT_END,
        radius=0.0035,
        rgba=BRACKET_RGBA,
    )

    camera_module = body.add_body(name=f"{camera_key}_camera_module")
    camera_module.pos = HEAD_CAMERA_MODULE_OFFSET
    add_visual_mesh_geom(
        camera_module,
        name=f"{camera_key}_camera_body_geom",
        mesh_name="camera_mesh",
        pos=CAMERA_MESH_POS,
        quat=CAMERA_MESH_QUAT,
        rgba=HEAD_COLORS[camera_key]["camera"],
    )
    camera_plate = camera_module.add_geom(name=f"{camera_key}_camera_plate")
    camera_plate.type = mujoco.mjtGeom.mjGEOM_BOX
    camera_plate.pos = [0.001, 0.0, -0.012]
    camera_plate.size = [0.009, 0.010, 0.0035]
    camera_plate.rgba = BRACKET_RGBA
    camera_plate.contype = 0
    camera_plate.conaffinity = 0

    # Use a standalone flange-like ring between the camera and endoscope so the
    # three stages read clearly as camera -> ring connector -> endoscope.
    ring_connector = camera_module.add_body(name=f"{camera_key}_ring_connector")
    ring_connector.pos = CAMERA_TO_RING_CONNECTOR_OFFSET

    ring_core = ring_connector.add_geom(name=f"{camera_key}_ring_connector_core")
    ring_core.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    ring_core.size = RING_CONNECTOR_CORE_SIZE
    ring_core.rgba = CONNECTOR_RGBA
    ring_core.contype = 0
    ring_core.conaffinity = 0

    rear_flange = ring_connector.add_geom(name=f"{camera_key}_ring_connector_rear_flange")
    rear_flange.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    rear_flange.pos = RING_CONNECTOR_FLANGE_REAR_POS
    rear_flange.size = RING_CONNECTOR_FLANGE_SIZE
    rear_flange.rgba = CONNECTOR_RGBA
    rear_flange.contype = 0
    rear_flange.conaffinity = 0

    front_flange = ring_connector.add_geom(name=f"{camera_key}_ring_connector_front_flange")
    front_flange.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    front_flange.pos = RING_CONNECTOR_FLANGE_FRONT_POS
    front_flange.size = RING_CONNECTOR_FLANGE_SIZE
    front_flange.rgba = CONNECTOR_RGBA
    front_flange.contype = 0
    front_flange.conaffinity = 0

    endoscope_module = ring_connector.add_body(name=f"{camera_key}_endoscope_module")
    endoscope_module.pos = RING_TO_ENDOSCOPE_OFFSET
    add_visual_mesh_geom(
        endoscope_module,
        name=f"{camera_key}_endoscope_body_geom",
        mesh_name="endoscope_mesh",
        pos=ENDOSCOPE_MESH_POS,
        rgba=HEAD_COLORS[camera_key]["endoscope"],
    )

    camera = camera_module.add_camera(name=CAMERA_NAMES[camera_key])
    camera.pos = camera_local_position(DEFAULT_CONFIG)
    camera.quat = camera_quaternion(DEFAULT_CONFIG, camera_key)
    camera.resolution = CAMERA_RESOLUTION
    camera.sensor_size = [DEFAULT_CONFIG.sensor_width_m, DEFAULT_CONFIG.sensor_height_m]
    camera.focal_pixel = camera_pixel_intrinsics(DEFAULT_CONFIG)[:2]
    camera.principal_pixel = camera_pixel_intrinsics(DEFAULT_CONFIG)[2:]


def add_projector_body(spec: mujoco.MjSpec) -> None:
    # The projector is mounted above the camera modules, with a short bracket
    # that keeps it on the stereo midline and out of the way of the endoscopes.
    rig_body = spec.body(TOOL_RIG_BODY_NAME)
    if rig_body is None:
        raise ValueError("Failed to find dual endoscope rig body in generated MJCF model.")

    mount_cap = rig_body.add_geom(name="projector_mount_cap")
    mount_cap.type = mujoco.mjtGeom.mjGEOM_BOX
    mount_cap.pos = PROJECTOR_MOUNT_CAP_POS
    mount_cap.size = PROJECTOR_MOUNT_CAP_SIZE
    mount_cap.rgba = BRACKET_RGBA
    mount_cap.contype = 0
    mount_cap.conaffinity = 0

    support = rig_body.add_geom(name="projector_support")
    support.type = mujoco.mjtGeom.mjGEOM_BOX
    support.pos = (PROJECTOR_SUPPORT_START_POS + projector_local_position(DEFAULT_CONFIG)) / 2.0
    support.size = [*PROJECTOR_SUPPORT_HALF_SIZE_XY, 0.012]
    support.rgba = BRACKET_RGBA
    support.contype = 0
    support.conaffinity = 0

    projector_body = rig_body.add_body(name=PROJECTOR_BODY_NAME)
    projector_body.pos = projector_local_position(DEFAULT_CONFIG)
    projector_body.quat = projector_quaternion(DEFAULT_CONFIG)

    housing = projector_body.add_geom(name="projector_housing")
    housing.type = mujoco.mjtGeom.mjGEOM_BOX
    housing.size = PROJECTOR_HOUSING_SIZE
    housing.rgba = PROJECTOR_RGBA
    housing.contype = 0
    housing.conaffinity = 0

    lens = projector_body.add_geom(name="projector_lens")
    lens.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    lens.pos = PROJECTOR_LENS_POS
    lens.size = PROJECTOR_LENS_SIZE
    lens.rgba = PROJECTOR_LENS_RGBA
    lens.contype = 0
    lens.conaffinity = 0


def add_endoscope_tool(spec: mujoco.MjSpec) -> None:
    link6 = spec.body("link_6")
    if link6 is None:
        raise ValueError("Failed to find link_6 in generated MJCF model.")

    rig_body = link6.add_body(name=TOOL_RIG_BODY_NAME)

    add_bracket_geom(
        rig_body,
        name="rig_mount",
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=BRACKET_MOUNT_POS,
        size=BRACKET_MOUNT_SIZE,
    )
    add_bracket_geom(
        rig_body,
        name="rig_spine",
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=BRACKET_SPINE_POS,
        size=BRACKET_SPINE_SIZE,
    )
    add_bracket_geom(
        rig_body,
        name="rig_crossbar",
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=BRACKET_CROSSBAR_POS,
        size=[
            BRACKET_CROSSBAR_HALF_THICKNESS,
            crossbar_half_span(DEFAULT_CONFIG),
            BRACKET_CROSSBAR_HALF_HEIGHT,
        ],
    )

    for camera_key, body_name in HEAD_BODY_NAMES.items():
        head_body = rig_body.add_body(name=body_name)
        head_body.pos = head_body_offsets(DEFAULT_CONFIG)[camera_key]
        add_sensor_head(head_body, camera_key)


def add_target_body(spec: mujoco.MjSpec) -> None:
    target_body = spec.worldbody.add_body(name=TARGET_BODY_NAME)
    target_body.pos = target_position(DEFAULT_CONFIG)

    board = target_body.add_geom(name="target_board")
    board.type = mujoco.mjtGeom.mjGEOM_BOX
    board.size = TARGET_BOARD_HALF_SIZE.tolist()
    board.material = PROJECTOR_MATERIAL_NAME
    board.rgba = (1.0, 1.0, 1.0, 1.0)
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


def apply_projector_runtime_config(model: mujoco.MjModel, config: EndoscopeCameraConfig) -> None:
    projector_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PROJECTOR_BODY_NAME)
    model.body_pos[projector_body_id] = projector_local_position(config)
    model.body_quat[projector_body_id] = projector_quaternion(config)

    projector_support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "projector_support")
    projector_vector = projector_local_position(config) - PROJECTOR_SUPPORT_START_POS
    projector_length = max(float(np.linalg.norm(projector_vector)), 1e-4)
    model.geom_pos[projector_support_id] = PROJECTOR_SUPPORT_START_POS + projector_vector / 2.0
    model.geom_quat[projector_support_id] = quat_align_z_to_vector(projector_vector)
    model.geom_size[projector_support_id] = [
        PROJECTOR_SUPPORT_HALF_SIZE_XY[0],
        PROJECTOR_SUPPORT_HALF_SIZE_XY[1],
        projector_length / 2.0,
    ]


def apply_visual_theme(model: mujoco.MjModel) -> None:
    robot_bodies = {"link_1", "link_2", "link_3", "link_4", "link_5", "link_6"}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name in robot_bodies:
            model.geom_rgba[geom_id] = ROBOT_ARM_RGBA


def write_projector_texture(model: mujoco.MjModel, texture_image: np.ndarray) -> None:
    texture_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, PROJECTOR_TEXTURE_NAME)
    texture_adr = int(model.tex_adr[texture_id])
    texture_size = int(model.tex_width[texture_id] * model.tex_height[texture_id] * model.tex_nchannel[texture_id])
    model.tex_data[texture_adr : texture_adr + texture_size] = texture_image.reshape(-1)


# Structured-light projector approximation:
# We approximate the projector by baking the external speckle PNG onto the front face
# of the target board texture using the projector pose and FOV as a pinhole projector.
def build_projector_approximation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EndoscopeCameraConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    texture = base_target_texture()
    pattern_rgb, resolved_pattern_path = load_projector_pattern_image(config.projector_pattern_path)

    state: dict[str, object] = {
        "mode": "structured-light projector approximation",
        "enabled": bool(config.projector_enable),
        "pattern_path": config.projector_pattern_path,
        "resolved_pattern_path": resolved_pattern_path,
        "texture_size": list(PROJECTOR_TEXTURE_SIZE),
        "target_receiver": "target_board_front_face",
    }

    if not config.projector_enable:
        state["status"] = "disabled"
        return texture, state

    if resolved_pattern_path is None:
        state["status"] = "pattern_missing"
        return texture, state

    projector_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PROJECTOR_BODY_NAME)
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)

    projector_pos = np.array(data.xpos[projector_body_id], dtype=float)
    projector_rot = np.array(data.xmat[projector_body_id], dtype=float).reshape(3, 3)
    target_pos = np.array(data.xpos[target_body_id], dtype=float)
    target_rot = np.array(data.xmat[target_body_id], dtype=float).reshape(3, 3)

    projector_right = projector_rot[:, 0]
    projector_up = projector_rot[:, 1]
    projector_forward = -projector_rot[:, 2]

    board_half_width = float(TARGET_BOARD_HALF_SIZE[1])
    board_half_height = float(TARGET_BOARD_HALF_SIZE[2])
    board_half_thickness = float(TARGET_BOARD_HALF_SIZE[0])
    board_face_origin = np.array([board_half_thickness, 0.0, 0.0], dtype=float)

    pattern_height, pattern_width, _ = pattern_rgb.shape
    projector_aspect = pattern_width / pattern_height
    tan_half_fovy = math.tan(math.radians(config.projector_fovy_deg) / 2.0)

    texture_width, texture_height = PROJECTOR_TEXTURE_SIZE
    ys = np.linspace(-board_half_width, board_half_width, texture_width, dtype=float)
    zs = np.linspace(board_half_height, -board_half_height, texture_height, dtype=float)
    yy, zz = np.meshgrid(ys, zs)
    local_points = np.stack(
        [
            np.full_like(yy, board_face_origin[0]),
            yy,
            zz,
        ],
        axis=-1,
    )
    world_points = target_pos + local_points @ target_rot.T
    relative = world_points - projector_pos

    x_proj = relative @ projector_right
    y_proj = relative @ projector_up
    z_proj = relative @ projector_forward

    half_height = np.maximum(z_proj * tan_half_fovy, 1e-6)
    half_width = np.maximum(half_height * projector_aspect, 1e-6)

    u = 0.5 + x_proj / (2.0 * half_width)
    v = 0.5 - y_proj / (2.0 * half_height)

    visible = (z_proj > 1e-4) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    if not np.any(visible):
        state["status"] = "projector_misses_target"
        return texture, state

    pixel_x = np.clip((u * (pattern_width - 1)).astype(int), 0, pattern_width - 1)
    pixel_y = np.clip((v * (pattern_height - 1)).astype(int), 0, pattern_height - 1)
    sampled_pattern = pattern_rgb[pixel_y, pixel_x]

    overlay_strength = 0.78
    texture_float = texture.astype(np.float32)
    sampled_float = sampled_pattern.astype(np.float32)
    texture_float[visible] = np.clip(
        texture_float[visible] * (1.0 - overlay_strength) + sampled_float[visible] * overlay_strength,
        0.0,
        255.0,
    )

    state["status"] = "ok"
    state["visible_fraction"] = round(float(np.mean(visible)), 6)
    state["projector_world_position"] = projector_pos.round(6).tolist()
    state["projector_forward"] = projector_forward.round(6).tolist()
    return texture_float.astype(np.uint8), state


def update_projector_approximation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EndoscopeCameraConfig,
) -> dict[str, object]:
    texture_image, state = build_projector_approximation(model, data, config)
    write_projector_texture(model, texture_image)
    return state


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
        add_projector_texture_assets(spec)
        add_endoscope_tool(spec)
        add_projector_body(spec)
        add_target_body(spec)
        add_scene_lights(spec)
        model = spec.compile()
        apply_visual_theme(model)
        return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RM65-B endoscope scene in MuJoCo.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Only load the model and print a summary.")
    mode.add_argument("--capture-only", action="store_true", help="Save stereo images and projector metadata, then exit.")
    parser.add_argument("--no-ui", action="store_true", help="Disable the control panel.")
    parser.add_argument(
        "--camera",
        choices=tuple(CAMERA_NAMES),
        default=DEFAULT_ACTIVE_CAMERA_KEY,
        help="Select which camera head to preview or capture.",
    )
    parser.add_argument(
        "--projector-pattern",
        type=str,
        help="External PNG path used by the structured-light projector approximation.",
    )
    parser.add_argument(
        "--disable-projector",
        action="store_true",
        help="Disable the structured-light projector approximation while keeping the projector body visible.",
    )
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
) -> dict[str, object]:
    freeze_state(data, qpos)

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
    model.body_pos[target_body_id] = target_position(config)

    focal_mx, focal_my, principal_mx, principal_my = camera_metric_intrinsics(config)
    for camera_key, offset in head_body_offsets(config).items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY_NAMES[camera_key])
        model.body_pos[body_id] = offset

    crossbar_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rig_crossbar")
    model.geom_size[crossbar_geom_id] = [
        BRACKET_CROSSBAR_HALF_THICKNESS,
        crossbar_half_span(config),
        BRACKET_CROSSBAR_HALF_HEIGHT,
    ]
    apply_projector_runtime_config(model, config)

    for camera_key, camera_name in CAMERA_NAMES.items():
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        model.cam_pos[camera_id] = camera_local_position(config)
        model.cam_quat[camera_id] = camera_quaternion(config, camera_key)
        model.cam_fovy[camera_id] = config.fovy_deg
        model.cam_sensorsize[camera_id] = [config.sensor_width_m, config.sensor_height_m]
        model.cam_intrinsic[camera_id] = [focal_mx, focal_my, principal_mx, principal_my]

    mujoco.mj_forward(model, data)
    return update_projector_approximation(model, data, config)


def endoscope_camera_parameters(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EndoscopeCameraConfig,
) -> dict[str, dict[str, float | list[float] | str]]:
    fx, fy, cx, cy = camera_pixel_intrinsics(config)
    parameters: dict[str, dict[str, float | list[float] | str]] = {}
    for camera_key, camera_name in CAMERA_NAMES.items():
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        parameters[camera_key] = {
            "name": camera_name,
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
    return parameters


def projector_parameters(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    projector_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PROJECTOR_BODY_NAME)
    return {
        "name": PROJECTOR_BODY_NAME,
        "world_position": np.array(data.xpos[projector_body_id], dtype=float).round(6).tolist(),
        "world_rotation_matrix": np.array(data.xmat[projector_body_id], dtype=float).round(6).reshape(3, 3).tolist(),
        "approximation_mode": "structured-light projector approximation",
    }


def render_endoscope_view(model: mujoco.MjModel, data: mujoco.MjData, camera_key: str) -> np.ndarray:
    width, height = CAMERA_RESOLUTION
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=CAMERA_NAMES[camera_key])
    image = renderer.render().copy()
    renderer.close()
    return image


def render_stereo_views(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    return {camera_key: render_endoscope_view(model, data, camera_key) for camera_key in CAMERA_NAMES}


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(path)


def save_structured_light_outputs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EndoscopeCameraConfig,
    active_camera_key: str,
) -> dict[str, Path]:
    # Capture both stereo images and persist the projector approximation state so
    # future structured-light steps can reuse the same file layout and metadata.
    outputs: dict[str, Path] = {}

    for camera_key, image in render_stereo_views(model, data).items():
        output_path = CAPTURE_DIR / f"{CAMERA_OUTPUT_STEMS[camera_key]}.png"
        save_png(output_path, image)
        outputs[camera_key] = output_path

    approximation_image, projector_state = build_projector_approximation(model, data, config)
    approximation_path = CAPTURE_DIR / "projector_approximation.png"
    save_png(approximation_path, approximation_image)
    outputs["projector_approximation"] = approximation_path

    metadata_path = CAPTURE_DIR / "structured_light_state.json"
    metadata = {
        "active_camera": active_camera_key,
        "camera_labels": CAMERA_LABELS,
        "config": asdict(config),
        "projector_state": projector_state,
        "approximation_mode": "structured-light projector approximation",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["metadata"] = metadata_path
    return outputs


def overlay_endoscope_view(
    handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EndoscopeCameraConfig,
    camera_key: str,
) -> None:
    image = render_endoscope_view(model, data, camera_key)
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
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Active Camera", CAMERA_LABELS[camera_key]),
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "Projector",
                f"{'ON' if config.projector_enable else 'OFF'} / {config.projector_fovy_deg:.1f} deg",
            ),
            (
                None,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "Rig Offset",
                f"{config.camera_x_m:.3f}, {config.camera_y_m:.3f}, {config.camera_z_m:.3f} m",
            ),
            (
                None,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "Head Spacing",
                f"{config.head_spacing_m * 1000.0:.1f} mm",
            ),
        ]
    )


def print_runtime_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    config: EndoscopeCameraConfig,
    active_camera_key: str,
    projector_state: dict[str, object],
) -> None:
    fx, fy, cx, cy = camera_pixel_intrinsics(config)
    print(f"Loaded model from {URDF_PATH}")
    print(
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"nbody={model.nbody}, ngeom={model.ngeom}, nmesh={model.nmesh}"
    )
    print(f"qpos={qpos.tolist()}")
    print(f"active_camera={CAMERA_LABELS[active_camera_key]}")
    print(f"head_spacing_m={config.head_spacing_m}")
    print(f"head_toe_in_deg={config.head_toe_in_deg}")
    print(f"camera_local_position_m={camera_local_position(config).round(6).tolist()}")
    print(f"camera_fovy_deg={config.fovy_deg}")
    print(f"camera_sensor_size_m={[config.sensor_width_m, config.sensor_height_m]}")
    print(
        "camera_intrinsics_pixels="
        f"{{'fx': {fx:.6f}, 'fy': {fy:.6f}, 'cx': {cx:.6f}, 'cy': {cy:.6f}}}"
    )
    print(f"camera_shared_yaw_pitch_roll_deg={[config.yaw_deg, config.pitch_deg, config.roll_deg]}")
    print(f"projector_local_position_m={projector_local_position(config).round(6).tolist()}")
    print(
        "projector_yaw_pitch_roll_deg="
        f"{[config.projector_yaw_deg, config.projector_pitch_deg, config.projector_roll_deg]}"
    )
    print(f"projector_fovy_deg={config.projector_fovy_deg}")
    print(f"projector_pattern_path={config.projector_pattern_path}")
    print(f"projector_enable={config.projector_enable}")
    print(f"target_position_m={target_position(config).round(6).tolist()}")
    print(f"target_sphere_center_m={target_sphere_center(config).round(6).tolist()}")
    print(f"cameras={endoscope_camera_parameters(model, data, config)}")
    print(f"projector={projector_parameters(model, data)}")
    print(f"projector_state={projector_state}")


class EndoscopeControlPanel:
    def __init__(self, config: EndoscopeCameraConfig, qpos: np.ndarray, active_camera_key: str):
        import tkinter as tk
        from tkinter import filedialog

        self._tk = tk
        self._filedialog = filedialog
        self._root = tk.Tk()
        self._root.title("Endoscope Camera Controls")
        self._root.geometry("620x1380")
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
        self._active_camera_var = tk.StringVar(value=active_camera_key)
        self._projector_enable_var = tk.BooleanVar(value=config.projector_enable)
        self._projector_pattern_var = tk.StringVar(value=config.projector_pattern_path)
        self._defaults = config
        self._default_joint_deg = joint_degrees_from_qpos(qpos)
        self._base_qpos = qpos.copy()

        self._active_camera_var.trace_add("write", self._mark_dirty)
        self._projector_enable_var.trace_add("write", self._mark_dirty)
        self._projector_pattern_var.trace_add("write", self._mark_dirty)
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
            ("head_spacing_mm", "Head Spacing (mm)", 40.0, 100.0, 1.0, config.head_spacing_m * 1000.0),
            ("camera_x_mm", "Camera X (mm)", -80.0, 80.0, 1.0, config.camera_x_m * 1000.0),
            ("camera_y_mm", "Camera Y (mm)", -80.0, 80.0, 1.0, config.camera_y_m * 1000.0),
            ("camera_z_mm", "Camera Z (mm)", 200.0, 600.0, 1.0, config.camera_z_m * 1000.0),
            ("fovy_deg", "Vertical FOV (deg)", 20.0, 110.0, 0.5, config.fovy_deg),
            ("yaw_deg", "Shared Yaw (deg)", -45.0, 90.0, 0.5, config.yaw_deg),
            ("toe_in_deg", "Head Toe-In (deg)", 0.0, 45.0, 0.5, config.head_toe_in_deg),
            ("pitch_deg", "Camera Pitch (deg)", -45.0, 45.0, 0.5, config.pitch_deg),
            ("roll_deg", "Camera Roll (deg)", -180.0, 180.0, 1.0, config.roll_deg),
            ("sensor_w_mm", "Sensor Width (mm)", 2.0, 12.0, 0.1, config.sensor_width_m * 1000.0),
            ("sensor_h_mm", "Sensor Height (mm)", 2.0, 12.0, 0.1, config.sensor_height_m * 1000.0),
            ("projector_x_mm", "Projector X (mm)", -40.0, 40.0, 1.0, config.projector_x_m * 1000.0),
            ("projector_y_mm", "Projector Y (mm)", -25.0, 25.0, 1.0, config.projector_y_m * 1000.0),
            ("projector_z_mm", "Projector Z (mm)", 35.0, 110.0, 1.0, config.projector_z_m * 1000.0),
            ("projector_yaw_deg", "Projector Yaw (deg)", -30.0, 30.0, 0.5, config.projector_yaw_deg),
            ("projector_pitch_deg", "Projector Pitch (deg)", -45.0, 45.0, 0.5, config.projector_pitch_deg),
            ("projector_roll_deg", "Projector Roll (deg)", -180.0, 180.0, 1.0, config.projector_roll_deg),
            ("projector_fovy_deg", "Projector FOV (deg)", 20.0, 120.0, 0.5, config.projector_fovy_deg),
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
                length=540,
                variable=var,
            ).grid(row=row * 2 + 1, column=0, sticky="we", pady=(0, 8))

        camera_select_row = len(controls) * 2
        tk.Label(frame, text="Active View", anchor="w").grid(row=camera_select_row, column=0, sticky="w", pady=(0, 4))
        select_frame = tk.Frame(frame)
        select_frame.grid(row=camera_select_row + 1, column=0, sticky="w", pady=(0, 8))
        for camera_key in CAMERA_NAMES:
            tk.Radiobutton(
                select_frame,
                text=CAMERA_LABELS[camera_key],
                value=camera_key,
                variable=self._active_camera_var,
            ).pack(side="left", padx=(0, 10))

        projector_row = camera_select_row + 2
        tk.Checkbutton(
            frame,
            text="Enable Structured-Light Projector Approximation (Optional)",
            variable=self._projector_enable_var,
        ).grid(row=projector_row, column=0, sticky="w", pady=(2, 6))

        pattern_frame = tk.Frame(frame)
        pattern_frame.grid(row=projector_row + 1, column=0, sticky="we", pady=(0, 8))
        tk.Label(pattern_frame, text="Projector Pattern PNG", anchor="w").pack(side="left")
        tk.Entry(pattern_frame, textvariable=self._projector_pattern_var, width=48).pack(side="left", padx=(8, 8))
        tk.Button(pattern_frame, text="Browse", command=self._browse_projector_pattern, width=10).pack(side="left")

        button_frame = tk.Frame(frame, pady=8)
        button_frame.grid(row=projector_row + 2, column=0, sticky="we")

        tk.Button(button_frame, text="Save PNG", command=self.request_capture, width=14).pack(side="left")
        tk.Button(button_frame, text="Reset Defaults", command=self.reset, width=16).pack(side="left", padx=8)
        tk.Button(button_frame, text="Close", command=self._on_close, width=10).pack(side="right")

        tk.Label(
            frame,
            text=(
                "Two cameras and one projector are mounted on the end-effector bracket.\n"
                "The bracket fixes the left and right cameras plus the projector.\n"
                "Each endoscope is mounted coaxially in front of its camera through a ring connector.\n"
                "The projector stays above the stereo pair on a short bracket. Leave approximation off if you only need placement."
            ),
            justify="left",
            anchor="w",
        ).grid(row=projector_row + 3, column=0, sticky="we", pady=(4, 8))

        tk.Label(frame, textvariable=self._status_var, justify="left", anchor="w", font=("Consolas", 10)).grid(
            row=projector_row + 4, column=0, sticky="we"
        )

    def _browse_projector_pattern(self) -> None:
        path = self._filedialog.askopenfilename(
            title="Select Projector Pattern PNG",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if path:
            self._projector_pattern_var.set(path)

    def _refresh_status(self) -> None:
        config = self.current_config()
        joint_deg = self.current_joint_deg()
        fx, fy, cx, cy = camera_pixel_intrinsics(config)
        camera_pos = camera_local_position(config)
        projector_pos = projector_local_position(config)
        sphere = target_sphere_center(config)
        resolved_pattern = resolve_projector_pattern_path(config.projector_pattern_path)
        self._status_var.set(
            "\n".join(
                [
                    f"Active camera: {CAMERA_LABELS[self.current_active_camera_key()]}",
                    "Joints: " + ", ".join(f"{name}={deg:.1f}" for name, deg in zip(JOINT_NAMES, joint_deg)),
                    f"Resolution: {CAMERA_RESOLUTION[0]} x {CAMERA_RESOLUTION[1]}",
                    f"fx={fx:.2f}px  fy={fy:.2f}px",
                    f"cx={cx:.1f}px  cy={cy:.1f}px",
                    f"Head spacing: {config.head_spacing_m * 1000.0:.1f} mm",
                    f"Toe-in: {config.head_toe_in_deg:.1f} deg",
                    f"Camera local pos: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}] m",
                    f"Projector local pos: [{projector_pos[0]:.3f}, {projector_pos[1]:.3f}, {projector_pos[2]:.3f}] m",
                    "Head layout: bracket -> camera -> ring connector -> endoscope",
                    "Projector mount: bracket-fixed above both camera modules",
                    f"Projector midline offset Y: {config.projector_y_m * 1000.0:.1f} mm",
                    f"Projector: {'ON' if config.projector_enable else 'OFF'}  FOV={config.projector_fovy_deg:.1f} deg",
                    f"Projector pattern: {resolved_pattern if resolved_pattern is not None else 'missing / not set'}",
                    f"Target sphere: [{sphere[0]:.3f}, {sphere[1]:.3f}, {sphere[2]:.3f}] m",
                ]
            )
        )

    def current_config(self) -> EndoscopeCameraConfig:
        return EndoscopeCameraConfig(
            head_spacing_m=self._vars["head_spacing_mm"].get() / 1000.0,
            head_toe_in_deg=self._vars["toe_in_deg"].get(),
            camera_x_m=self._vars["camera_x_mm"].get() / 1000.0,
            camera_y_m=self._vars["camera_y_mm"].get() / 1000.0,
            camera_z_m=self._vars["camera_z_mm"].get() / 1000.0,
            fovy_deg=self._vars["fovy_deg"].get(),
            yaw_deg=self._vars["yaw_deg"].get(),
            pitch_deg=self._vars["pitch_deg"].get(),
            roll_deg=self._vars["roll_deg"].get(),
            sensor_width_m=self._vars["sensor_w_mm"].get() / 1000.0,
            sensor_height_m=self._vars["sensor_h_mm"].get() / 1000.0,
            projector_x_m=self._vars["projector_x_mm"].get() / 1000.0,
            projector_y_m=self._vars["projector_y_mm"].get() / 1000.0,
            projector_z_m=self._vars["projector_z_mm"].get() / 1000.0,
            projector_yaw_deg=self._vars["projector_yaw_deg"].get(),
            projector_pitch_deg=self._vars["projector_pitch_deg"].get(),
            projector_roll_deg=self._vars["projector_roll_deg"].get(),
            projector_fovy_deg=self._vars["projector_fovy_deg"].get(),
            projector_pattern_path=self._projector_pattern_var.get().strip(),
            projector_enable=self._projector_enable_var.get(),
            target_x_m=self._vars["target_x_mm"].get() / 1000.0,
            target_y_m=self._vars["target_y_mm"].get() / 1000.0,
            target_z_m=self._vars["target_z_mm"].get() / 1000.0,
        )

    def current_active_camera_key(self) -> str:
        return self._active_camera_var.get()

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
        self._vars["head_spacing_mm"].set(defaults.head_spacing_m * 1000.0)
        self._vars["toe_in_deg"].set(defaults.head_toe_in_deg)
        self._vars["camera_x_mm"].set(defaults.camera_x_m * 1000.0)
        self._vars["camera_y_mm"].set(defaults.camera_y_m * 1000.0)
        self._vars["camera_z_mm"].set(defaults.camera_z_m * 1000.0)
        self._vars["fovy_deg"].set(defaults.fovy_deg)
        self._vars["yaw_deg"].set(defaults.yaw_deg)
        self._vars["pitch_deg"].set(defaults.pitch_deg)
        self._vars["roll_deg"].set(defaults.roll_deg)
        self._vars["sensor_w_mm"].set(defaults.sensor_width_m * 1000.0)
        self._vars["sensor_h_mm"].set(defaults.sensor_height_m * 1000.0)
        self._vars["projector_x_mm"].set(defaults.projector_x_m * 1000.0)
        self._vars["projector_y_mm"].set(defaults.projector_y_m * 1000.0)
        self._vars["projector_z_mm"].set(defaults.projector_z_m * 1000.0)
        self._vars["projector_yaw_deg"].set(defaults.projector_yaw_deg)
        self._vars["projector_pitch_deg"].set(defaults.projector_pitch_deg)
        self._vars["projector_roll_deg"].set(defaults.projector_roll_deg)
        self._vars["projector_fovy_deg"].set(defaults.projector_fovy_deg)
        self._projector_pattern_var.set(defaults.projector_pattern_path)
        self._projector_enable_var.set(defaults.projector_enable)
        self._vars["target_x_mm"].set(defaults.target_x_m * 1000.0)
        self._vars["target_y_mm"].set(defaults.target_y_m * 1000.0)
        self._vars["target_z_mm"].set(defaults.target_z_m * 1000.0)
        self._active_camera_var.set(DEFAULT_ACTIVE_CAMERA_KEY)
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

    config_dict = asdict(DEFAULT_CONFIG)
    if args.projector_pattern:
        config_dict["projector_pattern_path"] = args.projector_pattern
        config_dict["projector_enable"] = True
    if args.disable_projector:
        config_dict["projector_enable"] = False
    config = EndoscopeCameraConfig(**config_dict)
    active_camera_key = args.camera
    projector_state = apply_runtime_config(model, data, current_qpos, config)

    if args.check_only:
        print_runtime_summary(model, data, current_qpos, config, active_camera_key, projector_state)
        return

    if args.capture_only:
        output_paths = save_structured_light_outputs(model, data, config, active_camera_key)
        for key, output_path in output_paths.items():
            print(f"{key} saved to {output_path}")
        print_runtime_summary(model, data, current_qpos, config, active_camera_key, projector_state)
        return

    from mujoco import viewer

    control_panel = None if args.no_ui else EndoscopeControlPanel(config, current_qpos, active_camera_key)

    with viewer.launch_passive(model, data) as handle:
        overlay_endoscope_view(handle, model, data, config, active_camera_key)
        print_runtime_summary(model, data, current_qpos, config, active_camera_key, projector_state)

        while handle.is_running():
            if control_panel is not None:
                if not control_panel.update():
                    handle.close()
                    break
                if control_panel.consume_dirty():
                    config = control_panel.current_config()
                    active_camera_key = control_panel.current_active_camera_key()
                    current_qpos = control_panel.current_qpos()
                    projector_state = apply_runtime_config(model, data, current_qpos, config)
                    overlay_endoscope_view(handle, model, data, config, active_camera_key)
                if control_panel.consume_capture_request():
                    projector_state = apply_runtime_config(model, data, current_qpos, config)
                    output_paths = save_structured_light_outputs(model, data, config, active_camera_key)
                    for key, output_path in output_paths.items():
                        print(f"{key} saved to {output_path}")
                    print_runtime_summary(model, data, current_qpos, config, active_camera_key, projector_state)

            freeze_state(data, current_qpos)
            mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(args.dt)

    if control_panel is not None:
        control_panel.close()


if __name__ == "__main__":
    main()
