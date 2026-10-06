"""
Golden Sentinel - articulated mechanical face builder (Blender 4.2+ / 5.x).

Rebuilds the Meshy "Golden Sentinel" bust with every armour plate as its own
object. Plates hang under ~45 functional mechanical groups (brows, lids, cheeks,
forehead tiers, temples, fins, mask ...). Each group is a pair of empties:

    GRP_<name>.rest   fixed: sits at the group pivot, oriented to the group frame
                      (X lateral, Y up-the-surface, Z out of the plate)
    GRP_<name>        driven: local loc/rot = clamp(sum(control * gain))

The high level expression controls live as custom properties on the CTRL_FACE
empty (brow_l_raise, eye_l_open, cheek_l_raise, smile, surprise, concern,
jaw_open, head_yaw, ...).  The same rig.json drives the Godot runtime, so the
Blender drivers and the runtime evaluator share one definition.

The geometry is the untouched Meshy mesh (same vertices, same per-corner
normals); it is only split along the plate creases found by ../pipeline.

Run inside Blender:

    import runpy
    runpy.run_path("/path/to/build_robot_head.py", init_globals={
        "OBJ_PATH":  "/path/to/Meshy_AI_Golden_Sentinel_1006194324_generate.obj",
        "DATA_DIR":  "/path/to/Robot-head/data",
    })
"""
import bpy, os, json, lzma, io, math, time
import numpy as np
from mathutils import Matrix, Vector, Euler

OBJ_PATH = globals().get("OBJ_PATH", "")
DATA_DIR = globals().get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
BUILD_OPTS = globals().get("BUILD_OPTS", {})

COLL_NAME = "GoldenSentinel"


def log(*a):
    print("[GoldenSentinel]", *a)


# --------------------------------------------------------------------------- io

def read_obj(path):
    """Parse the Meshy OBJ in file order (identical to the segmentation pipeline)."""
    with open(path, "rb") as fh:
        lines = fh.read().split(b"\n")
    V = np.array(b" ".join(l[2:] for l in lines if l[:2] == b"v ").split(), dtype=np.float64).reshape(-1, 3)
    VN = np.array(b" ".join(l[3:] for l in lines if l[:3] == b"vn ").split(), dtype=np.float64).reshape(-1, 3)
    toks = b" ".join(l[2:] for l in lines if l[:2] == b"f ").replace(b"//", b" ").split()
    FF = np.array(toks, dtype=np.int64).reshape(-1, 6) - 1
    return V, FF[:, 0::2], VN, FF[:, 1::2]


def to_world(a):
    """Meshy OBJ is Y-up / faces +Z.  Blender: Z-up / faces -Y (same as the OBJ importer)."""
    a = np.asarray(a)
    return np.stack([a[..., 0], -a[..., 2], a[..., 1]], -1)


def load_labels(data_dir):
    with lzma.open(os.path.join(data_dir, "face_labels.u16.xz"), "rb") as fh:
        return np.frombuffer(fh.read(), dtype=np.uint16).astype(np.int64)


def load_core(data_dir):
    p = os.path.join(data_dir, "core.npz.xz")
    if not os.path.exists(p):
        return None
    with lzma.open(p, "rb") as fh:
        z = np.load(io.BytesIO(fh.read()))
        return z["V"].astype(np.float64), z["F"].astype(np.int64)


# --------------------------------------------------------------------- helpers

def get_collection(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def wipe_previous():
    root = bpy.data.collections.get(COLL_NAME)
    if root is None:
        return

    def walk(c):
        yield c
        for ch in c.children:
            yield from walk(ch)

    colls = list(walk(root))
    for c in colls:
        for ob in list(c.objects):
            data = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if data is not None and data.users == 0:
                for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
                    if data.name in coll and coll[data.name] == data:
                        coll.remove(data)
                        break
    for c in reversed(colls):
        bpy.data.collections.remove(c)
    for a in list(bpy.data.actions):
        if a.name.startswith("GS_"):
            bpy.data.actions.remove(a)


def principled(name, color, metallic=0.0, roughness=0.5, emission=None, strength=0.0, coat=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes.get("Principled BSDF")
    if b is None:
        nt.nodes.clear()
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(b.outputs[0], out.inputs[0])
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    if coat and "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    if emission is not None:
        key = "Emission Color" if "Emission Color" in b.inputs else "Emission"
        b.inputs[key].default_value = (*emission, 1)
        b.inputs["Emission Strength"].default_value = strength
    m.diffuse_color = (*color, 1)
    return m


def make_materials():
    return {
        "gold": principled("GS_Gold", (0.98, 0.64, 0.06), metallic=0.55, roughness=0.34, coat=0.6),
        "dark": principled("GS_DarkMech", (0.040, 0.028, 0.020), metallic=0.6, roughness=0.45),
        "brow": principled("GS_BrowBronze", (0.11, 0.040, 0.016), metallic=0.5, roughness=0.32, coat=0.5),
        "eye": principled("GS_EyeGlow", (1.0, 0.96, 0.86), roughness=0.2, emission=(1.0, 0.92, 0.70), strength=14.0),
        "collar": principled("GS_ShirtWhite", (0.86, 0.84, 0.79), roughness=0.75),
        "jacket": principled("GS_JacketBrown", (0.085, 0.036, 0.016), roughness=0.66),
        "plinth": principled("GS_Plinth", (0.06, 0.055, 0.05), metallic=0.2, roughness=0.5),
    }


CLASS_MATS = {  # (outer surface, inner shell)
    "plate": ("gold", "dark"), "neck_plate": ("gold", "dark"), "ear_disc": ("gold", "dark"),
    "ear_fin": ("gold", "dark"), "ear_hook": ("gold", "dark"), "brow": ("brow", "dark"),
    "under": ("dark", "dark"), "eye": ("eye", "dark"),
    "collar": ("collar", "collar"), "shirt": ("collar", "collar"), "jacket": ("jacket", "jacket"),
    "plinth": ("plinth", "plinth"),
}
SOLID_CLASSES = {"plate", "neck_plate", "ear_disc", "ear_fin", "ear_hook", "brow", "under", "eye"}
COLL_FOR = {"brow": "Brows", "eye": "Eyes", "ear_disc": "Ears", "ear_fin": "Ears", "ear_hook": "Ears",
            "neck_plate": "Neck", "under": "UnderLayer", "collar": "Body", "jacket": "Body",
            "shirt": "Body", "plinth": "Body"}


def build_mesh(name, Vw, F, loop_normals):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(Vw))
    me.vertices.foreach_set("co", Vw.astype(np.float32).ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(F), dtype=bool))
    if hasattr(me, "use_auto_smooth"):  # Blender < 4.1
        me.use_auto_smooth = True
    ln = loop_normals / (np.linalg.norm(loop_normals, axis=1, keepdims=True) + 1e-12)
    me.normals_split_custom_set(ln.astype(np.float32).tolist())
    return me


def new_empty(name, coll, display="PLAIN_AXES", size=0.03):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = display
    ob.empty_display_size = size
    coll.objects.link(ob)
    return ob


def place(ob, parent, parent_world, world):
    """Parent `ob` and give it world matrix `world` (parent world matrix passed explicitly)."""
    ob.parent = parent
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = parent_world.inverted() @ world if parent is not None else world


# ----------------------------------------------------------------------- build

def build(obj_path=OBJ_PATH, data_dir=DATA_DIR, opts=BUILD_OPTS):
    t0 = time.time()
    data_dir = os.path.abspath(data_dir)
    regions = json.load(open(os.path.join(data_dir, "regions.json")))
    rig = json.load(open(os.path.join(data_dir, "rig.json")))
    labels = load_labels(data_dir)
    V, F, VN, FN = read_obj(obj_path)
    assert len(labels) == len(F), f"label count {len(labels)} != faces {len(F)} - wrong OBJ?"
    chk = regions.get("_checksum")
    if chk is not None:
        s = float(np.round(V[:: max(1, len(V) // 997)].sum(), 3))
        assert abs(s - chk) < 1e-2, f"OBJ checksum mismatch {s} vs {chk} - this is not the Golden Sentinel OBJ"
    Vw, NW = to_world(V), to_world(VN)
    log("parsed", len(V), "verts", len(F), "faces", round(time.time() - t0, 1), "s")

    wipe_previous()
    scene = bpy.context.scene
    root = get_collection(COLL_NAME)
    sub = {k: get_collection(f"GS_{k}", root) for k in
           ("Controls", "Groups", "Plates", "Brows", "Eyes", "Ears", "Neck", "UnderLayer", "Body", "Stage")}
    M = make_materials()

    # ------------------------------------------------------------- hierarchy
    I = Matrix.Identity(4)
    head = rig["head"]
    W = {}  # world matrices of created nodes (rest pose)
    rig_root = new_empty("RIG_ROOT", sub["Controls"], "ARROWS", 0.15)
    W["ROOT"] = I
    neck = new_empty("NECK", sub["Controls"], "SPHERE", 0.05)
    W["NECK"] = Matrix.Translation(Vector(head["neck_pivot"]))
    place(neck, rig_root, W["ROOT"], W["NECK"])
    head_root = new_empty("HEAD_ROOT", sub["Controls"], "SPHERE", 0.08)
    W["HEAD"] = Matrix.Translation(Vector(head["pivot"]))
    place(head_root, neck, W["NECK"], W["HEAD"])
    nodes = {"ROOT": rig_root, "NECK": neck, "HEAD": head_root}

    ctrl = new_empty("CTRL_FACE", sub["Controls"], "CIRCLE", 0.12)
    place(ctrl, rig_root, W["ROOT"], Matrix.Translation((0.75, -0.6, 0.35)) @ Euler((math.pi / 2, 0, 0)).to_matrix().to_4x4())
    for c in rig["controls"]:
        ctrl[c["name"]] = float(c.get("neutral", 0.0))
        try:
            ctrl.id_properties_ui(c["name"]).update(min=c["range"][0], max=c["range"][1], soft_min=c["range"][0],
                                                   soft_max=c["range"][1], default=float(c.get("neutral", 0.0)))
        except Exception:
            pass

    group_nodes = {}
    for g in rig["groups"]:
        fr = Matrix([g["frame"][0], g["frame"][1], g["frame"][2]]).transposed()  # columns = axes
        rest_w = Matrix.Translation(Vector(g["pivot"])) @ fr.to_4x4()
        rest = new_empty(f"GRP_{g['name']}.rest", sub["Groups"], "PLAIN_AXES", 0.012)
        place(rest, nodes[g.get("parent", "HEAD")], W[g.get("parent", "HEAD")], rest_w)
        drv = new_empty(f"GRP_{g['name']}", sub["Groups"], "ARROWS" if g.get("designed") else "PLAIN_AXES", 0.03 if g.get("designed") else 0.015)
        place(drv, rest, rest_w, rest_w)
        drv.rotation_mode = "XYZ"
        drv["gs_group"] = g["name"]
        group_nodes[g["name"]] = (drv, rest_w)
    log("groups", len(group_nodes))

    # ------------------------------------------------------------- plates
    order = np.argsort(labels, kind="stable")
    starts = np.searchsorted(labels[order], np.arange(labels.max() + 2))
    thick = opts.get("plate_thickness", rig.get("plate_thickness", 0.008))
    made = 0
    for r in regions["regions"]:
        rid = r["id"]
        fidx = order[starts[rid]:starts[rid + 1]]
        if len(fidx) == 0:
            continue
        Fr = F[fidx]
        uv, inv = np.unique(Fr, return_inverse=True)
        pivot = np.array(r["pivot"])
        me = build_mesh(r["name"], Vw[uv] - pivot, inv.reshape(-1, 3), NW[FN[fidx]].reshape(-1, 3))
        outer, inner = CLASS_MATS.get(r["cls"], ("gold", "dark"))
        me.materials.append(M[outer])
        me.materials.append(M[inner])
        ob = bpy.data.objects.new(r["name"], me)
        sub[COLL_FOR.get(r["cls"], "Plates")].objects.link(ob)
        ob["gs_region"] = rid
        ob["gs_class"] = r["cls"]
        if r["cls"] in SOLID_CLASSES and opts.get("solidify", True):
            sm = ob.modifiers.new("PlateBacking", "SOLIDIFY")
            sm.thickness = thick
            sm.offset = -1.0
            sm.use_even_offset = False      # even-offset spikes on Meshy sliver triangles
            sm.thickness_clamp = 0.5
            sm.use_rim = True
            sm.material_offset = 1          # inner shell -> dark mech
            sm.material_offset_rim = 0      # rim keeps the plate material
        gnode, gw = group_nodes[r["group"]]
        place(ob, gnode, gw, Matrix.Translation(Vector(pivot)))
        made += 1
    log("plates", made, round(time.time() - t0, 1), "s")

    core = load_core(data_dir)
    if core is not None:
        cV, cF = core
        cme = bpy.data.meshes.new("GS_InnerCore")
        cme.from_pydata(to_world(cV).tolist(), [], cF.tolist())
        cme.polygons.foreach_set("use_smooth", np.ones(len(cF), dtype=bool))
        cme.materials.append(M["dark"])
        cob = bpy.data.objects.new("GS_InnerCore", cme)
        sub["UnderLayer"].objects.link(cob)
        place(cob, head_root, W["HEAD"], I)
        cob.hide_select = True

    setup_drivers(rig, ctrl, group_nodes, neck, head_root)
    if opts.get("stage", True):
        setup_stage(scene, sub["Stage"], rig)
    if opts.get("demo_animation", True):
        setup_expression_reel(rig, ctrl, scene)
    log("done", round(time.time() - t0, 1), "s")
    return ctrl


# ----------------------------------------------------------------------- rig

def _driver(owner, path, index, terms, lo, hi, ctrl):
    """terms: list of (control_name, neutral, gain).  value = clamp(sum(gain*(c-neutral)), lo, hi)."""
    fc = owner.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    parts = []
    for i, (cname, neutral, gain) in enumerate(terms):
        v = d.variables.new()
        v.name = f"v{i}"
        v.type = "SINGLE_PROP"
        v.targets[0].id_type = "OBJECT"
        v.targets[0].id = ctrl
        v.targets[0].data_path = f'["{cname}"]'
        parts.append(f"{gain:.6g}*(v{i}-{neutral:.4g})" if neutral else f"{gain:.6g}*v{i}")
    expr = "+".join(parts) if parts else "0"
    d.expression = f"clamp({expr},{lo:.6g},{hi:.6g})"
    return fc


def collect_terms(rig):
    """{group: {('loc'|'rot', axis): [(control, neutral, gain), ...]}}"""
    out = {}
    for c in rig["controls"]:
        for e in c.get("effects", []):
            for kind in ("loc", "rot"):
                vec = e.get(kind)
                if not vec:
                    continue
                for ax in range(3):
                    if abs(vec[ax]) < 1e-9:
                        continue
                    gain = vec[ax] if kind == "loc" else math.radians(vec[ax])
                    out.setdefault(e["group"], {}).setdefault((kind, ax), []).append((c["name"], c.get("neutral", 0.0), gain))
    return out


def setup_drivers(rig, ctrl, group_nodes, neck, head_root):
    lim = rig.get("limits", {"loc": 0.09, "rot_deg": 30})
    for gname, chans in collect_terms(rig).items():
        node = group_nodes[gname][0]
        for (kind, ax), terms in chans.items():
            if kind == "loc":
                _driver(node, "location", ax, terms, -lim["loc"], lim["loc"], ctrl)
            else:
                r = math.radians(lim["rot_deg"])
                _driver(node, "rotation_euler", ax, terms, -r, r, ctrl)
    # head / neck:  yaw -> Z, pitch (up +) -> -X, roll -> Y ; neck takes `neck_share`
    h = rig["head"]
    share = h.get("neck_share", 0.35)
    for node, k in ((neck, share), (head_root, 1.0 - share)):
        node.rotation_mode = "XYZ"
        for ax, cname, sign, key in ((2, "head_yaw", 1, "yaw"), (0, "head_pitch", -1, "pitch"), (1, "head_roll", 1, "roll")):
            lo, hi = [math.radians(x) * k for x in h["limits_deg"][key]]
            _driver(node, "rotation_euler", ax, [(cname, 0.0, sign * k)], min(lo, -hi), max(hi, -lo), ctrl)


def setup_stage(scene, coll, rig):
    st = rig.get("stage", {})
    cam_d = bpy.data.cameras.new("CAM_GoldenSentinel")
    cam_d.lens = st.get("lens", 85)
    cam = bpy.data.objects.new("CAM_GoldenSentinel", cam_d)
    coll.objects.link(cam)
    target = Vector(st.get("target", (0, 0, 0.22)))
    cam.location = Vector(st.get("cam", (1.55, -4.4, 0.62)))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    def light(name, loc, energy, color, size):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.color, ld.size = energy, color, size
        lo = bpy.data.objects.new(name, ld)
        coll.objects.link(lo)
        lo.location = Vector(loc)
        lo.rotation_euler = (target - lo.location).to_track_quat("-Z", "Y").to_euler()
        try:
            lo.visible_camera = False
        except Exception:
            pass

    light("LGT_Key", (-2.2, -2.8, 2.2), 260, (1.0, 0.93, 0.82), 2.0)
    light("LGT_Fill", (2.8, -2.4, 0.4), 70, (0.80, 0.86, 1.0), 3.0)
    light("LGT_Rim", (1.4, 2.6, 2.0), 300, (1.0, 0.86, 0.66), 1.5)
    light("LGT_RimL", (-2.2, 2.2, 0.9), 160, (0.9, 0.9, 1.0), 1.5)
    w = scene.world or bpy.data.worlds.new("GS_World")
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.03, 0.027, 0.025, 1)
        bg.inputs[1].default_value = 1.0
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass
    scene.render.resolution_x, scene.render.resolution_y = 1080, 1350


def apply_expression(rig, ctrl, name, weight=1.0):
    vals = rig["expressions"].get(name, {})
    for c in rig["controls"]:
        n0 = float(c.get("neutral", 0.0))
        ctrl[c["name"]] = n0 + (vals.get(c["name"], n0) - n0) * weight


def setup_expression_reel(rig, ctrl, scene, hold=24, blend=12):
    """Keyframe CTRL_FACE through every expression; a timeline marker names each pose."""
    f = 1
    names = list(rig["expressions"].keys())
    for m in list(scene.timeline_markers):
        if m.name in names:
            scene.timeline_markers.remove(m)
    for name in names + ["neutral"]:
        apply_expression(rig, ctrl, name)
        for c in rig["controls"]:
            ctrl.keyframe_insert(f'["{c["name"]}"]', frame=f, group="FaceControls")
            ctrl.keyframe_insert(f'["{c["name"]}"]', frame=f + hold, group="FaceControls")
        scene.timeline_markers.new(name, frame=f)
        f += hold + blend
    if ctrl.animation_data and ctrl.animation_data.action:
        ctrl.animation_data.action.name = "GS_ExpressionReel"
    scene.frame_start, scene.frame_end = 1, f - blend
    scene.frame_set(1)


if OBJ_PATH:
    build()
