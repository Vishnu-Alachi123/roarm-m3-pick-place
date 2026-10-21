"""
ROarm M3 Gripper Attachment — CadQuery Generator (v3)
=====================================================
Claw design: hollow triangular-prism shell (left + right + back walls, OPEN
front face) with horizontal rib slabs inside. The rib tips are the contact
surface. A centre post between each pair of adjacent ribs prevents mid-span
sag and gives the "little bit of rigidity in the middle" the design requires.

Cross-section (looking down, XY plane):
   back wall (+Y)
   ┌──────────────┐
   │              │   ← WALL_T thick
   └──────────────┘
   │  inner void  │   ← hollow (open to –Y)
   left wall      right wall
   (WALL_T thick) (WALL_T thick)
                  │
   –Y front: OPEN ↓  (rib tips here, contact the object)

Ribs are horizontal slabs (full inner width × full inner depth × RIB_T)
at N_RIBS evenly-spaced heights. Centre posts (width = CENTER_POST_W) run
between adjacent ribs at X = 0, spanning the full inter-rib gap.

Install:  pip install cadquery
Run:      python3 roarm_gripper_attachment.py
"""

import cadquery as cq

# ================================================================
# PARAMETERS  — caliper real arm and update FINGER_W / FINGER_D
# ================================================================
FINGER_W   = 16.0   # finger width  (X, mm)
FINGER_D   =  6.0   # finger depth  (Y, mm)
CLEARANCE  =  0.3   # clip fit clearance on all sides

WALL_SIDE  =  2.5   # collar left/right wall thickness
WALL_BACK  =  3.0   # collar back wall thickness
CLIP_H     = 18.0   # clip collar height

CLAW_H     = 55.0   # ribbed claw section height
TIP_W      =  9.0   # claw width at tip (Z = 0)

# Shell walls (claw body — left, right, back)
WALL_T     =  2.0   # mm

# Ribs
N_RIBS         =  5     # number of horizontal ribs
RIB_T          =  3.5   # rib slab thickness in Z (decrease → more flex)
RIB_MARGIN     =  7.0   # distance from tip/collar junction to outermost rib

# Centre post between each pair of adjacent ribs
CENTER_POST_W  =  2.5   # post width in X (mm)

# ================================================================
# DERIVED
# ================================================================
CLIP_W = FINGER_W + CLEARANCE + 2 * WALL_SIDE   # 21.8 mm
CLIP_D = FINGER_D + CLEARANCE + WALL_BACK        #  9.3 mm
TOTAL_H = CLAW_H + CLIP_H                         # 73.0 mm

CH_W  = FINGER_W + CLEARANCE    # finger channel width
CH_D  = FINGER_D + CLEARANCE    # finger channel depth
CH_CY = CLIP_D / 2 - WALL_BACK - CH_D / 2       # channel Y centre (≈ –1.5 mm)

# Interior dimensions (inside the hollow shell)
INNER_W_TIP = max(TIP_W  - 2 * WALL_T, 0.5)     #  5.0 mm at tip
INNER_W_TOP = CLIP_W - 2 * WALL_T                # 17.8 mm at top
INNER_D     = CLIP_D - WALL_T                    #  7.3 mm (front-to-back interior)


# ================================================================
# Part 1: Flexible TPU claw (moving jaw)
# ================================================================
def make_claw():
    """
    Hollow triangular-prism shell with horizontal ribs.

    Coordinate system
      Z = 0        → claw tip (narrowest point, bottom)
      Z = TOTAL_H  → top of clip collar
      –Y           → open front face (rib tips face the object here)
      +Y           → back wall (outer side)
    """

    # ── tapered outer solid ──────────────────────────────────────
    body = (
        cq.Workplane("XY")
        .rect(TIP_W, CLIP_D)
        .workplane(offset=CLAW_H)
        .rect(CLIP_W, CLIP_D)
        .loft()
    )

    # ── hollow out interior ──────────────────────────────────────
    # The inner void is the same taper but narrower in X (by 2×WALL_T on
    # each side) and shallower in Y (INNER_D instead of CLIP_D), shifted
    # WALL_T/2 toward the front (–Y) so WALL_T of back wall remains.
    # Both profiles are centred at (0, 0); the whole void is then translated.
    inner_void = (
        cq.Workplane("XY")
        .rect(INNER_W_TIP, INNER_D)
        .workplane(offset=CLAW_H)
        .rect(INNER_W_TOP, INNER_D)
        .loft()
    ).translate((0, -WALL_T / 2, 0))

    body = body.cut(inner_void)

    # ── horizontal ribs ─────────────────────────────────────────
    usable  = CLAW_H - 2 * RIB_MARGIN
    spacing = usable / max(N_RIBS - 1, 1)
    rib_z   = [RIB_MARGIN + i * spacing for i in range(N_RIBS)]

    for z_c in rib_z:
        t     = z_c / CLAW_H
        rib_w = INNER_W_TIP + (INNER_W_TOP - INNER_W_TIP) * t

        rib = (
            cq.Workplane("XY")
            .workplane(offset=z_c - RIB_T / 2)
            .center(0, -WALL_T / 2)     # interior Y centre
            .rect(rib_w, INNER_D)
            .extrude(RIB_T)
        )
        body = body.union(rib)

    # ── centre posts (mid-span rigidity between ribs) ────────────
    for i in range(N_RIBS - 1):
        z_lo  = rib_z[i]     + RIB_T / 2
        z_hi  = rib_z[i + 1] - RIB_T / 2
        z_gap = z_hi - z_lo
        if z_gap > 1.0:
            post = (
                cq.Workplane("XY")
                .workplane(offset=z_lo)
                .center(0, -WALL_T / 2)
                .rect(CENTER_POST_W, INNER_D)
                .extrude(z_gap)
            )
            body = body.union(post)

    # ── clip collar (full solid block) ───────────────────────────
    body = body.union(
        cq.Workplane("XY")
        .workplane(offset=CLAW_H - 0.1)
        .rect(CLIP_W, CLIP_D)
        .extrude(CLIP_H + 0.2)
    )

    # ── finger channel (open at Z = CLAW_H, slides onto finger) ──
    body = body.cut(
        cq.Workplane("XY")
        .workplane(offset=CLAW_H - 0.1)
        .center(0, CH_CY)
        .rect(CH_W, CH_D)
        .extrude(CLIP_H + 0.2)
    )

    return body


# ================================================================
# Part 2: Rigid PLA pad (fixed jaw)
# Bond a 1–2 mm rubber/silicone sheet to the flat inner face.
# ================================================================
def make_fixed_pad():
    PAD_H = 38.0   # contact face height

    body = (
        cq.Workplane("XY")
        .box(CLIP_W, CLIP_D, PAD_H + CLIP_H, centered=(True, True, False))
    )

    # Same finger channel as claw
    body = body.cut(
        cq.Workplane("XY")
        .workplane(offset=PAD_H - 0.1)
        .center(0, CH_CY)
        .rect(CH_W, CH_D)
        .extrude(CLIP_H + 0.2)
    )

    return body


# ================================================================
# Assembly: both parts facing each other as on the real arm
# ================================================================
def make_assembly(claw, pad, jaw_gap=25.0):
    """
    Rotate each part 90° so inner faces point toward each other,
    then translate outward by jaw_gap/2 + CLIP_D/2.
    """
    pad_pos = (
        pad
        .rotate((0, 0, 0), (0, 0, 1),  90)
        .translate((-jaw_gap / 2 - CLIP_D / 2, 0, 0))
    )
    claw_pos = (
        claw
        .rotate((0, 0, 0), (0, 0, 1), -90)
        .translate(( jaw_gap / 2 + CLIP_D / 2, 0, 0))
    )
    return pad_pos.union(claw_pos)


# ================================================================
# Export
# ================================================================
if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Building parts …")
    claw = make_claw()
    pad  = make_fixed_pad()
    asm  = make_assembly(claw, pad, jaw_gap=25.0)

    claw_path = os.path.join(out_dir, "roarm_claw_tpu.stl")
    pad_path  = os.path.join(out_dir, "roarm_fixed_pad_pla.stl")
    asm_path  = os.path.join(out_dir, "roarm_gripper_assembly.stl")

    cq.exporters.export(claw, claw_path)
    cq.exporters.export(pad,  pad_path)
    cq.exporters.export(asm,  asm_path)

    print("Done:")
    print(f"  {claw_path}")
    print(f"    → TPU 95A | 0 % infill | 2–3 walls | print upright")
    print(f"  {pad_path}")
    print(f"    → PLA | 20 % infill | standard | bond rubber to flat face")
    print(f"  {asm_path}")
    print(f"    → both parts assembled, jaw gap = 25 mm")
    print()
    print("Key tuning params:")
    print(f"  FINGER_W  = {FINGER_W} mm  ← caliper real arm, update if off")
    print(f"  FINGER_D  = {FINGER_D} mm")
    print(f"  CLEARANCE = {CLEARANCE} mm  ← increase if collar is too tight")
    print(f"  RIB_T     = {RIB_T} mm  ← decrease for more flex")
    print(f"  N_RIBS    = {N_RIBS}       ← increase for finer conformation")
