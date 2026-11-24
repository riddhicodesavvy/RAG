"""
Build a clean, synthetic, non-confidential technical manual corpus and
a fixed 105-question ground-truth benchmark for Enterprise RAG evaluation.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "benchmarks" / "datasets"
MANUALS_DIR = DATASET_DIR / "manuals"

MANUALS = {
    "MAN-HYD-101_Hydraulic_Press.txt": """# STANDARD OPERATING MANUAL: HYDRAULIC PRESS & CIRCUIT MAINTENANCE (DOC-ID: MAN-HYD-101)

[SEC-HYD-1] System Overview & Operating Pressure Envelope
The Series-4000 Heavy Forming Press operates using ISO VG 46 anti-wear hydraulic oil maintained between 38 C and 54 C during continuous duty cycles. Normal main cylinder working pressure is 21.5 MPa (3,118 psi). System over-pressure protection is governed by the proportional relief valve PRV-402-B, factory-calibrated to crack open at 24.0 MPa and reach full bypass flow at 25.2 MPa. Never adjust the locknut on PRV-402-B without a calibrated Class-0.5 digital pressure gauge connected to test port TP-01.

[SEC-HYD-2] Pump Cavitation Diagnostics & Suction Line Limits
Pump cavitation manifests as high-pitched metallic knocking above 82 dBA coupled with erratic ram advance speed. Vacuum at suction manifold gauge VG-02 must never exceed 17 kPa (5 inHg) under cold startup or 12 kPa at operating viscosity. Common root causes include clogged 100-mesh suction strainer STR-14, collapsed elastomeric inlet hose HOSE-IN-09, or reservoir oil level dropping below sight-glass mark L-MIN (185 liters). If cavitation persists after strainer replacement, inspect shaft seal ring SSR-22 for air ingress.

[SEC-HYD-3] Proportional Valve Replacement & Flushing Protocol
Before removing proportional directional valve PDV-118, depressurize accumulator ACC-04 by opening manual bleed needle valve NBV-03 three full turns counterclockwise and confirming zero reading on gauge PG-MAIN. Flush the manifold block using filtered ISO 4406 16/14/11 cleanliness flushing fluid for 25 minutes. Torque the four M10 Grade-12.9 mounting bolts of PDV-118 in a diagonal cross pattern to 68 Nm in two stages: first stage 34 Nm, final stage 68 Nm.

[SEC-HYD-4] Solenoid Coil Diagnostics (SOL-H88)
Directional pilot solenoid SOL-H88 operates on 24 VDC regulated supply with nominal coil resistance of 22.4 ohms at 20 C. Resistance readings below 18.0 ohms indicate inter-turn shorting, while readings above 30.0 ohms or infinite continuity indicate open winding failure. Minimum pull-in voltage under load is 20.4 VDC. Always verify flyback suppression diode D-41 across terminals A1 and A2 prior to energizing the PLC output card.

[SEC-HYD-5] Emergency Ram Descent & Thermal Shutdown Interlocks
Thermal switch TS-HIGH triggers an automatic pump motor trip when reservoir bulk temperature reaches 68 C, engaging latching relay K-9. To recover from a thermal trip, allow fluid to cool below 52 C, inspect shell-and-tube heat exchanger HEX-02 water flow rate (minimum 28 liters/min), and press reset button PB-RST-4. During power loss, counterbalance valve CBV-77 locks the upper platen mechanically; manual lowering requires turning override screw OVR-77 clockwise using a 6 mm hex key while maintaining visual clearance.
""",

    "MAN-WLD-204_Structural_Welding_QA.txt": """# STRUCTURAL WELDING QUALITY ASSURANCE SPECIFICATION (DOC-ID: MAN-WLD-204)

[SEC-WLD-1] Base Metal Preparation & Joint Fit-Up Tolerances
Structural carbon-steel girders conforming to ASTM A572 Grade 50 require bevel angles of 30 degrees (+5/-0 degrees) for single-V groove joints with a root opening of 3.0 mm (+/- 0.8 mm) and a root face of 1.6 mm. All mill scale, grease, paint, and moisture must be removed within 50 mm of the weld toe using power wire brushing or flap-disc grinding prior to tack welding. Tack welds showing cracks must be completely gouged out via carbon-arc gouging before root pass deposition.

[SEC-WLD-2] SMAW Consumable Control & Electrode Baking (E7018-H4R)
Low-hydrogen electrodes AWS E7018-H4R must remain sealed in hermetically crimped cans until point-of-use issue. Upon opening, electrodes shall be transferred immediately into portable holding ovens maintained between 120 C and 150 C. Any E7018-H4R electrode exposed to ambient workshop humidity (>60% RH) for more than 4.0 hours must be segregated and rebaked once only at 370 C for 2.0 hours. Wet or flaking flux coatings are permanently rejected and discarded into scrap bin BIN-RED-03.

[SEC-WLD-3] Preheat & Interpass Temperature Controls
For plate thicknesses between 25 mm and 38 mm on ASTM A572 Grade 50 steel, minimum preheat temperature prior to arc strike is 65 C, measured 75 mm laterally from the joint centerline using a calibrated contact thermocouple TC-08 or Tempilstik crayon. For thicknesses exceeding 38 mm, minimum preheat increases to 110 C. Maximum interpass temperature shall not exceed 230 C to preserve heat-affected zone (HAZ) Charpy V-notch impact toughness (minimum 27 J at -20 C).

[SEC-WLD-4] GTAW Root Pass Purge & Shielding Gas Purity
Stainless steel piping joints (ASTM A312 TP316L) welded via GTAW require internal argon back-purging using Purge Dam Kit PDK-60. Welding shall not commence until oxygen analyzer OA-900 confirms residual O2 concentration inside the pipe bore is below 0.05% (500 ppm). Primary torch shielding uses 99.995% pure Argon flowing at 14 to 18 L/min through a #7 alumina gas lens cup. Exceeding 22 L/min induces turbulent air aspiration and weld bead sugar-oxidation.

[SEC-WLD-5] Ultrasonic Nondestructive Testing (UT-SCAN-9) Acceptance Criteria
Completed full-penetration butt welds are cooled to ambient temperature (<40 C) and held for 24 hours (hydrogen-delayed cracking incubation period) before ultrasonic inspection per procedure UT-SCAN-9 using a 2.25 MHz, 60-degree shear-wave transducer. Any planar discontinuity exhibiting an amplitude exceeding +6 dB above the reference DAC curve (Class-A reject level), or slag inclusions exceeding 12 mm continuous length, mandates full excavation, re-welding under repair spec WPS-REP-02, and 100% re-inspection.
""",

    "MAN-ELE-309_LOTO_Substation.txt": """# HIGH-VOLTAGE SUBSTATION LOCKOUT / TAGOUT & PPE MANUAL (DOC-ID: MAN-ELE-309)

[SEC-ELE-1] Mandatory Lockout/Tagout Sequence (Procedure LOTO-77X)
All maintenance on 11 kV switchgear cubicles must follow the seven-step Zero Energy Verification protocol LOTO-77X: (1) notify control desk operator, (2) open downstream load feeders first, (3) rack out main vacuum circuit breaker VCB-11K to TEST/DISCONNECTED position, (4) apply personal padlock LOTO-PAD and red danger tag at mechanical racking shutter, (5) verify line-side dead state using calibrated live-line tester HV-DET-15KV tested before and after use (Live-Dead-Live test), (6) close integral earthing switch ES-11K, and (7) attach portable earthing cluster PEC-04.

[SEC-ELE-2] Arc Flash PPE Categories & Boundary Distances
Switchgear panel SWG-BUS-A carries an incident energy calculation of 14.8 cal/cm2 at a working distance of 450 mm, placing it in Arc Flash PPE Category 3 (minimum arc rating 25 cal/cm2). Authorized technicians inside the 1.85-meter arc flash boundary must wear multi-layer arc-rated coverall suit AR-25, balaclava hood with Class-2 tinted face shield, leather over-protectors, and hearing protection. Synthetic melting fabrics (polyester, nylon) are strictly prohibited under arc-rated garments.

[SEC-ELE-3] Dielectric Insulating Glove Inspection & Testing (CLASS-2-17KV)
Rubber insulating gloves designated CLASS-2-17KV are rated for maximum AC working voltage of 17,000 V (tested at 20,000 V proof voltage). Prior to every shift, perform a manual air-inflation roll test to detect micro-pinholes, ozone cracking, or embedded metallic splinters. Dielectric gloves must undergo laboratory electrical re-certification every 6 months per ASTM D120 / IEC 60903; any pair exceeding 180 days from its stamped test date stamped on cuff tag GLV-STAMP must be withdrawn from service immediately.

[SEC-ELE-4] Transformer Dissolved Gas Analysis (DGA) Action Limits
Oil-immersed 5 MVA step-down transformer TRF-5M undergoes semi-annual DGA sampling via bottom drain valve DV-01. Normal baseline total dissolved combustible gas (TDCG) is below 720 ppm. Acetylene (C2H2) concentrations rising above 35 ppm signify active internal high-energy arcing and mandate immediate off-line diagnostic testing (winding resistance and sweep frequency response analysis SFRA-200) before re-energization. Ethylene (C2H4) > 100 ppm indicates localized thermal overheating above 500 C.

[SEC-ELE-5] Battery Bank & DC Control Bus Maintenance
Substation 110 VDC trip-coil supply relies on 55 lead-acid VRLA cells (Model VRLA-110-AH) maintained at float charge voltage of 124.85 VDC (+/- 0.5%). Individual cell impedance rising >25% above installation baseline (2.15 mOhm) flags sulfation degradation requiring single-cell capacity discharge testing at C/10 rate (11 A). Technicians must use insulated 1000 V rated torque wrenches set to 14 Nm on inter-cell copper busbars to prevent catastrophic short-circuit faults.
""",

    "MAN-THM-415_Boiler_Feedwater.txt": """# INDUSTRIAL WATER-TUBE BOILER FEEDWATER CHEMISTRY MANUAL (DOC-ID: MAN-THM-415)

[SEC-THM-1] Deaerator & Dissolved Oxygen Scavenging
Steam boiler BLR-80T operates at 4.2 MPa drum pressure (saturation temperature 253 C). Feedwater leaving thermal deaerator DA-01 must maintain dissolved oxygen (DO) strictly below 7 ppb (0.007 mg/L) to arrest pitting corrosion in economizer tubes. Residual oxygen is scavenged chemically via continuous metered dosing of catalyzed carbohydrazide solution (SCAV-CHZ-10) into the deaerator storage drop-leg, targeting a residual carbohydrazide concentration of 50 to 100 ppb at the boiler economizer inlet.

[SEC-THM-2] Drum Water pH, Phosphate & Silica Limits
Boiler drum water pH must be maintained within 9.2 to 9.8 using coordinated tri-sodium phosphate (Na3PO4) dosing to eliminate caustic gouging and acid phosphate corrosion. Total drum silica (SiO2) must remain below 2.5 ppm at 4.2 MPa to prevent volatile silica carryover and glass-like deposition on downstream steam turbine blades. Cation conductivity of saturated steam sample point SS-03 must stay below 0.20 uS/cm; spikes above 0.50 uS/cm indicate condenser tube leakage.

[SEC-THM-3] Continuous & Intermittent Blowdown Sequencing (BDV-12A)
Drum total dissolved solids (TDS) are controlled below 800 ppm via automated continuous blowdown valve CBD-09 modulated by conductivity transmitter CT-401. Manual bottom mud-drum blowdown uses quick-opening/slow-closing tandem valves BDV-12A and BDV-12B every 8-hour shift for strictly 12 seconds duration. Sequence rule: open slow-closing valve BDV-12B fully first, cycle quick-opening valve BDV-12A open for 12 seconds, close BDV-12A first, then close BDV-12B last to absorb seat wear on the outer sacrificial valve.

[SEC-THM-4] Acid Cleaning & Passivation Criteria
Internal boiler tube chemical cleaning (inhibited 5% hydrochloric acid at 65 C) is triggered when waterside deposit weight density measured on crown-side tube samples exceeds 150 g/m2. Following acid flush and citric acid neutralization, tubes are passivated using a sodium nitrite / ammonia bath at pH 9.5 for 6 hours to establish a continuous protective magnetite (Fe3O4) film before cold hydrostatic testing at 6.3 MPa (1.5x MAWP).

[SEC-THM-5] Low-Water Level Cutoff Interlock (LWCO-01)
Primary drum level is monitored by differential pressure transmitter DPT-301 and dual independent conductivity probes LWCO-01 and LWCO-02. A drop of drum water level below -75 mm from normal waterline triggers a low-water alarm; dropping below -125 mm initiates an immediate Master Fuel Trip (MFT), closing natural gas double-block-and-bleed valves DBB-1/2 within 1.2 seconds. Never bypass LWCO-01 during live firing; probe blowdown verification must be logged once per shift.
""",

    "MAN-CRN-520_Overhead_Crane.txt": """# OVERHEAD BRIDGE CRANE & RIGGING OPERATIONS MANUAL (DOC-ID: MAN-CRN-520)

[SEC-CRN-1] Rated Capacity & Proof Load Testing (SWL-25T)
Double-girder electric overhead traveling (EOT) crane CRN-BAY3 carries a Safe Working Load (SWL) rating of 25.0 metric tonnes on main hoist MH-01 and 5.0 metric tonnes on auxiliary hoist AH-02. Annual statutory load testing requires lifting a calibrated test weight of 31.25 tonnes (125% SWL overload) 150 mm off the floor and holding for 10 minutes with zero brake creep and maximum girder mid-span deflection below L/800 (22.5 mm for 18-meter span).

[SEC-CRN-2] Wire Rope Inspection & Discard Criteria
Main hoist MH-01 uses 18 mm diameter 6x36 IWRC galvanized steel wire rope. Immediate rope removal and replacement is mandatory under any of four discard conditions per ISO 4309: (a) 6 or more randomly distributed broken wires in one rope lay length (115 mm), (b) 3 or more broken wires in one strand within one lay, (c) outer wire diameter reduction exceeding 7% (measured below 16.74 mm using a calibrated groove caliper), or (d) any visible kinking, birdcaging, or heat discoloration.

[SEC-CRN-3] Rigging Sling Angle Derating Factors
When rigging loads with a two-leg wire-rope bridle sling, included horizontal sling angles below 60 degrees drastically amplify leg tension. At a 60-degree angle to the horizontal, load factor is 1.155 (86.6% efficiency). At 45 degrees, tension multiplier rises to 1.414 (70.7% efficiency). Rigging at sling angles below 30 degrees to the horizontal (tension multiplier 2.000) is strictly prohibited without an engineered spreader beam SB-15T approved by the site rigging engineer.

[SEC-CRN-4] Wind Speed Interlocks & Storm Tie-Downs
For gantry or semi-gantry outdoor bay transfers, anemometer ANEM-01 continuously monitors 3-second gust velocity. When sustained wind speed reaches 15.0 m/s (54 km/h), an amber deck beacon flashes, requiring completion of the current lift within 3 minutes. At 20.0 m/s (72 km/h), crane motion is automatically inhibited and operators must move the bridge to designated storm parking bay SP-01, engage rail clamps RC-A/B, and lower the hook block to 2.0 m above ground without load.

[SEC-CRN-5] Hoist Limit Switches & Anti-Two-Block Testing
Upper hoist travel is protected by primary rotary cam limit switch HLS-01 (set 450 mm below trolley frame) and secondary weighted counterweight over-hoist switch HLS-02 (set 200 mm below trolley frame). Every shift start requires an unloaded functional bump test of HLS-01 before lifting any payload. Failure of HLS-01 to arrest upward hook travel places crane CRN-BAY3 in mandatory Red-Tag out-of-service lock state until electrical maintenance recalibrates the cam contacts.
"""
}

def build_questions():
    qs = []
    qid = 1

    def add(category, query, doc, section, phrases, reference_answer):
        nonlocal qid
        qs.append({
            "id": f"Q-{qid:03d}",
            "category": category,
            "query": query,
            "answerable": doc is not None,
            "ground_truth_doc": doc,
            "ground_truth_section": section,
            "required_evidence_phrases": phrases,
            "reference_answer": reference_answer,
        })
        qid += 1

    # --- 1. DIRECT FACTUAL (25 questions, 5 per manual) ---
    add("direct_factual", "What hydraulic oil grade and temperature range are specified for the Series-4000 press?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-1", ["ISO VG 46", "38 C and 54 C"],
        "ISO VG 46 anti-wear hydraulic oil maintained between 38 C and 54 C.")
    add("direct_factual", "At what pressure does proportional relief valve PRV-402-B crack open and reach full bypass?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-1", ["24.0 MPa", "25.2 MPa"],
        "PRV-402-B cracks open at 24.0 MPa and reaches full bypass flow at 25.2 MPa.")
    add("direct_factual", "What is the maximum allowable vacuum on suction manifold gauge VG-02 during cold startup?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-2", ["17 kPa", "5 inHg"],
        "Vacuum at suction manifold gauge VG-02 must never exceed 17 kPa (5 inHg) during cold startup.")
    add("direct_factual", "What is the nominal coil resistance and minimum pull-in voltage for solenoid SOL-H88?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-4", ["22.4 ohms", "20.4 VDC"],
        "Nominal coil resistance is 22.4 ohms at 20 C, and minimum pull-in voltage under load is 20.4 VDC.")
    add("direct_factual", "At what reservoir temperature does thermal switch TS-HIGH trip the pump motor?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-5", ["68 C", "K-9"],
        "Thermal switch TS-HIGH triggers an automatic pump motor trip at 68 C, engaging latching relay K-9.")

    add("direct_factual", "What bevel angle and root opening are required for ASTM A572 Grade 50 single-V groove joints?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-1", ["30 degrees", "3.0 mm"],
        "Bevel angles of 30 degrees (+5/-0 degrees) and a root opening of 3.0 mm (+/- 0.8 mm).")
    add("direct_factual", "What temperature range must portable holding ovens maintain for E7018-H4R electrodes?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-2", ["120 C and 150 C", "E7018-H4R"],
        "Portable holding ovens must maintain E7018-H4R electrodes between 120 C and 150 C.")
    add("direct_factual", "What is the minimum preheat temperature for ASTM A572 Grade 50 plates over 38 mm thick?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-3", ["110 C", "38 mm"],
        "Minimum preheat temperature increases to 110 C for thicknesses exceeding 38 mm.")
    add("direct_factual", "What maximum oxygen concentration is allowed inside TP316L pipe bores before GTAW welding?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-4", ["0.05%", "500 ppm"],
        "Residual O2 concentration must be below 0.05% (500 ppm) before starting GTAW welding.")
    add("direct_factual", "How long must butt welds be held after cooling before ultrasonic testing under UT-SCAN-9?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-5", ["24 hours", "UT-SCAN-9"],
        "Welds must cool below 40 C and be held for 24 hours before UT-SCAN-9 inspection.")

    add("direct_factual", "What is the calculated incident energy and Arc Flash PPE Category for switchgear panel SWG-BUS-A?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-2", ["14.8 cal/cm2", "Category 3"],
        "14.8 cal/cm2 at 450 mm working distance, placing SWG-BUS-A in Arc Flash PPE Category 3.")
    add("direct_factual", "How often must CLASS-2-17KV rubber insulating gloves undergo laboratory electrical re-certification?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-3", ["6 months", "180 days"],
        "Dielectric gloves must be re-certified every 6 months (withdrawn if >180 days from test date).")
    add("direct_factual", "What acetylene concentration in transformer TRF-5M indicates active internal arcing?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-4", ["35 ppm", "C2H2"],
        "Acetylene (C2H2) concentrations rising above 35 ppm signify active internal high-energy arcing.")
    add("direct_factual", "What is the float charge voltage and cell count for the substation 110 VDC battery bank?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-5", ["55 lead-acid", "124.85 VDC"],
        "55 lead-acid VRLA cells maintained at a float charge voltage of 124.85 VDC (+/- 0.5%).")
    add("direct_factual", "What torque must be applied to inter-cell copper busbars using insulated 1000 V wrenches?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-5", ["14 Nm", "1000 V"],
        "Torque wrenches must be set to 14 Nm on inter-cell copper busbars.")

    add("direct_factual", "What is the maximum allowable dissolved oxygen leaving deaerator DA-01 for boiler BLR-80T?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-1", ["7 ppb", "0.007 mg/L"],
        "Dissolved oxygen leaving DA-01 must remain strictly below 7 ppb (0.007 mg/L).")
    add("direct_factual", "What drum water pH and maximum silica limits apply at 4.2 MPa?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-2", ["9.2 to 9.8", "2.5 ppm"],
        "Drum water pH must stay between 9.2 and 9.8, and total drum silica must remain below 2.5 ppm.")
    add("direct_factual", "What deposit weight density triggers chemical acid cleaning of boiler tubes?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-4", ["150 g/m2", "5% hydrochloric"],
        "Acid cleaning is triggered when waterside deposit density exceeds 150 g/m2.")
    add("direct_factual", "At what drum level drop does LWCO-01 initiate a Master Fuel Trip?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-5", ["-125 mm", "1.2 seconds"],
        "Dropping below -125 mm initiates an immediate Master Fuel Trip, closing DBB-1/2 within 1.2 seconds.")
    add("direct_factual", "What is the normal operating drum TDS limit controlled by continuous blowdown valve CBD-09?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-3", ["800 ppm", "CBD-09"],
        "Drum total dissolved solids (TDS) are controlled below 800 ppm.")

    add("direct_factual", "What are the Safe Working Load ratings for main hoist MH-01 and auxiliary hoist AH-02?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-1", ["25.0 metric tonnes", "5.0 metric tonnes"],
        "25.0 metric tonnes on main hoist MH-01 and 5.0 metric tonnes on auxiliary hoist AH-02.")
    add("direct_factual", "What proof test load weight and hold duration are required for annual crane certification?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-1", ["31.25 tonnes", "10 minutes"],
        "A calibrated test weight of 31.25 tonnes (125% SWL) held for 10 minutes.")
    add("direct_factual", "What minimum wire diameter requires immediate discard of the 18 mm hoist wire rope?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-2", ["16.74 mm", "7%"],
        "Outer wire diameter reduction exceeding 7% (measured below 16.74 mm) mandates replacement.")
    add("direct_factual", "At what wind speed does anemometer ANEM-01 automatically inhibit crane motion?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-4", ["20.0 m/s", "72 km/h"],
        "At 20.0 m/s (72 km/h), crane motion is automatically inhibited.")
    add("direct_factual", "What heights below the trolley frame are limit switches HLS-01 and HLS-02 positioned?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-5", ["450 mm", "200 mm"],
        "Primary switch HLS-01 is set 450 mm below the trolley frame and secondary HLS-02 at 200 mm.")

    # --- 2. PROCEDURAL QUESTIONS (20 questions) ---
    add("procedural", "What steps must be taken before removing proportional directional valve PDV-118?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-3", ["NBV-03", "25 minutes", "68 Nm"],
        "Depressurize ACC-04 via NBV-03 three turns, confirm zero on PG-MAIN, flush 25 min, torque M10 bolts to 34 then 68 Nm.")
    add("procedural", "How should an operator torque the four M10 mounting bolts on valve PDV-118?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-3", ["diagonal cross pattern", "34 Nm", "68 Nm"],
        "Torque in a diagonal cross pattern in two stages: first stage 34 Nm, final stage 68 Nm.")
    add("procedural", "What is the recovery procedure after thermal switch TS-HIGH trips the hydraulic pump?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-5", ["52 C", "28 liters/min", "PB-RST-4"],
        "Allow fluid to cool below 52 C, verify HEX-02 water flow >= 28 L/min, and press reset PB-RST-4.")
    add("procedural", "How do you manually lower the upper platen if power fails?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-5", ["OVR-77", "6 mm hex key"],
        "Turn override screw OVR-77 clockwise using a 6 mm hex key while maintaining visual clearance.")

    add("procedural", "How should E7018-H4R electrodes be handled if exposed to >60% workshop humidity for 5 hours?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-2", ["rebaked once only", "370 C for 2.0 hours"],
        "Segregate and rebake once only at 370 C for 2.0 hours (or discard if flux is wet/flaking).")
    add("procedural", "What surface cleaning steps are required before tack welding ASTM A572 Grade 50 steel?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-1", ["50 mm", "wire brushing"],
        "Remove mill scale, grease, paint, and moisture within 50 mm of the weld toe via power brushing or grinding.")
    add("procedural", "What steps are required to repair a weld exhibiting slag inclusions longer than 12 mm?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-5", ["excavation", "WPS-REP-02", "100% re-inspection"],
        "Full excavation, re-welding under repair spec WPS-REP-02, and 100% ultrasonic re-inspection.")
    add("procedural", "How should stainless steel pipe interiors be purged before GTAW root welding?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-4", ["PDK-60", "OA-900", "0.05%"],
        "Back-purge with argon using Purge Dam Kit PDK-60 until analyzer OA-900 reads below 0.05% O2.")

    add("procedural", "List the seven-step Zero Energy Verification sequence LOTO-77X for 11 kV switchgear.",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-1", ["VCB-11K", "Live-Dead-Live", "ES-11K", "PEC-04"],
        "Notify operator, open downstream feeders, rack out VCB-11K, apply LOTO-PAD, Live-Dead-Live test with HV-DET-15KV, close ES-11K, attach PEC-04.")
    add("procedural", "What daily shift check is mandatory before wearing CLASS-2-17KV dielectric gloves?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-3", ["air-inflation roll test", "GLV-STAMP"],
        "Perform a manual air-inflation roll test for pinholes/cracking and verify cuff tag GLV-STAMP is within 180 days.")
    add("procedural", "What diagnostic actions are required if transformer TRF-5M DGA shows >35 ppm acetylene?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-4", ["SFRA-200", "winding resistance"],
        "Take transformer offline and perform winding resistance and sweep frequency response analysis (SFRA-200).")
    add("procedural", "What test is triggered when a VRLA battery cell's impedance rises >25% above baseline?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-5", ["C/10 rate", "11 A"],
        "Conduct single-cell capacity discharge testing at C/10 rate (11 A).")

    add("procedural", "What is the exact opening and closing valve sequence for manual mud-drum blowdown?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-3", ["BDV-12B", "BDV-12A", "12 seconds"],
        "Open slow-closing BDV-12B first, open BDV-12A for 12 seconds, close BDV-12A first, then close BDV-12B last.")
    add("procedural", "How are boiler tubes passivated after hydrochloric acid cleaning?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-4", ["citric acid", "sodium nitrite", "6.3 MPa"],
        "Neutralize with citric acid, passivate using sodium nitrite/ammonia at pH 9.5 for 6 hours, then hydrotest at 6.3 MPa.")
    add("procedural", "How is oxygen scavenged in the feedwater downstream of deaerator DA-01?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-1", ["carbohydrazide", "50 to 100 ppb"],
        "Continuously meter catalyzed carbohydrazide solution SCAV-CHZ-10 to target 50-100 ppb residual at the economizer inlet.")
    add("procedural", "What daily shift check is mandatory for low-water level cutoff probes LWCO-01 and LWCO-02?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-5", ["probe blowdown", "logged once per shift"],
        "Perform and log probe blowdown verification once per shift without bypassing LWCO-01.")

    add("procedural", "What actions must the crane operator take when sustained wind speed reaches 20.0 m/s?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-4", ["SP-01", "RC-A/B", "2.0 m"],
        "Move bridge to storm parking bay SP-01, engage rail clamps RC-A/B, and lower unloaded hook to 2.0 m above ground.")
    add("procedural", "What pre-shift test must be executed before lifting any payload with CRN-BAY3?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-5", ["unloaded functional bump test", "HLS-01"],
        "Perform an unloaded functional bump test of upper hoist limit switch HLS-01.")
    add("procedural", "What is required to legally rig a load using sling angles below 30 degrees to the horizontal?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-3", ["spreader beam SB-15T", "rigging engineer"],
        "Strictly prohibited without an engineered spreader beam SB-15T approved by the site rigging engineer.")
    add("procedural", "How is structural girder deflection evaluated during the annual 31.25-tonne crane proof test?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-1", ["L/800", "22.5 mm"],
        "Mid-span deflection must remain below L/800 (22.5 mm for the 18-meter span) with zero brake creep.")

    # --- 3. PARAPHRASED / VOCABULARY-MISMATCH QUESTIONS (20 questions) ---
    add("paraphrased", "Why would a hydraulic press make loud metallic banging noises and move jerkily?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-2", ["cavitation", "STR-14", "HOSE-IN-09"],
        "Pump cavitation caused by clogged strainer STR-14, collapsed hose HOSE-IN-09, low oil (<185 L), or air ingress via SSR-22.")
    add("paraphrased", "How do I release stored hydraulic energy safely before unbolting a control valve?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-3", ["ACC-04", "NBV-03", "PG-MAIN"],
        "Bleed accumulator ACC-04 by opening needle valve NBV-03 three turns until gauge PG-MAIN reads zero.")
    add("paraphrased", "My 24V directional actuator won't shift; what electrical resistance indicates a burnt-out or shorted coil?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-4", ["18.0 ohms", "30.0 ohms"],
        "Below 18.0 ohms indicates an inter-turn short; above 30.0 ohms or infinite resistance indicates an open winding.")
    add("paraphrased", "What keeps the heavy upper press head from falling down if the electricity goes out?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-5", ["counterbalance valve CBV-77", "mechanically"],
        "Counterbalance valve CBV-77 mechanically locks the upper platen during power loss.")

    add("paraphrased", "How do I keep stainless steel inside pipes from turning crusty and oxidized during TIG welding?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-4", ["back-purging", "0.05%", "14 to 18 L/min"],
        "Use argon back-purging below 0.05% O2 and keep torch argon flow between 14 and 18 L/min to avoid turbulence.")
    add("paraphrased", "Why can't we weld fast and hot without waiting between passes on thick structural steel?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-3", ["230 C", "Charpy V-notch"],
        "Exceeding 230 C interpass temperature degrades heat-affected zone Charpy impact toughness.")
    add("paraphrased", "What happens if low-hydrogen stick welding rods sit out on a humid workshop bench all day?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-2", ["4.0 hours", "370 C"],
        "Exposure over 4.0 hours above 60% RH requires single-time rebaking at 370 C for 2 hours or rejection if flux flakes.")
    add("paraphrased", "Why do inspectors wait a whole day after welding cools before running sound-wave flaw checks?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-5", ["24 hours", "hydrogen-delayed cracking"],
        "The 24-hour hold accounts for the hydrogen-delayed cracking incubation period prior to UT-SCAN-9.")

    add("paraphrased", "Why are technicians banned from wearing polyester or nylon t-shirts under high-voltage flash suits?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-2", ["Synthetic melting fabrics", "prohibited"],
        "Synthetic fabrics like polyester and nylon can melt onto skin under arc-rated garments during an arc flash.")
    add("paraphrased", "How do electricians prove a high-voltage circuit is truly dead before touching busbars?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-1", ["Live-Dead-Live", "HV-DET-15KV"],
        "Perform a Live-Dead-Live test using calibrated live-line tester HV-DET-15KV before closing the earthing switch.")
    add("paraphrased", "Which gas found in transformer oil tells us electrical sparks are jumping inside the tank?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-4", ["Acetylene", "35 ppm"],
        "Acetylene (C2H2) above 35 ppm indicates active internal high-energy arcing.")
    add("paraphrased", "Why must we check the date inked on rubber electrical gloves even if they look clean?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-3", ["180 days", "6 months"],
        "Dielectric gloves must be lab-recertified every 6 months; pairs >180 days past stamp must be withdrawn.")

    add("paraphrased", "What chemical damage occurs on steam turbine blades if boiler silica concentrations climb too high?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-2", ["volatile silica carryover", "glass-like deposition"],
        "Volatile silica carryover causes glass-like deposition on downstream steam turbine blades.")
    add("paraphrased", "Why do we open the slow blowdown valve before the fast valve, and close it last?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-3", ["seat wear", "sacrificial valve"],
        "Cycling BDV-12A inside open BDV-12B concentrates throttling and seat wear onto the outer sacrificial valve.")
    add("paraphrased", "How do engineers detect if cooling water is leaking into the pure steam condenser?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-2", ["Cation conductivity", "0.50 uS/cm"],
        "Cation conductivity at sample point SS-03 spiking above 0.50 uS/cm indicates condenser tube leakage.")
    add("paraphrased", "What protective iron oxide layer are we trying to grow inside freshly acid-washed boiler tubes?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-4", ["magnetite", "Fe3O4"],
        "Passivation with sodium nitrite/ammonia establishes a protective magnetite (Fe3O4) film.")

    add("paraphrased", "How much extra force hits each lifting sling leg when riggers lift with a shallow 30-degree bridle?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-3", ["2.000", "30 degrees"],
        "At 30 degrees to horizontal, the tension multiplier doubles to 2.000, requiring spreader beam SB-15T.")
    add("paraphrased", "How many snapped strands or wires force us to throw away a crane hoist cable?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-2", ["6 or more", "3 or more"],
        "6+ broken wires in one lay (115 mm) or 3+ broken wires in a single strand within one lay.")
    add("paraphrased", "What keeps the crane hook from slamming upward into the overhead trolley drum?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-5", ["HLS-01", "HLS-02"],
        "Primary rotary cam switch HLS-01 (450 mm below trolley) and secondary counterweight switch HLS-02 (200 mm below).")
    add("paraphrased", "When gusty weather starts picking up outside, at what speed do we get an early warning versus total shutdown?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-4", ["15.0 m/s", "20.0 m/s"],
        "Amber warning beacon flashes at 15.0 m/s (54 km/h); automatic motion inhibit occurs at 20.0 m/s (72 km/h).")

    # --- 4. EXACT TECHNICAL IDENTIFIER QUERIES (15 questions, 3 per manual) ---
    add("exact_id", "What is the part specification and role of SSR-22?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-2", ["SSR-22", "shaft seal ring"],
        "SSR-22 is the shaft seal ring inspected for air ingress during persistent cavitation.")
    add("exact_id", "Explain the function of D-41 and PB-RST-4 in MAN-HYD-101.",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-4", ["D-41", "flyback suppression diode"],
        "D-41 is the flyback suppression diode on SOL-H88 terminals; PB-RST-4 resets thermal trip relay K-9.")
    add("exact_id", "Where is test port TP-01 used and what gauge accuracy is required?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-1", ["TP-01", "Class-0.5"],
        "TP-01 is used to verify pressure with a Class-0.5 calibrated digital gauge before adjusting PRV-402-B.")

    add("exact_id", "What is scrap bin BIN-RED-03 designated for in MAN-WLD-204?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-2", ["BIN-RED-03", "Wet or flaking flux"],
        "BIN-RED-03 receives rejected electrodes with wet or flaking flux coatings.")
    add("exact_id", "What equipment is specified as TC-08 and OA-900?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-3", ["TC-08", "thermocouple"],
        "TC-08 is a calibrated contact thermocouple for preheat measurement; OA-900 is the oxygen analyzer for GTAW purging.")
    add("exact_id", "What transducer frequency and angle are required by UT-SCAN-9?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-5", ["2.25 MHz", "60-degree"],
        "UT-SCAN-9 uses a 2.25 MHz, 60-degree shear-wave transducer.")

    add("exact_id", "What are the model and rating of VRLA-110-AH cells?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-5", ["VRLA-110-AH", "2.15 mOhm"],
        "55 lead-acid VRLA-110-AH cells with installation impedance baseline of 2.15 mOhm.")
    add("exact_id", "What components are PEC-04 and DV-01 in MAN-ELE-309?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-1", ["PEC-04", "portable earthing cluster"],
        "PEC-04 is the portable earthing cluster attached in step 7 of LOTO-77X; DV-01 is transformer TRF-5M's DGA drain valve.")
    add("exact_id", "What proof test voltage applies to CLASS-2-17KV gloves under ASTM D120?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-3", ["20,000 V", "17,000 V"],
        "CLASS-2-17KV gloves are rated for 17,000 V AC max working voltage and tested at 20,000 V proof voltage.")

    add("exact_id", "What chemical is dosed as SCAV-CHZ-10?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-1", ["SCAV-CHZ-10", "carbohydrazide"],
        "SCAV-CHZ-10 is catalyzed carbohydrazide solution dosed into the deaerator storage drop-leg.")
    add("exact_id", "What instruments are CT-401 and DPT-301?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-3", ["CT-401", "conductivity transmitter"],
        "CT-401 is the conductivity transmitter modulating CBD-09; DPT-301 is the drum level differential pressure transmitter.")
    add("exact_id", "What are DBB-1/2 and how quickly must they close on trip?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-5", ["DBB-1/2", "1.2 seconds"],
        "DBB-1/2 are natural gas double-block-and-bleed valves closing within 1.2 seconds of a Master Fuel Trip.")

    add("exact_id", "What is the wire rope construction and lay length of main hoist MH-01?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-2", ["6x36 IWRC", "115 mm"],
        "18 mm diameter 6x36 IWRC galvanized wire rope with a rope lay length of 115 mm.")
    add("exact_id", "What are SP-01 and RC-A/B used for in MAN-CRN-520?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-4", ["SP-01", "RC-A/B"],
        "SP-01 is the storm parking bay and RC-A/B are the rail clamps engaged during 20.0 m/s wind shutdown.")
    add("exact_id", "When is spreader beam SB-15T required?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-3", ["SB-15T", "30 degrees"],
        "SB-15T is required when rigging angles fall below 30 degrees to the horizontal.")

    # --- 5. AMBIGUOUS / UNDERSPECIFIED QUESTIONS (10 questions) ---
    add("ambiguous", "What pressure limit applies?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-1", ["21.5 MPa", "24.0 MPa"],
        "Ambiguous across press (21.5 MPa working / 24.0 MPa relief) and boiler (4.2 MPa working / 6.3 MPa hydrotest).")
    add("ambiguous", "What is the required torque specification?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-3", ["68 Nm"],
        "Depends on component: PDV-118 valve bolts require 68 Nm; substation battery busbars require 14 Nm.")
    add("ambiguous", "How long should we wait before performing the inspection?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-5", ["24 hours"],
        "Weld UT inspection requires a 24-hour delay; crane proof load test requires a 10-minute hold; glove lab recert is 6 months.")
    add("ambiguous", "What temperature threshold triggers a shutdown or trip?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-5", ["68 C"],
        "Hydraulic press thermal trip triggers at 68 C; welding maximum interpass is 230 C; transformer overheating flag at 500 C.")
    add("ambiguous", "What is the maximum allowed ppm limit?",
        "MAN-WLD-204_Structural_Welding_QA.txt", "SEC-WLD-4", ["500 ppm"],
        "Depends on context: GTAW purge O2 <= 500 ppm; DGA C2H2 <= 35 ppm; drum silica <= 2.5 ppm; drum TDS <= 800 ppm.")
    add("ambiguous", "Which valve needs to be opened first?",
        "MAN-THM-415_Boiler_Feedwater.txt", "SEC-THM-3", ["BDV-12B"],
        "During mud-drum blowdown, open slow-closing BDV-12B first; for accumulator bleed, open NBV-03.")
    add("ambiguous", "What PPE is required for this task?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-2", ["Category 3", "AR-25"],
        "Substation switchgear SWG-BUS-A mandates Category 3 (AR-25 suit, Class-2 shield, leather protectors).")
    add("ambiguous", "What is the voltage rating mentioned in the manual?",
        "MAN-ELE-309_LOTO_Substation.txt", "SEC-ELE-1", ["11 kV"],
        "Switchgear is 11 kV; dielectric gloves rated 17 kV; DC battery bank is 110 VDC (float 124.85 VDC); solenoid is 24 VDC.")
    add("ambiguous", "When do we replace the component?",
        "MAN-CRN-520_Overhead_Crane.txt", "SEC-CRN-2", ["16.74 mm", "6 or more"],
        "Hoist wire rope is discarded at 6+ broken wires in one lay or <16.74 mm diameter; gloves retired past 180 days.")
    add("ambiguous", "What flow rate should be checked?",
        "MAN-HYD-101_Hydraulic_Press.txt", "SEC-HYD-5", ["28 liters/min"],
        "Heat exchanger HEX-02 water flow >= 28 L/min; GTAW shielding argon flow 14-18 L/min.")

    # --- 6. OUT-OF-SCOPE / UNANSWERABLE QUESTIONS (15 questions) ---
    oos = [
        "How do I configure Kubernetes horizontal pod autoscaling for a microservice?",
        "What is the corporate employee reimbursement policy for domestic flight tickets?",
        "How do React useEffect hooks avoid infinite re-rendering loops?",
        "What was the quarterly revenue growth of Bridge & Roof Co. in FY 2024?",
        "Which chemotherapy dosage applies to stage-3 non-small-cell lung cancer?",
        "How do you configure PostgreSQL Row-Level Security policies for multi-tenant tables?",
        "What are the ingredients and baking time for sourdough focaccia bread?",
        "How do I initialize a ChromaDB vector collection in Python?",
        "What is the optimal learning rate schedule for fine-tuning Llama-3-8B with LoRA?",
        "Which visa documentation is required for a University of Toronto study permit?",
        "How do you calculate the Black-Scholes option pricing delta for European calls?",
        "What torque applies to a Toyota Corolla cylinder head bolt?",
        "How are municipal property taxes assessed in Chennai metropolitan wards?",
        "What is the exact chemical formula of aspirin and its synthesis yield?",
        "How do you configure BGP peering routes on a Cisco IOS XR router?",
    ]
    for q in oos:
        add("out_of_scope", q, None, None, [],
            "INSUFFICIENT_CONTEXT: The provided technical manuals do not contain information to answer this question.")

    return qs


def main():
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, text in MANUALS.items():
        (MANUALS_DIR / fname).write_text(text.strip() + "\n", encoding="utf-8")

    questions = build_questions()
    out_q = DATASET_DIR / "evaluation_questions.json"
    out_q.write_text(json.dumps({
        "dataset_name": "synthetic_industrial_sop_benchmark_v1",
        "corpus_files": sorted(MANUALS.keys()),
        "total_questions": len(questions),
        "category_counts": {
            cat: sum(1 for q in questions if q["category"] == cat)
            for cat in ["direct_factual", "procedural", "paraphrased", "exact_id", "ambiguous", "out_of_scope"]
        },
        "questions": questions,
    }, indent=2), encoding="utf-8")
    print(f"Generated {len(MANUALS)} synthetic manuals and {len(questions)} benchmark questions at {out_q}")


if __name__ == "__main__":
    main()

