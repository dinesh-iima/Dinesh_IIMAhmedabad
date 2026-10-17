"""Exports each synthetic patient's EHR as an HL7 FHIR R4 Bundle (the format Synthea emits
and that India's Ayushman Bharat Digital Mission uses for health records), and reads it back.

Reading the EHR back from FHIR, rather than from the flat table, proves the twin can
ingest a standards-based record from any hospital system.
"""
import json
from pathlib import Path

LOINC = {
    "hba1c_pct": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "fpg_mgdl": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
    "waist_cm": ("8280-0", "Waist Circumference at umbilicus by Tape measure", "cm"),
    "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
    "ldl": ("18262-6", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "mg/dL"),
    "hdl": ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL"),
    "triglycerides": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    "egfr": ("62238-1", "Glomerular filtration rate/1.73 sq M.predicted (CKD-EPI)", "mL/min/{1.73_m2}"),
    "resting_hr": ("40443-4", "Heart rate --resting", "/min"),
}
MEDS = {
    "on_metformin": ("6809", "Metformin 500 MG Oral Tablet"),
    "on_sulfonylurea": ("25789", "Glimepiride 2 MG Oral Tablet"),
    "on_dpp4i": ("593411", "Sitagliptin 100 MG Oral Tablet"),
}
CONDITIONS = {
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "dyslipidaemia": ("E78.5", "Hyperlipidaemia, unspecified"),
}


def _obs(pid, key, value, date):
    code, display, unit = LOINC[key]
    return {"resourceType": "Observation", "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
            "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": date,
            "valueQuantity": {"value": value, "unit": unit}}


def to_bundle(r):
    pid = r["patient_id"]
    entries = [{"resourceType": "Patient", "id": pid,
                "name": [{"text": r["display_name"]}],
                "gender": "male" if r["sex"] == "M" else "female",
                "extension": [{"url": "age-years", "valueInteger": int(r["age"])}],
                "address": [{"city": r["city"], "country": "IN"}],
                "meta": {"tag": [{"code": "synthetic", "display": "Synthetic record, not a real person"}]}}]
    entries.append({"resourceType": "Condition", "subject": {"reference": f"Patient/{pid}"},
                    "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": r["icd10"],
                                         "display": r["diagnosis"]}]},
                    "extension": [{"url": "duration-years", "valueDecimal": float(r["diabetes_duration_y"])}]})
    for flag, (code, disp) in CONDITIONS.items():
        if r[flag]:
            entries.append({"resourceType": "Condition", "subject": {"reference": f"Patient/{pid}"},
                            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": code, "display": disp}]}})
    for key in LOINC:
        entries.append(_obs(pid, key, float(r[key]), "2026-08-20"))
    entries.append(_obs(pid, "hba1c_pct", float(r["hba1c_6m_ago"]), "2026-02-18"))
    entries.append(_obs(pid, "hba1c_pct", float(r["hba1c_12m_ago"]), "2025-08-22"))
    for flag, (rx, disp) in MEDS.items():
        if r[flag]:
            entries.append({"resourceType": "MedicationStatement", "status": "active",
                            "subject": {"reference": f"Patient/{pid}"},
                            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                                                      "code": rx, "display": disp}]}})
    if r["family_history_t2d"]:
        entries.append({"resourceType": "FamilyMemberHistory", "patient": {"reference": f"Patient/{pid}"},
                        "status": "completed", "relationship": {"text": "Parent"},
                        "condition": [{"code": {"text": "Type 2 diabetes mellitus"}}]})
    entries.append({"resourceType": "Observation", "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "69548-6",
                                         "display": "Genetic variant assessment"}]},
                    "subject": {"reference": f"Patient/{pid}"},
                    "component": [
                        {"code": {"text": "TCF7L2 rs7903146 T allele count"}, "valueInteger": int(r["tcf7l2_rs7903146_risk_alleles"])},
                        {"code": {"text": "Type 2 diabetes polygenic risk score (z)"}, "valueDecimal": float(r["t2d_prs"])}]})
    return {"resourceType": "Bundle", "type": "collection", "id": f"bundle-{pid}",
            "entry": [{"resource": e} for e in entries]}


def write_bundles(cohort, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for _, r in cohort.iterrows():
        (out / f"{r['patient_id']}.json").write_text(json.dumps(to_bundle(r), indent=1))


def read_bundle(path):
    """Parse a FHIR bundle into the flat EHR feature dict the twin uses."""
    b = json.loads(Path(path).read_text())
    res = [e["resource"] for e in b["entry"]]
    rec = {"on_metformin": False, "on_sulfonylurea": False, "on_dpp4i": False,
           "hypertension": False, "dyslipidaemia": False, "family_history_t2d": False}
    code2key = {v[0]: k for k, v in LOINC.items()}
    hba1c = []
    for r in res:
        t = r["resourceType"]
        if t == "Patient":
            rec["patient_id"] = r["id"]
            rec["sex"] = "M" if r["gender"] == "male" else "F"
            rec["age"] = r["extension"][0]["valueInteger"]
        elif t == "Condition":
            c = r["code"]["coding"][0]["code"]
            if c in ("E11.9", "R73.03"):
                rec["icd10"] = c
                rec["diabetes_duration_y"] = r["extension"][0]["valueDecimal"]
            elif c == "I10":
                rec["hypertension"] = True
            elif c == "E78.5":
                rec["dyslipidaemia"] = True
        elif t == "Observation":
            code = r["code"]["coding"][0]["code"]
            if code == "69548-6":
                rec["tcf7l2_rs7903146_risk_alleles"] = r["component"][0]["valueInteger"]
                rec["t2d_prs"] = r["component"][1]["valueDecimal"]
            elif code in code2key:
                key = code2key[code]
                if key == "hba1c_pct":
                    hba1c.append((r["effectiveDateTime"], r["valueQuantity"]["value"]))
                else:
                    rec[key] = r["valueQuantity"]["value"]
        elif t == "MedicationStatement":
            rx = r["medicationCodeableConcept"]["coding"][0]["code"]
            for flag, (code, _) in MEDS.items():
                if rx == code:
                    rec[flag] = True
        elif t == "FamilyMemberHistory":
            rec["family_history_t2d"] = True
    hba1c.sort()
    rec["hba1c_pct"] = hba1c[-1][1]
    rec["hba1c_trend_12m"] = round(hba1c[-1][1] - hba1c[0][1], 2)
    return rec
