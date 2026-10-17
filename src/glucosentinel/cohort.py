"""Synthetic Indian cardiometabolic cohort: demographics, EHR fields, hidden physiology and lifestyle.

Each synthetic person gets
  * EHR-visible attributes (age, sex, BMI, waist, diagnoses, medications, family history,
    TCF7L2 genotype, polygenic risk score, lipids, eGFR, resting vitals),
  * hidden physiological parameters (insulin sensitivity, beta-cell response, basal glucose ...)
    that are causally linked to the EHR attributes, and
  * hidden lifestyle traits (diet region, portion size, walking habit, sleep and stress
    propensity, meal timing, logging compliance).

The twin never sees the hidden parameters. It has to infer them from EHR + wearable data.
No real person's data is used anywhere.
"""
import numpy as np
import pandas as pd

FIRST_M = ["Ramesh", "Suresh", "Arjun", "Vikram", "Rahul", "Manoj", "Prakash", "Karthik", "Imran",
           "Harish", "Mahesh", "Sanjay", "Deepak", "Ravi", "Anil", "Joseph", "Gurpreet", "Nitin",
           "Bhavesh", "Hitesh", "Sunil", "Ajay", "Ashok", "Venkat", "Farhan", "Pranav"]
FIRST_F = ["Lakshmi", "Priya", "Anjali", "Meena", "Kavita", "Sunita", "Fatima", "Divya", "Pooja",
           "Rekha", "Geeta", "Shalini", "Asha", "Nirmala", "Hetal", "Neha", "Radha", "Sneha",
           "Mary", "Harpreet", "Jyoti", "Usha", "Swati", "Bhavna"]
SURNAME_BY_REGION = {
    "south": ["Iyer", "Reddy", "Nair", "Rao", "Krishnan", "Subramanian", "Menon", "Gowda"],
    "west": ["Patel", "Shah", "Desai", "Joshi", "Mehta", "Kulkarni", "Parmar", "Trivedi"],
    "north": ["Sharma", "Singh", "Gupta", "Verma", "Kapoor", "Malhotra", "Yadav", "Khan"],
    "east": ["Das", "Banerjee", "Ghosh", "Mukherjee", "Sahoo", "Bose", "Mohanty", "Roy"],
}
CITY_BY_REGION = {
    "south": ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Coimbatore"],
    "west": ["Ahmedabad", "Surat", "Mumbai", "Pune", "Vadodara"],
    "north": ["Delhi", "Lucknow", "Jaipur", "Chandigarh", "Kanpur"],
    "east": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi"],
}


def make_cohort(n=120, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pre = rng.random() < 0.15                      # prediabetes vs diagnosed T2D
        z = rng.normal(-1.0 if pre else 0.25, 0.6 if pre else 0.9)  # latent metabolic severity
        sex = "M" if rng.random() < 0.58 else "F"
        region = rng.choice(["south", "west", "north", "east"], p=[0.33, 0.32, 0.25, 0.10])
        age = int(np.clip(rng.normal(44 if pre else 53, 9), 28, 75))
        bmi = float(np.clip(rng.normal(25.8 + 1.4 * z, 3.2), 19.0, 40.0))
        waist = float(np.clip((86 if sex == "M" else 80) + 2.4 * (bmi - 24) + rng.normal(0, 4), 64, 135))
        duration = 0.0 if pre else float(np.clip(rng.gamma(2.0, 3.2) + 0.15 * (age - 50), 0.2, 25))
        tcf7l2 = int(rng.binomial(2, 0.32))           # rs7903146 T-allele count
        prs = float(rng.normal(0.45 * z, 0.85))       # standardised T2D polygenic risk score
        fam = bool(rng.random() < np.clip(0.42 + 0.12 * z, 0.1, 0.9))
        htn = bool(rng.random() < np.clip(0.33 + 0.10 * z + 0.008 * (age - 50), 0.05, 0.9))
        dyslip = bool(rng.random() < np.clip(0.48 + 0.08 * z, 0.1, 0.9))

        # ---- hidden physiology (causally linked to EHR attributes) ----
        SI = 4.6e-4 * np.exp(-0.42 * z - 0.03 * (bmi - 25) - 0.06 * prs + rng.normal(0, 0.18))
        if pre:
            SI *= 1.15
        gamma = 0.068 * np.exp(-0.22 * z - 0.028 * duration - 0.11 * tcf7l2 + rng.normal(0, 0.18))
        Gb = float(np.clip(116 + 27 * (z + 1.0) + rng.normal(0, 8), 92, 230))
        if pre:
            Gb = float(np.clip(rng.normal(109, 5), 98, 122))

        # ---- medications and their effect on true physiology ----
        metformin = (not pre) and rng.random() < 0.86
        su = (not pre) and z > 0.7 and rng.random() < 0.55
        dpp4 = (not pre) and (not su) and rng.random() < 0.28
        if metformin:
            Gb *= 0.91; SI *= 1.10
        if su:
            gamma *= 1.35
        if dpp4:
            gamma *= 1.15; Gb *= 0.97

        params = dict(
            weight=float(bmi * (1.70 if sex == "M" else 1.57) ** 2),
            SG=float(0.014 * np.exp(rng.normal(0, 0.15))),
            SI=float(SI), p2=0.025, n=0.12, gamma=float(gamma),
            Ib=float(7.0 + 0.25 * (bmi - 22) + rng.normal(0, 1.0)),
            Gb=float(Gb),
            ex=float(0.009 * np.exp(rng.normal(0, 0.25))),
            dawn=float(np.clip(6 + 9 * max(z, 0) + rng.normal(0, 4), 0, 35)),
        )
        params["V"] = 1.6 * params["weight"]

        rest_hr = float(np.clip(68 + 2.5 * z + 3 * htn + rng.normal(0, 5), 52, 92))
        rmssd = float(np.clip(42 - 0.35 * (age - 40) - 0.5 * duration - 2.5 * z + rng.normal(0, 5), 10, 70))

        # ---- hidden lifestyle traits ----
        life = dict(
            region=region,
            portion=float(np.exp(rng.normal(0.05, 0.13))),
            walk_after_meal=float(rng.beta(2, 4)),
            morning_walk=float(rng.beta(2, 3)),
            activity_level=float(np.exp(rng.normal(0, 0.35))),
            sweet_freq=float(np.clip(rng.beta(2, 9), 0, 0.5)),
            snack_prob=float(rng.uniform(0.55, 0.95)),
            poor_sleep_prop=float(rng.beta(2.2, 6)),
            stress_prop=float(rng.beta(2, 6.5)),
            log_compliance=float(rng.uniform(0.70, 0.95)),
            t_breakfast=float(rng.uniform(7.4, 9.4)),
            t_lunch=float(rng.uniform(12.7, 14.2)),
            t_snack=float(rng.uniform(16.4, 17.8)),
            t_dinner=float(rng.uniform(19.8, 21.8)),
            t_bed=float(rng.uniform(22.4, 24.4)),
            t_wake=float(rng.uniform(5.7, 7.4)),
        )

        first = rng.choice(FIRST_M if sex == "M" else FIRST_F)
        surname = rng.choice(SURNAME_BY_REGION[region])
        rows.append(dict(
            patient_id=f"PT-{i + 1:04d}",
            display_name=f"{first} {surname[0]}.",
            sex=sex, age=age, region=region, city=str(rng.choice(CITY_BY_REGION[region])),
            diagnosis="Prediabetes" if pre else "Type 2 diabetes mellitus",
            icd10="R73.03" if pre else "E11.9",
            diabetes_duration_y=round(duration, 1),
            bmi=round(bmi, 1), waist_cm=round(waist, 1),
            hypertension=htn, dyslipidaemia=dyslip, family_history_t2d=fam,
            tcf7l2_rs7903146_risk_alleles=tcf7l2, t2d_prs=round(prs, 2),
            on_metformin=metformin, on_sulfonylurea=su, on_dpp4i=dpp4,
            resting_hr=round(rest_hr, 1), baseline_rmssd=round(rmssd, 1),
            sbp=int(np.clip(rng.normal(128 + 10 * htn + 2 * z, 10), 100, 185)),
            dbp=int(np.clip(rng.normal(80 + 6 * htn, 7), 60, 115)),
            ldl=int(np.clip(rng.normal(112 + 12 * dyslip, 22), 50, 220)),
            hdl=int(np.clip(rng.normal(43 - 2 * z - 3 * (sex == "M"), 7), 22, 80)),
            triglycerides=int(np.clip(rng.normal(150 + 30 * z + 25 * dyslip, 40), 60, 450)),
            egfr=int(np.clip(rng.normal(96 - 0.6 * (age - 40) - 3 * z, 10), 30, 125)),
            _latent_severity=float(z), _params=params, _life=life,
        ))
    return pd.DataFrame(rows)
