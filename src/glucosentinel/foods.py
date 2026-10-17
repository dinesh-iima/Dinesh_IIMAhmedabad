"""Indian meal library used by the simulator, the meal logger and the what-if engine.

Carbohydrate values are typical home portions (grams of available carbohydrate).
GI values are approximate glycaemic index bands drawn from published Indian food
GI tables; they are used only to split carbs into fast and slow absorbing pools.
"""

FOODS = {
    # breakfast
    "Idli (4) with sambar": dict(carbs=60, gi=70, slot="breakfast", region="south"),
    "Plain dosa with chutney": dict(carbs=55, gi=75, slot="breakfast", region="south"),
    "Ragi dosa with chutney": dict(carbs=45, gi=55, slot="breakfast", region="south"),
    "Pongal": dict(carbs=58, gi=68, slot="breakfast", region="south"),
    "Poha": dict(carbs=50, gi=65, slot="breakfast", region="west"),
    "Thepla (3) with curd": dict(carbs=48, gi=55, slot="breakfast", region="west"),
    "Upma": dict(carbs=50, gi=65, slot="breakfast", region="all"),
    "Aloo paratha (2) with curd": dict(carbs=65, gi=65, slot="breakfast", region="north"),
    "Besan chilla (2)": dict(carbs=30, gi=40, slot="breakfast", region="north"),
    "Oats porridge": dict(carbs=40, gi=55, slot="breakfast", region="all"),
    "Bread omelette": dict(carbs=35, gi=70, slot="breakfast", region="east"),
    # lunch
    "Rice with sambar and poriyal": dict(carbs=90, gi=75, slot="lunch", region="south"),
    "Curd rice": dict(carbs=70, gi=70, slot="lunch", region="south"),
    "Rice with dal and sabzi": dict(carbs=85, gi=72, slot="lunch", region="east"),
    "Gujarati thali (rotli, dal, bhaat, shaak)": dict(carbs=95, gi=68, slot="lunch", region="west"),
    "Chapati (3) with dal and sabzi": dict(carbs=65, gi=55, slot="lunch", region="north"),
    "Veg biryani": dict(carbs=95, gi=70, slot="lunch", region="all"),
    "Bajra or jowar roti with dal": dict(carbs=55, gi=48, slot="lunch", region="west"),
    "Brown rice with rajma": dict(carbs=70, gi=55, slot="lunch", region="north"),
    # evening snack
    "Chai with sugar and biscuits": dict(carbs=30, gi=70, slot="snack", region="all"),
    "Samosa": dict(carbs=35, gi=65, slot="snack", region="north"),
    "Khakhra with chai": dict(carbs=25, gi=60, slot="snack", region="west"),
    "Murukku with filter coffee": dict(carbs=32, gi=70, slot="snack", region="south"),
    "Fruit bowl": dict(carbs=20, gi=45, slot="snack", region="all"),
    "Roasted chana": dict(carbs=18, gi=30, slot="snack", region="all"),
    "Sweet (gulab jamun or jalebi)": dict(carbs=45, gi=85, slot="snack", region="all"),
    # dinner
    "Chapati (2) with sabzi": dict(carbs=55, gi=55, slot="dinner", region="north"),
    "Rice with rasam": dict(carbs=75, gi=75, slot="dinner", region="south"),
    "Dosa with sambar": dict(carbs=55, gi=75, slot="dinner", region="south"),
    "Khichdi with kadhi": dict(carbs=60, gi=60, slot="dinner", region="west"),
    "Millet khichdi": dict(carbs=50, gi=50, slot="dinner", region="all"),
    "Rice with fish curry": dict(carbs=75, gi=72, slot="dinner", region="east"),
}

REGION_WEIGHT = {"south": 3.0, "west": 3.0, "north": 3.0, "east": 3.0}


def menu_for(slot, region):
    """Return (names, weights) for a meal slot, favouring the patient's region."""
    names, weights = [], []
    for name, f in FOODS.items():
        if f["slot"] != slot:
            continue
        if f["region"] == region:
            w = REGION_WEIGHT[region]
        elif f["region"] == "all":
            w = 1.2
        else:
            w = 0.25
        names.append(name)
        weights.append(w)
    return names, weights


def fast_fraction(gi):
    """Share of a meal's carbs that behaves like fast-absorbing glucose."""
    return float(min(max((gi - 35.0) / 55.0, 0.05), 0.95))


# Low-GI swaps a doctor can try in the what-if simulator
SWAPS = {
    "Rice with sambar and poriyal": "Bajra or jowar roti with dal",
    "Curd rice": "Millet khichdi",
    "Rice with dal and sabzi": "Brown rice with rajma",
    "Gujarati thali (rotli, dal, bhaat, shaak)": "Bajra or jowar roti with dal",
    "Veg biryani": "Brown rice with rajma",
    "Plain dosa with chutney": "Ragi dosa with chutney",
    "Idli (4) with sambar": "Ragi dosa with chutney",
    "Poha": "Besan chilla (2)",
    "Aloo paratha (2) with curd": "Besan chilla (2)",
    "Rice with rasam": "Millet khichdi",
    "Sweet (gulab jamun or jalebi)": "Fruit bowl",
    "Chai with sugar and biscuits": "Roasted chana",
}


def personal_portion(meal_log):
    """Patient's usual portion size relative to the library serving, learned from their meal log."""
    import numpy as np
    r = [c / FOODS[d]["carbs"] for d, c in zip(meal_log.dish, meal_log.carbs_logged_g) if d in FOODS]
    return float(np.clip(np.median(r), 0.6, 1.6)) if r else 1.0
