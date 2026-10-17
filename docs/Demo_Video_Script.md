# GlucoSentinel demo video: script and shot list (about 21 minutes)

**Format:** record the PowerPoint in slideshow mode with voice (PowerPoint: Slide Show > Record), then switch to the live dashboard for the demo segment. Upload to YouTube as **Unlisted** and paste the link into section 8 of the README.

**Before recording**
1. `pip install -r requirements.txt`
2. `streamlit run dashboard/app.py` and open http://localhost:8501 in a browser at 100% zoom.
3. In the sidebar keep patient **PT-0054 (Gurpreet K.)**, day **Thu 10 Sep 2026**, time **12.50 h**, validation mode **on**.
4. Open `docs/GlucoSentinel_Presentation.pptx`. The narration below matches each slide's speaker notes.

Speaking pace: about 140 words a minute. Times are cumulative.

---

## Part A. The presentation (0:00 to 14:30)

### 0:00 to 0:50 | Slide 1, cover
Hello, I am Dinesh D S from IIM Ahmedabad, and this is GlucoSentinel. A sentinel is a guard who warns before danger arrives, and that is exactly what this does. It is a proof-of-concept digital twin for type 2 diabetes. It fuses a patient's health record with five-minute wearable data, builds a calibrated virtual copy of that patient, and warns of a glucose spike up to two hours before it happens. On 24 test patients the model never saw, it reaches an AUROC of 0.961, warns 85 percent of spikes at least 30 minutes ahead, and gives a median warning of 95 minutes. In the next twenty minutes I will cover the problem, the data, the twin engine, the AI model and the results, and then give a live demo of the doctor console.

### 0:50 to 2:20 | Slide 2, executive summary
Here is the whole story on one slide. The problem: India has 101 million people with diabetes and 136 million with prediabetes. What hurts them between visits are post-meal spikes, which HbA1c averages away. In our cohort two patients share an HbA1c of about 7 percent, yet one spends twice as long above 180. And today's tools react: a CGM trend alarm warns only 28 percent of spikes thirty minutes ahead.
Our solution starts from the EHR, calibrates a personal physiological twin over seven days of wearable data, keeps it in sync with every CGM reading, and puts a LightGBM model on top of 81 fused features.
The outcome on 24 unseen patients: AUROC 0.961, 85 percent of spikes warned thirty or more minutes ahead, a median lead of 95 minutes, and under one false alert a day. The twin also answers what-if questions; for example, swapping rice for bajra roti lowers this patient's predicted peak from 328 to 220. And everything regenerates from one command.

### 2:20 to 4:00 | Slide 3, problem
Why does this matter? The ICMR-INDIAB survey of over 113,000 adults found 101 million Indians with diabetes, 136 million with prediabetes and 315 million with hypertension. Indians also develop diabetes at lower body weight and younger ages, and our plates are carbohydrate heavy.
The damage happens between clinic visits, after meals. HbA1c, the number doctors track, is an average, so it hides spikes. Look at these two patients: both near 7 percent HbA1c, but Bhavna spends 24 percent of her day above 180 and Rahul 11 percent.
Current tools react. A CGM trend alarm only sees the rise once it starts, so it warned just 28 percent of spikes thirty minutes ahead, with a median lead of 15 minutes.
The bottom left is the anatomy of a missed spike: a rice-heavy lunch, glucose climbs, the alarm fires at 180 when it is already rising fast, the peak reaches 250 to 330, and this repeats every day. The gap we fill: doctors are overloaded, data sits in silos, there is no look-ahead before the meal, and no way to test advice on this patient first.

### 4:00 to 5:30 | Slide 4, solution
GlucoSentinel is a hybrid of physiology and machine learning. On the left is the twin loop. Step A: the EHR sets a prior, so on day one we already have a reasonable physiology from HbA1c, fasting glucose, BMI, years since diagnosis, TCF7L2 genotype and medications. Step B: over a seven-day onboarding week, the twin calibrates insulin sensitivity, beta-cell response, basal glucose and the dawn effect to this patient's own CGM. Step C: every five-minute CGM reading nudges the twin, like a weather model assimilating observations. Step D: the twin runs forward to forecast and to test what-if scenarios.
In the middle is what users get: the patient receives a nudge before the meal, and the doctor gets a risk card, a forecast, a why panel, a what-if lab and a weekly review.
On the right is how this differs from a CGM app or a pure ML model. The ablation proves the twin matters: without it, AUROC falls from 0.961 to 0.930 and the median warning from 95 to 75 minutes.

### 5:30 to 7:00 | Slide 5, data
The brief asks us to fuse two streams, and both are here. Top left, the static stream: each patient's record is an HL7 FHIR R4 bundle, the format Synthea produces and India's ABDM uses, with ICD-10 diagnoses, LOINC-coded labs including three HbA1c values, RxNorm medications, family history and two genetic markers, the TCF7L2 risk allele and a polygenic risk score.
Top right, the dynamic stream for one patient over 30 hours: CGM every five minutes, steps, heart rate, HRV and sleep stages, plus a meal log.
Because real patient data is restricted by the DPDP Act and HIPAA, everything is synthetic. Our simulator gives each person hidden physiology linked to their EHR, a daily life with regional Indian meals, walks, sleep and stress, a one-minute glucose-insulin simulation, and a realistic sensor model with an eight-minute lag, noise, dropouts and imperfect meal logging. The cohort: 120 patients, 483,840 readings and 5,200 logged meals from a library of 32 Indian dishes with glycaemic index.

### 7:00 to 9:00 | Slide 6, twin engine
This is the engine. Top left is the physiology core, an extended Bergman minimal model: glucose, insulin action and insulin secretion, with a two-pool gut model driven by each dish's glycaemic index, plus exercise uptake, the dawn effect, and sleep debt and stress, which lower insulin sensitivity.
Calibration happens in the onboarding week. We simulate 625 candidate physiologies at once, then zoom into 256 around the best, scoring each on how well it forecasts the patient's own CGM at 30, 60 and 120 minutes, with a penalty for straying far from the EHR prior. That takes about five and a half CPU-seconds per patient.
The scatter shows it works: recovered insulin sensitivity correlates at 0.92 with the hidden truth.
Live, every CGM reading nudges the twin, so its 60-minute forecast error is 27 mg/dL versus 43 for assuming no change, and the twin alone already reaches an AUROC of 0.921.
And it can simulate. For this patient a rice lunch peaks at 328; swapping to bajra roti brings it to 220; a 20-minute walk alone helps less. That is the personal insight a doctor needs.

### 9:00 to 10:45 | Slide 7, AI model
Now the AI layer. Every five minutes we build 81 features in four families: 23 from the EHR, 40 from wearables and CGM, 5 describing the patient's routine learned in week one, and 13 from the twin, such as the peak the twin expects if the patient eats their usual meal.
The label is the onset of an excursion above 180 lasting at least 15 minutes, within the next two hours.
The evaluation is strict: 84 patients for training, 12 for validation, and 24 test patients the model never saw. Week one is only for calibrating the twin and learning routine; everything is trained and scored on week two.
The model is LightGBM with early stopping, plus four forecasters for the glucose path. The alert threshold was chosen on validation patients only, to catch as many spikes as possible at about one false alert a day.
In the SHAP view, the usual spike rate at this hour and the twin's predicted peak dominate. The EHR looks small directly because its value flows through the twin's prior.

### 10:45 to 12:30 | Slide 8, results
Here are the results on the 24 test patients. Top left, the ablation: EHR plus time of day alone reaches 0.904 AUROC; wearables alone 0.930; fusing the two, 0.930; adding the twin and routine lifts it to 0.961, with AUPRC rising from 0.848 to 0.908. A CGM trend alarm, which is what devices do today, reaches only 0.642.
Top right, the clinical view across 582 spike episodes: GlucoSentinel warns 85 percent of spikes at least 30 minutes ahead versus 28 percent for the trend alarm; the median lead is 95 minutes versus 15; and false alerts are 0.82 a day versus 1.62.
Bottom left, glucose forecasts: at 60 minutes the error is 19 mg/dL, versus 43 for no change and 27 for the twin alone. The Clarke error grid puts 93.5 percent of 60-minute forecasts in zone A and 99.6 percent in A plus B.
And performance holds across women and men, regions and diagnosis groups.

### 12:30 to 13:45 | Slide 9, one patient, one day
Let us make it concrete with one test patient, Gurpreet, 47, with diabetes for twelve and a half years. The dark line is his CGM, the light line is what GlucoSentinel forecast sixty minutes earlier, and the bottom strip is the risk against the alert threshold.
Three alerts: at 8:20, an hour before the breakfast spike; at 12:30, 105 minutes before the lunch spike; and at 20:35, 70 minutes before dinner. He did not log his dinner, yet routine and twin features still caught it.
Why did the 12:30 alert fire? This is his usual spike hour on five of seven days, and the twin predicts a peak of 262 if he eats his usual lunch.
We are honest about failures too: 8 percent of spikes get no warning, mostly off-routine meals such as festival sweets, and 7 percent are warned less than 30 minutes ahead.

### 13:45 to 14:30 | Slide 10, dashboard overview
This is the doctor console. The top shows the FHIR record, the virtual patient card and the risk card. The chart shows twelve hours of CGM, the twin's path, GlucoSentinel's forecast, steps and risk. A doctor's workflow takes about a minute: pick a patient, read the risk, see why, test a plan, send a nudge, and review the week. Let me show it live.

---

## Part B. Live demo of the doctor console (14:30 to 19:30)

Switch to the browser with the dashboard open.

**14:30 | The patient view (1 min).** "This is GlucoSentinel's doctor console running on the test cohort. I have selected Gurpreet K., a test patient, on Thursday 10 September at 12:30." Point to the three cards:
* Patient record: "This comes straight from the FHIR bundle: type 2 diabetes for 12.5 years, BMI 22.1, HbA1c 6.7%, one TCF7L2 risk allele."
* Virtual patient: "The calibrated twin says his insulin sensitivity is at the 73rd percentile but his beta-cell response is at the 17th. So he is a low insulin secretor: lean, but his pancreas cannot keep up with a big carb load. That pattern is often described in South Asians. Sensitivity is 1.42 times what the EHR alone suggested, which shows the twin learned from his data."
* Risk card: "Spike risk 78 percent, high, alert sent. Glucose is 120 now, and a spike above 180 is expected in about two hours."

**15:30 | The forecast chart (1 min).** "The dark line is the last twelve hours of CGM. The light blue line is the twin's physiology path if he eats nothing. The dark points are GlucoSentinel's forecasts at 30, 60, 90 and 120 minutes, rising above 180 because it expects his usual lunch. Validation mode is on, so the dotted line shows what actually happened: the spike arrived at about 14:15. Below are steps and the risk curve; the same risk pattern caught his breakfast spike this morning."

**16:30 | Why the model thinks so (45 s).** "The why panel lists the drivers in plain words: his usual spike rate at this hour is 0.71, the twin's peak if he eats his usual meal is 262, he usually eats about 69 grams of carbs in the next two hours. The context card adds last night's sleep, steps today and HRV versus his baseline."

**17:15 | Replay the day (30 s).** Drag the time slider to 08:00, then 20:30. "I can replay any moment. At 8 the risk is building before breakfast; at 20:30 it rises again before dinner, even though he did not log that dinner."
Return the slider to 12:30.

**17:45 | What-if simulator (1 min 15 s).** Click the What-if simulator tab. "Now the doctor tests a plan on the twin before the patient tries it. As usual, curd rice peaks at 279. The suggested plan, millet khichdi plus a 20-minute walk, peaks at 211, 68 points lower." Change the planned meal to Bajra or jowar roti with dal and set Walk to 20 minutes; read out the new peak. Switch Sleep to Poor night: "a poor night raises the peak, which is why sleep is part of the twin." Set it back. "The message to the patient is drafted automatically and is editable; the doctor sends it with one click. In this prototype nothing is actually sent."

**19:00 | Week review and record (30 s).** Click Week review (AGP). "Before the next visit: the ambulatory glucose profile, 86 percent time in range, 14 percent above 180, a glucose management indicator of 6.7 percent, and alerts per day." Click EHR record and expand the FHIR bundle: "and this is the raw FHIR bundle the twin ingests."

---

## Part C. Impact, roadmap and close (19:30 to 21:15)

Switch back to the slides.

### 19:30 to 20:30 | Slide 11, impact
The impact pathway is simple: an alert about 95 minutes ahead gives time to act, by swapping, eating less or walking; in the twin, one swap lowered the peak by 108 mg/dL; and less time above range follows. Long-term benefits on HbA1c and complications must be proven in a trial, and we say so. Patients get advice before the meal, doctors get one screen instead of four thousand readings, insurers get a measurable outcome, and because records go in and out as FHIR R4, it fits ABDM. We built in responsible-AI checks, and we state our limitations: synthetic data flatters accuracy, routine features need a stable first week, there is no hypoglycaemia module yet, and medication what-ifs are illustrative.

### 20:30 to 21:15 | Slide 12, roadmap and cost; close
Over the next twelve months: test retrospectively on real CGM datasets, add a hypoglycaemia module and photo and voice meal logging in Indian languages, get ethics approval, run a 50-patient three-month clinic pilot, and integrate with ABDM. On cost, the CGM sensor dominates, at about 3,100 to 4,600 rupees for fifteen days at current Indian retail prices. The AI is nearly free: calibrating 120 twins took 331 seconds on two CPU cores, and scoring 20,000 readings takes 53 milliseconds.
Everything, including code, synthetic data, models, this deck and the architecture diagram, is in the public GitHub repository under the MIT licence, and one command, python run_all.py, rebuilds it all. Thank you.

---

## Optional extra (adds about 1 minute): code walkthrough
Show the repository in the browser: `src/glucosentinel/twin.py` (the PatientTwin class with prior, calibrate, filter and what_if), `reports/metrics.json` (every number in the deck), and a terminal running `python run_all.py` and `pytest -q`.
