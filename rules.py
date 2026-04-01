"""
rules.py  —  Deterministic rule engine
----------------------------------------
Combines llm-planner's original 2 rules with clinical rules derived from
the measurement types Maria can track (maria_paper domain).
"""


class Rules:
    def __init__(self, persona: dict, task: dict):
        self._p = persona
        self.task = task
        self.process()

    def _get(self, key):
        return self._p.get(key, -1)

    def _add(self, task_str: str):
        if any(v == task_str for v in self.task.values()):
            return
        self.task[str(len(self.task) + 1)] = task_str

    # ── Original llm-planner rules ─────────────────────────────────────────────

    def sleep_blood_pressure(self):
        if self._get("Average_Sleeping_Hours") not in (-1, None) and self._get("Has_High_Blood_Pressure") == 1:
            if self._get("Average_Sleeping_Hours") < 8:
                self._add("Try to get more sleep to reduce blood pressure")

    def exercise_weight(self):
        if self._get("Daily_Exercise") not in (-1, None) and self._get("Weight") not in (-1, None):
            if self._get("Daily_Exercise") != 1 and self._get("Weight") > 90:
                self._add("Try to do more exercise to reduce weight")

    # ── Clinical rules from maria_paper measurement domain ────────────────────

    def hba1c_diabetes(self):
        hba1c = self._get("Glycated_Hemoglobin")
        if hba1c == -1:
            return
        if hba1c > 6.5:
            self._add("Consult your doctor: HbA1c above 6.5% — diabetes screening recommended")
        if self._get("Has_Diabetes") == 1 and hba1c > 7.0:
            self._add("Schedule endocrinology review: HbA1c above target (>7%) for diabetic patient")

    def bmi_risk(self):
        bmi = self._get("BMI")
        if bmi == -1:
            return
        if bmi >= 30:
            self._add("Seek nutritional counselling: BMI indicates obesity (≥30)")
        elif bmi >= 25:
            self._add("Consider a balanced diet plan: BMI is in the overweight range (25–29.9)")

    def cholesterol(self):
        ldl = self._get("LDL")
        hdl = self._get("HDL")
        if ldl != -1 and ldl > 130:
            self._add("Discuss LDL reduction with your doctor: LDL above 130 mg/dL")
        if hdl != -1 and hdl < 40:
            self._add("Increase physical activity to raise HDL: current HDL below 40 mg/dL")

    def framingham(self):
        score = self._get("Framingham_Score")
        if score == -1:
            return
        if score >= 20:
            self._add("Urgent cardiology review: Framingham 10-year cardiovascular risk ≥20%")
        elif score >= 10:
            self._add("Lifestyle modification programme recommended: Framingham risk 10–19%")

    def abdominal_obesity(self):
        ac = self._get("Abdominal_Circumference_cm")
        if ac != -1 and ac > 94:
            self._add("Reduce abdominal circumference through diet and exercise: central obesity detected")

    def hypertension(self):
        sbp = self._get("Blood_Pressure_Systolic")
        if sbp != -1 and sbp >= 140:
            self._add("Seek hypertension evaluation: systolic BP ≥140 mmHg")

    def smoking(self):
        if self._get("Smokes") == 1:
            self._add("Enrol in a smoking cessation programme to significantly reduce cardiovascular risk")

    def stress_sleep(self):
        stress = self._get("Stress_Level")
        sleep = self._get("Average_Sleeping_Hours")
        if stress not in (-1, None) and sleep not in (-1, None):
            if stress >= 7 and sleep < 7:
                self._add("Address high stress + poor sleep: consider mindfulness or mental health support")

    def process(self):
        self.sleep_blood_pressure()
        self.exercise_weight()
        self.hba1c_diabetes()
        self.bmi_risk()
        self.cholesterol()
        self.framingham()
        self.abdominal_obesity()
        self.hypertension()
        self.smoking()
        self.stress_sleep()
        return self.task