"""
reset.py
--------
Resets all session state files to their initial empty values.
Run this between patients or when you want a clean slate.
"""

import json

PERSONA_DEFAULT = {
    "Age": -1,
    "Weight": -1,
    "Average_Sleeping_Hours": -1,
    "Has_Diabetes": -1,
    "Has_High_Blood_Pressure": -1,
    "Daily_Exercise": -1,
    "Height_cm": -1,
    "BMI": -1,
    "Glycated_Hemoglobin": -1,
    "Blood_Pressure_Systolic": -1,
    "Blood_Pressure_Diastolic": -1,
    "HDL": -1,
    "LDL": -1,
    "Framingham_Score": -1,
    "Abdominal_Circumference_cm": -1,
    "Stress_Level": -1,
    "Diet_Quality": -1,
    "Smokes": -1,
    "Alcohol_Consumption": -1,
}

with open("persona.json", "w") as f:
    json.dump(PERSONA_DEFAULT, f, indent=4)

with open("conversation.json", "w") as f:
    json.dump({"Conversation": {}}, f, indent=4)

with open("task.json", "w") as f:
    json.dump({}, f, indent=4)

with open("persona_history.json", "w") as f:
    json.dump([], f, indent=4)

with open("diagnosis_memory.json", "w") as f:
    json.dump(
        {
            "patient_messages": [],
            "symptoms_raw": [],
            "symptoms_normalized": [],
            "unmatched_symptoms": [],
            "candidate_diseases": [],
            "recommended_tasks": [],
            "metadata": {
                "updated_at": None,
                "patient_message_count": 0,
                "extractor": "lightweight-kiddi-adapter",
            },
        },
        f,
        indent=4,
    )

print("All session state files reset to defaults.")
