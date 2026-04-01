"""
persona.py  —  Extended UserHealthProfile
------------------------------------------
Combines llm-planner's original 6 fields with maria_paper's full clinical
measurement set (the same measurements Maria can save via save_measurement tool).
"""

from pydantic import BaseModel, Field


class UserHealthProfile(BaseModel):
    # ── llm-planner original fields ────────────────────────────────────────────
    Age: int = Field(default=-1)
    Weight: float = Field(default=-1)
    Average_Sleeping_Hours: float = Field(default=-1)
    Has_Diabetes: int = Field(default=-1, description="1=yes, 0=no, -1=unknown")
    Has_High_Blood_Pressure: int = Field(default=-1, description="1=yes, 0=no, -1=unknown")
    Daily_Exercise: int = Field(default=-1, description="1=yes, 0=no, -1=unknown")

    # ── maria_paper clinical fields ────────────────────────────────────────────
    Height_cm: float = Field(default=-1)
    BMI: float = Field(default=-1)
    Glycated_Hemoglobin: float = Field(default=-1, description="HbA1c %")
    Blood_Pressure_Systolic: int = Field(default=-1, description="mmHg")
    Blood_Pressure_Diastolic: int = Field(default=-1, description="mmHg")
    HDL: float = Field(default=-1, description="mg/dL")
    LDL: float = Field(default=-1, description="mg/dL")
    Framingham_Score: float = Field(default=-1, description="10-year CVD risk %")
    Abdominal_Circumference_cm: float = Field(default=-1)

    # ── Lifestyle / subjective ─────────────────────────────────────────────────
    Stress_Level: int = Field(default=-1, description="1-10 self-reported, -1=unknown")
    Diet_Quality: int = Field(default=-1, description="1-10 self-reported, -1=unknown")
    Smokes: int = Field(default=-1, description="1=yes, 0=no, -1=unknown")
    Alcohol_Consumption: int = Field(default=-1, description="weekly drinks, -1=unknown")