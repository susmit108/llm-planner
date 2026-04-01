"""
tools.py
--------
Groq tool (function) definitions that mirror every action maria_paper's
resolve_function dispatched via the OpenAI Assistants `requires_action` loop.

Each tool maps 1-to-1 to a handler in business/action_business.py.

These are passed as the `tools` parameter to every Groq chat completion so
Maria can trigger DB side-effects mid-conversation, exactly as in the original.
"""

MARIA_TOOLS = [
    # ── Medical measurements ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_measurement",
            "description": (
                "Persist a clinical measurement for the current patient. "
                "Call this whenever the user reports a new exam result or body measurement "
                "(weight, blood pressure, BMI, HbA1c, HDL, LDL, Framingham score, "
                "abdominal circumference, height)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "PESO", "ALTURA", "IMC",
                            "HEMOGLOBINA_GLICADA", "PRESSAO_ARTERIAL",
                            "HDL", "LDL", "ESCORE_FRAMINGHAM",
                            "CIRCUNFERENCIA_ABDOMINAL"
                        ],
                        "description": "Measurement type key (Portuguese enum, kept from original)."
                    },
                    "value": {
                        "type": "number",
                        "description": "Numeric value of the measurement (use for quantitative types)."
                    },
                    "text_value": {
                        "type": "string",
                        "description": "Text value (use for PRESSAO_ARTERIAL e.g. '120/80')."
                    }
                },
                "required": ["type"]
            }
        }
    },

    # ── Alarms / reminders ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "create_alarm",
            "description": (
                "Schedule a reminder alarm for the patient. "
                "Use when the user wants to be reminded about medication, "
                "exercise, appointments, or any recurring health task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What the alarm is for (e.g. 'Take metformin 500mg')."
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in DD/MM/YYYY format."
                    },
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM format (24h)."
                    }
                },
                "required": ["description", "date", "time"]
            }
        }
    },

    # ── Medical appointment scheduling ────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "schedule_medical_appointment",
            "description": (
                "Register a medical appointment or exam for the patient. "
                "Use when the user mentions a scheduled visit, exam, or consultation "
                "they want tracked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Appointment type: 'consulta' or 'exame'."
                    },
                    "doctor_name": {
                        "type": "string",
                        "description": "Doctor or specialist name (optional)."
                    },
                    "exam": {
                        "type": "string",
                        "description": "Exam name if type is 'exame' (optional)."
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in DD/MM/YYYY format."
                    },
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM format (optional)."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the appointment (optional)."
                    }
                },
                "required": ["type", "date"]
            }
        }
    },

    # ── Food / dietary data ───────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_food_data",
            "description": (
                "Record dietary information for the patient: food preferences, "
                "allergies, intolerances, eating habits, meal frequency, restrictions, "
                "or cultural/religious dietary rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "PREFERENCIA_POSITIVA", "PREFERENCIA_NEGATIVA",
                            "ALERGIA", "INTOLERANCIA", "HABITOS",
                            "FREQUENCIA", "RESTRICOES",
                            "PREFERENCIAS_CULTURAIS_RELIGIOSAS", "HORARIOS"
                        ],
                        "description": "Food data category."
                    },
                    "value": {
                        "type": "string",
                        "description": "Description of the food item or habit."
                    }
                },
                "required": ["type", "value"]
            }
        }
    },

    # ── Proactive scheduling ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "schedule_proactive_message",
            "description": (
                "Queue a proactive outreach message to be sent to the patient at a "
                "later time. Use when you identify a health topic that warrants a "
                "follow-up message (e.g. after detecting a worsening trend)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of why this message is being queued."
                    },
                    "theme": {
                        "type": "string",
                        "description": "Health theme (e.g. 'diabetes', 'hypertension', 'sleep')."
                    },
                    "relevance": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Clinical relevance of this outreach."
                    },
                    "message": {
                        "type": "string",
                        "description": "Full text of the proactive message to send."
                    },
                    "message_excerpt": {
                        "type": "string",
                        "description": "Short excerpt / subject line of the message."
                    },
                    "date": {
                        "type": "string",
                        "description": "ISO datetime string for when to send (e.g. '2025-06-15T09:00:00')."
                    }
                },
                "required": ["description", "theme", "relevance", "message", "message_excerpt", "date"]
            }
        }
    },
]