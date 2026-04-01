"""
business/action_business.py
----------------------------
Resolves Groq tool calls to actual side-effects.

This mirrors the role of maria_paper's `resolve_function` (which was called
inside the OpenAI Assistants `requires_action` loop). Here it is called from
Maria's agentic run loop after Groq returns a `tool_calls` finish reason.

When a real SQLAlchemy `db` session is provided (production mode), operations
are persisted to PostgreSQL, exactly as in the original.

When `db` is None (Streamlit standalone mode), operations are stored in an
in-memory dict that is passed back to the UI — so the demo still works without
a database.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# In-memory fallback store (used when db=None)
# ---------------------------------------------------------------------------

_memory_store: dict[str, list] = {
    "measurements": [],
    "alarms": [],
    "appointments": [],
    "food_data": [],
    "proactive_messages": [],
}


def get_memory_store() -> dict:
    """Return a copy of the in-memory store (for Streamlit display)."""
    return {k: list(v) for k, v in _memory_store.items()}


def clear_memory_store() -> None:
    for key in _memory_store:
        _memory_store[key].clear()


# ---------------------------------------------------------------------------
# Individual action handlers
# ---------------------------------------------------------------------------

def _save_measurement(args: dict, user_id: int, db=None) -> str:
    mtype = args.get("type")
    value = args.get("value")
    text_value = args.get("text_value")

    record = {
        "user_id": user_id,
        "type": mtype,
        "value": value,
        "text_value": text_value,
        "created": datetime.now().isoformat(),
    }

    if db is not None:
        try:
            from models import Measurement, MeasurementTypeEnum  # noqa: PLC0415
            m = Measurement(
                user_id=user_id,
                type=MeasurementTypeEnum[mtype],
                value=value,
                text_value=text_value,
                from_hapvida=False,
                date=datetime.now(),
                created=datetime.now(),
            )
            db.add(m)
            db.commit()
            return f"Measurement '{mtype}' saved successfully (value={value or text_value})."
        except Exception as e:
            return f"DB error saving measurement: {e}"
    else:
        _memory_store["measurements"].append(record)
        return f"Measurement '{mtype}' recorded (value={value or text_value})."


def _create_alarm(args: dict, user_id: int, db=None) -> str:
    description = args.get("description", "")
    date = args.get("date", "")
    time = args.get("time", "")

    record = {
        "user_id": user_id,
        "description": description,
        "date": date,
        "time": time,
        "created": datetime.now().isoformat(),
    }

    if db is not None:
        try:
            from models import Alarm  # noqa: PLC0415
            from datetime import datetime as dt  # noqa: PLC0415
            try:
                dt_obj = dt.strptime(f"{date} {time}", "%d/%m/%Y %H:%M")
            except ValueError:
                dt_obj = dt.now()
            alarm = Alarm(
                owner_id=user_id,
                description=description,
                date=date,
                time=time,
                date_time=dt_obj,
                message_sent=False,
                created=dt.now(),
            )
            db.add(alarm)
            db.commit()
            return f"Alarm set: '{description}' on {date} at {time}."
        except Exception as e:
            return f"DB error creating alarm: {e}"
    else:
        _memory_store["alarms"].append(record)
        return f"Alarm set: '{description}' on {date} at {time}."


def _schedule_medical_appointment(args: dict, user_id: int, db=None) -> str:
    appt_type = args.get("type", "consulta")
    doctor = args.get("doctor_name", "")
    exam = args.get("exam", "")
    date = args.get("date", "")
    time = args.get("time", "")
    reason = args.get("reason", "")

    record = {
        "user_id": user_id,
        "type": appt_type,
        "doctor_name": doctor,
        "exam": exam,
        "date": date,
        "time": time,
        "reason": reason,
        "created": datetime.now().isoformat(),
    }

    if db is not None:
        try:
            from models import MedicalScheduling  # noqa: PLC0415
            ms = MedicalScheduling(
                user_id=user_id,
                type=appt_type,
                doctor_name=doctor or None,
                exam=exam or None,
                date=date,
                time=time or None,
                reason=reason or None,
                processed=False,
                sent=False,
                created=datetime.now(),
            )
            db.add(ms)
            db.commit()
            label = doctor or exam or appt_type
            return f"Appointment scheduled: {label} on {date}{' at ' + time if time else ''}."
        except Exception as e:
            return f"DB error scheduling appointment: {e}"
    else:
        _memory_store["appointments"].append(record)
        label = doctor or exam or appt_type
        return f"Appointment scheduled: {label} on {date}{' at ' + time if time else ''}."


def _save_food_data(args: dict, user_id: int, db=None) -> str:
    ftype = args.get("type", "HABITOS")
    value = args.get("value", "")

    record = {
        "user_id": user_id,
        "type": ftype,
        "value": value,
        "created": datetime.now().isoformat(),
    }

    if db is not None:
        try:
            from models import FoodData, FoodDataTypeEnum  # noqa: PLC0415
            fd = FoodData(
                user_id=user_id,
                type=FoodDataTypeEnum[ftype],
                value=value,
                created=datetime.now(),
            )
            db.add(fd)
            db.commit()
            return f"Food data saved: [{ftype}] {value}."
        except Exception as e:
            return f"DB error saving food data: {e}"
    else:
        _memory_store["food_data"].append(record)
        return f"Food data recorded: [{ftype}] {value}."


def _schedule_proactive_message(args: dict, user_id: int, db=None) -> str:
    description = args.get("description", "")
    theme = args.get("theme", "")
    relevance = args.get("relevance", "medium")
    message = args.get("message", "")
    excerpt = args.get("message_excerpt", "")
    date_str = args.get("date", datetime.now().isoformat())

    record = {
        "user_id": user_id,
        "description": description,
        "theme": theme,
        "relevance": relevance,
        "message": message,
        "message_excerpt": excerpt,
        "date": date_str,
        "created": datetime.now().isoformat(),
    }

    if db is not None:
        try:
            from models import ProactiveScheduling  # noqa: PLC0415
            from datetime import datetime as dt  # noqa: PLC0415
            try:
                dt_obj = dt.fromisoformat(date_str)
            except ValueError:
                dt_obj = dt.now()
            ps = ProactiveScheduling(
                user_id=user_id,
                description=description,
                theme=theme,
                relevance=relevance,
                date=dt_obj,
                message=message,
                message_excerpt=excerpt,
                created=dt.now(),
                analysed=False,
                sent=False,
            )
            db.add(ps)
            db.commit()
            return f"Proactive message queued: '{description}' ({theme}, {relevance} relevance)."
        except Exception as e:
            return f"DB error scheduling proactive message: {e}"
    else:
        _memory_store["proactive_messages"].append(record)
        return f"Proactive message queued: '{description}' ({theme}, {relevance} relevance)."


# ---------------------------------------------------------------------------
# Dispatcher — called by Maria's agentic loop
# ---------------------------------------------------------------------------

_HANDLERS = {
    "save_measurement": _save_measurement,
    "create_alarm": _create_alarm,
    "schedule_medical_appointment": _schedule_medical_appointment,
    "save_food_data": _save_food_data,
    "schedule_proactive_message": _schedule_proactive_message,
}


def resolve_function(function_name: str, arguments: str | dict, user_id: int, db=None) -> str:
    """
    Dispatch a tool call from Maria's agentic loop.

    Parameters
    ----------
    function_name : str
        Name of the tool as defined in tools.py.
    arguments : str | dict
        JSON string or dict of tool arguments from Groq.
    user_id : int
        Current patient's user ID.
    db : SQLAlchemy session or None
        If provided, persists to PostgreSQL; otherwise uses in-memory store.

    Returns
    -------
    str
        Human-readable result string fed back to the model as the tool result.
    """
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Error: could not parse arguments for '{function_name}'."
    else:
        args = arguments

    handler = _HANDLERS.get(function_name)
    if handler is None:
        return f"Unknown function '{function_name}'."

    return handler(args, user_id, db)