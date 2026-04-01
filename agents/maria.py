"""
agents/maria.py
---------------
Maria — the medical conversational agent.

Original architecture (maria_paper):
  OpenAI Assistants API → persistent thread → requires_action loop → resolve_function

This port preserves ALL of that behaviour on Groq:
  - Per-user rolling conversation thread (in memory, same as OpenAI thread)
  - System prompt injected with live medical data on every turn
  - Agentic tool-call loop: Groq returns finish_reason='tool_calls' →
    resolve_function executes each tool → results fed back → model continues
    (equivalent to the original `runStatus == 'requires_action'` while-loop)
  - Max 10 tool-call rounds per turn to prevent infinite loops

The agent does NOT know about the llm-planner pipeline; that runs separately
in the background and its output is injected via `extra_context`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytz
from groq import Groq

from tools import MARIA_TOOLS
from business.action_business import resolve_function
from config import get_groq_model


# ---------------------------------------------------------------------------
# Maria
# ---------------------------------------------------------------------------

class Maria:
    """
    Medical conversational agent backed by Groq.

    Parameters
    ----------
    client : Groq
        Configured Groq client.
    model : str
        Groq model string. Defaults to GROQ_MODEL from .env.
    max_tool_rounds : int
        Maximum tool-call iterations per user turn before forcing a final reply.
    """

    MAX_TOOL_ROUNDS = 10

    def __init__(
        self,
        client: Groq,
        model: str | None = None,
    ):
        self.client = client
        self.model = model or get_groq_model()
        # Per-user rolling threads: { user_id -> [{"role":…,"content":…}, …] }
        self._threads: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # Date helpers (kept from original)
    # ------------------------------------------------------------------

    def _traduzir_dia_semana(self, dia_em_ingles: str) -> str:
        return {
            "Monday": "Monday", "Tuesday": "Tuesday",
            "Wednesday": "Wednesday", "Thursday": "Thursday",
            "Friday": "Friday", "Saturday": "Saturday", "Sunday": "Sunday",
        }.get(dia_em_ingles, dia_em_ingles)

    # ------------------------------------------------------------------
    # Medical data retrieval (kept from original get_medical_data)
    # ------------------------------------------------------------------

    def get_medical_data(self, user, db=None) -> tuple[str, str, str]:
        """
        Build patient history strings from the DB.
        Returns (sobre_paciente, first_data_str, last_data_str).
        Falls back to empty strings when db/user is None (standalone mode).
        """
        if db is None or user is None:
            return "", "", ""

        try:
            from models import Measurement, MeasurementTypeEnum  # noqa

            def _q(order):
                return (
                    db.query(Measurement)
                    .filter_by(user_id=user.id)
                    .distinct(Measurement.type)
                    .order_by(Measurement.type.asc(), order)
                    .all()
                )

            first_measurement = (
                db.query(Measurement)
                .filter_by(user_id=user.id)
                .order_by(Measurement.created.asc())
                .first()
            )
            first_measurements = _q(Measurement.created.asc())
            current_measurements = _q(Measurement.created.desc())

            def _pick(lst, t):
                return next((x for x in lst if x.type == t), None)

            def _val(m):
                if m is None:
                    return "Not provided"
                v = m.get_value()
                return str(v) if v is not None else "Not provided"

            def _block(mlist):
                return {
                    "altura": _val(_pick(mlist, MeasurementTypeEnum.ALTURA)),
                    "peso": _val(_pick(mlist, MeasurementTypeEnum.PESO)),
                    "imc": _val(_pick(mlist, MeasurementTypeEnum.IMC)),
                    "hba1c": _val(_pick(mlist, MeasurementTypeEnum.HEMOGLOBINA_GLICADA)),
                    "pa": _val(_pick(mlist, MeasurementTypeEnum.PRESSAO_ARTERIAL)),
                    "hdl": _val(_pick(mlist, MeasurementTypeEnum.HDL)),
                    "ldl": _val(_pick(mlist, MeasurementTypeEnum.LDL)),
                    "framingham": _val(_pick(mlist, MeasurementTypeEnum.ESCORE_FRAMINGHAM)),
                    "ca": _val(_pick(mlist, MeasurementTypeEnum.CIRCUNFERENCIA_ABDOMINAL)),
                }

            cur = _block(current_measurements)
            fst = _block(first_measurements)

            def _fmt_block(b, label):
                return (
                    f"{label}:\n"
                    f"  Height: {b['altura']} cm  |  Weight: {b['peso']} kg  |  BMI: {b['imc']}\n"
                    f"  HbA1c: {b['hba1c']}  |  Blood Pressure: {b['pa']}\n"
                    f"  HDL: {b['hdl']}  |  LDL: {b['ldl']}\n"
                    f"  Framingham Score: {b['framingham']}  |  Waist Circumference: {b['ca']} cm\n"
                )

            primeira_consulta = ""
            first_data_str = ""
            if first_measurement and first_measurements:
                fist_date = first_measurement.created.strftime("%d/%m/%Y")
                primeira_consulta = _fmt_block(fst, f"First visit ({fist_date})")
                first_data_str = (
                    f"first visit - height: {fst['altura']}, weight: {fst['peso']}, "
                    f"BMI: {fst['imc']}, HbA1c: {fst['hba1c']}, BP: {fst['pa']}, "
                    f"HDL: {fst['hdl']}, LDL: {fst['ldl']}"
                )

            last_data_str = (
                f"recent data - height: {cur['altura']}, weight: {cur['peso']}, "
                f"BMI: {cur['imc']}, HbA1c: {cur['hba1c']}, BP: {cur['pa']}, "
                f"HDL: {cur['hdl']}, LDL: {cur['ldl']}"
            )

            recente = _fmt_block(cur, "Most recent visit")

            sobre_paciente = (
                (primeira_consulta + "\n" if primeira_consulta else "")
                + recente
            ) or "No clinical history recorded."

            return sobre_paciente, first_data_str, last_data_str

        except Exception:
            return "", "", ""

    # ------------------------------------------------------------------
    # System prompt (equivalent to get_instructions_2 in original)
    # ------------------------------------------------------------------

    def _build_system_prompt(self, user=None, db=None, extra_context: str = "") -> str:
        timezone = pytz.timezone("America/Fortaleza")
        current_datetime = datetime.now(tz=timezone).strftime("%d/%m/%Y %H:%M")
        day_of_week = self._traduzir_dia_semana(datetime.now().strftime("%A"))

        name = "User"
        sobre_paciente = ""
        if user is not None:
            name = f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip()
            sobre_paciente, _, _ = self.get_medical_data(user, db)

        # ── Clinical system prompt ───────────────────────────────────────────
        # The original had a redacted block ("PASTE HERE System Template Prompt").
        # This reconstructs the spirit of that prompt from what the code reveals:
        # Maria is an empathetic health assistant who knows the patient's clinical
        # history and can take actions (save measurements, set alarms, etc.)
        system = f"""You are MarIA, an empathetic and responsible virtual health assistant for the Viver Bem program.
You support patients with the ongoing monitoring of their health, well-being, and lifestyle.

Patient: {name}

=== CLINICAL HISTORY ===
{sobre_paciente if sobre_paciente else "No clinical history recorded yet."}

=== ADDITIONAL CONTEXT (profile and system-generated tasks) ===
{extra_context if extra_context else "No additional context available."}

=== TEMPORAL CONTEXT ===
Current date and time: {current_datetime}
Day of the week: {day_of_week}
If the user asks to schedule a reminder for a past date or time,
ask for a new date or time.

=== YOUR CAPABILITIES ===
You can perform real actions for the patient:
- Save clinical measurements (weight, BP, HbA1c, HDL, LDL, BMI, etc.)
- Create alarms and medication reminders
- Record scheduled appointments and exams
- Record preferences, restrictions, and eating habits
- Schedule proactive follow-up messages

Always confirm with the user before saving data, and clearly inform them when an action has been completed.
Be accurate, welcoming, and clinical when necessary. Never invent medical data."""

        return system.strip()

    # ------------------------------------------------------------------
    # Thread management (equivalent to OpenAI thread_id per phone)
    # ------------------------------------------------------------------

    def _get_thread(self, user_id: int) -> list[dict]:
        return self._threads.setdefault(user_id, [])

    def _push(self, user_id: int, role: str, content) -> None:
        self._threads.setdefault(user_id, []).append({"role": role, "content": content})
        # Rolling window of 30 messages (~15 turns)
        if len(self._threads[user_id]) > 30:
            self._threads[user_id] = self._threads[user_id][-30:]

    def clear_thread(self, user_id: int) -> None:
        self._threads[user_id] = []

    # ------------------------------------------------------------------
    # Agentic run loop  ←  this is the core that was missing before
    #
    # Mirrors the original:
    #   run = client.beta.threads.runs.create(...)
    #   while runStatus.status != 'completed':
    #       if runStatus.status == 'requires_action':
    #           for tool in tools_to_call:
    #               response = resolve_function(...)
    #               tool_output_response.append(...)
    #           runStatus = client.beta.threads.runs.submit_tool_outputs(...)
    # ------------------------------------------------------------------

    def run(
        self,
        content: str,
        user_id: int = 0,
        user=None,
        db=None,
        extra_context: str = "",
    ) -> str:
        """
        Send a patient message through the full agentic loop and return Maria's reply.

        The loop:
          1. Append user message to thread
          2. Call Groq with MARIA_TOOLS
          3. If finish_reason == 'tool_calls':
               - Execute each tool via resolve_function
               - Append tool results to thread
               - Call Groq again (repeat up to MAX_TOOL_ROUNDS)
          4. When finish_reason == 'stop', return the text response

        Parameters
        ----------
        content : str
            The patient's message text.
        user_id : int
            ID used to maintain per-user thread state.
        user : ORM User object or None
        db : SQLAlchemy session or None
        extra_context : str
            Current persona + task snapshot from llm-planner (injected into system prompt).
        """
        system_prompt = self._build_system_prompt(user=user, db=db, extra_context=extra_context)
        self._push(user_id, "user", content)

        for _round in range(self.MAX_TOOL_ROUNDS):
            messages = [{"role": "system", "content": system_prompt}] + self._get_thread(user_id)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=MARIA_TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1024,
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message

            # ── Tool calls requested (≡ requires_action in original) ──────────
            if finish_reason == "tool_calls":
                # Append the assistant's tool-call turn to thread
                self._push(user_id, "assistant", message.content or "")

                tool_results = []
                for tc in message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments
                    print(f"[Maria tool call] {fn_name}({fn_args})")

                    result = resolve_function(
                        function_name=fn_name,
                        arguments=fn_args,
                        user_id=user_id,
                        db=db,
                    )
                    print(f"[Maria tool result] {result}")

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # Append all tool results to thread so the model sees them
                for tr in tool_results:
                    self._threads[user_id].append(tr)

                # Loop: call model again with tool results
                continue

            # ── Normal completion ─────────────────────────────────────────────
            reply = (message.content or "").strip()
            self._push(user_id, "assistant", reply)
            return reply

        # Fallback if tool loop exhausted
        fallback = "Sorry, there was an internal error while processing your request. Please try again."
        self._push(user_id, "assistant", fallback)
        return fallback

    # Back-compat alias
    def chat(self, user_message: str, user_id: int = 0, user=None, db=None, extra_context: str = "") -> str:
        return self.run(content=user_message, user_id=user_id, user=user, db=db, extra_context=extra_context)
