from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.gpu_anomaly import GPUAnomalyMetadata, enrich_with_gpu_anomaly_scores
from src.incident_memory import IncidentMemory
from src.rca_engine import RCAAnalysis, analyze_incident
from src.reporting import build_rca_markdown
from src.vllm_client import is_vllm_configured, vllm_api_key, vllm_base_url, vllm_model


@dataclass
class AgentResult:
    analysis: RCAAnalysis
    report_markdown: str
    agent_trace: list[str]


class LearningAgent:
    name = "Continuous Learning Agent"

    def save_feedback(
        self,
        memory: IncidentMemory,
        incident_id: str,
        service: str,
        selected_root_cause: str,
        actual_root_cause: str | None,
        agent_root_cause: str,
        correctness: str,
        notes: str,
        evidence_summary: str,
    ) -> None:
        memory.save_feedback(
            incident_id=incident_id,
            service=service,
            selected_root_cause=selected_root_cause,
            actual_root_cause=actual_root_cause,
            agent_root_cause=agent_root_cause,
            correctness=correctness,
            notes=notes,
            evidence_summary=evidence_summary,
        )


class GPUTimeSeriesAnomalyAgent:
    name = "GPU Time-Series Anomaly Agent"

    def enrich(self, telemetry: pd.DataFrame) -> tuple[pd.DataFrame, GPUAnomalyMetadata]:
        return enrich_with_gpu_anomaly_scores(telemetry)


class UnifiedRCAAgent:
    """LangChain/vLLM agent that coordinates RCA tools and incident learning."""

    def __init__(self, memory: IncidentMemory) -> None:
        self.memory = memory
        self.learning_agent = LearningAgent()
        self.gpu_anomaly_agent = GPUTimeSeriesAnomalyAgent()

    def investigate(
        self,
        incident: dict[str, Any],
        telemetry: pd.DataFrame,
        reference_sources: list[dict[str, str]],
    ) -> AgentResult:
        trace = ["Accepted incident context"]
        if is_vllm_configured():
            result = self._run_langchain_agent(incident, telemetry, reference_sources)
            if result is not None:
                return result

        return self._run_local_agent(incident, telemetry, reference_sources, trace)

    def _run_local_agent(
        self,
        incident: dict[str, Any],
        telemetry: pd.DataFrame,
        reference_sources: list[dict[str, str]],
        trace: list[str] | None = None,
    ) -> AgentResult:
        trace = trace or ["Accepted incident context"]
        telemetry, gpu_metadata = self.gpu_anomaly_agent.enrich(telemetry)
        analysis = analyze_incident(incident, telemetry, self.memory, reference_sources)
        report = build_rca_markdown(incident, analysis, reference_sources)
        trace.extend(
            [
                (
                    "Agent tool: gpu_time_series_anomaly_detection "
                    f"({gpu_metadata.model} on {gpu_metadata.device}, rows={gpu_metadata.rows_scored})"
                ),
                "Agent tool: correlate_cross_tower_data",
                "Agent tool: retrieve_incident_memory",
                "Agent tool: retrieve_known_rca_patterns",
                "Agent tool: generate_grounded_rca",
                "Correlated compute, storage, network, and application evidence",
                "Generated grounded human-readable RCA",
                "Applied incident-memory learning adjustments",
            ]
        )
        return AgentResult(analysis=analysis, report_markdown=report, agent_trace=trace)

    def _run_langchain_agent(
        self,
        incident: dict[str, Any],
        telemetry: pd.DataFrame,
        reference_sources: list[dict[str, str]],
    ) -> AgentResult | None:
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None

        state: dict[str, Any] = {"analysis": None, "telemetry": telemetry}

        @tool
        def run_gpu_time_series_anomaly_detection(_: str = "") -> str:
            """Score multivariate telemetry anomalies with the GPU time-series anomaly agent."""
            enriched, metadata = self.gpu_anomaly_agent.enrich(telemetry)
            state["telemetry"] = enriched
            return json.dumps(
                {
                    "model": metadata.model,
                    "device": metadata.device,
                    "rows_scored": metadata.rows_scored,
                    "enabled": metadata.enabled,
                    "reason": metadata.reason,
                },
                indent=2,
            )

        @tool
        def correlate_cross_tower_data(_: str = "") -> str:
            """Correlate compute, storage, network, and application telemetry for the selected incident."""
            active_telemetry = state["telemetry"]
            if "gpu_anomaly_score" not in active_telemetry.columns:
                active_telemetry, _ = self.gpu_anomaly_agent.enrich(active_telemetry)
                state["telemetry"] = active_telemetry
            analysis = analyze_incident(incident, active_telemetry, self.memory, reference_sources)
            state["analysis"] = analysis
            return json.dumps(
                {
                    "primary_root_cause": analysis.primary.title,
                    "confidence": analysis.primary.confidence,
                    "evidence": [item.to_dict() for item in analysis.evidence],
                    "alternatives": [
                        {"title": item.title, "confidence": item.confidence}
                        for item in analysis.alternatives
                    ],
                },
                indent=2,
            )

        @tool
        def retrieve_incident_memory(_: str = "") -> str:
            """Retrieve resolved incident feedback used for continuous learning."""
            records = self.memory.load_all()
            return json.dumps(records[-5:], indent=2)

        @tool
        def generate_grounded_rca(_: str = "") -> str:
            """Generate the final human-readable RCA using current analysis and reference sources."""
            analysis = state["analysis"] or analyze_incident(
                incident,
                state["telemetry"],
                self.memory,
                reference_sources,
            )
            state["analysis"] = analysis
            return build_rca_markdown(incident, analysis, reference_sources)

        llm = ChatOpenAI(
            model=vllm_model(),
            base_url=vllm_base_url(),
            api_key=vllm_api_key(),
            temperature=0.1,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Unified Observability RCA Agent. Use tools to correlate "
                    "cross-tower data, retrieve incident memory, and generate grounded RCA. "
                    "Do not invent evidence.",
                ),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        tools = [
            run_gpu_time_series_anomaly_detection,
            correlate_cross_tower_data,
            retrieve_incident_memory,
            generate_grounded_rca,
        ]
        executor = AgentExecutor(
            agent=create_tool_calling_agent(llm, tools, prompt),
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
        )
        response = executor.invoke(
            {
                "input": (
                    "Investigate this incident end to end. First correlate cross-tower telemetry, "
                    "but run GPU time-series anomaly detection before correlation. "
                    "Then retrieve memory and generate the final RCA.\n"
                    f"Incident: {incident}"
                )
            }
        )
        analysis = state["analysis"] or analyze_incident(
            incident,
            state["telemetry"],
            self.memory,
            reference_sources,
        )
        report = str(response.get("output") or build_rca_markdown(incident, analysis, reference_sources))
        trace = [
            "LangChain AgentExecutor invoked with vLLM",
            "Tool: run_gpu_time_series_anomaly_detection",
            "Tool: correlate_cross_tower_data",
            "Tool: retrieve_incident_memory",
            "Tool: generate_grounded_rca",
            f"Intermediate tool steps: {len(response.get('intermediate_steps', []))}",
        ]
        return AgentResult(analysis=analysis, report_markdown=report, agent_trace=trace)

    def learn_from_feedback(
        self,
        incident: dict[str, Any],
        analysis: RCAAnalysis,
        selected_root_cause: str,
        correctness: str,
        notes: str,
    ) -> None:
        self.learning_agent.save_feedback(
            memory=self.memory,
            incident_id=incident["incident_id"],
            service=incident["service"],
            selected_root_cause=selected_root_cause,
            actual_root_cause=None,
            agent_root_cause=analysis.primary.title,
            correctness=correctness,
            notes=notes,
            evidence_summary=analysis.primary.summary,
        )
