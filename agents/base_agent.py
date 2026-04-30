import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Any
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("finsight.agents")

@dataclass
class AgentConfig:
    role: str
    goal: str
    backstory: str
    tools: list[Callable] = field(default_factory=list)
    temperature: float = 0.2
    verbose: bool = True

@dataclass
class Task:
    description: str
    agent_config: AgentConfig
    context: str = ""
    expected_output: str = ""

@dataclass
class TaskResult:
    task_description: str
    agent_role: str
    output: str
    success: bool

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        if config.verbose:
            log.info(f"Agent created: [{config.role}]")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    def _llm_call(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=self.config.temperature
        )
        return response.choices[0].message.content.strip()

    def _run_tools(self, task: Task) -> dict[str, Any]:
        results = {}
        for tool in self.config.tools:
            try:
                log.info(f"[{self.config.role}] Running tool: {tool.__name__}")
                result = tool()
                results[tool.__name__] = result
            except Exception as e:
                log.error(f"Tool {tool.__name__} failed: {e}")
                results[tool.__name__] = f"Tool failed: {e}"
        return results

    def execute(self, task: Task) -> TaskResult:
        log.info(f"[{self.config.role}] Executing: {task.description[:60]}")

        tool_output = ""
        if self.config.tools:
            tool_results = self._run_tools(task)
            tool_output = "\n".join([
                f"Tool '{k}' returned:\n{v}"
                for k, v in tool_results.items()
            ])

        system = f"""You are a {self.config.role}.
Your goal: {self.config.goal}
Background: {self.config.backstory}

Always be specific, factual, and concise.
Format your output clearly."""

        user_parts = [f"Task: {task.description}"]
        if task.context:
            user_parts.append(f"Context from previous agent:\n{task.context}")
        if tool_output:
            user_parts.append(f"Tool results:\n{tool_output}")
        if task.expected_output:
            user_parts.append(f"Expected output format: {task.expected_output}")

        user = "\n\n".join(user_parts)

        try:
            output = self._llm_call(system, user)
            log.info(f"[{self.config.role}] Task complete ({len(output)} chars)")
            return TaskResult(
                task_description=task.description,
                agent_role=self.config.role,
                output=output,
                success=True
            )
        except Exception as e:
            log.error(f"[{self.config.role}] Failed: {e}")
            return TaskResult(
                task_description=task.description,
                agent_role=self.config.role,
                output=f"Agent failed: {e}",
                success=False
            )

class Crew:
    def __init__(self, agents: list[Agent], tasks: list[Task], verbose: bool = True):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        log.info(f"Crew created: {len(agents)} agents, {len(tasks)} tasks")

    def run(self) -> list[TaskResult]:
        results = []
        previous_output = ""

        log.info("=" * 50)
        log.info("CREW KICKOFF")
        log.info("=" * 50)

        for i, task in enumerate(self.tasks):
            if self.verbose:
                log.info(f"\nTask {i+1}/{len(self.tasks)}: {task.description[:60]}")
                log.info(f"Assigned to: [{task.agent_config.role}]")

            task.context = previous_output

            agent = Agent(task.agent_config)
            result = agent.execute(task)
            results.append(result)

            previous_output = f"Output from [{result.agent_role}]:\n{result.output}"

            if self.verbose:
                log.info(f"[{result.agent_role}] Output preview: {result.output[:100]}...")

        log.info("=" * 50)
        log.info(f"CREW COMPLETE — {len(results)} tasks finished")
        log.info("=" * 50)

        return results