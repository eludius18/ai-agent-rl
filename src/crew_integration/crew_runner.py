from crewai import Crew
from crew_integration.tasks import analyze_task, rl_task, validate_task

def run_crew_flow():
    crew = Crew(
        agents=[
            analyze_task.agent,
            rl_task.agent,
            validate_task.agent
        ],
        tasks=[
            analyze_task,
            rl_task,
            validate_task
        ],
        verbose=True
    )
    result = crew.kickoff()
    print("\n🧠 Final Decision:\n", result)
