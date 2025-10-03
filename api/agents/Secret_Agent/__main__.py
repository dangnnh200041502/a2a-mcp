import uvicorn

from a2a.types import AgentSkill, AgentCard, AgentCapabilities
import click
from a2a.server.request_handlers import DefaultRequestHandler

from agents.Secret_Agent.agent_executor import SecretAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

@click.command()
@click.option('--host', default='localhost', help='Host for the agent server')
@click.option('--port', default=10003, help='Port for the agent server')
def main(host: str, port: int):
    """
    Main function to create and run the secret agent.
    """
    skill = AgentSkill(
        id="secret_information_skill",
        name="Confidential Information Specialist",
        description="Expert in retrieving confidential information from private documents and databases. Specializes in accessing owner details, company information, personal data, and sensitive business intelligence that other agents cannot access.",
        tags=[
            "secret", "confidential", "private", "email", "owner", "information", 
            "personal", "business", "data", "retrieval", "search", "document",
            "GreenGrow", "TechWave", "company", "founder", "contact", "details",
            "sensitive", "internal", "proprietary", "classified", "secure"
        ],
        examples=[
            "What is the owner's email address?",
            "Who is the owner of this company?",
            "What is the owner's name and contact information?",
            "When was GreenGrow Innovations founded?",
            "What is TechWave and who owns it?",
            "Tell me about the company owner",
            "What confidential information do you have?",
            "Who founded this business?",
            "What are the owner's personal details?",
            "Show me the company's private information"
        ]
    )

    agent_card = AgentCard(
        name="confidential_information_agent",
        description=(
            "Specialized confidential information retrieval agent with exclusive access to private documents and databases. "
            "Capable of retrieving owner details, company information, personal data, and sensitive business intelligence "
            "that other agents cannot access. Uses advanced document search and data extraction techniques."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SecretAgentExecutor(),
        task_store=InMemoryTaskStore()
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    uvicorn.run(server.build(), host=host, port=port)

if __name__ == "__main__":
    main()
