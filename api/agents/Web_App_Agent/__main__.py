import uvicorn

from a2a.types import AgentSkill, AgentCard, AgentCapabilities
import click
from a2a.server.request_handlers import DefaultRequestHandler

from agents.Web_App_Agent.agent_executor import WebsiteBuilderSimpleAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

@click.command()
@click.option('--host', default='localhost', help='Host for the agent server')
@click.option('--port', default=10000, help='Port for the agent server')
def main(host: str, port: int):
    """
    Main function to create and run the website builder agent.
    """
    skill = AgentSkill(
        id="web_app_builder_skill",
        name="Web Development Specialist",
        description="Expert web developer that creates complete, functional web applications from natural language descriptions. Handles all web-related tasks including HTML, CSS, JavaScript, responsive design, and interactive features.",
        tags=[
            "web", "website", "webapp", "webpage", "web page", "web application", 
            "html", "css", "javascript", "frontend", "web development", "web design", 
            "calculator", "todo list", "landing page", "portfolio", "dashboard", 
            "form", "interactive", "responsive", "bootstrap", "ui", "ux"
        ],
        examples=[
            "Create a simple calculator web application",
            "Build a to-do list web app with add, delete, and complete functionality",
            "Generate a landing page for a startup with call-to-action button",
            "Make a portfolio website with image gallery",
            "Create a contact form with validation",
            "Build a dashboard with charts and data visualization",
            "Design a responsive e-commerce product page",
            "Create a blog layout with sidebar and comments"
        ]
    )

    agent_card = AgentCard(
        name="web_development_agent",
        description=(
            "Specialized web development agent that creates complete, functional web applications from natural language descriptions. "
            "Uses a sophisticated multi-agent workflow (planner, architect, coder) to deliver production-ready HTML/CSS/JavaScript code. "
            "Handles everything from simple static pages to complex interactive web applications with responsive design and modern UI/UX."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=WebsiteBuilderSimpleAgentExecutor(),
        task_store=InMemoryTaskStore()
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    uvicorn.run(server.build(), host=host, port=port)

if __name__ == "__main__":
    main()