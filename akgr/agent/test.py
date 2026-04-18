from smolagents import Tool, CodeAgent, OpenAIServerModel


class EchoTool(Tool):
    name = "echo"
    description = "Returns the input text as-is."
    inputs = {
        "text": {"type": "string", "description": "Text to echo"}
    }
    output_type = "string"

    def forward(self, text: str) -> str:
        return text


if __name__ == "__main__":
    model = OpenAIServerModel(
        model_id="Qwen/Qwen3-235B-A22B",
        api_base="https://api.deepinfra.com/v1/openai",
        api_key="8b7BpAmXY0fLfQsQyF3lkFCvayTqHjdc",
    )
    agent = CodeAgent(
        tools=[EchoTool()],
        model=model,
    )
    result = agent.run("Use the echo tool to echo: 'hello world'")
    print(result)
