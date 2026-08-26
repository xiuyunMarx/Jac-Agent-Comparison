import from jaclang.byllm.lib {Model}
glob llm = Model(model_name = "ollama_chat/glm-5.2");


node Plan{
    def make_plan(question: str) -> list[str] by llm();

}
node Explore{}
node Edit{}
node Verify{}
node Finish{}