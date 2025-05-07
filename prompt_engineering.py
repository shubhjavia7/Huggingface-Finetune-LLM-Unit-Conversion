from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here.
        """
        messages = [
            {"role": "system", "content": "You are a helpful unit conversion assistant. Answer with step-by-step reasoning and include the numerical (only include decimals) final result inside <answer></answer>. Be concise."},
            {"role": "user", "content": "Can you convert 3 km to meters?"},
            {"role": "assistant", "content": "1 km is 1000 meters, so 3 km is 3 * 1000 which is 3000.0 meters. <answer>3000.0</answer>"},
            {"role": "user", "content": "What is 3 centuries as a measure of weeks"},
            {"role": "assistant", "content": "1 century is 100 years. 1 year is 52.1775 weeks. 3 centuries is 300 years. 300 years * 52.1775 weeks is 15,653.25 weeks. <answer>15653.25</answer>"},
            {"role": "user", "content": "How much is 10 pint when converted to milliliter?"},
            {"role": "assistant", "content": "1 US pint is approximately 473.176473 milliliters. So, 10 pints = 10 × 473.176473 which is 4731.76473 milliliters. <answer>4731.76473</answer>"},
            {"role": "user", "content": "Can you provide the conversion value from B to bit for 4 units?"},
            {"role": "assistant", "content": "1 byte (B) is 8 bits. So, 4 bytes = 4 × 8 = 32 bits. <answer>32.0</answer>"},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
