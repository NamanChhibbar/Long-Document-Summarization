import torch
from bert_score import BERTScorer


class Evaluator:

	def __init__(
			self, pipelines, texts_summaries: tuple[str]|list[tuple[str]],
			device: str|torch.device|None=None
		) -> None:
		if not isinstance(texts_summaries, list):
			texts_summaries = [texts_summaries]
		self.pipelines = pipelines
		self.texts = [pair[0] for pair in texts_summaries]
		self.summaries = [pair[1] for pair in texts_summaries]
		self.bert_scorer = BERTScorer(lang="en", device=device)
		self.generated_summaries = []
	
	def generate_summaries(self) -> None:
		summaries = self.generated_summaries
		for pipeline in self.pipelines:
			summary = pipeline(self.texts)
			summaries.extend(summary)

	def get_bertscore(self) -> list[torch.Tensor]:
		if not self.generated_summaries:
			print("Generating summaries")
			self.generate_summaries()
		summaries = self.summaries
		num_pipelines = len(self.pipelines)
		summaries *= num_pipelines
		metrics = self.bert_scorer.score(self.generated_summaries, summaries)
		metrics = [
			metric.reshape((num_pipelines, -1)).mean(dim=1)
			for metric in metrics
		]
		return metrics
