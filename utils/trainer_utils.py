'''
Contains utilities for `trainer.py`.
'''

from math import ceil
from time import perf_counter

import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from utils.helpers import count_words, show_exception, clear_stdout


class SummarizationDataset:
  '''
  Creates an iterable batched dataset of text (and summary) encodings.

  ## Parameters
  `texts`: List of texts
  `encoder`: Encoder used to encode `texts`
  `batch_size`: Maximum number of text encodings in a batch
  `summaries`: List of summaries
  `summary_max_tokens`: Maximum tokens in summary encodings
  `shuffle`: Shuffle batches before iterating
  `seed`: Manual seed for output reproducibility
  '''

  def __init__(
    self,
    texts: list[str],
    encoder: 'Encoder', # type: ignore
    batch_size: int,
    summaries: list[str] | None = None,
    summary_max_tokens: int = 0,
    shuffle: bool = False,
    seed: int | None = None
  ) -> None:
    # Check if texts and summaries are of same length
    # if summaries are provided
    if summaries is not None and len(texts) != len(summaries):
      raise ValueError('Length of "texts" and "summaries" must be equal')
    # This enables dynamic batching
    perm = np.argsort([count_words(text) for text in texts])
    texts = np.array(texts)[perm]
    if summaries is not None:
      summaries = np.array(summaries)[perm]
    # Store batches of texts and summaries in a numpy array
    num_batches = self.num_batches = ceil(len(texts) / batch_size)
    self.text_batches = np.zeros(num_batches, dtype=object)
    self.summary_batches = None if summaries is None \
      else np.zeros(num_batches, dtype=object)
    for i in range(num_batches):
      text_batch = texts[i*batch_size:(i+1)*batch_size].tolist()
      self.text_batches[i] = text_batch
      if summaries is not None:
        summary_batch = summaries[i*batch_size:(i+1)*batch_size].tolist()
        self.summary_batches[i] = summary_batch
    # Use numpy array as a cache
    self.cache = np.zeros(num_batches, dtype=object)
    self.encoder = encoder
    self.batch_size = batch_size
    self.summary_max_tokens = summary_max_tokens
    self.shuffle = shuffle
    self.seed = seed
    np.random.seed(seed)
    self.it = None

  def __len__(self) -> int:
    return self.num_batches

  def __getitem__(self, ind: int) -> BatchEncoding:
    encoder = self.encoder
    cache = self.cache
    # Check if input is cached
    if cache[ind]:
      return cache[ind]
    # Encode texts using encoder and summaries using tokenizer
    text_batches = self.text_batches
    summary_batches = self.summary_batches
    texts = text_batches[ind]
    encodings = encoder(texts)
    if summary_batches is not None:
      tokenizer = encoder.tokenizer
      summaries = summary_batches[ind]
      summ_encodings = tokenizer(
        summaries,
        padding=True,
        max_length=self.summary_max_tokens,
        truncation=True,
        return_tensors='pt'
      )['input_ids']
      # Set padding token ids to -100 (ignored id in attention)
      filt = summ_encodings == tokenizer.pad_token_id
      summ_encodings[filt] = -100
      encodings['labels'] = summ_encodings
    # Create batch encoding
    batch_encodings = BatchEncoding(encodings)
    # Save to cache and delete text batch
    cache[ind] = batch_encodings
    text_batches[ind] = 0
    if summary_batches is not None:
      summary_batches[ind] = 0
    return batch_encodings

  def __iter__(self):
    # Shuffle batches if specified
    if self.shuffle:
      permutation = np.random.permutation(self.num_batches)
      self.text_batches = self.text_batches[permutation]
      self.cache = self.cache[permutation]
      if self.summary_batches is not None:
        self.summary_batches = self.summary_batches[permutation]
    self.it = 0
    return self

  def __next__(self) -> BatchEncoding:
    # Check if iterator is initialized
    it = self.it
    if it is not None:
      raise ValueError('Iterator not initialized')
    # Check if iterations are completed
    if it == self.num_batches:
      raise StopIteration()
    self.it += 1
    return self[it]


def train_model(
  model,
  dataset: SummarizationDataset,
  epochs: int,
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
  device: str | torch.device = 'cpu',
  flt_prec: int = 4,
  spaces: int = 100
) -> list[int]:
  # Set model to training mode
  model = model.to(device)
  model.train(True)
  epoch_losses = []
  num_batches = len(dataset)
  for epoch in range(epochs):
    # Track total epoch loss and time
    epoch_loss = 0
    epoch_time = 0
    for batch, inputs in enumerate(dataset):
      try:
        start = perf_counter()
        inputs = inputs.to(device)
        loss = model(**inputs).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_taken = (perf_counter() - start) * 1000
      except Exception as e:
        show_exception(e)
        print('Training terminated')
        model.train(False)
        return epoch_losses, False
      epoch_loss += loss.item()
      epoch_time += time_taken
      # Calculate remaining time
      seconds = int(
        epoch_time * (num_batches * (epochs - epoch) / (batch + 1) - 1)
      ) // 1000
      minutes = seconds // 60
      hours = minutes // 60
      days = hours // 24
      time_remaining = f'{seconds % 60}s'
      if minutes:
        time_remaining = f'{minutes % 60}m {time_remaining}'
      if hours:
        time_remaining = f'{hours % 24}h {time_remaining}'
      if days:
        time_remaining = f'{days}d {time_remaining}'
      clear_stdout(spaces)
      print(
        f'Epoch [{epoch+1}/{epochs}]',
        f'Batch [{batch+1}/{num_batches}]',
        f'Time [{round(time_taken, flt_prec)} ms/batch]',
        f'Loss [{round(loss.item(), flt_prec)}]',
        f'Time remaining [{time_remaining}]',
        end = None
      )
    epoch_loss = epoch_loss / num_batches
    epoch_time = epoch_time / num_batches
    epoch_losses.append(epoch_loss)
    if scheduler is not None:
      scheduler.step(epoch_loss)
    clear_stdout(spaces)
    print(
      f'Epoch [{epoch+1}/{epochs}]',
      f'Average loss [{round(epoch_loss, flt_prec)}]',
      f'Avergage time [{round(epoch_time, flt_prec)} ms/batch]'
    )
  model.train(False)
  return epoch_losses, True
