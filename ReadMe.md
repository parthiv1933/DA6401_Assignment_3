---

# Recurrent Neural Network based Seq2Seq model

This repository provides a Python program for a sequence-to-sequence model built with PyTorch. It allows you to use different rnn cell types (LSTM, RNN, GRU) in both the encoder and decoder parts of the model, making it useful for various sequence prediction tasks.

## Classes


### Seq2Seq
- **Data Members**:
  - `output_index_size`: target vocabulary size.
  - `encoder`: Encoder class's instance.
  - `decoder`: Decoder class's instance.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `teacher_force_ratio`: teacher forcing's ratio during training. 

- **Methods**:
  - `__init__()`: Method to initialize seq2Seq model.
  - `forward()`: Method which performs forward pass through the seq2Seq model.


### Decoder
- **Data Members**:
  - `embedding_size`: Embedding layer size.
  - `bidirectional`: Boolean informing if the RNN is bidirectional.
  - `hidden_size`: Hidden state in the RNN size.
  - `num_layers`: Number of the layers in the RNN.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `input_size`: Input vocabulary size.
  - `output_size`: Size of the output vocabulary.
  - `dropout`: Regularization's dropout rate.

  
- **Methods**:
  - `__init__()`: Method to initialize decoder.
  - `forward()`: Method which performs forward pass through the decoder.

### Encoder
- **Data Members**:
 
  - `embedding_size`: Embedding layer size.
  - `bidirectional`: Boolean informing if the RNN is bidirectional.
  - `hidden_size`: Hidden state in the RNN size.
  - `num_layers`: Number of the layers in the RNN.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `input_size`: Input vocabulary size.
  - `dropout`: Regularization's dropout rate.
 
 
  
- **Methods**:
  - `__init__()`: Method to initialize encoder.
  - `forward()`: Method which performs forward pass through the encoder.


## Training functions

### beam_search()
- **Arguments**:
  - `model`: The Seq2Seq model we're training.
  - `max_length`: Input sequence maximum length.
  - `input_seq`: Input sequence for translation.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `output_index_reversed`: Integers to characters for output vocabulary's reverse mapping.
  - `beam_width`: Beam search's beam width.
  - `input_index`: Input vocabulary's mapping from characters to integers.
  - `output_index`: output vocabulary's mapping from characters to integers.
  - `length_penalty`: Beam search's length penalty.
 

- **Returns**:
  - `str`: Produced output sequence.


### train()
- **Arguments**:

  - `model`: The Seq2Seq model we're training.
  - `input_data`:  Input data batch for training.
  - `output_data`: Target data batch for training.
  - `optimizer`: Training's optimizer.
  - `output_index`: output vocabulary's mapping from characters to integers.
  - `output_index_reversed`: Integers to characters for output vocabulary's reverse mapping.
  - `cell_type`:  What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `max_len`: Sequences maximum length.
  - `val_input_data`: Input data batch in validation.
  - `val_output_data`: Target data batch validation.
  - `df_val`:  Validation data's dataFrame.
  - `num_epochs`: How many times we'll train the model on entire dataset.
  - `input_index`: Input vocabulary's mapping from characters to integers.
  - `beam_width`: Beam search's beam width.
  - `criterion`: Training's loss criterion.
  - `length_penalty`: Beam search's length penalty.
  - `wandb_log`: should we log to weights and biases [0 for no ,1 for yes].

- **Returns**:
  - `model`: Trained Seq2Seq model.
  - `beam_val`: Validation accuracy using beam search.



---

## Output Metrics

During training and validation, the following output metrics are provided:

- **Train Accuracy**: Character-level accuracy of predictions on training data.
- **Train Loss**: Average loss during training.
- **Validation Loss**: Average loss during validation.
- **Validation Accuracy Word**: Word-level accuracy of predictions on validation data using beam search..
- **Validation Accuracy Char**: Character-level accuracy of predictions on validation data.
- **Correct Prediction**: Number of correct predictions out of total validation samples.

These metrics provide insights into the Seq2Seq model's performance during training and validation. Character-level accuracy evaluates how accurately individual characters are predicted, while word-level accuracy assesses the correctness of entire output sequences.


## Usage

To train the Seq2Seq model with different RNN cell types, use the `train.py` script with the following command-line arguments:

| Argument                      | Description                                                               | Default Value                              |
| ----------------------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| -d, --datapath                | data folder path                                                         | '/kaggle/input/dakshina-dataset-ass-3'|
| -l, --lang                    | Language for which training is to be done                                | 'hi'                                       |
| -emb_size , --embadding_size  | Size of embedding                                                        | 512                                         |
| -hdn_size, --hidden_size      | Size of hidden                                                           | 512                                         |
| -nl, --num_layers             | Number of  the layers                                                    | 2                                           |
| -cell, --cell_type            | Cell type :RNN, LSTM or GRU                                              | 'GRU'                                      |
| -dp, --dropout                | Dropout rate                                                             | 0.3                                         |
| -lr, --learning_rate          | Learning rate                                                            | 0.001                                        |
| -b, --batch_size              | Batch size                                                               | 64                                          |
| -e, --epochs                  | Number of the epochs                                                     | 10                                          |
| -o, --optimizer               | Optimizer : adam, sgd, rmsprop, nadam or adagrad                         | 'adagrad'                                   |
| -bw, --beam_width             | Width of the beam search                                                 | 1                                           |
| -lp, --length_penalty         | Beam search's Length penalty                                             | 0.6                                         |
| -tfr, --teacher_forcing_ratio | Ratio of teacher forcing                                                 | 0.7                                         |
| -bi_dir, --bidirectional      | Use of bidirectional encoder or not                                      | True                                        |
| -wl, --wandb_log              | should we log to WandB (0 for no, 1 for yes)                             | 0                                           |
| -wp, --wandb_project          | Project name used to track experiments in weights and Biases dashboard   | 'DL_A3'                           |
| -we, --wandb_entity           | Wandb Entity used to track experiments in weights and biases dashboard   | 'cs24m030'                                  |


Example command to run the training script:

```bash
python train_vaniilla.py -d 'your/dataset/path/up/to//kaggle/input/dakshina-dataset-ass-3' -l 'hi'
```

---

# BASE Seq2Seq MODEL WITH ATTENTION

This repository provides a Python implementation of a sequence-to-sequence (Seq2Seq) model with an attention mechanism for sequence prediction tasks. Built with PyTorch, the model uses attention to focus on important parts of the input sequence during decoding.


## Usage

follow the steps below to to use this model with attention:

1. Import the necessary classes code.

2. Create an instance of the Attention class with the necessary parameter:

   - `hidden_size`: Hidden state's size.

3. Create an instance of the decoder class with necessary parameters:

  - `embedding_size`: Embedding layer size.
  - `bidirectional`: Boolean informing if the RNN is bidirectional.
  - `hidden_size`: Hidden state in the RNN size.
  - `num_layers`: Number of the layers in the RNN.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `input_size`: Input vocabulary size.
  - `output_size`: Size of the output vocabulary.
  - `dropout`: Regularization's dropout rate.

4. Create an instance of the encoder class with necessary parameters:

  - `embedding_size`: Embedding layer size.
  - `bidirectional`: Boolean informing if the RNN is bidirectional.
  - `hidden_size`: Hidden state in the RNN size.
  - `num_layers`: Number of the layers in the RNN.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `input_size`: Input vocabulary size.
  - `dropout`: Regularization's dropout rate.

5. create an instance of the Seq2Seq class with necessary parameters:

  - `output_index_size`: target vocabulary size.
  - `encoder`: Encoder class's instance.
  - `decoder`: Decoder class's instance.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `teacher_force_ratio`: teacher forcing's ratio during training. 

6. Train the Seq2Seq model using the `train()` function with the required arguments:

  - `model`: The Seq2Seq model we're training.
  - `input_data`:  Input data batch for training.
  - `output_data`: Target data batch for training.
  - `optimizer`: Training's optimizer.
  - `output_index`: output vocabulary's mapping from characters to integers.
  - `output_index_reversed`: Integers to characters for output vocabulary's reverse mapping.
  - `cell_type`:  What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `max_len`: Sequences maximum length.
  - `val_input_data`: Input data batch in validation.
  - `val_output_data`: Target data batch validation.
  - `df_val`:  Validation data's dataFrame.
  - `length_penalty`: Beam search's length penalty.
  - `num_epochs`: How many times we'll train the model on entire dataset.
  - `input_index`: Input vocabulary's mapping from characters to integers.
  - `beam_width`: Beam search's beam width.
  - `criterion`: Training's loss criterion.
  - `wandb_log`: should we log to weights and biases [0 for no ,1 for yes].

## Classes

### Seq2Seq

- **Data Members**:

  - `output_index_size`: target vocabulary size.
  - `encoder`: Encoder class's instance.
  - `decoder`: Decoder class's instance.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `teacher_force_ratio`: teacher forcing's ratio during training. 

- **Methods**:
  - `__init__()`: Method to initialize seq2Seq model.
  - `forward()`: Method which performs forward pass through the seq2Seq model.


### Attention

- **Data Members**:
  - `hidden_size`: Hidden state's size.

- **Methods**:
  
  - `dot_score()`: finds  dot product attention scores between decoder's hidden state and encoder's outputs.
  - `__init__()`: Method to initialize the Attention mechanism.
  - `forward()`: Method to performs forward pass through Attention mechanism.

### Decoder

- **Data Members**:

  - `embedding_size`: Embedding layer size.
  - `bidirectional`: Boolean informing if the RNN is bidirectional.
  - `hidden_size`: Hidden state in the RNN size.
  - `num_layers`: Number of the layers in the RNN.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `input_size`: Input vocabulary size.
  - `output_size`: Size of the output vocabulary.
  - `dropout`: Regularization's dropout rate.

- **Methods**:
  - `__init__()`: Method to initialize decoder.
  - `forward()`: Method which performs forward pass through the decoder.


### Encoder

- **Data Members**:

  - `embedding_size`: Embedding layer size.
  - `bidirectional`: Boolean informing if the RNN is bidirectional.
  - `hidden_size`: Hidden state in the RNN size.
  - `num_layers`: Number of the layers in the RNN.
  - `cell_type`: What type  of RNN cell used 'LSTM', 'GRU' or 'RNN'.
  - `input_size`: Input vocabulary size.
  - `dropout`: Regularization's dropout rate.
 
- **Methods**:
  - `__init__()`: Method to initialize encoder.
  - `forward()`: Method which performs forward pass through encoder.


## Output Metrics

During training and validation, the following output metrics are provided:

- **Train Accuracy**: Character-level accuracy of predictions on training data.
- **Train Loss**: Average loss during training.
- **Validation Loss**: Average loss during validation.
- **Validation Accuracy Word**: Word-level accuracy of predictions on validation data using beam search..
- **Validation Accuracy Char**: Character-level accuracy of predictions on validation data.
- **Correct Prediction**: Number of correct predictions out of total validation samples.

These metrics provide insights into the Seq2Seq model's performance during training and validation. Character-level accuracy evaluates how accurately individual characters are predicted, while word-level accuracy assesses the correctness of entire output sequences.



## Command-line Arguments

| Argument                      | Description                                                               | Default Value                              |
| ----------------------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| -d, --datapath                | data folder path                                                         | '/kaggle/input/dakshina-dataset-ass-3'|
| -l, --lang                    | Language for which training is to be done                                | 'hin'                                       |
| -emb_size , --embadding_size  | Size of embedding                                                        | 256                                         |
| -hdn_size, --hidden_size      | Size of hidden                                                           | 512                                         |
| -nl, --num_layers             | Number of  the layers                                                    | 2                                           |
| -cell, --cell_type            | Cell type :RNN, LSTM or GRU                                              | 'LSTM'                                      |
| -dp, --dropout                | Dropout rate                                                             | 0.5                                         |
| -lr, --learning_rate          | Learning rate                                                            | 0.005                                        |
| -b, --batch_size              | Batch size                                                               | 32                                          |
| -e, --epochs                  | Number of the epochs                                                     | 10                                          |
| -o, --optimizer               | Optimizer : adam, sgd, rmsprop, nadam or adagrad                         | 'adagrad'                                   |
| -bw, --beam_width             | Width of the beam search                                                 | 1                                           |
| -lp, --length_penalty         | Beam search's Length penalty                                             | 0.6                                         |
| -tfr, --teacher_forcing_ratio | Ratio of teacher forcing                                                 | 0.7                                         |
| -bi_dir, --bidirectional      | Use of bidirectional encoder or not                                      | True                                        |
| -wl, --wandb_log              | should we log to WandB (0 for no, 1 for yes)                             | 0                                           |
| -wp, --wandb_project          | Project name used to track experiments in weights and Biases dashboard   | 'DL_A3'                           |
| -we, --wandb_entity           | Wandb Entity used to track experiments in weights and biases dashboard   | 'cs24m030'                                  |


## Example Usage

```bash
python train_attention.py -d 'your/dataset/path/up/to//kaggle/input/dakshina-dataset-ass-3' -l 'hi'
```

