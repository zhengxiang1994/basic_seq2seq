# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# hyper parameters
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001


def extract_character_vocab(data):
    """
    construct mapping
    """
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    set_words = list(set([character for line in data.split('\n') for character in line]))
    # add special words to dictionary
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


def process_decoder_input(data, vocab_to_int, batch_size):
    """
    add <GO>, remove the last character
    :param data:
    :param vocab_to_int:
    :param batch_size:
    :return:
    """
    # cut掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


def pad_sentence_batch(sentence_batch, pad_int):
    """
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    :param pad_int: <PAD>对应索引号
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


def get_inputs():
    """
    model input
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    # define the max length of sequence
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size, encoding_embedding_size):
    """
    encoder layer
    :param input_data: input tensor
    :param rnn_size: the dimension of rnn hidden layer
    :param num_layers: the number of rnn cell
    :param source_sequence_length: the length of
    :param source_vocab_size: the size of source dictionary
    :param encoding_embedding_size: the size of embedding
    :return:
    """
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    cells = []
    for i in range(num_layers):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cells.append(lstm_cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length, dtype=tf.float32)
    return encoder_output, encoder_state


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    """
    decoder layer
    :param target_letter_to_int: the mapping of target data
    :param decoding_embedding_size: the size of embedding
    :param num_layers: the number of rnn cell
    :param rnn_size: the dimension of rnn hidden layer
    :param target_sequence_length: the length of target sequence
    :param max_target_sequence_length: the max length of target sequence
    :param encoder_state: the state of encoder
    :param decoder_input: the input of decoder
    :return:
    """
    # embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # construct rnn cells
    cells = []
    for i in range(num_layers):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cells.append(lstm_cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    # full connect layer, will get a trainable variable size [hidden_size x output_size]
    output_layer = tf.layers.Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # training decoder
    with tf.variable_scope('decoder'):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # predicting decoder, share parameters with training decoder
    with tf.variable_scope('decoder', reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     target_letter_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper, encoder_state, output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            impute_finished=True,
                                                                            maximum_iterations=max_target_sequence_length)
        return training_decoder_output, predicting_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size, encoder_embedding_size, decoder_embedding_size, rnn_size,
                  num_layers):
    # obtain the output of encoder
    _, encoder_state = get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                                         encoding_embedding_size)
    # the input of decoder after pre-process
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    # feed the state and input to decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)
    return training_decoder_output, predicting_decoder_output


if __name__ == '__main__':
    # load data
    with open('./data/letters_source.txt', 'r', encoding='utf-8') as f:
        source_data = f.read()

    with open('./data/letters_target.txt', 'r', encoding='utf-8') as f:
        target_data = f.read()

    # print(source_data)

    # construct mapping
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)
    # print(source_letter_to_int)

    # convert letter to int
    source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
                   for letter in line] for line in source_data.split('\n')]
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                   for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]
    # print(source_int[:5])
    # print(target_int[:5])

    # construct graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        # the input of model
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
        training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                           targets,
                                                                           lr,
                                                                           target_sequence_length,
                                                                           max_target_sequence_length,
                                                                           source_sequence_length,
                                                                           len(source_letter_to_int),
                                                                           len(target_letter_to_int),
                                                                           encoding_embedding_size,
                                                                           decoding_embedding_size,
                                                                           rnn_size,
                                                                           num_layers)
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    # train
    # 将数据集分割为train和validation
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]
    # 留出一个batch进行验证
    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
        get_batches(valid_target, valid_source, batch_size,
                    source_letter_to_int['<PAD>'],
                    target_letter_to_int['<PAD>']))

    display_step = 50  # 每隔50轮输出loss

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                source_letter_to_int['<PAD>'],
                                target_letter_to_int['<PAD>'])):

                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})

                if batch_i % display_step == 0:
                    # 计算validation loss
                    validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'.format(
                        epoch_i,
                        epochs,
                        batch_i,
                        len(train_source) // batch_size,
                        loss,
                        validation_loss[0]))

        # output sentences
        pred_idx = sess.run(predicting_logits, feed_dict={input_data: valid_targets_batch,
                                                          lr: learning_rate,
                                                          target_sequence_length: valid_sources_lengths * 2,
                                                          source_sequence_length: valid_sources_lengths})
        # print(pred_idx)
        for line in pred_idx[:10]:
            for idx in line:
                print(target_int_to_letter[idx], end='')
            print('\n')

        print('------ground-truth------')
        for line in valid_targets_batch[:10]:
            for idx in line:
                print(target_int_to_letter[idx], end='')
            print('\n')
