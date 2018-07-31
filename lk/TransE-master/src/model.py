import math
import timeit
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from dataset import KnowledgeGraph


class TransE:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator, model, log):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        self.model = model
        self.logging=log
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)

        # 实体关系embedding
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

            # 投影矩阵
            if self.model == 'TransR':
                self.transfer_matrix = tf.get_variable(name="transfer_matrix",
                                                       shape=[kg.n_relation, self.embedding_dim * self.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        # 实体关系embedding标准化
        # with tf.name_scope('normalization'):
        #     self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
        #     self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            # 计算正负样例的分数值
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            # 计算loss值
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    # 计算正负样例的分数值
    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            # [batch_size,embedding_size]
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])

            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
            #self.logging.info(head_pos.shape, tail_pos.shape, relation_pos.shape)
            if self.model == 'TransR':
                matrix_pos = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, triple_pos[:, 2]),
                                        [-1, self.embedding_dim, self.embedding_dim])
                matrix_neg = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, triple_neg[:, 2]),
                                        [-1, self.embedding_dim, self.embedding_dim])
        with tf.name_scope('link'):
            if self.model == 'TransR':
                h_p = tf.reshape(tf.matmul(matrix_pos,tf.reshape(head_pos, [-1, self.embedding_dim, 1])),
                                 [-1, self.embedding_dim])
                t_p = tf.reshape(tf.matmul(matrix_pos,tf.reshape(tail_pos, [-1, self.embedding_dim, 1])),
                                 [-1, self.embedding_dim])
                h_n = tf.reshape(tf.matmul(matrix_neg,tf.reshape(head_neg, [-1, self.embedding_dim, 1])),
                                 [-1, self.embedding_dim])
                t_n = tf.reshape(tf.matmul(matrix_neg,tf.reshape(tail_neg, [-1, self.embedding_dim, 1])),
                                 [-1, self.embedding_dim])
                distance_pos = h_p + relation_pos - t_p
                distance_neg = h_n + relation_neg - t_n
            else:
                distance_pos = head_pos + relation_pos - tail_pos
                distance_neg = head_neg + relation_neg - tail_neg
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            # tf.summary.scalar("loss",loss)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])

            #self.logging.info(head.shape,tail.shape,relation.shape)
            if self.model == 'TransR':
                matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, eval_triple[2]),
                                        [self.embedding_dim, self.embedding_dim])
                #self.logging.info(matrix.shape)
        with tf.name_scope('link'):
            if self.model == 'TransR':

                h = tf.reshape(tf.matmul(matrix,tf.reshape(head, [self.embedding_dim, 1])),
                                 [self.embedding_dim])
                t = tf.reshape(tf.matmul(matrix,tf.reshape(tail, [self.embedding_dim, 1])),
                               [self.embedding_dim])

                distance_head_prediction = tf.transpose(tf.matmul(matrix,tf.transpose(self.entity_embedding))) + relation - t
                distance_tail_prediction = h + relation - tf.transpose(tf.matmul(matrix,tf.transpose(self.entity_embedding)))

            else:
                distance_head_prediction = self.entity_embedding + relation - tail
                distance_tail_prediction = head + relation - self.entity_embedding

        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)

        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer,epoch):
        start = timeit.default_timer()
        epoch_loss = 0

        for batch_pos, batch_neg in self.kg.get_training_batch(self.batch_size):
            batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss

        self.logging.info('\tepoch {:<4d} {:>5.2f} s\tloss:{:.3f}'.format(epoch,timeit.default_timer() - start,
                                                                           epoch_loss/self.kg.n_training_triple))

        # self.check_norm(session=session)

    def launch_evaluation(self, session, summary_writer, epoch):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        #self.logging.info('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(
                fetches=[self.idx_head_prediction, self.idx_tail_prediction], \
                feed_dict={self.eval_triple: eval_triple})

            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            # self.logging.info('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                    n_used_eval_triple,
            #                                                    self.kg.n_test_triple), end='\r')

        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        #self.logging.info('-----Joining all rank calculator-----')
        eval_result_queue.join()
        #self.logging.info('-----All rank calculation accomplished-----')
        #self.logging.info('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        self.logging.info('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple

        self.logging.info('Head prediction MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))

        self.logging.info('Tail prediction MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))

        self.logging.info('    Average     MeanRank: {:.3f}, Hits@10: {:.3f}\n'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        self.logging.info('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple

        self.logging.info('Head prediction MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))

        self.logging.info('Tail prediction MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))

        self.logging.info('    Average     MeanRank: {:.3f}, Hits@10: {:.3f}\n'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        self.logging.info('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        self.logging.info('-----Finish evaluation-----\n')

        summary = tf.Summary()
        summary.value.add(tag="raw_meanrank", simple_value=(head_meanrank_raw + tail_meanrank_raw) / 2)
        summary.value.add(tag="raw_hits10", simple_value=(head_hits10_raw + tail_hits10_raw) / 2)
        summary.value.add(tag="filter_meanrank", simple_value=(head_meanrank_filter + tail_meanrank_filter) / 2)
        summary.value.add(tag="filter_hits10", simple_value=(head_hits10_filter + tail_hits10_filter) / 2)
        # step代表横轴坐标
        summary_writer.add_summary(summary, epoch)

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        self.logging.info('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        self.logging.info('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
